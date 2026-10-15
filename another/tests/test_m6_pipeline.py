import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from services.image_preprocessing.pipeline import ImagePipeline, PipelineConfig, process_batch
from services.image_preprocessing import cli as m6_cli
from services.image_preprocessing.adapters import SpandrelRestorer, _match_source_size

class Restore:
    model_id="fake-restorer"
    def __init__(self,events=None,color=(12,34,56),size=None): self.events=events; self.color=color; self.size=size; self.seen=[]
    def restore(self,image):
        if self.events is not None: self.events.append(("restore",))
        self.seen.append((image.mode,image.size))
        return Image.new("RGB",self.size or image.size,self.color)
class Detect:
    model_id="fake-detector"
    def __init__(self,boxes,events=None): self.boxes=boxes; self.events=events
    def detect(self,image,confidence):
        if self.events is not None: self.events.append(("detect",confidence,image.mode))
        return self.boxes
def setup(boxes,**cfg): return ImagePipeline(Restore(),Detect(boxes),PipelineConfig(**cfg))

def test_stage_order_rgb_and_defaults():
    events=[]; g=Restore(events); d=Detect([{"bbox_xyxy":[20,20,40,40],"confidence":.9}],events)
    result=ImagePipeline(g,d).process(np.zeros((100,100,3),dtype=np.uint8))
    assert result.status=="ok" and [x[0] for x in events]==["restore","detect"]
    assert events[1][1]==.5 and g.seen[0][0]=="RGB"

def test_realesrgan_output_is_resized_to_source_dimensions():
    assert _match_source_size(Image.new("RGB",(400,400)),(100,75)).size==(100,75)
    original=Image.new("RGB",(100,75)); assert _match_source_size(original,original.size) is original

def test_spandrel_tiled_restore_bounds_and_covers_large_image():
    import torch
    import torch.nn.functional as functional
    class X4(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls=[]
        def forward(self,x): self.calls.append(tuple(x.shape[-2:])); return functional.interpolate(x,scale_factor=4,mode="nearest")
    restorer=SpandrelRestorer.__new__(SpandrelRestorer); restorer.model_id="test-x4"; restorer.torch=torch; restorer.device="cpu"; restorer.model=X4(); restorer.scale=4; restorer.tile_size=32; restorer.tile_pad=4
    color=(45,110,205); source=Image.new("RGB",(70,50),color)
    tensor=torch.from_numpy(np.asarray(source,dtype=np.float32)/255).permute(2,0,1).unsqueeze(0)
    result=restorer._restore_tiled(tensor)
    assert result.size==(70,50) and result.mode=="RGB" and len(restorer.model.calls)==6
    assert max(h*w for h,w in restorer.model.calls)<=40*40
    assert max(abs(a-b) for a,b in zip(result.getpixel((35,25)),color))<=1

def test_crop_is_1_3x_centered_and_coordinates_returned():
    source=Image.new("RGB",(100,100),(8,9,10)); result=setup([{"bbox_xyxy":[30,30,50,50],"confidence":.8}]).process(source)
    face=result.faces[0]
    assert face.bbox_xyxy==[30,30,50,50] and face.crop_bbox_xyxy==[27,27,53,53] and face.crop.size==(26,26)
    assert face.crop.tobytes()==source.crop((27,27,53,53)).tobytes()

def test_clamp_all_four_edges():
    boxes=[[0,0,20,20],[80,0,100,20],[0,80,20,100],[80,80,100,100]]
    result=setup([{"bbox_xyxy":box,"confidence":1} for box in boxes]).process(Image.new("RGB",(100,100)))
    assert [f.crop_bbox_xyxy for f in result.faces]==[[0,0,23,23],[77,0,100,23],[0,77,23,100],[77,77,100,100]]

def test_no_face_keep_or_error():
    image=Image.new("RGB",(12,8)); keep=setup([]).process(image); err=setup([],no_face_policy="error").process(image)
    assert keep.status=="no_face" and keep.enhanced.size==image.size and err.status=="error" and "no_face" in err.error

def test_limit_long_side_without_upscale():
    pipeline=setup([]); large=pipeline.process(Image.new("RGB",(6000,3000))); small=pipeline.process(Image.new("RGB",(200,100)))
    assert large.enhanced.size==(4096,2048) and small.enhanced.size==(200,100)

def test_general_restoration_size_change_is_rejected_before_detection():
    events=[]; general=Restore(events,size=(400,400)); detector=Detect([],events)
    result=ImagePipeline(general,detector).process(Image.new("RGB",(100,100)))
    assert result.status=="error" and "preserve dimensions" in result.error and [e[0] for e in events]==["restore"]

def test_detection_confidence_is_filtered_and_validated():
    assert setup([{"bbox_xyxy":[1,1,5,5],"confidence":.49}]).process(Image.new("RGB",(10,10))).status=="no_face"
    malformed=setup([{"bbox_xyxy":[1,1,5,5],"confidence":float("nan")}]).process(Image.new("RGB",(10,10)))
    assert malformed.status=="error" and "confidence" in malformed.error

@pytest.mark.parametrize("config",[PipelineConfig(confidence=1.1),PipelineConfig(padding=float("nan")),PipelineConfig(max_side=0),PipelineConfig(numpy_color="XYZ")])
def test_invalid_pipeline_config_rejected(config):
    with pytest.raises(ValueError): ImagePipeline(Restore(),Detect([]),config)
@pytest.mark.parametrize("array",[np.zeros((2,2),np.uint8),np.zeros((2,2,4),np.uint8),np.zeros((2,2,3),np.float32)])
def test_bad_array_rejected(array): assert setup([]).process(array).status=="error"
def test_bbox_empty_rejected():
    result=setup([{"bbox_xyxy":[2,2,2,5],"confidence":.7}]).process(Image.new("RGB",(10,10)))
    assert result.status=="error" and "empty bbox" in result.error
def test_path_input_rgb():
    path=Path("database/image/archive/DATASET/test/2/test_2303_aligned.jpg")
    if not path.is_file(): pytest.skip("repository RAF-DB path fixture unavailable")
    assert setup([]).process(path).status=="no_face"
def test_batch_continues_after_a_bad_file():
    from pathlib import Path
    results=process_batch(setup([]),[Path("missing-m6-input.jpg"),Image.new("RGB",(10,10))])
    assert results[0][1].status=="error" and results[1][1].status=="no_face"

def test_cli_recurses_exports_results_and_continues_after_bad_input(tmp_path,monkeypatch):
    input_dir=tmp_path/"input"; (input_dir/"nested").mkdir(parents=True)
    Image.new("RGB",(12,10),(2,3,4)).save(input_dir/"nested"/"good.png"); (input_dir/"broken.jpg").write_bytes(b"not an image")
    output_dir=tmp_path/"output"
    for checkpoint in ("general.pth","face-yolo.pt"): (tmp_path/checkpoint).write_bytes(b"test checkpoint marker")
    class General(Restore):
        def __init__(self,*args,**kwargs): super().__init__()
    class Yolo(Detect):
        def __init__(self,*args,**kwargs): super().__init__([{"bbox_xyxy":[2,2,8,8],"confidence":.9}])
    monkeypatch.setattr(m6_cli,"SpandrelRestorer",General); monkeypatch.setattr(m6_cli,"YoloDetector",Yolo)
    code=m6_cli.main([str(input_dir),"--output",str(output_dir),"--general-weights",str(tmp_path/"general.pth"),"--yolo-weights",str(tmp_path/"face-yolo.pt")])
    assert code==1 and (output_dir/"enhanced_images"/"nested"/"good__png.png").is_file()
    good=json.loads((output_dir/"metadata"/"nested"/"good__png.json").read_text(encoding="utf-8"))
    broken=json.loads((output_dir/"metadata"/"broken__jpg.json").read_text(encoding="utf-8"))
    assert good["status"]=="ok" and good["face_count"]==1
    assert (output_dir/"faces"/"nested"/"good__png_face_01.png").is_file()
    assert len(good["model_artifacts"][0]["sha256"])==64
    assert broken["status"]=="error" and not (output_dir/"enhanced_images"/"broken__jpg.png").exists()
