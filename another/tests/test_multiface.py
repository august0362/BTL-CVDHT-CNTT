"""Focused M7 checks for multi-face limits, original-image crops, and serial inference."""
from io import BytesIO
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from web import orchestrator, predata_experiment as predata
from web.face_preprocessor import FaceDetectionPreprocessor, FaceDetectionResult, NoFaceDetectedError


class TensorLike:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
class Boxes:
    def __init__(self, xyxy, conf): self.xyxy, self.conf = TensorLike(xyxy), TensorLike(conf)
    def __len__(self): return len(self.xyxy.value)
class Prediction:
    def __init__(self, boxes): self.boxes = boxes
class Detector:
    def __init__(self, boxes, conf): self.boxes, self.conf = boxes, conf
    def predict(self, image, *, conf, imgsz, verbose=False): return [Prediction(Boxes(self.boxes,self.conf))]


def _detection(image, bbox):
    x1,y1,x2,y2=bbox
    crop=(x1-2,y1-2,x2+2,y2+2)
    return FaceDetectionResult(image.copy(), image.crop(crop), bbox, crop, .9, image.size,
                               (crop[2]-crop[0],crop[3]-crop[1]),1.)

def _views(count=2):
    source=Image.new("RGB",(64,48),(1,1,1))
    faces=[]
    for i in range(count):
        bbox=(5+i*24,5,20+i*24,20)
        faces.append(predata.MultiFaceItem(i+1,_detection(source,bbox),source.copy()))
    return predata.MultiFaceViews(source,faces=faces,detection_complete=True)

def _valid(marker):
    return {"status":"ok","faces":[{"predictions":[{"model_id":"ma3d_rafdb","status":"ok","top1":{"label":str(marker),"prob":.75}}]}]}


def test_process_many_caps_at_thirty_largest_then_assigns_spatial_order():
    boxes=[]; conf=[]
    for i in range(35):
        size=20+i; row,col=divmod(i,5); x,y=col*190+5,row*125+5
        boxes.append([x,y,x+size,y+size]); conf.append(.5+i/100)
    processor=FaceDetectionPreprocessor(detector=Detector(boxes,conf),conf_threshold=.5,crop_scale=1.3)
    result=processor.process_many(Image.new("RGB",(1000,1000)),max_faces=100)
    indices=list(range(5,35)); indices.sort(key=lambda i:(boxes[i][1],boxes[i][0],-conf[i]))
    assert len(result)==30 and [face.bbox_xyxy for face in result]==[tuple(boxes[i]) for i in indices]


def test_reading_order_ignores_small_vertical_jitter_within_a_row():
    # Real YOLO tops in one row differ by a few pixels (seen 2026-10-03: y1 = 0,1,1,1,2,4);
    # IDs must still run left-to-right per row, then top-to-bottom.
    tops=[0,1,1,1,2,4]; xs=[0,800,160,645,321,480]
    boxes=[[x,y,x+120,y+125] for x,y in zip(xs,tops)]+[[x,160+y,x+120,285+y] for x,y in zip(xs[::-1],tops)]
    processor=FaceDetectionPreprocessor(detector=Detector(boxes,[.9]*len(boxes)),conf_threshold=.5,crop_scale=1.3)
    result=processor.process_many(Image.new("RGB",(1000,500)),max_faces=30)
    centres=[((f.bbox_xyxy[1]+f.bbox_xyxy[3])//2>=150, f.bbox_xyxy[0]) for f in result]
    assert centres==sorted(centres), centres


def test_real_esrgan_boxes_keep_original_image_crop_pixels(monkeypatch):
    source=Image.new("RGB",(64,48))
    for y in range(48):
        for x in range(64): source.putpixel((x,y),(x,y,(x+y)%256))
    detection=_detection(Image.eval(source,lambda value:255-value),(8,8,24,24))
    class General:
        def restore(self,image): return Image.eval(image,lambda value:255-value)
    class FakeDetector:
        calls=0
        def process_many(self,image,*,max_faces):
            self.calls+=1
            if self.calls==1:
                assert image.getpixel((8,8))==source.getpixel((8,8))
                raise NoFaceDetectedError("no original face")
            assert image.getpixel((8,8)) != source.getpixel((8,8))
            return [detection]
    class Ctx:
        def __enter__(self): return self
        def __exit__(self,*args): return False
    stack=predata._ModelStack(general=General(),detector=FakeDetector())
    monkeypatch.setattr(predata,"_get_multiface_model_stack",lambda:stack)
    monkeypatch.setattr(predata,"_limit_windows_cpu_cores",lambda *_a,**_k:Ctx())
    views=predata._build_multiface_views(source,30)
    assert views.detection_source=="RealESRGAN + YOLO"
    assert views.faces[0].detection.cropped_face.tobytes()==source.crop(detection.crop_bbox_xyxy).tobytes()


def test_two_faces_run_serially_and_isolate_per_face_errors(monkeypatch):
    events=[]; views=_views(3); call_count=0
    def analyze(data,**kwargs):
        nonlocal call_count
        call_count+=1
        crop=Image.open(BytesIO(data)).convert("RGB")
        marker=crop.getpixel((0,0))[0]
        events.append(("analyze",marker,kwargs["model_ids"],kwargs["max_faces"]))
        if call_count==2: raise RuntimeError("face inference failed")
        return _valid(marker)
    monkeypatch.setattr(predata,"build_multiface_views",lambda *a,**k:views)
    monkeypatch.setattr(orchestrator,"analyze",analyze)
    state=orchestrator.analyze_multiface(views.source_image)
    assert state["kind"]=="faces" and [r["face_id"] for r in state["rows"]]==[1,2,3]
    assert [event[0] for event in events]==["analyze"]*3
    assert state["rows"][1]["result"]["error"]["message"]=="RuntimeError: face inference failed"


def test_single_and_zero_yolo_faces_do_not_enter_multiface_ma3d(monkeypatch):
    for count,kind in ((1,"single"),(0,"baseline")):
        views=_views(count)
        monkeypatch.setattr(predata,"build_multiface_views",lambda *a,**k:views)
        calls=[]
        monkeypatch.setattr(orchestrator,"analyze",lambda *a,**k:calls.append(k))
        state=orchestrator.analyze_multiface(views.source_image)
        assert state["kind"]==kind and state["rows"]==[] and calls==[]


def test_roll_correct_upload_applied_and_below_threshold(monkeypatch):
    source=Image.new("RGB",(80,40),(20,30,40))
    crop=Image.new("RGB",(20,30),(100,90,80))
    from types import SimpleNamespace
    views=SimpleNamespace(faces=[SimpleNamespace(detection=SimpleNamespace(cropped_face=crop))])
    class Fake:
        def __init__(self,roll,applied): self.roll,self.applied=roll,applied
        def correct_roll(self,image,threshold):
            assert image is crop and threshold==15
            return type("Correction",(),{"roll_deg":self.roll,"rotation_deg":-self.roll if self.applied else 0,
                "threshold_deg":threshold,"applied":self.applied,"image":image})()
    import services.frontalization as frontalization
    monkeypatch.setattr(orchestrator,"roll_correction_settings",lambda:(True,15.))
    monkeypatch.setattr(frontalization,"get_tddfa_v2_frontalizer",lambda:Fake(90,True))
    corrected,info=orchestrator.roll_correct_upload(source,views)
    assert corrected.size==(40,80) and info=={"roll_deg":90.,"rotation_deg":-90.,"threshold_deg":15.}
    assert corrected is not source
    expected=source.rotate(-90,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(20,30,40))
    assert corrected.tobytes()==expected.tobytes()
    monkeypatch.setattr(frontalization,"get_tddfa_v2_frontalizer",lambda:Fake(3,False))
    same,info=orchestrator.roll_correct_upload(source,views)
    assert same is source and info["roll_deg"]==3


def test_roll_correct_upload_disabled_face_count_and_exception(monkeypatch):
    source=Image.new("RGB",(80,40),(20,30,40))
    import services.frontalization as frontalization
    monkeypatch.setattr(orchestrator,"roll_correction_settings",lambda:(False,15.))
    assert orchestrator.roll_correct_upload(source,_views(1))==(source,None)
    monkeypatch.setattr(orchestrator,"roll_correction_settings",lambda:(True,15.))
    assert orchestrator.roll_correct_upload(source,_views(0))==(source,None)
    assert orchestrator.roll_correct_upload(source,_views(2))==(source,None)
    failed_detection=_views(1);failed_detection.detection_complete=False
    assert orchestrator.roll_correct_upload(source,failed_detection)==(source,None)
    class Broken:
        def correct_roll(self,*args): raise RuntimeError("fake 3DDFA failure")
    monkeypatch.setattr(frontalization,"get_tddfa_v2_frontalizer",lambda:Broken())
    unchanged,info=orchestrator.roll_correct_upload(source,_views(1))
    assert unchanged is source and info=={"error":"RuntimeError: fake 3DDFA failure"}


def test_multiface_roll_corrects_each_crop_before_analysis(monkeypatch):
    views=_views(2); received=[]
    class Fake:
        def correct_roll(self,image,threshold):
            received.append(image)
            return type("Correction",(),{"roll_deg":25.,"rotation_deg":-25.,"threshold_deg":threshold,
                                          "applied":True,"image":image.rotate(-25,expand=True)})()
    import services.frontalization as frontalization
    monkeypatch.setattr(orchestrator,"roll_correction_settings",lambda:(True,15.))
    monkeypatch.setattr(frontalization,"get_tddfa_v2_frontalizer",lambda:Fake())
    monkeypatch.setattr(orchestrator,"analyze",lambda data,**kwargs:_valid(Image.open(BytesIO(data)).size[0]))
    state=orchestrator.analyze_multiface(views.source_image,views=views)
    assert received==[item.detection.cropped_face for item in views.faces]
    assert all(row["roll_info"]["applied"] for row in state["rows"])
    assert all(row["processed"].size != views.faces[i].detection.cropped_face.size
               for i,row in enumerate(state["rows"]))
    class Broken:
        def correct_roll(self,*args): raise RuntimeError("face roll failed")
    monkeypatch.setattr(frontalization,"get_tddfa_v2_frontalizer",lambda:Broken())
    failed=orchestrator.analyze_multiface(views.source_image,views=views)
    assert all(row["roll_info"]["error"]=="RuntimeError: face roll failed" for row in failed["rows"])
    assert all(row["processed"] is item.detection.cropped_face for row,item in zip(failed["rows"],views.faces))


def test_removed_symbols_are_not_exported():
    from services.image_preprocessing import adapters, cli
    forbidden="Code"+"FormerRestorer"
    assert not hasattr(adapters,forbidden)
    assert not hasattr(cli,forbidden)
    assert not hasattr(predata,"restore_multiface_crop")

if __name__=="__main__":
    import pytest
    raise SystemExit(pytest.main([str(Path(__file__).resolve()),"-q"]))
