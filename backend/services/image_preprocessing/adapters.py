"""Optional real model adapters; weights are never downloaded implicitly."""
from pathlib import Path
import logging
from time import perf_counter
import numpy as np
from PIL import Image

_LOG = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda/mps without assuming an accelerator is available."""
    import torch
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "auto":
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable")
    return requested


def _match_source_size(image: Image.Image, source_size: tuple[int, int]) -> Image.Image:
    """Keep image-coordinate detections valid when a restoration model changes scale."""
    if image.size == source_size: return image
    return image.resize(source_size, Image.Resampling.LANCZOS)

class YoloDetector:
    model_id = "yolo26_face (local checkpoint)"
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        path=Path(checkpoint)
        if not path.is_file(): raise FileNotFoundError(f"face detector checkpoint not found: {path}")
        from ultralytics import YOLO
        self.model=YOLO(str(path))
        self.device=resolve_device(device)
    def detect(self,image:Image.Image,confidence:float):
        predictions=self.model.predict(np.asarray(image.convert("RGB")),conf=confidence,device=self.device,verbose=False)
        if not predictions: return []
        pred=predictions[0]
        if pred.boxes is None: raise RuntimeError("YOLO output has no boxes field")
        return [{"bbox_xyxy":b.xyxy[0].cpu().numpy().tolist(),"confidence":float(b.conf[0].cpu())} for b in pred.boxes]

class SpandrelRestorer:
    """Spandrel loader; xN super-resolution outputs are resampled to input size."""
    def __init__(self,checkpoint:str|Path,model_id:str,device:str="auto",tile_size:int=384,tile_pad:int=32):
        import torch
        from spandrel import ModelLoader
        self.model_id,self.torch=model_id,torch
        self.device=resolve_device(device)
        if tile_size < 64 or tile_pad < 0:
            raise ValueError("tile_size must be >=64 and tile_pad must be >=0")
        self.tile_size,self.tile_pad=int(tile_size),int(tile_pad)
        path=Path(checkpoint)
        if not path.is_file(): raise FileNotFoundError(f"general-restoration checkpoint not found: {path}")
        descriptor=ModelLoader().load_from_file(str(path))
        self.scale=int(descriptor.scale)
        self.model=descriptor.eval().to(self.device)

    def _restore_tiled(self, tensor):
        """Run super-resolution on overlapping tiles and downsample each tile immediately.

        x4 super-resolution can allocate a very large full-resolution output and feature maps
        for multi-megapixel inputs. Keeping only one padded tile's tensors alive bounds
        peak inference memory while the final image remains at source resolution.
        """
        width,height=tensor.shape[-1],tensor.shape[-2]
        scale=self.scale
        output=Image.new("RGB",(width,height))
        total=((width+self.tile_size-1)//self.tile_size)*((height+self.tile_size-1)//self.tile_size)
        tile_index=0
        started=perf_counter()
        for top in range(0,height,self.tile_size):
            bottom=min(top+self.tile_size,height)
            for left in range(0,width,self.tile_size):
                right=min(left+self.tile_size,width)
                padded_left=max(0,left-self.tile_pad)
                padded_top=max(0,top-self.tile_pad)
                padded_right=min(width,right+self.tile_pad)
                padded_bottom=min(height,bottom+self.tile_pad)
                patch=tensor[...,padded_top:padded_bottom,padded_left:padded_right]
                with self.torch.inference_mode():
                    restored=self.model(patch)
                crop_left=(left-padded_left)*scale
                crop_top=(top-padded_top)*scale
                crop_right=crop_left+(right-left)*scale
                crop_bottom=crop_top+(bottom-top)*scale
                core=restored[...,crop_top:crop_bottom,crop_left:crop_right]
                pixels=(core.squeeze(0).permute(1,2,0).float().clamp(0,1).mul(255).round()
                        .byte().cpu().numpy())
                tile=Image.fromarray(pixels,"RGB").resize((right-left,bottom-top),Image.Resampling.LANCZOS)
                output.paste(tile,(left,top))
                del patch,restored,core,pixels,tile
                tile_index+=1
        _LOG.info("%s: tiled restoration %sx%s as %s tiles (%s px, pad %s) in %.1f ms",
                  self.model_id,width,height,total,self.tile_size,self.tile_pad,
                  (perf_counter()-started)*1000)
        return output

    def restore(self,image:Image.Image):
        t=self.torch
        source=image.convert("RGB")
        x=t.from_numpy(np.asarray(source,dtype=np.float32)/255).permute(2,0,1).unsqueeze(0).to(self.device)
        if source.width*source.height > 1_000_000 and self.scale > 1:
            restored=self._restore_tiled(x)
        else:
            with t.inference_mode(): y=self.model(x)
            if isinstance(y,(tuple,list)): y=y[0]
            a=y.squeeze(0).permute(1,2,0).float().clamp(0,1).cpu().numpy()
            restored=Image.fromarray(np.rint(a*255).astype(np.uint8),"RGB")
        # M6 keeps detector coordinates in the restored image's original pixel space.
        del x
        return _match_source_size(restored,source.size)
