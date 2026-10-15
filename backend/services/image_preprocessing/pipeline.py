from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
import math
import numpy as np
from PIL import Image, ImageOps

class Detector(Protocol):
    model_id: str
    def detect(self, image: Image.Image, confidence: float) -> list[dict[str, Any]]: ...
class Restorer(Protocol):
    model_id: str
    def restore(self, image: Image.Image) -> Image.Image: ...

@dataclass
class PipelineConfig:
    confidence: float = 0.5
    padding: float = 0.15
    max_side: int = 4096
    no_face_policy: str = "keep"
    numpy_color: str = "RGB"
@dataclass
class FaceResult:
    bbox_xyxy: list[int]
    crop_bbox_xyxy: list[int]
    confidence: float
    crop: Image.Image
@dataclass
class PipelineResult:
    enhanced: Image.Image | None
    faces: list[FaceResult] = field(default_factory=list)
    input_dimensions: list[int] | None = None
    output_dimensions: list[int] | None = None
    model_ids: dict[str, str] = field(default_factory=dict)
    stage_times_ms: dict[str, float] = field(default_factory=dict)
    status: str = "error"
    error: str | None = None

def _input_rgb(image: str | Path | Image.Image | np.ndarray, numpy_color: str) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as src: return ImageOps.exif_transpose(src).convert("RGB").copy()
    if isinstance(image, Image.Image): return ImageOps.exif_transpose(image).convert("RGB").copy()
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("NumPy input must be non-empty HxWx3 uint8")
        if numpy_color not in ("RGB", "BGR"): raise ValueError("numpy_color must be RGB or BGR")
        arr = image if numpy_color == "RGB" else image[..., ::-1]
        return Image.fromarray(np.ascontiguousarray(arr), "RGB")
    raise TypeError("input must be a path, PIL image, or NumPy RGB array")

def _xyxy(box: Any, width: int, height: int) -> list[int]:
    v = np.asarray(box, dtype=np.float64).reshape(-1)
    if len(v) != 4 or not np.isfinite(v).all(): raise ValueError("detector returned malformed bbox_xyxy")
    x1, y1, x2, y2 = v
    x1, x2 = sorted((int(np.floor(x1)), int(np.ceil(x2))))
    y1, y2 = sorted((int(np.floor(y1)), int(np.ceil(y2))))
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1: raise ValueError("detector returned empty bbox")
    return [x1, y1, x2, y2]

def _padded(box: list[int], width: int, height: int, padding: float) -> list[int]:
    if not math.isfinite(padding) or padding < 0: raise ValueError("padding must be finite and non-negative")
    x1,y1,x2,y2 = box; bw,bh=x2-x1,y2-y1; cx,cy=(x1+x2)/2,(y1+y2)/2
    return [max(0,int(np.floor(cx-bw*(.5+padding)))), max(0,int(np.floor(cy-bh*(.5+padding)))),
            min(width,int(np.ceil(cx+bw*(.5+padding)))), min(height,int(np.ceil(cy+bh*(.5+padding))))]

def _limit(image: Image.Image, max_side: int) -> Image.Image:
    if max_side < 1: raise ValueError("max_side must be positive")
    w,h=image.size; scale=min(1.,max_side/max(w,h))
    return image if scale == 1 else image.resize((max(1,int(w*scale)),max(1,int(h*scale))),Image.Resampling.LANCZOS)

class ImagePipeline:  # standalone M6 processing orchestration
    def __init__(self, general_restorer: Restorer, detector: Detector, config: PipelineConfig|None=None):
        self.general_restorer,self.detector=general_restorer,detector
        self.config=config or PipelineConfig()
        if self.config.no_face_policy not in ("keep","error"): raise ValueError("no_face_policy must be keep or error")
        if not math.isfinite(self.config.padding) or self.config.padding < 0: raise ValueError("padding must be finite and non-negative")
        if not math.isfinite(self.config.confidence) or not 0 <= self.config.confidence <= 1: raise ValueError("confidence must be in [0, 1]")
        if self.config.max_side < 1: raise ValueError("max_side must be positive")
        if self.config.numpy_color not in ("RGB", "BGR"): raise ValueError("numpy_color must be RGB or BGR")
    def process(self, source: str|Path|Image.Image|np.ndarray) -> PipelineResult:
        r=PipelineResult(enhanced=None,model_ids={"general":self.general_restorer.model_id,"detector":self.detector.model_id})
        try:
            original=_input_rgb(source,self.config.numpy_color); r.input_dimensions=[original.width,original.height]
            t=perf_counter(); enhanced=self.general_restorer.restore(original.copy()); r.stage_times_ms["general_restoration"]=(perf_counter()-t)*1000
            if not isinstance(enhanced,Image.Image): raise TypeError("general restorer must return PIL.Image")
            enhanced=enhanced.convert("RGB")
            if enhanced.size != original.size: raise ValueError("general restoration must preserve dimensions; bbox mapping is undefined")
            r.enhanced=enhanced; r.output_dimensions=[enhanced.width,enhanced.height]
            t=perf_counter(); detections=self.detector.detect(enhanced.copy(),self.config.confidence); r.stage_times_ms["face_detection"]=(perf_counter()-t)*1000
            t=perf_counter()
            for d in detections:
                try:
                    confidence=float(d["confidence"])
                    if not math.isfinite(confidence) or not 0 <= confidence <= 1: raise ValueError("detector confidence must be finite and in [0, 1]")
                    if confidence < self.config.confidence: continue
                    box=_xyxy(d.get("bbox_xyxy"),enhanced.width,enhanced.height); cb=_padded(box,enhanced.width,enhanced.height,self.config.padding)
                    r.faces.append(FaceResult(box,cb,confidence,original.crop(tuple(cb)).convert("RGB")))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"detector returned malformed face: {exc}") from exc
            r.stage_times_ms["face_cropping"]=(perf_counter()-t)*1000
            t=perf_counter(); r.enhanced=_limit(enhanced,self.config.max_side); r.output_dimensions=[r.enhanced.width,r.enhanced.height]
            r.stage_times_ms["output_size_limit"]=(perf_counter()-t)*1000
            if not r.faces and self.config.no_face_policy == "error": raise RuntimeError("no_face")
            r.status="ok" if r.faces else "no_face"
        except Exception as e: r.status,r.error="error",f"{type(e).__name__}: {e}"
        return r

def process_batch(pipeline: ImagePipeline, sources):
    """Process every input independently; a bad file does not stop the batch."""
    return [(source, pipeline.process(source)) for source in sources]
