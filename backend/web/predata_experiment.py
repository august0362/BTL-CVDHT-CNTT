"""In-memory image views for optional preprocessing ablation in the web UI.

The model stack is initialized lazily once per Python process. No image or
intermediate result is written to disk by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable
import ctypes
import logging
import os

from PIL import Image, ImageDraw, ImageOps

from services.image_preprocessing.adapters import SpandrelRestorer, YoloDetector
from web.face_preprocessor import (
    MAX_FACES_PER_IMAGE,
    FaceDetectionPreprocessor,
    FaceDetectionResult,
    NoFaceDetectedError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG = logging.getLogger(__name__)
GENERAL_CHECKPOINT = PROJECT_ROOT / "model" / "realesr-general-x4v3" / "realesr-general-x4v3.pth"
YOLO_CHECKPOINT = PROJECT_ROOT / "model" / "yolo26_face" / "yolo26_face.pt"


@dataclass
class PredataViews:
    """Input, detector enhancement, and original-image YOLO crop."""

    source_image: Image.Image
    enhanced_image: Image.Image | None = None
    enhanced_detection: FaceDetectionResult | None = None
    branch_errors: dict[str, str | None] = field(default_factory=lambda: {
        "enhanced_detection": None,
    })
    stage_times_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class MultiFaceItem:
    face_id: int
    detection: FaceDetectionResult
    reference_image: Image.Image


@dataclass
class MultiFaceViews:
    source_image: Image.Image
    faces: list[MultiFaceItem] = field(default_factory=list)
    enhanced_image: Image.Image | None = None
    detection_source: str | None = None
    detection_complete: bool = False
    errors: list[str] = field(default_factory=list)
    stage_times_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class _ModelStack:
    general: Any = None
    yolo: Any = None
    detector: FaceDetectionPreprocessor | None = None
    load_errors: dict[str, str] = field(default_factory=dict)


_STACK: _ModelStack | None = None
_STACK_LOCK = Lock()


def _build_model_stack() -> _ModelStack:
    """Load local adapters for the optional detector-stage panel."""
    stack = _ModelStack()
    _load_stack_components(stack, ("general", "detector"))
    return stack


def _load_stack_components(stack: _ModelStack, components: tuple[str, ...]) -> None:
    """Load requested adapters once; one unavailable model does not disable others."""
    if "general" in components and stack.general is None and "general" not in stack.load_errors:
        try:
            stack.general = SpandrelRestorer(
                GENERAL_CHECKPOINT,
                "realesr-general-x4v3 via Spandrel (same-size output)",
            )
        except Exception as exc:
            stack.load_errors["general"] = f"{type(exc).__name__}: {exc}"
    if "detector" in components and stack.detector is None and "detector" not in stack.load_errors:
        try:
            stack.yolo = YoloDetector(YOLO_CHECKPOINT)
            # Reuse the model already loaded by the M6 adapter.
            stack.detector = FaceDetectionPreprocessor(
                detector=stack.yolo.model,
                conf_threshold=0.5,
                crop_scale=1.3,
            )
        except Exception as exc:
            stack.load_errors["detector"] = f"{type(exc).__name__}: {exc}"


def _get_model_stack() -> _ModelStack:
    global _STACK
    if _STACK is None:
        with _STACK_LOCK:
            if _STACK is None:
                _STACK = _ModelStack()
    with _STACK_LOCK:
        _load_stack_components(_STACK, ("general", "detector"))
    return _STACK


def _get_multiface_model_stack() -> _ModelStack:
    """Load YOLO only; RealESRGAN is initialized after an original-image miss."""
    global _STACK
    if _STACK is None:
        with _STACK_LOCK:
            if _STACK is None:
                _STACK = _ModelStack()
    with _STACK_LOCK:
        _load_stack_components(_STACK, ("detector",))
    return _STACK


def _elapsed(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)


def _crops_from_original(detections: list[FaceDetectionResult], source: Image.Image) -> list[FaceDetectionResult]:
    """Retain YOLO metadata while taking downstream pixels from the upload."""
    return [replace(item, cropped_face=source.crop(item.crop_bbox_xyxy).convert("RGB"),
                    crop_size=source.crop(item.crop_bbox_xyxy).size) for item in detections]


def _reference_preview(source: Image.Image, bbox: tuple[int, int, int, int], face_id: int) -> Image.Image:
    """Create a compact original-photo thumbnail with one face location marked."""
    scale = min(512 / source.width, 512 / source.height, 1.0)
    size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    preview = source.resize(size, Image.Resampling.LANCZOS) if size != source.size else source.copy()
    x1, y1, x2, y2 = bbox
    mapped = tuple(round(value * scale) for value in (x1, y1, x2, y2))
    draw = ImageDraw.Draw(preview)
    draw.rectangle((mapped[0], mapped[1], mapped[2] - 1, mapped[3] - 1),
                   outline=(0, 200, 255), width=max(2, round(3 * scale)))
    draw.text((mapped[0] + 3, max(0, mapped[1] - 13)), f"Face {face_id}", fill=(0, 80, 160))
    return preview


@contextmanager
def _limit_windows_cpu_cores(limit: int = 4):
    """Keep optional CPU restoration from occupying every logical CPU on Windows."""
    original_mask = None
    kernel32 = None
    try:
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_current = kernel32.GetCurrentProcess
            get_current.restype = ctypes.c_void_p
            get_mask = kernel32.GetProcessAffinityMask
            get_mask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            get_mask.restype = ctypes.c_int
            set_mask = kernel32.SetProcessAffinityMask
            set_mask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            set_mask.restype = ctypes.c_int
            handle = get_current()
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            if get_mask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                original_mask = process_mask.value
                remaining = original_mask
                limited_mask = 0
                for _ in range(max(1, limit)):
                    bit = remaining & -remaining
                    if not bit:
                        break
                    limited_mask |= bit
                    remaining ^= bit
                if limited_mask and limited_mask != original_mask:
                    if set_mask(handle, ctypes.c_size_t(limited_mask)):
                        _LOG.info("Pre-data CPU affinity limited to %s logical cores", limited_mask.bit_count())
                    else:
                        original_mask = None
        yield
    finally:
        if original_mask is not None and kernel32 is not None:
            try:
                kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(original_mask))
            except Exception:
                pass


def build_predata_views(image: Image.Image) -> PredataViews:
    """Build views while limiting CPU thread fan-out so the web host stays responsive."""
    import torch

    original_threads = torch.get_num_threads()
    limit_threads = not torch.cuda.is_available() and original_threads > 4
    with _limit_windows_cpu_cores(4):
        if limit_threads:
            torch.set_num_threads(4)
        try:
            return _build_predata_views(image)
        finally:
            if limit_threads:
                torch.set_num_threads(original_threads)


def build_multiface_views(
    image: Image.Image,
    max_faces: int = MAX_FACES_PER_IMAGE,
    progress_callback: Callable[[str], None] | None = None,
) -> MultiFaceViews:
    """Run YOLO on same-size enhanced pixels; crop the original upload downstream."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces < 1:
        raise ValueError("max_faces must be a positive integer")
    max_faces = min(max_faces, MAX_FACES_PER_IMAGE)
    import torch

    original_threads = torch.get_num_threads()
    limit_threads = not torch.cuda.is_available() and original_threads > 4
    with _limit_windows_cpu_cores(4):
        if limit_threads:
            torch.set_num_threads(4)
        try:
            return _build_multiface_views(image, max_faces, progress_callback)
        finally:
            if limit_threads:
                torch.set_num_threads(original_threads)


def _build_multiface_views(
    image: Image.Image,
    max_faces: int,
    progress_callback: Callable[[str], None] | None = None,
) -> MultiFaceViews:
    source = ImageOps.exif_transpose(image).convert("RGB").copy()
    result = MultiFaceViews(source_image=source)
    stack = _get_multiface_model_stack()
    detector = stack.detector

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    detections: list[FaceDetectionResult] = []
    inference_failed = False
    if detector is None:
        result.errors.append(stack.load_errors.get("detector", "YOLO face detector unavailable"))
        inference_failed = True
    else:
        report("Đang chạy YOLO trên ảnh gốc…")
        started = perf_counter()
        try:
            detections = detector.process_many(source.copy(), max_faces=max_faces)
            detections = _crops_from_original(detections, source)
            result.detection_source = "YOLO trên ảnh gốc"
            result.stage_times_ms["yolo"] = _elapsed(started)
        except NoFaceDetectedError as exc:
            result.errors.append(f"YOLO trên ảnh gốc: {exc}")
            # ADR-028: restoration is loaded and run only after a confirmed zero-face result.
            report("YOLO không thấy mặt; đang chạy RealESRGAN để thử lại…")
            with _STACK_LOCK:
                _load_stack_components(stack, ("general",))
            if stack.general is None:
                result.errors.append(stack.load_errors.get("general", "RealESRGAN unavailable"))
                inference_failed = True
            else:
                started = perf_counter()
                try:
                    enhanced = stack.general.restore(source.copy())
                    if not isinstance(enhanced, Image.Image) or enhanced.size != source.size:
                        raise ValueError("RealESRGAN must return a PIL image at source dimensions")
                    enhanced = enhanced.convert("RGB")
                    result.enhanced_image = enhanced
                except Exception as exc:
                    result.errors.append(f"RealESRGAN: {type(exc).__name__}: {exc}")
                    result.stage_times_ms["realesrgan"] = _elapsed(started)
                    inference_failed = True
                else:
                    result.stage_times_ms["realesrgan"] = _elapsed(started)
                    report("Đang chạy YOLO trên ảnh đã làm nét…")
                    started = perf_counter()
                    try:
                        detections = detector.process_many(enhanced.copy(), max_faces=max_faces)
                        detections = _crops_from_original(detections, source)
                        result.detection_source = "RealESRGAN + YOLO"
                    except NoFaceDetectedError as exc:
                        result.errors.append(f"RealESRGAN + YOLO: {exc}")
                    except Exception as exc:
                        result.errors.append(f"RealESRGAN + YOLO: {type(exc).__name__}: {exc}")
                        inference_failed = True
                    result.stage_times_ms["yolo"] = _elapsed(started)
        except Exception as exc:
            # A broken original-image YOLO pass is not a zero-face result and must not invoke ESRGAN.
            result.errors.append(f"YOLO trên ảnh gốc: {type(exc).__name__}: {exc}")
            result.stage_times_ms["yolo"] = _elapsed(started)
            inference_failed = True
        if not detections:
            result.stage_times_ms.setdefault("yolo", 0.0)
        result.detection_complete = bool(detections) or not inference_failed

    if detections:
        if len(detections) == 1:
            detection = detections[0]
            reference = _reference_preview(source, detection.bbox_xyxy, 1)
            result.faces.append(MultiFaceItem(1, detection, reference))
            return result

        for face_id, detection in enumerate(detections, start=1):
            result.faces.append(MultiFaceItem(
                face_id=face_id,
                detection=detection,
                reference_image=_reference_preview(source, detection.bbox_xyxy, face_id),
            ))
        return result

    return result


def _build_predata_views(image: Image.Image) -> PredataViews:
    """Build detector enhancement and original-image crop views in RAM.

    Args:
        image: Input PIL image in any supported mode. All returned images are
            independent RGB PIL images.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    source = ImageOps.exif_transpose(image).convert("RGB").copy()
    result = PredataViews(source_image=source)
    stack = _get_model_stack()

    started = perf_counter()
    if stack.general is None:
        result.branch_errors["enhanced_detection"] = stack.load_errors.get(
            "general", "general restorer unavailable"
        )
    else:
        try:
            enhanced = stack.general.restore(source.copy())
            if not isinstance(enhanced, Image.Image):
                raise TypeError("general restorer must return a PIL image")
            enhanced = enhanced.convert("RGB")
            if enhanced.size != source.size:
                raise ValueError("general restorer changed image dimensions")
            result.enhanced_image = enhanced
        except Exception as exc:
            result.branch_errors["enhanced_detection"] = f"{type(exc).__name__}: {exc}"
    result.stage_times_ms["general_restoration"] = _elapsed(started)

    if result.enhanced_image is not None:
        started = perf_counter()
        if stack.detector is None:
            result.branch_errors["enhanced_detection"] = stack.load_errors.get(
                "detector", "detector model unavailable"
            )
        else:
            try:
                result.enhanced_detection = _crops_from_original(
                    [stack.detector.process(result.enhanced_image.copy())], source
                )[0]
            except Exception as exc:
                result.branch_errors["enhanced_detection"] = f"{type(exc).__name__}: {exc}"
        result.stage_times_ms["enhanced_detection"] = _elapsed(started)
    else:
        result.stage_times_ms["enhanced_detection"] = 0.0

    return result
