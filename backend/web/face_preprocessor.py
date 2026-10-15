"""Optional YOLO26 face detection utilities, isolated from the default FER flow.

The returned crop is an inspection/downstream candidate. It is deliberately not
fed into EMOCA by the current orchestrator until the coordinate mapping and
preprocessing A/B are accepted (ADR-021).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "yolo26_face" / "yolo26_face.pt"
MAX_FACES_PER_IMAGE = 30
ColorOrder = Literal["BGR", "RGB"]
ImageInput = Union[str, Path, Image.Image, NDArray[np.uint8]]


class NoFaceDetectedError(RuntimeError):
    """Raised when the detector returns no face above the configured threshold."""


class FaceDetector(Protocol):
    def predict(self, source: Any, *, conf: float, imgsz: int, verbose: bool = False) -> Any: ...


@dataclass(frozen=True)
class FaceDetectionResult:
    """Single selected face and image products, all image pixels are RGB."""

    annotated_image: Image.Image
    cropped_face: Image.Image
    bbox_xyxy: tuple[int, int, int, int]
    crop_bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    image_size: tuple[int, int]
    crop_size: tuple[int, int]
    inference_ms: float


def _load_rgb(image_input: ImageInput, array_color_order: ColorOrder) -> Image.Image:
    if isinstance(image_input, Image.Image):
        image = ImageOps.exif_transpose(image_input).convert("RGB")
    elif isinstance(image_input, (str, Path)):
        try:
            with Image.open(image_input) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Không đọc được ảnh đầu vào: {image_input}") from exc
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8 or image_input.ndim != 3 or image_input.shape[2] != 3:
            raise ValueError("NumPy image phải có dtype uint8 và shape H×W×3.")
        pixels = image_input if array_color_order == "RGB" else image_input[:, :, ::-1]
        image = Image.fromarray(np.ascontiguousarray(pixels), mode="RGB")
    else:
        raise TypeError("image_input phải là path, PIL Image hoặc numpy.ndarray uint8.")

    if image.width < 1 or image.height < 1:
        raise ValueError("Ảnh đầu vào rỗng.")
    return image


def _reading_order(boxes: list[tuple]) -> list[tuple]:
    """Order (x1, y1, x2, y2, conf) boxes top-to-bottom, then left-to-right within a row.

    Sorting on raw y1 scrambles a row because box tops jitter by a few pixels; boxes whose
    vertical centres lie within half the median face height of a row's first box share it.
    """
    if not boxes:
        return []
    heights = sorted(box[3] - box[1] for box in boxes)
    tolerance = heights[len(heights) // 2] / 2
    by_centre = sorted(boxes, key=lambda box: ((box[1] + box[3]) / 2, box[0]))
    rows: list[list[tuple]] = []
    for box in by_centre:
        centre = (box[1] + box[3]) / 2
        if rows and abs(centre - (rows[-1][0][1] + rows[-1][0][3]) / 2) <= tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [box for row in rows for box in sorted(row, key=lambda box: (box[0], -box[4]))]


class FaceDetectionPreprocessor:
    """Detect faces and return one confidence-selected crop plus annotation.

    Args:
        model_path: YOLO26 face checkpoint. The checkpoint is intentionally not
            downloaded implicitly; obtain and verify it using the documented
            model source before enabling this optional utility.
        conf_threshold: Minimum detection confidence, in [0, 1].
        detector: Optional YOLO-compatible detector, useful for integration
            adapters and deterministic tests without loading a checkpoint.
        imgsz: YOLO inference size (default 640).
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.5,
        detector: FaceDetector | None = None,
        imgsz: int = 640,
        crop_scale: float = 1.3,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold phải nằm trong [0, 1].")
        if imgsz < 1:
            raise ValueError("imgsz phải là số nguyên dương.")
        self.model_path = Path(model_path)
        self.conf_threshold = float(conf_threshold)
        if not math.isfinite(crop_scale) or crop_scale < 1.0:
            raise ValueError("crop_scale phải hữu hạn và >= 1.0.")
        self.crop_scale = float(crop_scale)
        self.imgsz = int(imgsz)
        if detector is not None:
            self.detector = detector
        else:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Thiếu checkpoint YOLO26-face: {self.model_path}. "
                    "Xem docs/RUNBOOK_DEMO.md phần M5-01 để chuẩn bị model."
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Cần cài dependency ultralytics trong runtime web để dùng YOLO26-face."
                ) from exc
            LOGGER.info("Loading YOLO26-face checkpoint: %s", self.model_path)
            self.detector = YOLO(str(self.model_path))

    def process(
        self,
        image_input: ImageInput,
        *,
        array_color_order: ColorOrder = "BGR",
    ) -> FaceDetectionResult:
        """Return RGB annotation/crop and selected face metadata in original pixels."""
        if array_color_order not in ("BGR", "RGB"):
            raise ValueError("array_color_order chỉ nhận 'BGR' hoặc 'RGB'.")
        image = _load_rgb(image_input, array_color_order)

        started = time.perf_counter()
        predictions = self.detector.predict(
            image, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        if predictions is None:
            raise RuntimeError("YOLO26-face trả output prediction None không hợp lệ.")
        try:
            prediction_count = len(predictions)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("YOLO26-face trả output prediction sai cấu trúc.") from exc
        if prediction_count == 0:
            raise NoFaceDetectedError("YOLO26-face không phát hiện khuôn mặt.")

        try:
            prediction = predictions[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("YOLO26-face không đọc được prediction đầu tiên.") from exc
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            raise RuntimeError("YOLO26-face trả prediction thiếu trường boxes.")
        boxes = prediction.boxes
        try:
            box_count = len(boxes)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("YOLO26-face trả danh sách boxes sai cấu trúc.") from exc
        if box_count == 0:
            raise NoFaceDetectedError("YOLO26-face không phát hiện khuôn mặt.")

        try:
            xyxy = np.asarray(boxes.xyxy.detach().cpu().numpy(), dtype=np.float64)
            confidences = np.asarray(boxes.conf.detach().cpu().numpy(), dtype=np.float64)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "YOLO26-face trả bounding box/confidence sai định dạng.") from exc
        if (xyxy.ndim != 2 or xyxy.shape[1:] != (4,) or
                confidences.ndim != 1 or len(xyxy) != len(confidences) or
                len(xyxy) != box_count):
            raise RuntimeError("YOLO26-face trả bounding box/confidence không hợp lệ.")
        if not np.isfinite(confidences).all() or np.any((confidences < 0) | (confidences > 1)):
            raise RuntimeError("YOLO26-face trả confidence không hữu hạn hoặc ngoài [0, 1].")

        # Single-face policy: choose highest confidence; stable first-index tie-break.
        selected = int(np.argmax(confidences))
        raw_box = np.asarray(xyxy[selected], dtype=np.float64)
        if raw_box.shape != (4,) or not np.isfinite(raw_box).all():
            raise RuntimeError("YOLO26-face trả bounding box không hữu hạn hoặc sai shape.")
        x1, y1, x2, y2 = (int(np.floor(raw_box[0])), int(np.floor(raw_box[1])),
                          int(np.ceil(raw_box[2])), int(np.ceil(raw_box[3])))
        x1 = min(max(x1, 0), image.width)
        y1 = min(max(y1, 0), image.height)
        x2 = min(max(x2, 0), image.width)
        y2 = min(max(y2, 0), image.height)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("Bounding box YOLO26-face rỗng sau khi giới hạn theo ảnh.")

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        crop_w, crop_h = (x2 - x1) * self.crop_scale, (y2 - y1) * self.crop_scale
        crop_x1, crop_y1 = math.floor(cx - crop_w / 2), math.floor(cy - crop_h / 2)
        crop_x2, crop_y2 = math.ceil(cx + crop_w / 2), math.ceil(cy + crop_h / 2)
        crop_x1 = min(max(crop_x1, 0), image.width)
        crop_y1 = min(max(crop_y1, 0), image.height)
        crop_x2 = min(max(crop_x2, 0), image.width)
        crop_y2 = min(max(crop_y2, 0), image.height)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise RuntimeError("Vùng crop YOLO26-face rỗng sau khi giới hạn theo ảnh.")

        confidence = float(confidences[selected])
        crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
        crop = image.crop(crop_bbox)
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 200, 255), width=max(2, image.width // 500))
        label = f"face {confidence:.2f}"
        try:
            font = ImageFont.load_default()
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            label_y = max(0, y1 - (bottom - top) - 4)
            draw.rectangle((x1, label_y, x1 + right - left + 6, label_y + bottom - top + 4), fill=(0, 200, 255))
            draw.text((x1 + 3, label_y + 2), label, fill=(0, 0, 0), font=font)
        except (AttributeError, OSError):  # defensive for older Pillow/font backends
            pass

        LOGGER.info(
            "YOLO26-face inference %.2f ms; selected bbox=%s conf=%.4f; crop=%dx%d RGB",
            inference_ms, (x1, y1, x2, y2), confidence, crop.width, crop.height,
        )
        return FaceDetectionResult(
            annotated_image=annotated,
            cropped_face=crop,
            bbox_xyxy=(x1, y1, x2, y2),
            crop_bbox_xyxy=crop_bbox,
            confidence=confidence,
            image_size=image.size,
            crop_size=crop.size,
            inference_ms=round(inference_ms, 3),
        )

    def process_many(
        self,
        image_input: ImageInput,
        *,
        array_color_order: ColorOrder = "BGR",
        max_faces: int = MAX_FACES_PER_IMAGE,
    ) -> list[FaceDetectionResult]:
        """Detect and crop up to 30 faces, ordered top-to-bottom then left-to-right.

        If more faces are present, the largest 30 are retained before assigning
        spatially stable order. Every returned bbox is expressed in source-image
        pixels, and each crop includes the configured padding around its face.
        """
        if array_color_order not in ("BGR", "RGB"):
            raise ValueError("array_color_order must be 'BGR' or 'RGB'.")
        if isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces < 1:
            raise ValueError("max_faces must be a positive integer.")
        max_faces = min(max_faces, MAX_FACES_PER_IMAGE)
        image = _load_rgb(image_input, array_color_order)

        started = time.perf_counter()
        predictions = self.detector.predict(
            image, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        if predictions is None:
            raise RuntimeError("YOLO26-face returned None predictions.")
        try:
            if len(predictions) == 0:
                raise NoFaceDetectedError("YOLO26-face did not detect any faces.")
            prediction = predictions[0]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("YOLO26-face returned an invalid prediction structure.") from exc
        if not hasattr(prediction, "boxes") or prediction.boxes is None:
            raise RuntimeError("YOLO26-face prediction has no boxes.")
        boxes = prediction.boxes
        try:
            box_count = len(boxes)
            xyxy = np.asarray(boxes.xyxy.detach().cpu().numpy(), dtype=np.float64)
            confidences = np.asarray(boxes.conf.detach().cpu().numpy(), dtype=np.float64)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError("YOLO26-face returned malformed boxes or confidences.") from exc
        if box_count == 0:
            raise NoFaceDetectedError("YOLO26-face did not detect any faces.")
        if (xyxy.ndim != 2 or xyxy.shape[1:] != (4,) or confidences.ndim != 1
                or len(xyxy) != len(confidences) or len(xyxy) != box_count):
            raise RuntimeError("YOLO26-face boxes and confidences have inconsistent shapes.")
        if not np.isfinite(xyxy).all() or not np.isfinite(confidences).all():
            raise RuntimeError("YOLO26-face returned non-finite coordinates or confidence.")
        if np.any((confidences < 0) | (confidences > 1)):
            raise RuntimeError("YOLO26-face confidence is outside [0, 1].")

        candidates: list[tuple[int, int, int, int, float]] = []
        for raw_box, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = (
                int(np.floor(raw_box[0])), int(np.floor(raw_box[1])),
                int(np.ceil(raw_box[2])), int(np.ceil(raw_box[3])),
            )
            x1, x2 = min(max(x1, 0), image.width), min(max(x2, 0), image.width)
            y1, y2 = min(max(y1, 0), image.height), min(max(y2, 0), image.height)
            if x2 > x1 and y2 > y1:
                candidates.append((x1, y1, x2, y2, float(confidence)))
        if not candidates:
            raise NoFaceDetectedError("YOLO26-face boxes were empty after clipping to the image.")

        # Match EMOCA's priority for the largest faces when the hard cap is hit.
        candidates.sort(key=lambda box: -((box[2] - box[0]) * (box[3] - box[1])))
        candidates = candidates[:max_faces]
        candidates = _reading_order(candidates)

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, image.width // 500)
        for index, (x1, y1, x2, y2, confidence) in enumerate(candidates, start=1):
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=(0, 200, 255), width=line_width)
            label = f"face {index} {confidence:.2f}"
            try:
                font = ImageFont.load_default()
                left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
                label_y = max(0, y1 - (bottom - top) - 4)
                draw.rectangle((x1, label_y, x1 + right - left + 6, label_y + bottom - top + 4),
                               fill=(0, 200, 255))
                draw.text((x1 + 3, label_y + 2), label, fill=(0, 0, 0), font=font)
            except (AttributeError, OSError):
                pass

        results: list[FaceDetectionResult] = []
        for x1, y1, x2, y2, confidence in candidates:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            crop_w, crop_h = (x2 - x1) * self.crop_scale, (y2 - y1) * self.crop_scale
            crop_x1, crop_y1 = math.floor(cx - crop_w / 2), math.floor(cy - crop_h / 2)
            crop_x2, crop_y2 = math.ceil(cx + crop_w / 2), math.ceil(cy + crop_h / 2)
            crop_x1 = min(max(crop_x1, 0), image.width)
            crop_y1 = min(max(crop_y1, 0), image.height)
            crop_x2 = min(max(crop_x2, 0), image.width)
            crop_y2 = min(max(crop_y2, 0), image.height)
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue
            crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
            crop = image.crop(crop_bbox)
            results.append(FaceDetectionResult(
                # All detections share one annotation so a 20-face image does
                # not retain 20 full-resolution copies of the same source.
                annotated_image=annotated,
                cropped_face=crop,
                bbox_xyxy=(x1, y1, x2, y2),
                crop_bbox_xyxy=crop_bbox,
                confidence=confidence,
                image_size=image.size,
                crop_size=crop.size,
                inference_ms=round(inference_ms, 3),
            ))
        if not results:
            raise NoFaceDetectedError("YOLO26-face could not produce a non-empty crop.")
        LOGGER.info("YOLO26-face detected %d faces (cap %d) in %.2f ms", len(results), max_faces, inference_ms)
        return results
