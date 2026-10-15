"""Opt-in per-face comparison with and without 3DDFA-V2 frontalization."""
from __future__ import annotations

import io
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from PIL import Image


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _roll_threshold_deg() -> float:
    """Read the provisional ADR-028 roll threshold from project configuration."""
    import yaml

    config = Path(__file__).resolve().parents[1] / "configs" / "frontalization.yaml"
    value = (yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("roll_threshold_deg")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 90:
        raise ValueError("configs/frontalization.yaml roll_threshold_deg must be in (0, 90]")
    return float(value)


def run_frontalization_experiment(views: Any, *, emoca: dict | None, engine_mode: str,
                                  progress_callback: Callable[[str], None] | None = None) -> dict:
    """Run crop -> EMOCA and crop -> roll-only 2D correction -> EMOCA per face."""
    faces = list(getattr(views, "faces", ()) or ())
    if not faces:
        return {"rows": [], "times_ms": {}, "errors": ["Không có YOLO face crop để chạy thử nghiệm."]}
    try:
        from services.frontalization import get_tddfa_v2_frontalizer
        from .orchestrator import analyze
    except ImportError:
        from services.frontalization import get_tddfa_v2_frontalizer
        from orchestrator import analyze

    rows: list[dict] = []
    aggregate: dict[str, float] = {}
    errors: list[str] = []
    frontalizer = None
    threshold_deg = _roll_threshold_deg()
    for index, item in enumerate(faces, start=1):
        crop = item.detection.cropped_face.convert("RGB")
        face_id = item.face_id
        row = {"face_id": face_id, "crop": crop, "corrected": None,
               "pose": None, "result_before": None, "result_after": None,
               "skipped_reason": None,
               "errors": {}, "times_ms": {"roll_estimate": 0.0,
               "analyze_before": 0.0, "analyze_after": 0.0}}
        def report(stage: str) -> None:
            if progress_callback:
                progress_callback(f"Mặt {index}/{len(faces)} · {stage}")
        # Keep both branches for this face contiguous; don't batch one stage over all faces.
        report("đang phân tích crop đối chứng…")
        started = perf_counter()
        try:
            row["result_before"] = analyze(_png(crop), emoca=emoca, engine_mode=engine_mode, max_faces=1)
        except Exception as exc:
            row["errors"]["analyze_before"] = f"{type(exc).__name__}: {exc}"
        row["times_ms"]["analyze_before"] = (perf_counter()-started)*1000

        report("đang đo roll bằng 3DDFA-V2…")
        started = perf_counter()
        try:
            if frontalizer is None:
                frontalizer = get_tddfa_v2_frontalizer()
            correction = frontalizer.correct_roll(crop, threshold_deg)
            row["corrected"] = correction.image
            row["pose"] = {"roll_deg": correction.roll_deg, "yaw_deg": correction.yaw_deg,
                           "pitch_deg": correction.pitch_deg, "applied": correction.applied,
                           "rotation_deg": correction.rotation_deg, "threshold_deg": correction.threshold_deg}
            row["times_ms"]["roll_estimate"] = float(correction.time_ms)
        except Exception as exc:
            row["errors"]["roll_estimate"] = f"{type(exc).__name__}: {getattr(exc, 'message', str(exc))}"
            row["times_ms"]["roll_estimate"] = (perf_counter()-started)*1000
        corrected_image = row["corrected"]
        if corrected_image is not None and not row["pose"]["applied"]:
            row["result_after"] = row["result_before"]
            row["skipped_reason"] = "roll_below_threshold"
        elif corrected_image is not None:
            report("đang phân tích crop đã sửa roll…")
            started = perf_counter()
            try:
                row["result_after"] = analyze(_png(corrected_image), emoca=emoca,
                                              engine_mode=engine_mode, max_faces=1)
            except Exception as exc:
                row["errors"]["analyze_after"] = f"{type(exc).__name__}: {exc}"
            row["times_ms"]["analyze_after"] = (perf_counter()-started)*1000
        else:
            row["errors"].setdefault("analyze_after", "Không có ảnh đã sửa roll để phân tích.")
        for key, value in row["times_ms"].items():
            aggregate[key] = aggregate.get(key, 0.0) + value
        for stage, message in row["errors"].items():
            errors.append(f"Face {face_id} · {stage}: {message}")
        rows.append(row)
    return {"rows": rows, "times_ms": {key: round(value, 2) for key, value in aggregate.items()}, "errors": errors}
