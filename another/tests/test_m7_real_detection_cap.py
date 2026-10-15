"""Heavy local-only real YOLO cap acceptance; scenes and measurements stay in RAM (venv web)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

MODEL = ROOT / "backend" / "model" / "yolo26_face" / "yolo26_face.pt"
DATASET = ROOT / "database" / "image" / "archive" / "DATASET" / "test"
CELL = 128
GAP = 32


def _dataset_images() -> list[Path]:
    return sorted(path for path in DATASET.rglob("*")
                  if path.is_file() and path.name.lower().endswith("_aligned.jpg"))


def _scene(face_count: int, faces: list[Image.Image]) -> Image.Image:
    columns = min(6, face_count) if face_count else 1
    rows = max(1, (face_count + columns - 1) // columns)
    width = columns * CELL + (columns - 1) * GAP
    height = rows * CELL + (rows - 1) * GAP
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    for index, face in enumerate(faces[:face_count]):
        row, col = divmod(index, columns)
        canvas.paste(face, (col * (CELL + GAP), row * (CELL + GAP)))
    return canvas


def main() -> int:
    if not MODEL.is_file() or not DATASET.is_dir():
        print("SKIP test_m7_real_detection_cap: YOLO checkpoint or RAF-DB aligned test directory is missing")
        return 0
    paths = _dataset_images()
    if len(paths) < 36:
        print(f"SKIP test_m7_real_detection_cap: need 36 aligned RAF-DB crops, found {len(paths)}")
        return 0

    from web.orchestrator import prepare_multiface_views

    faces = []
    for path in paths[:36]:
        with Image.open(path) as source:
            faces.append(source.convert("RGB").resize((CELL, CELL), Image.Resampling.LANCZOS))

    measurements = []
    scenes = (("0", 0), ("1", 1), ("2", 2), ("30", 30), ("36", 36))
    for scene_name, pasted in scenes:
        image = _scene(pasted, faces)
        started = time.perf_counter()
        views = prepare_multiface_views(image, max_faces=30)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        count = len(views.faces)
        assert count <= 30, f"scene {scene_name}: detector returned {count} faces (cap is 30)"
        if pasted == 0:
            assert count == 0, f"zero-face scene returned {count} faces"
            assert views.detection_complete is True, "zero-face detection did not complete"
        if pasted == 36 and count >= 30:
            assert count == 30, f"36-face scene should retain exactly 30 when cap is reached, got {count}"

        boxes = [item.detection.bbox_xyxy for item in views.faces]
        # Ground truth, not the implementation's sort key: each box's centre falls in a known
        # grid cell (row, col); IDs must follow reading order of those cells.
        pitch = CELL + GAP
        cells = [(int(((b[1] + b[3]) / 2) // pitch), int(((b[0] + b[2]) / 2) // pitch)) for b in boxes]
        assert cells == sorted(cells), \
            f"scene {scene_name}: face IDs are not top-to-bottom/left-to-right by grid cell: {cells}"
        assert [item.face_id for item in views.faces] == list(range(1, count + 1)), \
            f"scene {scene_name}: IDs are not sequential"
        source = views.source_image
        for item in views.faces:
            expected = source.crop(item.detection.crop_bbox_xyxy).convert("RGB")
            assert item.detection.cropped_face.tobytes() == expected.tobytes(), \
                f"scene {scene_name}, face {item.face_id}: crop pixels differ from original-image bbox"

        realesrgan_ran = "realesrgan" in views.stage_times_ms
        original_detection = "ảnh gốc" in (views.detection_source or "")
        if count and original_detection:
            assert not realesrgan_ran, "RealESRGAN ran even though YOLO found faces on the original image"
        measurements.append({
            "scene": scene_name,
            "pasted_faces": pasted,
            "detected_faces": count,
            "recall": (round(count / pasted, 4) if pasted else None),
            "detection_source": views.detection_source,
            "realesrgan_ran": realesrgan_ran,
            "stage_times_ms": views.stage_times_ms,
            "elapsed_ms": elapsed_ms,
            "face_ids": [item.face_id for item in views.faces],
            "bbox_xyxy": boxes,
        })
        print(f"PASS real_detection_scene_{scene_name}")

    print("JSON " + json.dumps({"status": "PASS", "scenes": measurements,
                               "user_images_written": False}, ensure_ascii=False, sort_keys=True))
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL test_m7_real_detection_cap: {type(exc).__name__}: {exc}")
        raise
