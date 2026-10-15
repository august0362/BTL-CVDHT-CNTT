"""Heavy real-model M6 CLI flow plus paired YOLO benchmark.

venv web

Run with the project venv after local checkpoints are available:
  python tests/test_m6_real_integration.py

Exercises the actual CLI on a sample photo, verifies saved metadata and original
pixel crops, then compares YOLO on 21 RAF-DB images before/after x4v3. CLI image
outputs are confined to a temporary directory under out_m6_cli and removed
automatically. The paired benchmark writes numeric results only to %TEMP%.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from statistics import mean
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

GENERAL = ROOT / "model/realesr-general-x4v3/realesr-general-x4v3.pth"
YOLO = ROOT / "model/yolo26_face/yolo26_face.pt"
SAMPLE = ROOT / "database/image/afraid.jpg"
DATA = ROOT / "database/image/archive/DATASET/test"
REPORT = Path(tempfile.gettempdir()) / "M6_REAL_BENCHMARK_2026-10-03.json"


def samples() -> list[tuple[int, Path]]:
    chosen: list[tuple[int, Path]] = []
    for label in range(1, 8):
        folder = DATA / str(label)
        files = sorted((p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                       key=lambda p: p.name.casefold())
        if not files:
            raise FileNotFoundError(f"No RAF-DB images found under {folder}")
        for index in sorted({0, len(files) // 2, len(files) - 1}):
            chosen.append((label, files[index]))
    return chosen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exercise_cli(Image, ImageOps, cli) -> dict:
    output_parent = ROOT / "out_m6_cli"
    output_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="real-integration-", dir=output_parent) as temporary:
        output = Path(temporary)
        code = cli.main([
            str(SAMPLE), "--output", str(output),
            "--general-weights", str(GENERAL), "--yolo-weights", str(YOLO),
            "--device", "cpu", "--confidence", "0.5", "--padding", "0.15",
            "--max-side", "4096",
        ])
        assert code == 0, f"M6 CLI returned {code}"
        metadata_path = output / "metadata" / "afraid__jpg.json"
        enhanced_path = output / "enhanced_images" / "afraid__jpg.png"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert enhanced_path.is_file()
        assert metadata["status"] == "ok"
        assert metadata["config"]["confidence"] == 0.5
        assert metadata["config"]["padding"] == 0.15
        assert metadata["config"]["device"] == "cpu"
        assert metadata["face_count"] == len(metadata["faces"])
        assert set(metadata["stage_times_ms"]) == {
            "general_restoration", "face_detection", "face_cropping", "output_size_limit"
        }
        artifacts = {item["role"]: item["sha256"] for item in metadata["model_artifacts"]}
        assert artifacts == {"general_restoration": _sha256(GENERAL), "face_detection": _sha256(YOLO)}

        with Image.open(SAMPLE) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        with Image.open(enhanced_path) as opened:
            enhanced = opened.convert("RGB")
        assert enhanced.size == original.size and enhanced.mode == "RGB"
        crop_sizes = []
        for index, face in enumerate(metadata["faces"], start=1):
            crop_path = output / "faces" / f"afraid__jpg_face_{index:02d}.png"
            assert crop_path.is_file()
            expected = original.crop(tuple(face["crop_bbox_xyxy"])).convert("RGB")
            with Image.open(crop_path) as opened:
                crop = opened.convert("RGB")
            assert crop.size == expected.size and crop.tobytes() == expected.tobytes()
            crop_sizes.append(list(crop.size))
        assert len(list((output / "faces").glob("*.png"))) == metadata["face_count"]
        return {
            "file": SAMPLE.name,
            "status": metadata["status"],
            "face_count": metadata["face_count"],
            "crop_dimensions": crop_sizes,
            "stage_times_ms": metadata["stage_times_ms"],
        }


def main() -> int:
    missing = [str(p) for p in (GENERAL, YOLO, SAMPLE) if not p.exists()]
    if missing:
        print("SKIP test_m6_real_integration: missing " + ", ".join(missing))
        return 0
    try:
        import torch
        from PIL import Image, ImageOps
        from services.image_preprocessing import cli
        from services.image_preprocessing.adapters import SpandrelRestorer, YoloDetector
    except ImportError as exc:
        print(f"SKIP test_m6_real_integration: dependency missing: {exc}")
        return 0

    torch.set_num_threads(min(torch.get_num_threads(), 2))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    cli_flow = _exercise_cli(Image, ImageOps, cli)
    try:
        chosen = samples()
    except FileNotFoundError:
        comparison = {
            "sample_count": 0,
            "full_cli_single_sample": cli_flow,
            "warning": "RAF-DB archive is absent; paired detector benchmark was skipped.",
        }
        REPORT.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        print("  SKIP paired 21-image comparison: RAF-DB archive unavailable")
        print("ok test_m6_real_integration: real CLI")
        print("ALL PASS")
        return 0

    general = SpandrelRestorer(GENERAL, "realesr-general-x4v3", "cpu")
    detector = YoloDetector(YOLO, "cpu")
    rows = []
    for label, path in chosen:
        with Image.open(path) as opened:
            original = opened.convert("RGB")
        baseline = detector.detect(original, 0.5)
        enhanced = general.restore(original)
        restored = detector.detect(enhanced, 0.5)
        assert enhanced.size == original.size, f"{path.name}: dimensions changed"
        rows.append({
            "label": label,
            "file": path.name,
            "baseline_detected": bool(baseline),
            "baseline_max_conf": max((float(d["confidence"]) for d in baseline), default=None),
            "restored_detected": bool(restored),
            "restored_max_conf": max((float(d["confidence"]) for d in restored), default=None),
        })

    before = sum(row["baseline_detected"] for row in rows)
    after = sum(row["restored_detected"] for row in rows)
    comparison = {
        "sample_count": len(rows),
        "full_cli_single_sample": cli_flow,
        "selection": "first/middle/last filename in each RAF-DB test label folder 1..7",
        "confidence_threshold": 0.5,
        "baseline_detected": before,
        "baseline_recall_proxy": before / len(rows),
        "x4v3_then_yolo_detected": after,
        "x4v3_then_yolo_recall_proxy": after / len(rows),
        "delta_images": after - before,
        "baseline_mean_confidence_on_detections": mean([r["baseline_max_conf"] for r in rows if r["baseline_max_conf"] is not None]) if before else None,
        "x4v3_mean_confidence_on_detections": mean([r["restored_max_conf"] for r in rows if r["restored_max_conf"] is not None]) if after else None,
        "newly_detected_after_restoration": [r["file"] for r in rows if r["restored_detected"] and not r["baseline_detected"]],
        "lost_after_restoration": [r["file"] for r in rows if r["baseline_detected"] and not r["restored_detected"]],
        "per_label": {
            str(label): {
                "count": sum(row["label"] == label for row in rows),
                "baseline_detected": sum(row["label"] == label and row["baseline_detected"] for row in rows),
                "x4v3_detected": sum(row["label"] == label and row["restored_detected"] for row in rows),
            }
            for label in range(1, 8)
        },
        "rows": rows,
        "warning": "RAF-DB has no face-box ground truth here; this is face-present image recall proxy, not bbox IoU/precision.",
    }
    REPORT.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: comparison[k] for k in (
        "sample_count", "full_cli_single_sample", "confidence_threshold", "baseline_detected", "baseline_recall_proxy",
        "x4v3_then_yolo_detected", "x4v3_then_yolo_recall_proxy", "delta_images",
        "baseline_mean_confidence_on_detections", "x4v3_mean_confidence_on_detections",
        "newly_detected_after_restoration", "lost_after_restoration", "per_label",
    )}, indent=2))
    print(f"Benchmark detail written: {REPORT}")
    print("ok test_m6_real_integration: real CLI plus paired 21-image YOLO comparison")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
