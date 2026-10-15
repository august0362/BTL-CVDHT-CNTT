"""Heavy local-only M7 acceptance; run in the venv web environment, with all image data in RAM."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def _health() -> dict:
    with urllib.request.urlopen("http://127.0.0.1:8765/v1/health", timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def _image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _baseline_signature(result: dict) -> tuple:
    signature = []
    for face in result.get("faces") or []:
        for prediction in face.get("predictions") or []:
            top1 = prediction.get("top1") or {}
            signature.append((prediction.get("model_id"), prediction.get("status"),
                              top1.get("label"), float(top1.get("prob", 0.0))))
    return tuple(signature)


def _baseline_probability_delta(before: tuple, after: tuple) -> float:
    if len(before) != len(after):
        raise AssertionError("the optional M7 pass changed the number of baseline predictions")
    deltas = []
    for first, second in zip(before, after):
        if first[:3] != second[:3]:
            raise AssertionError(f"the optional M7 pass changed a baseline model/top label: {first} -> {second}")
        deltas.append(abs(first[3] - second[3]))
    return max(deltas, default=0.0)


class _PeakRss:
    def __init__(self):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("psutil is required to record M7 process memory") from exc
        self.process = psutil.Process()
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(0.1):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except Exception:
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.peak = max(self.peak, self.process.memory_info().rss)


def run() -> int:
    first = ROOT / "database/image/archive/DATASET/test/1/test_0002_aligned.jpg"
    second = ROOT / "database/image/archive/DATASET/test/2/test_0274_aligned.jpg"
    checkpoints = (
        ROOT / "model/realesr-general-x4v3/realesr-general-x4v3.pth",
        ROOT / "model/yolo26_face/yolo26_face.pt",
    )
    if not first.is_file() or not second.is_file() or not all(path.is_file() for path in checkpoints):
        print("SKIP test_m7_real_acceptance: local M7 samples or model weights are missing")
        return 0
    try:
        health_before = _health()
    except Exception as exc:
        print(f"SKIP test_m7_real_acceptance: local EMOCA is unavailable ({type(exc).__name__}: {exc})")
        return 0
    if health_before.get("status") != "up" or health_before.get("mode") != "real":
        print("SKIP test_m7_real_acceptance: EMOCA health is not real/up")
        return 0

    from web.orchestrator import analyze, analyze_multiface

    settings = {"mode": "real", "url": "http://127.0.0.1:8765", "timeout_s": 60}
    face_a = Image.open(first).convert("RGB").resize((160, 160), Image.Resampling.LANCZOS)
    face_b = Image.open(second).convert("RGB").resize((160, 160), Image.Resampling.LANCZOS)
    group = Image.new("RGB", (340, 180), (245, 245, 245))
    group.paste(face_a, (5, 10))
    group.paste(face_b, (175, 10))

    started = time.perf_counter()
    baseline_before = analyze(_image_bytes(group), emoca=settings, engine_mode="real")
    if baseline_before.get("status") != "ok":
        raise AssertionError(f"original-image baseline failed: {baseline_before.get('error')}")
    baseline_ids = {prediction.get("model_id")
                    for face in baseline_before.get("faces", [])
                    for prediction in face.get("predictions", [])}
    if not {"poster_rafdb", "ma3d_rafdb"}.issubset(baseline_ids):
        raise AssertionError(f"baseline did not run both original-image models: {sorted(baseline_ids)}")
    if baseline_before.get("image") != {"width": 340, "height": 180}:
        raise AssertionError(f"baseline response lost the uploaded-image dimensions: {baseline_before.get('image')}")
    print("PASS real_original_baseline")

    single_started = time.perf_counter()
    single_state = analyze_multiface(face_a, emoca=settings, engine_mode="real", max_faces=30)
    if single_state.get("kind") != "single" or single_state.get("rows"):
        raise AssertionError("the real one-face branch must reuse the baseline with no second MA3D row")
    single_seconds = time.perf_counter() - single_started
    print("PASS real_one_face_keeps_baseline")

    blank = Image.new("RGB", (320, 240), (190, 190, 190))
    zero_started = time.perf_counter()
    zero_state = analyze(_image_bytes(blank), emoca=settings, engine_mode="real")
    if zero_state.get("image") != {"width": 320, "height": 240}:
        raise AssertionError("zero-face analysis did not preserve the original upload dimensions")
    zero_seconds = time.perf_counter() - zero_started
    print("PASS real_zero_face_uses_original_baseline")

    with _PeakRss() as rss:
        multi_started = time.perf_counter()
        multi_state = analyze_multiface(group, emoca=settings, engine_mode="real", max_faces=30)
        multi_seconds = time.perf_counter() - multi_started
    rows = multi_state.get("rows") or []
    if multi_state.get("kind") != "faces" or [row.get("face_id") for row in rows] != [1, 2]:
        raise AssertionError(f"real two-face result did not return sequential IDs: {multi_state.get('kind')}")
    boxes = [item.detection.bbox_xyxy for item in multi_state["views"].faces]
    if boxes != sorted(boxes, key=lambda box: (box[1], box[0])):
        raise AssertionError(f"real face IDs are not top-to-bottom, left-to-right: {boxes}")
    multi_predictions = []
    for row in rows:
        prediction = next((prediction for face in row["result"].get("faces", [])
                           for prediction in face.get("predictions", [])
                           if prediction.get("model_id") == "ma3d_rafdb"), None)
        if not prediction or prediction.get("status") != "ok" or not prediction.get("top1"):
            raise AssertionError(f"real MA3D prediction failed for face {row.get('face_id')}")
        multi_predictions.append({"face_id": row["face_id"], "bbox_original_px": boxes[row["face_id"] - 1],
                                  "crop_size": row["processed"].size, "top1": prediction["top1"]})
    print("PASS real_two_face_sequential_inference")

    baseline_after = analyze(_image_bytes(group), emoca=settings, engine_mode="real")
    baseline_before_signature = _baseline_signature(baseline_before)
    baseline_after_signature = _baseline_signature(baseline_after)
    max_baseline_delta = _baseline_probability_delta(baseline_before_signature, baseline_after_signature)
    if max_baseline_delta > 1e-5:
        raise AssertionError(f"baseline probability drift {max_baseline_delta:.8g} exceeds 1e-5")
    print("PASS baseline_stable_after_multiface")
    health_after = _health()
    total_seconds = time.perf_counter() - started
    print({
        "status": "PASS",
        "image": {"width": group.width, "height": group.height, "faces_created_in_memory": 2},
        "baseline_model_ids": sorted(baseline_ids),
        "baseline_outputs_stable": True,
        "baseline_max_probability_delta": round(max_baseline_delta, 8),
        "single_face": {"kind": single_state["kind"], "seconds": round(single_seconds, 3)},
        # ADR-027: zero faces run the baseline analyze() on the original image (no fallback chain).
        "no_face_fallback": {"path": "baseline", "status": zero_state.get("status"),
                             "seconds": round(zero_seconds, 3)},
        "multi_face": {"kind": multi_state["kind"], "seconds": round(multi_seconds, 3),
                       "faces": multi_predictions, "preprocessing_errors": multi_state["views"].errors},
        "windows_web_process_rss_mb": {"before": round(rss.before / 1048576, 1),
                                        "peak": round(rss.peak / 1048576, 1),
                                        "peak_delta": round((rss.peak - rss.before) / 1048576, 1)},
        "emoca_gpu": {"device": health_after.get("device"),
                      "vram_reserved_peak_mb": health_after.get("vram_reserved_peak_mb"),
                      "gpu_free_mb_after": health_after.get("gpu_free_mb")},
        "total_seconds": round(total_seconds, 3),
        "user_images_written": False,
    })
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        print(f"FAIL test_m7_real_acceptance: {type(exc).__name__}: {exc}")
        raise
