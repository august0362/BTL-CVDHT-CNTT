from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from PIL import Image


CLASSES = ("surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral")
MODELS = ("poster_rafdb", "ma3d_rafdb")
DEFAULT_ANGLES = (0, 45, 90)


def image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def rotate(image: Image.Image, angle: int) -> Image.Image:
    """Match tests/bench_frontalization_rafdb.py::_rotate byte-for-byte in behavior."""
    if angle == 0:
        return image
    mean = tuple(int(value) for value in image.resize((1, 1), Image.BILINEAR).getpixel((0, 0)))
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=mean)


def rotate_any(image: Image.Image, angle: float) -> Image.Image:
    """Same rotation/fill as `rotate`, for a float angle (ADR-028 roll correction)."""
    if abs(angle) < 1e-6:
        return image
    mean = tuple(int(value) for value in image.resize((1, 1), Image.BILINEAR).getpixel((0, 0)))
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=mean)


def predictions(result: dict | None) -> dict:
    output = {}
    for face in (result or {}).get("faces") or []:
        for prediction in face.get("predictions") or []:
            model_id = prediction.get("model_id")
            if model_id not in MODELS or model_id in output:
                continue
            if prediction.get("status") == "ok":
                labels = prediction.get("labels") or []
                probs = prediction.get("probs") or []
                top = prediction.get("top1") or {}
                if len(labels) != len(probs) or not labels:
                    output[model_id] = {"error": "invalid_output"}
                else:
                    output[model_id] = {
                        "label": str(top.get("label", labels[max(range(len(probs)), key=lambda i: probs[i])])),
                        "probs": {str(label): round(float(prob), 5) for label, prob in zip(labels, probs)},
                    }
            else:
                error = prediction.get("error") or {}
                output[model_id] = {"error": str(error.get("code") or prediction.get("status") or "prediction_error"),
                                    "message": str(error.get("message") or "")[:300]}
    top_error = (result or {}).get("error") or {}
    fallback = {"error": str(top_error.get("code") or "no_prediction"), "message": str(top_error.get("message") or "")[:300]}
    for model_id in MODELS:
        output.setdefault(model_id, dict(fallback))
    return output


def parse_angles(value: str | Iterable[int]) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()] if isinstance(value, str) else [int(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("angles must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError("angles must contain at least one unique integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark raw and 3DDFA-V2 frontalized RAF-DB images.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--emoca-url", default=None)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Limit sorted images before sharding; 0 means all.")
    parser.add_argument("--angles", default="0,45,90", help="Comma-separated in-plane angles (default: 0,45,90).")
    parser.add_argument("--mode", choices=("frontal3d", "roll2d"), default="frontal3d",
                        help="frontal3d: ADR-026 3D warp; roll2d: ADR-028 roll estimate + 2D rotation only.")
    parser.add_argument("--min-roll", type=float, default=5.0,
                        help="roll2d: below this |roll| the corrected image equals the input, so the raw "
                             "result is reused instead of a second analyze (thresholds are applied post hoc).")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("require --shards >= 1 and 0 <= --shard-index < --shards")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    try:
        angles = parse_angles(args.angles)
    except ValueError as exc:
        parser.error(str(exc))

    bundle = args.bundle_root.resolve()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    os.environ["MA3D_DEVICE"] = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "auto"
    from services.frontalization import get_tddfa_v2_frontalizer
    from web.orchestrator import analyze, check_emoca
    import inspect
    # The Kaggle Dataset snapshot (2026-09-30) predates M7's `max_faces` argument; that
    # version is single-face by default, so only pass it when the staged code accepts it.
    analyze_extra = {"max_faces": 1} if "max_faces" in inspect.signature(analyze).parameters else {}

    labels_path = bundle / "dataset" / "test_labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        labels = {str(row["image"]): int(row["label"]) for row in csv.DictReader(stream)}
    # Canonicalize the key with its RAF-DB class directory, matching the local reference
    # (`k/filename.jpg`) and preventing a same-named file in another class from colliding.
    entries = sorted(((f"{label}/{Path(image).name}", label) for image, label in labels.items()),
                     key=lambda item: (item[1], item[0]))
    if args.limit:
        entries = entries[:args.limit]
    entries = [entry for index, entry in enumerate(entries) if index % args.shards == args.shard_index]

    emoca = {"mode": "real", "url": args.emoca_url or os.environ.get("EMOCA_URL", "http://127.0.0.1:8765"), "timeout_s": 120}
    health = check_emoca(emoca)
    if not health.get("ok"):
        raise RuntimeError(f"EMOCA is not healthy; refusing mock fallback: {health}")
    frontalizer = get_tddfa_v2_frontalizer()

    out_path = args.out_jsonl.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    done.add((str(row["image"]), int(row["angle"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    todo = [(image, label, angle) for image, label in entries for angle in angles if (image, angle) not in done]
    print(f"Shard {args.shard_index}/{args.shards}: images={len(entries)} rows_pending={len(todo)} angles={angles}", flush=True)

    def infer(image: Image.Image) -> tuple[dict, int, int]:
        retries = 0
        started = time.perf_counter()
        for attempt in range(3):
            result = analyze(image_bytes(image), emoca=emoca, engine_mode="real", **analyze_extra)
            parsed = predictions(result)
            unavailable = "service_unavailable" in str(parsed.get("ma3d_rafdb", {}).get("error", ""))
            if not unavailable or attempt == 2:
                return parsed, round((time.perf_counter() - started) * 1000), retries
            retries += 1
            time.sleep(5)
        raise AssertionError("unreachable")

    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8", buffering=1) as stream:
        for local_index, (relative, label_index, angle) in enumerate(todo, 1):
            row = {"image": relative, "label": CLASSES[label_index - 1], "angle": angle, "mode": args.mode,
                   "pose": None, "occluded": None,
                   "timings": {"ms_raw": None, "ms_frontalize": None, "ms_frontal": None},
                   "raw": {}, "frontal": {}, "emoca_retries": {"raw": 0, "frontal": 0}}
            image_path = bundle / "dataset" / "DATASET" / "test" / relative
            try:
                with Image.open(image_path) as source:
                    rotated = rotate(source.convert("RGB"), angle)
                raw_started = time.perf_counter()
                try:
                    row["raw"], row["timings"]["ms_raw"], row["emoca_retries"]["raw"] = infer(rotated)
                except Exception as exc:
                    row["timings"]["ms_raw"] = round((time.perf_counter() - raw_started) * 1000)
                    row["raw"] = {mid: {"error": "analyze_failed"} for mid in MODELS}
                    row["raw_error"] = f"{type(exc).__name__}: {exc}"
                frontal_started = time.perf_counter()
                if args.mode == "roll2d":
                    # ADR-028: estimate roll with 3DDFA-V2 and undo it with a plain 2D rotation.
                    # Sign measured on RAF-DB crops: rotate(+a) -> roll ~ +a, so rotate(-roll) fixes it.
                    try:
                        _, pose = frontalizer.estimate(rotated)
                        roll = float(pose[2])
                        row["pose"] = [round(float(pose[0]), 2), round(float(pose[1]), 2), round(roll, 2)]
                        row["timings"]["ms_frontalize"] = round((time.perf_counter() - frontal_started) * 1000)
                        if abs(roll) < args.min_roll:
                            row["roll_applied_deg"] = 0.0
                            row["frontal"] = json.loads(json.dumps(row["raw"]))
                            row["frontal_reused_raw"] = True
                        else:
                            row["roll_applied_deg"] = round(-roll, 2)
                            corrected = rotate_any(rotated, -roll)
                            infer_started = time.perf_counter()
                            try:
                                row["frontal"], row["timings"]["ms_frontal"], row["emoca_retries"]["frontal"] = infer(corrected)
                            except Exception as exc:
                                row["timings"]["ms_frontal"] = round((time.perf_counter() - infer_started) * 1000)
                                row["frontal"] = {mid: {"error": "analyze_failed"} for mid in MODELS}
                                row["frontal_analyze_error"] = f"{type(exc).__name__}: {exc}"
                    except Exception as exc:
                        row["timings"]["ms_frontalize"] = round((time.perf_counter() - frontal_started) * 1000)
                        row["frontal_error"] = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
                        row["frontal"] = {mid: {"error": "roll_estimate_failed"} for mid in MODELS}
                else:
                    try:
                        result = frontalizer.frontalize(rotated)
                        row["timings"]["ms_frontalize"] = round(result.time_ms)
                        row["pose"] = [round(result.yaw_deg, 2), round(result.pitch_deg, 2), round(result.roll_deg, 2)]
                        row["occluded"] = round(result.occluded_fraction, 4)
                    except Exception as exc:
                        row["timings"]["ms_frontalize"] = round((time.perf_counter() - frontal_started) * 1000)
                        row["frontal_error"] = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
                        row["frontal"] = {mid: {"error": "frontalize_failed"} for mid in MODELS}
                    else:
                        infer_started = time.perf_counter()
                        try:
                            row["frontal"], row["timings"]["ms_frontal"], row["emoca_retries"]["frontal"] = infer(result.image)
                        except Exception as exc:
                            row["timings"]["ms_frontal"] = round((time.perf_counter() - infer_started) * 1000)
                            row["frontal"] = {mid: {"error": "analyze_failed"} for mid in MODELS}
                            row["frontal_analyze_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                row["input_error"] = f"{type(exc).__name__}: {exc}"
                row["raw"] = {mid: {"error": "input_failed"} for mid in MODELS}
                row["frontal"] = {mid: {"error": "input_failed"} for mid in MODELS}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if local_index % 25 == 0 or local_index == len(todo):
                rate = (time.perf_counter() - started) / local_index
                # Progress file readable from Kaggle's Output panel while the cell is still running.
                progress = {"shard": args.shard_index, "done": local_index, "pending_total": len(todo),
                            "already_done_before_start": len(done), "percent": round(100 * local_index / len(todo), 2),
                            "seconds_per_row": round(rate, 3),
                            "eta_minutes": round(rate * (len(todo) - local_index) / 60, 1),
                            "updated_unix": round(time.time())}
                out_path.with_name(f"progress_shard{args.shard_index}.json").write_text(
                    json.dumps(progress, indent=2), encoding="utf-8")
                print(f"shard={args.shard_index} progress={local_index}/{len(todo)} image={relative} angle={angle} rate={rate:.2f}s/row", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
