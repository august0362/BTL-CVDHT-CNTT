from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image


VARIANTS = ("MA3DReCo_noYOLO", "MA3DRe_noYOLO", "MA3DCo_noYOLO")


def image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def get_ma3d_prediction(result: dict) -> dict:
    for face in result.get("faces") or []:
        for prediction in face.get("predictions") or []:
            if prediction.get("model_id") != "ma3d_rafdb":
                continue
            if prediction.get("status") == "ok":
                top = prediction.get("top1") or {}
                probs = [float(value) for value in prediction.get("probs", [])]
                if len(probs) != 7:
                    return {"ok": False, "error": f"invalid_output: expected 7 probabilities, got {len(probs)}"}
                return {
                    "ok": True,
                    "index": int(top["index"]),
                    "confidence": float(top["prob"]),
                    "probs": probs,
                }
            error = prediction.get("error") or {}
            return {"ok": False, "error": f"{error.get('code', 'prediction_error')}: {error.get('message', '')}"}
    error = result.get("error") or {}
    warnings = result.get("warnings") or []
    return {"ok": False, "error": f"{error.get('code', 'analyze_error')}: {error.get('message', warnings)}"}


def stage_record(status: str, started: float, error: str | None = None) -> dict:
    return {
        "status": status,
        "error": error,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate only the three no-YOLO preprocessing branches.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--emoca-url", required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test cap; 0 means the complete test set.")
    args = parser.parse_args()

    bundle = args.bundle_root.resolve()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    os.environ["MA3D_DEVICE"] = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "auto"

    from services.image_preprocessing.adapters import CodeFormerRestorer, SpandrelRestorer
    from web.orchestrator import analyze, check_emoca

    labels_path = bundle / "dataset" / "test_labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        labels = {row["image"]: int(row["label"]) for row in csv.DictReader(stream)}
    entries = sorted(((label, name) for name, label in labels.items()), key=lambda item: (item[0], item[1]))
    if args.limit:
        entries = entries[:args.limit]
    entries = [entry for index, entry in enumerate(entries) if index % args.shards == args.shard_index]

    emoca = {"mode": "real", "url": args.emoca_url, "timeout_s": 120}
    health = check_emoca(emoca)
    if not health.get("ok"):
        raise RuntimeError(f"EMOCA is not healthy; refusing mock fallback: {health}")

    general_path = project / "model" / "realesr-general-x4v3" / "realesr-general-x4v3.pth"
    codeformer_path = project / "model" / "codeformer" / "codeformer.pth"
    codeformer_root = project / "third_party" / "CodeFormer"
    general = face = None
    load_errors: dict[str, str] = {}
    try:
        general = SpandrelRestorer(general_path, "RealESRGAN Compact (local checkpoint)")
    except Exception as exc:
        load_errors["realesrgan"] = f"{type(exc).__name__}: {exc}"
    try:
        face = CodeFormerRestorer(codeformer_path, codeformer_root)
    except Exception as exc:
        load_errors["codeformer"] = f"{type(exc).__name__}: {exc}"

    out_path = args.out_jsonl.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[int, str]] = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    predictions = row.get("variants", {})
                    if all(predictions.get(name, {}).get("ok") for name in VARIANTS):
                        done.add((int(row["label"]), str(row["file"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    pending = [entry for entry in entries if entry not in done]
    print(
        f"Shard {args.shard_index}/{args.shards}: images={len(entries)} pending={len(pending)}; "
        f"RealESRGAN={'ready' if general else 'unavailable'}; "
        f"CodeFormer={'ready' if face else 'unavailable'}; EMOCA={emoca['url']}",
        flush=True,
    )

    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8", buffering=1) as stream:
        for local_index, (label, filename) in enumerate(pending, 1):
            item_started = time.perf_counter()
            image_path = bundle / "dataset" / "DATASET" / "test" / str(label) / filename
            row: dict = {
                "label": label,
                "file": filename,
                "dimensions": None,
                "variants": {},
                "stages": {},
                "emoca_retries": 0,
            }
            try:
                with Image.open(image_path) as source:
                    original = source.convert("RGB").copy()
                row["dimensions"] = list(original.size)

                # Run each restorer on the full, already aligned RAF-DB face image.
                # No YOLO model, face detector, or detector crop is used here.
                real_image = None
                stage_started = time.perf_counter()
                if general is None:
                    row["stages"]["realesrgan"] = stage_record(
                        "error", stage_started, load_errors.get("realesrgan", "RealESRGAN unavailable")
                    )
                else:
                    try:
                        real_image = general.restore(original.copy())
                        if not isinstance(real_image, Image.Image):
                            raise TypeError("RealESRGAN must return a PIL image")
                        real_image = real_image.convert("RGB")
                        if real_image.size != original.size:
                            raise ValueError(f"RealESRGAN changed image size {original.size} -> {real_image.size}")
                        row["stages"]["realesrgan"] = stage_record("ok", stage_started)
                    except Exception as exc:
                        row["stages"]["realesrgan"] = stage_record(
                            "error", stage_started, f"{type(exc).__name__}: {exc}"
                        )

                raw_codeformer = None
                stage_started = time.perf_counter()
                if face is None:
                    row["stages"]["codeformer_raw"] = stage_record(
                        "error", stage_started, load_errors.get("codeformer", "CodeFormer unavailable")
                    )
                else:
                    try:
                        raw_codeformer = face.restore(original.copy())
                        if not isinstance(raw_codeformer, Image.Image):
                            raise TypeError("CodeFormer must return a PIL image")
                        raw_codeformer = raw_codeformer.convert("RGB")
                        row["stages"]["codeformer_raw"] = stage_record("ok", stage_started)
                    except Exception as exc:
                        row["stages"]["codeformer_raw"] = stage_record(
                            "error", stage_started, f"{type(exc).__name__}: {exc}"
                        )

                real_codeformer = None
                stage_started = time.perf_counter()
                if real_image is None:
                    row["stages"]["codeformer_after_realesrgan"] = stage_record(
                        "skipped", stage_started, "RealESRGAN stage failed"
                    )
                elif face is None:
                    row["stages"]["codeformer_after_realesrgan"] = stage_record(
                        "error", stage_started, load_errors.get("codeformer", "CodeFormer unavailable")
                    )
                else:
                    try:
                        real_codeformer = face.restore(real_image.copy())
                        if not isinstance(real_codeformer, Image.Image):
                            raise TypeError("CodeFormer must return a PIL image")
                        real_codeformer = real_codeformer.convert("RGB")
                        row["stages"]["codeformer_after_realesrgan"] = stage_record("ok", stage_started)
                    except Exception as exc:
                        row["stages"]["codeformer_after_realesrgan"] = stage_record(
                            "error", stage_started, f"{type(exc).__name__}: {exc}"
                        )

                branch_images = {
                    "MA3DReCo_noYOLO": real_codeformer,
                    "MA3DRe_noYOLO": real_image,
                    "MA3DCo_noYOLO": raw_codeformer,
                }
                branch_stages = {
                    "MA3DReCo_noYOLO": "emoca_ma3d_realesrgan_codeformer",
                    "MA3DRe_noYOLO": "emoca_ma3d_realesrgan",
                    "MA3DCo_noYOLO": "emoca_ma3d_codeformer",
                }
                for variant in VARIANTS:
                    branch_image = branch_images[variant]
                    stage_name = branch_stages[variant]
                    stage_started = time.perf_counter()
                    if branch_image is None:
                        row["variants"][variant] = {
                            "ok": False,
                            "error": "preprocessing_unavailable: required restoration stage failed",
                        }
                        row["stages"][stage_name] = stage_record(
                            "skipped", stage_started, "required preprocessing stage failed"
                        )
                        continue
                    try:
                        result = None
                        retries = 0
                        prediction = {}
                        for attempt in range(3):
                            result = analyze(
                                image_bytes(branch_image), emoca=emoca,
                                engine_mode="real", model_ids=["ma3d_rafdb"],
                            )
                            prediction = get_ma3d_prediction(result)
                            if prediction.get("ok") or "service_unavailable" not in prediction.get("error", ""):
                                break
                            retries += 1
                            if attempt < 2:
                                time.sleep(5)
                        row["emoca_retries"] += retries
                        row["variants"][variant] = prediction
                        if prediction.get("ok"):
                            row["stages"][stage_name] = stage_record("ok", stage_started)
                        else:
                            row["stages"][stage_name] = stage_record(
                                "error", stage_started, prediction.get("error", "unknown MA3D error")
                            )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        row["variants"][variant] = {"ok": False, "error": error}
                        row["stages"][stage_name] = stage_record("error", stage_started, error)

                row["elapsed_seconds"] = round(time.perf_counter() - item_started, 3)
            except Exception as exc:
                row["fatal_error"] = f"{type(exc).__name__}: {exc}"
                row["elapsed_seconds"] = round(time.perf_counter() - item_started, 3)
                for variant in VARIANTS:
                    row["variants"].setdefault(variant, {"ok": False, "error": row["fatal_error"]})

            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if local_index % 25 == 0 or local_index == len(pending):
                elapsed = time.perf_counter() - started
                per_image = elapsed / max(1, local_index)
                values = " ".join(
                    f"{name}:{row['variants'].get(name, {}).get('index', 'ERR')}"
                    for name in VARIANTS
                )
                print(
                    f"shard={args.shard_index} progress={local_index}/{len(pending)} "
                    f"file={filename} {values} item={row['elapsed_seconds']:.2f}s "
                    f"rate={per_image:.2f}s/image",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
