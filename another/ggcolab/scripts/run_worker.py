from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image


VARIANT_ORDER = ["POSTER", "MA3DNRe", "MA3DNCo", "MA3D", "MA3DReCo"]


def png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def get_prediction(result: dict, model_id: str) -> dict:
    for face in result.get("faces") or []:
        for prediction in face.get("predictions") or []:
            if prediction.get("model_id") == model_id:
                if prediction.get("status") == "ok":
                    top = prediction.get("top1") or {}
                    return {
                        "ok": True,
                        "index": int(top["index"]),
                        "confidence": float(top["prob"]),
                        "probs": [float(value) for value in prediction.get("probs", [])],
                    }
                error = prediction.get("error") or {}
                return {"ok": False, "error": f"{error.get('code', 'prediction_error')}: {error.get('message', '')}"}
    error = result.get("error") or {}
    warnings = result.get("warnings") or []
    return {"ok": False, "error": f"{error.get('code', 'analyze_error')}: {error.get('message', warnings)}"}


def analyze_with_retry(analyze, check_emoca, image: Image.Image, model_ids: list[str], emoca: dict) -> tuple[dict, int]:
    last = {}
    retries = 0
    for attempt in range(3):
        last = analyze(png_bytes(image), emoca=emoca, engine_mode="real", model_ids=model_ids)
        predictions = [get_prediction(last, model_id) for model_id in model_ids]
        failed_service = any(
            (not prediction["ok"]) and "service_unavailable" in prediction.get("error", "")
            for prediction in predictions
        )
        if not failed_service:
            return last, retries
        retries += 1
        if attempt < 2:
            health = check_emoca(emoca)
            print(f"EMOCA temporary error; retry={attempt + 1}; healthy={health.get('ok')}", flush=True)
            time.sleep(5)
    return last, retries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--emoca-url", required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test cap; 0 means full test set.")
    args = parser.parse_args()

    bundle = args.bundle_root.resolve()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    os.environ["MA3D_DEVICE"] = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "auto"

    from web.orchestrator import analyze, check_emoca
    from web.predata_experiment import build_predata_views

    with (bundle / "dataset" / "test_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = {row["image"]: int(row["label"]) for row in csv.DictReader(stream)}
    entries = sorted(((label, name) for name, label in labels.items()), key=lambda item: (item[0], item[1]))
    if args.limit:
        entries = entries[: args.limit]
    entries = [entry for index, entry in enumerate(entries) if index % args.shards == args.shard_index]

    emoca = {"mode": "real", "url": args.emoca_url, "timeout_s": 120}
    health = check_emoca(emoca)
    if not health.get("ok"):
        raise RuntimeError(f"EMOCA is not healthy; refusing mock fallback: {health}")

    out_path = args.out_jsonl.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[int, str], dict] = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    variants = row.get("variants")
                    complete = (
                        not row.get("fatal_error")
                        and isinstance(variants, dict)
                        and all(
                            variant in variants and isinstance(variants[variant], dict)
                            for variant in VARIANT_ORDER
                        )
                    )
                    if complete:
                        done[(int(row["label"]), str(row["file"]))] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    pending = [entry for entry in entries if entry not in done]
    print(
        f"Shard {args.shard_index}/{args.shards}: test={len(entries)} pending={len(pending)} "
        f"EMOCA={emoca['url']} real; device={os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}",
        flush=True,
    )

    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8", buffering=1) as stream:
        for local_index, (label, name) in enumerate(pending, 1):
            item_started = time.perf_counter()
            image_path = bundle / "dataset" / "DATASET" / "test" / str(label) / name
            row = {"label": label, "file": name, "variants": {}, "emoca_retries": 0}
            try:
                with Image.open(image_path) as source:
                    original = source.convert("RGB").copy()
                row["dimensions"] = list(original.size)

                baseline, retries = analyze_with_retry(
                    analyze, check_emoca, original, ["poster_rafdb", "ma3d_rafdb"], emoca
                )
                row["emoca_retries"] += retries
                row["variants"]["POSTER"] = get_prediction(baseline, "poster_rafdb")
                row["variants"]["MA3D"] = get_prediction(baseline, "ma3d_rafdb")
                if not any("emoca_real:" in str(warning) for warning in baseline.get("warnings") or []):
                    if row["variants"]["MA3D"].get("ok"):
                        row["variants"]["MA3D"]["warning"] = "real EMOCA marker absent"

                views = build_predata_views(original)
                row["preprocess_times_ms"] = views.stage_times_ms
                row["preprocess_errors"] = views.branch_errors
                branch_inputs = {
                    "MA3DNRe": views.raw_codeformer_crop if views.raw_detection else None,
                    "MA3DNCo": views.enhanced_detection.cropped_face if views.enhanced_detection else None,
                    "MA3DReCo": views.enhanced_codeformer_crop,
                }
                for variant, variant_image in branch_inputs.items():
                    if variant_image is None:
                        detail = views.branch_errors.get(
                            "raw_codeformer" if variant == "MA3DNRe" else
                            "enhanced_detection" if variant == "MA3DNCo" else "enhanced_codeformer"
                        )
                        row["variants"][variant] = {"ok": False, "error": f"preprocessing_unavailable: {detail or 'no crop'}"}
                        continue
                    result, retries = analyze_with_retry(
                        analyze, check_emoca, variant_image, ["ma3d_rafdb"], emoca
                    )
                    row["emoca_retries"] += retries
                    row["variants"][variant] = get_prediction(result, "ma3d_rafdb")
                row["elapsed_seconds"] = round(time.perf_counter() - item_started, 3)
            except Exception as exc:  # Keep the complete test denominator and make each failure auditable.
                row["fatal_error"] = f"{type(exc).__name__}: {exc}"
                row["elapsed_seconds"] = round(time.perf_counter() - item_started, 3)

            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if not row.get("fatal_error") and all(
                variant in row["variants"] and isinstance(row["variants"][variant], dict)
                for variant in VARIANT_ORDER
            ):
                done[(label, name)] = row

            if local_index % 10 == 0 or local_index == len(pending):
                elapsed = time.perf_counter() - started
                rate = elapsed / max(1, local_index)
                eta = rate * (len(pending) - local_index)
                values = " ".join(
                    f"{variant}:{row['variants'].get(variant, {}).get('index', 'ERR')}"
                    for variant in VARIANT_ORDER
                )
                print(
                    f"shard={args.shard_index} progress={local_index}/{len(pending)} "
                    f"file={name} {values} item={row['elapsed_seconds']:.2f}s "
                    f"rate={rate:.2f}s/image ETA={eta / 60:.1f}m",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
