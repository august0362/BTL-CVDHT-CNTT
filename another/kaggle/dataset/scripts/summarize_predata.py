from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean


VARIANTS = (
    "MA3DReCo_noYOLO",
    "MA3DRe_noYOLO",
    "MA3DCo_noYOLO",
)
VARIANT_LABELS = {
    "MA3DReCo_noYOLO": "RealESRGAN + CodeFormer + MA3D",
    "MA3DRe_noYOLO": "RealESRGAN + MA3D",
    "MA3DCo_noYOLO": "CodeFormer + MA3D",
}
VARIANT_DIRS = {
    "MA3DReCo_noYOLO": "RealESRGAN_CodeFormer_MA3D",
    "MA3DRe_noYOLO": "RealESRGAN_MA3D",
    "MA3DCo_noYOLO": "CodeFormer_MA3D",
}
STAGES = (
    "realesrgan",
    "codeformer_raw",
    "codeformer_after_realesrgan",
    "emoca_ma3d_realesrgan_codeformer",
    "emoca_ma3d_realesrgan",
    "emoca_ma3d_codeformer",
)
LABELS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]


def load_latest_rows(results_dir: Path) -> dict[tuple[int, str], dict]:
    rows: dict[tuple[int, str], dict] = {}
    files = sorted(results_dir.glob("shard_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No shard_*.jsonl files in {results_dir}")
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    rows[(int(row["label"]), str(row["file"]))] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    return rows


def valid_prediction(row: dict, variant: str) -> dict | None:
    prediction = row.get("variants", {}).get(variant, {})
    if prediction.get("ok") and len(prediction.get("probs", [])) == 7:
        return prediction
    return None


def calculate_metrics(rows: list[dict], variant: str, expected: int) -> dict:
    valid = [(row, valid_prediction(row, variant)) for row in rows]
    valid = [(row, pred) for row, pred in valid if pred is not None]
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for row, prediction in valid:
        true_index = int(row["label"]) - 1
        pred_index = int(prediction["index"])
        if true_index in range(7) and pred_index in range(7):
            confusion[true_index][pred_index] += 1

    per_class = []
    f1_values = []
    for index, name in enumerate(LABELS):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(confusion[row][index] for row in range(7))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1_values.append(f1 * 100)
        per_class.append({
            "label": name,
            "support": support,
            "precision_pct": precision * 100,
            "recall_pct": recall * 100 if support else None,
            "f1_pct": f1 * 100 if support else None,
        })

    correct = sum(confusion[i][i] for i in range(7))
    wrong = len(valid) - correct
    failed = expected - len(valid)
    confidences = [float(pred["confidence"]) for _, pred in valid]
    return {
        "variant": variant,
        "description": VARIANT_LABELS[variant],
        "n_test": expected,
        "n_valid": len(valid),
        "n_failed": failed,
        "n_wrong_valid_predictions": wrong,
        "n_correct": correct,
        "coverage_pct": 100 * len(valid) / expected if expected else None,
        "accuracy_valid_pct": 100 * correct / len(valid) if valid else None,
        "accuracy_all_pct": 100 * correct / expected if expected else None,
        "macro_f1_valid_pct": mean(f1_values) if f1_values else None,
        "mean_confidence_valid_pct": 100 * mean(confidences) if confidences else None,
        "per_class": per_class,
        "confusion_matrix_true_rows_predicted_columns": confusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the three no-YOLO preprocessing ablations.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--examples-per-model", type=int, default=10)
    parser.add_argument("--expected-images", type=int, default=0,
                        help="Expected denominator for a smoke subset; defaults to the full test set.")
    args = parser.parse_args()

    bundle = args.bundle_root.resolve()
    results_dir = args.results_dir.resolve()
    rows_by_key = load_latest_rows(results_dir)
    with (bundle / "dataset" / "test_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = {row["image"]: int(row["label"]) for row in csv.DictReader(stream)}
    ordered_rows = [
        rows_by_key[(label, filename)]
        for label, filename in sorted(((value, name) for name, value in labels.items()), key=lambda x: (x[0], x[1]))
        if (label, filename) in rows_by_key
    ]
    expected = args.expected_images or len(labels)
    if expected < 1 or expected > len(labels):
        raise SystemExit(f"--expected-images must be between 1 and {len(labels)}")
    metrics = {variant: calculate_metrics(ordered_rows, variant, expected) for variant in VARIANTS}

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "dataset": "RAF-DB aligned test images",
        "processed_rows": len(ordered_rows),
        "expected_rows": expected,
        "all_images_processed": len(ordered_rows) == expected,
        "flows": {variant: VARIANT_LABELS[variant] for variant in VARIANTS},
        "metrics": metrics,
        "error_images": {
            variant: {
                "count": metrics[variant]["n_wrong_valid_predictions"],
                "selection": f"Top {args.examples_per_model} wrong predictions by confidence, descending",
                "directory": str(results_dir / "misclassified" / VARIANT_DIRS[variant]),
            }
            for variant in VARIANTS
        },
        "note": "Existing POSTER/MA3D/YOLO-based results were not rerun or overwritten.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metric_path = results_dir / "metrics.csv"
    metric_columns = [
        "variant", "description", "n_test", "n_valid", "n_failed", "n_wrong_valid_predictions",
        "n_correct", "coverage_pct", "accuracy_valid_pct", "accuracy_all_pct",
        "macro_f1_valid_pct", "mean_confidence_valid_pct",
    ]
    with metric_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_columns)
        writer.writeheader()
        for variant in VARIANTS:
            writer.writerow({key: metrics[variant][key] for key in metric_columns})

    stage_counts: dict[str, Counter] = {stage: Counter() for stage in STAGES}
    stage_errors: dict[str, Counter] = {stage: Counter() for stage in STAGES}
    for row in ordered_rows:
        for stage in STAGES:
            record = row.get("stages", {}).get(stage)
            if not isinstance(record, dict):
                stage_counts[stage]["missing"] += 1
                stage_errors[stage]["stage record missing"] += 1
                continue
            status = str(record.get("status", "missing"))
            stage_counts[stage][status] += 1
            if status == "error":
                stage_errors[stage][str(record.get("error") or "unspecified error")] += 1

    stage_csv = results_dir / "stage_errors.csv"
    with stage_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stage", "status", "count", "error"])
        writer.writeheader()
        for stage in STAGES:
            for status, count in sorted(stage_counts[stage].items()):
                errors = stage_errors[stage]
                if status == "error" and errors:
                    for error, error_count in errors.most_common():
                        writer.writerow({"stage": stage, "status": status, "count": error_count, "error": error})
                else:
                    writer.writerow({"stage": stage, "status": status, "count": count, "error": ""})

    predictions_path = results_dir / "predictions.csv"
    prediction_columns = ["label", "label_name", "file"]
    for variant in VARIANTS:
        prediction_columns.extend([f"{variant}_valid", f"{variant}_predicted", f"{variant}_confidence", f"{variant}_error"])
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_columns)
        writer.writeheader()
        for row in ordered_rows:
            label = int(row["label"])
            flat = {"label": label, "label_name": LABELS[label - 1], "file": row["file"]}
            for variant in VARIANTS:
                prediction = row.get("variants", {}).get(variant, {})
                flat[f"{variant}_valid"] = bool(prediction.get("ok"))
                flat[f"{variant}_predicted"] = LABELS[int(prediction["index"])] if prediction.get("ok") else ""
                flat[f"{variant}_confidence"] = prediction.get("confidence", "")
                flat[f"{variant}_error"] = prediction.get("error", "")
            writer.writerow(flat)

    misclassified_root = results_dir / "misclassified"
    for variant in VARIANTS:
        output_dir = misclassified_root / VARIANT_DIRS[variant]
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wrong_rows = []
        for row in ordered_rows:
            prediction = valid_prediction(row, variant)
            if prediction is not None and int(prediction["index"]) != int(row["label"]) - 1:
                wrong_rows.append((row, prediction))
        wrong_rows.sort(key=lambda item: (-float(item[1]["confidence"]), str(item[0]["file"])))
        selected = wrong_rows[:max(0, args.examples_per_model)]
        metadata_path = output_dir / "metadata.csv"
        with metadata_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "rank", "file", "true_label", "predicted_label", "confidence", "probability_true_label",
            ])
            writer.writeheader()
            for rank, (row, prediction) in enumerate(selected, 1):
                true_label = int(row["label"])
                predicted_label = int(prediction["index"]) + 1
                source = bundle / "dataset" / "DATASET" / "test" / str(true_label) / str(row["file"])
                destination_name = (
                    f"{rank:02d}_true-{LABELS[true_label - 1]}_pred-{LABELS[predicted_label - 1]}_"
                    f"conf-{float(prediction['confidence']):.3f}_{Path(row['file']).name}"
                )
                if source.is_file():
                    shutil.copy2(source, output_dir / destination_name)
                writer.writerow({
                    "rank": rank,
                    "file": row["file"],
                    "true_label": LABELS[true_label - 1],
                    "predicted_label": LABELS[predicted_label - 1],
                    "confidence": prediction["confidence"],
                    "probability_true_label": prediction["probs"][true_label - 1],
                })

    print(f"Processed rows: {len(ordered_rows)}/{expected}")
    print("Stage status/error counts:")
    for stage in STAGES:
        status_counts = ", ".join(f"{status}={count}" for status, count in sorted(stage_counts[stage].items()))
        print(f"  {stage}: {status_counts or 'no records'}")
        for error, count in stage_errors[stage].most_common(5):
            print(f"    {count} × {error}")
    print("Model results:")
    for variant in VARIANTS:
        result = metrics[variant]
        print(
            f"  {variant}: wrong={result['n_wrong_valid_predictions']}, failed={result['n_failed']}, "
            f"coverage={result['coverage_pct']:.2f}%, accuracy(valid)={result['accuracy_valid_pct']}, "
            f"accuracy(all)={result['accuracy_all_pct']}%, macro_f1={result['macro_f1_valid_pct']}%"
        )
        print(f"    Top misclassified examples: {results_dir / 'misclassified' / VARIANT_DIRS[variant]}")
    print(f"Wrote {summary_path}, {metric_path}, {stage_csv}, and {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
