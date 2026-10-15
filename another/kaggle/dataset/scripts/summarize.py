from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


VARIANTS = ["POSTER", "MA3DNRe", "MA3DNCo", "MA3D", "MA3DReCo"]
LABEL_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]


def load_rows(paths: list[Path]) -> dict[tuple[int, str], dict]:
    rows: dict[tuple[int, str], dict] = {}
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    rows[(int(row["label"]), str(row["file"]))] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    return rows


def metrics(rows: list[dict], variant: str, expected_rows: int) -> dict:
    valid = [
        row for row in rows
        if (row.get("variants", {}).get(variant, {}).get("ok")
            and len(row["variants"][variant].get("probs", [])) == 7)
    ]
    cm = [[0 for _ in range(7)] for _ in range(7)]
    for row in valid:
        true = int(row["label"]) - 1
        pred = int(row["variants"][variant]["index"])
        if true in range(7) and pred in range(7):
            cm[true][pred] += 1
    per_class = []
    for index in range(7):
        tp = cm[index][index]
        fn = sum(cm[index]) - tp
        fp = sum(cm[row][index] for row in range(7)) - tp
        support = sum(cm[index])
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        if support:
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else 0.0
            )
        else:
            f1 = None
        per_class.append({"label": LABEL_NAMES[index], "support_valid": support,
                          "precision_pct": 100 * precision if precision is not None else None,
                          "recall_pct": 100 * recall if recall is not None else None,
                          "f1_pct": 100 * f1 if f1 is not None else None})

    correct = sum(cm[index][index] for index in range(7))
    confidences = [float(row["variants"][variant]["confidence"]) for row in valid]
    true_probabilities = [float(row["variants"][variant]["probs"][int(row["label"]) - 1]) for row in valid]
    nll = [-math.log(max(1e-12, p)) for p in true_probabilities]
    ece = 0.0
    if valid:
        for bin_index in range(10):
            group = [row for row in valid if min(9, int(row["variants"][variant]["confidence"] * 10)) == bin_index]
            if not group:
                continue
            group_accuracy = sum(row["variants"][variant]["index"] == int(row["label"]) - 1 for row in group) / len(group)
            group_confidence = mean(row["variants"][variant]["confidence"] for row in group)
            ece += len(group) / len(valid) * abs(group_accuracy - group_confidence)

    recalls = [item["recall_pct"] for item in per_class if item["support_valid"] > 0]
    f1s = [item["f1_pct"] for item in per_class if item["support_valid"] > 0]
    denominator = expected_rows
    return {
        "n_valid": len(valid),
        "n_test_rows": denominator,
        "coverage_pct": 100 * len(valid) / denominator if denominator else None,
        "n_correct": correct,
        "accuracy_valid_pct": 100 * correct / len(valid) if valid else None,
        "accuracy_all_images_pct": 100 * correct / denominator if denominator else None,
        "mean_class_accuracy_valid_pct": mean(recalls) if recalls else None,
        "macro_recall_valid_pct": mean(recalls) if recalls else None,
        "macro_f1_valid_pct": mean(f1s) if f1s else None,
        "mean_top1_confidence_pct": 100 * mean(confidences) if confidences else None,
        "mean_true_label_probability_pct": 100 * mean(true_probabilities) if true_probabilities else None,
        "negative_log_likelihood": mean(nll) if nll else None,
        "ece_10bin_pct": 100 * ece if valid else None,
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": cm,
    }


def paired_metrics(rows: list[dict], left: str, right: str) -> dict:
    """Compare two models on identical images where both produced valid scores."""
    shared = [
        row for row in rows
        if all(
            row.get("variants", {}).get(name, {}).get("ok")
            and len(row["variants"][name].get("probs", [])) == 7
            for name in (left, right)
        )
    ]
    left_correct = sum(row["variants"][left]["index"] == int(row["label"]) - 1 for row in shared)
    right_correct = sum(row["variants"][right]["index"] == int(row["label"]) - 1 for row in shared)
    right_wins = sum(
        row["variants"][right]["index"] == int(row["label"]) - 1
        and row["variants"][left]["index"] != int(row["label"]) - 1
        for row in shared
    )
    left_wins = sum(
        row["variants"][left]["index"] == int(row["label"]) - 1
        and row["variants"][right]["index"] != int(row["label"]) - 1
        for row in shared
    )

    discordant = right_wins + left_wins
    if discordant:
        # Two-sided exact McNemar/binomial test, evaluated in log space.
        from math import exp, lgamma, log

        lower_tail = sum(
            exp(lgamma(discordant + 1) - lgamma(k + 1) - lgamma(discordant - k + 1)
                - discordant * log(2))
            for k in range(min(right_wins, left_wins) + 1)
        )
        mcnemar_p = min(1.0, 2.0 * lower_tail)
    else:
        mcnemar_p = 1.0

    n = len(shared)
    both_wrong = sum(
        row["variants"][left]["index"] != int(row["label"]) - 1
        and row["variants"][right]["index"] != int(row["label"]) - 1
        for row in shared
    )
    return {
        "left": left,
        "right": right,
        "n_shared_valid": n,
        "left_accuracy_shared_pct": 100 * left_correct / n if n else None,
        "right_accuracy_shared_pct": 100 * right_correct / n if n else None,
        "right_minus_left_accuracy_pp": 100 * (right_correct - left_correct) / n if n else None,
        "left_only_correct": left_wins,
        "right_only_correct": right_wins,
        "both_correct": n - left_wins - right_wins - both_wrong,
        "both_wrong": both_wrong,
        "mcnemar_exact_two_sided_p": mcnemar_p,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-name", default="full")
    parser.add_argument(
        "--expected-images", type=int, default=0,
        help="Expected denominator for this run; default is the full label CSV count.",
    )
    args = parser.parse_args()
    bundle = args.bundle_root.resolve()
    out_dir = args.results_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with (bundle / "dataset" / "test_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = {row["image"]: int(row["label"]) for row in csv.DictReader(stream)}
    expected_images = args.expected_images or len(labels)
    if expected_images < 1 or expected_images > len(labels):
        raise SystemExit(f"--expected-images must be between 1 and {len(labels)}")
    ordered = sorted(((label, name) for name, label in labels.items()), key=lambda item: (item[0], item[1]))
    rows_by_key = load_rows([out_dir / "shard_0.jsonl", out_dir / "shard_1.jsonl"])
    rows = [rows_by_key[key] for key in ordered if key in rows_by_key]
    dataset_class_counts = {str(label): sum(value == label for value in labels.values()) for label in range(1, 8)}
    processed_class_counts = {
        str(label): sum(int(row["label"]) == label for row in rows)
        for label in range(1, 8)
    }
    result_metrics = {variant: metrics(rows, variant, expected_images) for variant in VARIANTS}
    paired = [
        paired_metrics(rows, VARIANTS[left], VARIANTS[right])
        for left in range(len(VARIANTS))
        for right in range(left + 1, len(VARIANTS))
    ]
    summary = {
        "dataset": "RAF-DB local test archive",
        "sample": args.sample_name,
        "expected_test_images": len(labels),
        "expected_run_images": expected_images,
        "processed_rows": len(rows),
        "all_images_processed": len(rows) == expected_images,
        "dataset_class_counts": dataset_class_counts,
        "processed_class_counts": processed_class_counts,
        "label_order": LABEL_NAMES,
        "metrics_definitions": {
            "accuracy_valid_pct": "correct / valid predictions; read with coverage_pct",
            "accuracy_all_images_pct": "correct / expected run images; failed branches and missing rows count as incorrect",
            "mean_class_accuracy_valid_pct": "mean of per-class recalls among valid predictions; conditional on successful predictions",
            "macro_recall_valid_pct": "unweighted recall over classes represented among valid predictions",
            "macro_f1_valid_pct": "unweighted F1 over classes represented among valid predictions",
            "mean_top1_confidence_pct": "mean maximum softmax probability on valid predictions",
            "mean_true_label_probability_pct": "mean softmax probability assigned to the ground-truth class",
            "ece_10bin_pct": "expected calibration error using 10 equal-width top-1 confidence bins",
        },
        "model_flows": json.loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))["flows"],
        "results": result_metrics,
        "paired_comparisons": paired,
        "paired_comparison_note": (
            "Accuracy deltas and exact two-sided McNemar p-values use only the shared valid images; "
            "interpret them alongside each model's full-test coverage."
        ),
        "local_validation_references": {
            "POSTER": {"accuracy_pct": 90.44, "split": "validation", "source": "model/ma3d/log/raf_POSTER.txt"},
            "MA3D": {"accuracy_pct": 90.67, "split": "validation", "source": "model/ma3d/log/raf_MA3D.txt"},
        },
        "validation_comparison_note": (
            "These repository log values are validation accuracy, not paper-reported test metrics; "
            "do not subtract them from this full-test accuracy as if the protocols matched."
        ),
        "rows_contain_images": False,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metric_path = out_dir / "metrics.csv"
    with metric_path.open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["variant", "n_valid", "coverage_pct", "accuracy_valid_pct", "accuracy_all_images_pct",
                   "mean_class_accuracy_valid_pct", "macro_recall_valid_pct", "macro_f1_valid_pct", "mean_top1_confidence_pct",
                   "mean_true_label_probability_pct", "negative_log_likelihood", "ece_10bin_pct"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for variant in VARIANTS:
            writer.writerow({"variant": variant, **{key: result_metrics[variant][key] for key in columns[1:]}})

    image_path = out_dir / "per_image_predictions.csv"
    fields = ["label", "file"]
    for variant in VARIANTS:
        fields.extend([f"{variant}_valid", f"{variant}_top1_index", f"{variant}_top1_confidence", f"{variant}_error"])
    with image_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {"label": row["label"], "file": row["file"]}
            for variant in VARIANTS:
                prediction = row.get("variants", {}).get(variant, {})
                flat[f"{variant}_valid"] = prediction.get("ok", False)
                flat[f"{variant}_top1_index"] = prediction.get("index", "")
                flat[f"{variant}_top1_confidence"] = prediction.get("confidence", "")
                flat[f"{variant}_error"] = prediction.get("error", "")
            writer.writerow(flat)

    paired_path = out_dir / "paired_comparisons.csv"
    paired_fields = [
        "left", "right", "n_shared_valid", "left_accuracy_shared_pct",
        "right_accuracy_shared_pct", "right_minus_left_accuracy_pp",
        "left_only_correct", "right_only_correct", "both_correct", "both_wrong",
        "mcnemar_exact_two_sided_p",
    ]
    with paired_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired)

    print(f"processed_rows={len(rows)}/{expected_images} (full_dataset={len(labels)})")
    for variant in VARIANTS:
        current = result_metrics[variant]
        print(
            f"{variant}: coverage={current['coverage_pct']:.2f}% "
            f"accuracy(valid)={current['accuracy_valid_pct']}% "
            f"accuracy(all)={current['accuracy_all_images_pct']}% "
            f"macro_f1={current['macro_f1_valid_pct']}% "
            f"mean_confidence={current['mean_top1_confidence_pct']}%"
        )
    print(f"Wrote {summary_path}, {metric_path}, {image_path}, and {paired_path}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
