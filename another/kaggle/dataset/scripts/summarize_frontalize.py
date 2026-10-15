from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ("surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral")
MODELS = ("poster_rafdb", "ma3d_rafdb")
BRANCHES = ("raw", "frontal")


def load_rows(results_dir: Path) -> list[dict]:
    latest: dict[tuple[str, int], dict] = {}
    files = sorted(results_dir.glob("shard_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No shard_*.jsonl files in {results_dir}")
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    latest[(str(row["image"]), int(row["angle"]))] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    return sorted(latest.values(), key=lambda row: (str(row["image"]), int(row["angle"])))


def prediction(row: dict, branch: str, model: str) -> dict | None:
    value = (row.get(branch) or {}).get(model)
    if not isinstance(value, dict) or "label" not in value:
        return None
    return value


def metrics(rows: list[dict], branch: str, model: str) -> dict:
    n = len(rows)
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    valid = 0
    correct = 0
    true_probs = []
    for row in rows:
        true_label = str(row.get("label", ""))
        if true_label not in CLASSES:
            continue
        true_index = CLASSES.index(true_label)
        pred = prediction(row, branch, model)
        if pred is None or pred.get("label") not in CLASSES:
            # Missing or failed predictions count as wrong and as a false negative for truth.
            continue
        valid += 1
        pred_index = CLASSES.index(pred["label"])
        confusion[true_index][pred_index] += 1
        correct += int(pred_index == true_index)
        probs = pred.get("probs") or {}
        try:
            true_probs.append(float(probs.get(true_label, 0.0)))
        except (TypeError, ValueError):
            true_probs.append(0.0)

    per_class = {}
    f1_scores = []
    for index, name in enumerate(CLASSES):
        tp = confusion[index][index]
        support = sum(1 for row in rows if row.get("label") == name)
        predicted = sum(confusion[actual][index] for actual in range(len(CLASSES)))
        # Failed predictions belong in actual support but not predicted support.
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "f1": f1, "f1_pct": 100 * f1}
        f1_scores.append(f1)
    return {
        "n": n,
        "n_valid": valid,
        "n_failed": n - valid,
        "coverage": valid / n if n else None,
        "accuracy_valid": correct / valid if valid else None,
        "accuracy_all": correct / n if n else None,
        "macro_f1": sum(f1_scores) / len(CLASSES) if n else None,
        "mean_p_true_label": sum(true_probs) / len(true_probs) if true_probs else None,
        "per_class_f1": per_class,
        "confusion_matrix": confusion,
    }


def exact_mcnemar(fixed: int, broken: int) -> float | None:
    discordant = fixed + broken
    if discordant == 0:
        return 1.0
    tail = min(fixed, broken)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def paired(rows: list[dict], model: str) -> dict:
    fixed = broken = 0
    for row in rows:
        truth = row.get("label")
        raw = prediction(row, "raw", model)
        frontal = prediction(row, "frontal", model)
        raw_correct = raw is not None and raw.get("label") == truth
        frontal_correct = frontal is not None and frontal.get("label") == truth
        fixed += int(not raw_correct and frontal_correct)
        broken += int(raw_correct and not frontal_correct)
    return {"fixed_wrong_to_right": fixed, "broken_right_to_wrong": broken,
            "discordant": fixed + broken, "mcnemar_exact_two_sided_p": exact_mcnemar(fixed, broken)}


def pose_bin(row: dict, dimension: str) -> str | None:
    pose = row.get("pose")
    if not isinstance(pose, list) or len(pose) < 2:
        return None
    value = abs(float(pose[0 if dimension == "yaw" else 1]))
    return "<15" if value < 15 else ("15-30" if value < 30 else ">=30")


def metric_record(angle: int | None, group_type: str, group: str, branch: str, model: str, rows: list[dict]) -> dict:
    result = metrics(rows, branch, model)
    return {"angle": angle, "group_type": group_type, "group": group, "branch": branch,
            "model": model, **result}


def summarize(results_dir: Path) -> dict:
    rows = load_rows(results_dir)
    metric_rows = []
    paired_rows = []
    by_angle: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_angle[int(row["angle"])].append(row)
    for angle, subset in sorted(by_angle.items()):
        for branch in BRANCHES:
            for model in MODELS:
                metric_rows.append(metric_record(angle, "angle", str(angle), branch, model, subset))
        for model in MODELS:
            paired_rows.append({"angle": angle, "model": model, **paired(subset, model)})

    natural = by_angle.get(0, [])
    pose_groups = {"yaw": ("<15", "15-30", ">=30"), "pitch": ("<15", "15-30", ">=30")}
    for dimension, bins in pose_groups.items():
        for bin_name in bins:
            subset = [row for row in natural if pose_bin(row, dimension) == bin_name]
            if not subset:
                continue
            for branch in BRANCHES:
                for model in MODELS:
                    metric_rows.append(metric_record(0, dimension, bin_name, branch, model, subset))

    stage_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    for row in rows:
        for key in ("raw_error", "frontal_error", "frontal_analyze_error", "input_error"):
            if row.get(key):
                stage_counts[key.removesuffix("_error") + ":error"] += 1
                error_counts[f"{key.removesuffix('_error')}|{row[key]}"] += 1
        for branch in BRANCHES:
            for model in MODELS:
                pred = (row.get(branch) or {}).get(model) or {}
                if "error" in pred:
                    stage_counts[f"{branch}:{model}:error"] += 1
                    error_counts[f"{branch}:{model}|{pred['error']}"] += 1
                else:
                    stage_counts[f"{branch}:{model}:ok"] += 1
        timings = row.get("timings") or {}
        for stage in ("ms_raw", "ms_frontalize", "ms_frontal"):
            stage_counts[f"{stage}:{'ok' if stage in timings else 'missing'}"] += 1

    summary = {
        "dataset": "RAF-DB aligned test split",
        "n_rows": len(rows),
        "angles": sorted(by_angle),
        "metric_rows": metric_rows,
        "paired_comparisons": paired_rows,
        "stage_counts": dict(sorted(stage_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "note": "Failed predictions count as incorrect for accuracy(all), macro F1, and paired flips; coverage uses valid predictions.",
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(results_dir / "metrics.csv", metric_rows)
    write_predictions_csv(results_dir / "predictions.csv", rows)
    print_markdown(metric_rows, paired_rows, stage_counts, error_counts)
    return summary


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    columns = ["angle", "group_type", "group", "branch", "model", "n", "n_valid", "n_failed", "coverage",
               "accuracy_valid", "accuracy_all", "macro_f1", "mean_p_true_label"] + [f"f1_{name}" for name in CLASSES]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in columns if not key.startswith("f1_")}
            flat.update({f"f1_{name}": row["per_class_f1"][name]["f1"] for name in CLASSES})
            writer.writerow(flat)


def write_predictions_csv(path: Path, rows: list[dict]) -> None:
    columns = ["image", "label", "angle", "yaw", "pitch", "roll", "occluded", "ms_raw", "ms_frontalize", "ms_frontal", "frontal_error", "frontal_analyze_error", "raw_error", "input_error"]
    for branch in BRANCHES:
        for model in MODELS:
            columns.extend([f"{branch}_{model}_label", f"{branch}_{model}_probs", f"{branch}_{model}_error"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            pose = row.get("pose") or [None, None, None]
            flat = {"image": row.get("image"), "label": row.get("label"), "angle": row.get("angle"),
                    "yaw": pose[0], "pitch": pose[1], "roll": pose[2], "occluded": row.get("occluded"),
                    "frontal_error": row.get("frontal_error", ""),
                    "frontal_analyze_error": row.get("frontal_analyze_error", ""),
                    "raw_error": row.get("raw_error", ""), "input_error": row.get("input_error", "")}
            flat.update(row.get("timings") or {})
            for branch in BRANCHES:
                for model in MODELS:
                    pred = (row.get(branch) or {}).get(model) or {}
                    flat[f"{branch}_{model}_label"] = pred.get("label", "")
                    flat[f"{branch}_{model}_probs"] = json.dumps(pred.get("probs", {}), ensure_ascii=False, separators=(",", ":"))
                    flat[f"{branch}_{model}_error"] = pred.get("error", "")
            writer.writerow(flat)


def print_markdown(metric_rows: list[dict], paired_rows: list[dict], stages: Counter, errors: Counter) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n### Per angle × branch × model")
    print("| angle | branch | model | n | coverage | accuracy(valid) | accuracy(all) | macro F1 | mean p(true) |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for row in metric_rows:
        if row["group_type"] != "angle":
            continue
        pct = lambda value: "—" if value is None else f"{100 * value:.2f}%"
        mean_p = "—" if row["mean_p_true_label"] is None else f"{row['mean_p_true_label']:.4f}"
        print(f"| {row['group']} | {row['branch']} | {row['model']} | {row['n']} | {pct(row['coverage'])} | {pct(row['accuracy_valid'])} | {pct(row['accuracy_all'])} | {pct(row['macro_f1'])} | {mean_p} |")
    print("\n### Per-class F1 (%)")
    print("| group type | group | angle | branch | model | " + " | ".join(CLASSES) + " |")
    print("|---|---|---:|---|---|" + "---:|" * len(CLASSES))
    for row in metric_rows:
        f1s = " | ".join(f"{100 * row['per_class_f1'][name]['f1']:.2f}" for name in CLASSES)
        print(f"| {row['group_type']} | {row['group']} | {row['angle'] if row['angle'] is not None else ''} | {row['branch']} | {row['model']} | {f1s} |")
    print("\n### Angle 0 pose bins (raw vs frontal)")
    print("| dimension | |pose| bin | model | n | branch | coverage | accuracy(valid) | accuracy(all) | macro F1 | mean p(true) |")
    print("|---|---|---|---:|---|---:|---:|---:|---:|---:|")
    for row in metric_rows:
        if row["group_type"] == "angle":
            continue
        pct = lambda value: "—" if value is None else f"{100 * value:.2f}%"
        mean_p = "—" if row["mean_p_true_label"] is None else f"{row['mean_p_true_label']:.4f}"
        print(f"| {row['group_type']} | {row['group']} | {row['model']} | {row['n']} | {row['branch']} | {pct(row['coverage'])} | {pct(row['accuracy_valid'])} | {pct(row['accuracy_all'])} | {pct(row['macro_f1'])} | {mean_p} |")
    print("\n### Paired raw → frontal")
    print("| angle | model | fixed (wrong→right) | broken (right→wrong) | exact McNemar p |")
    print("|---:|---|---:|---:|---:|")
    for row in paired_rows:
        print(f"| {row['angle']} | {row['model']} | {row['fixed_wrong_to_right']} | {row['broken_right_to_wrong']} | {row['mcnemar_exact_two_sided_p']:.6g} |")
    print("\n### Stage/error counts")
    print("| count | stage/status or error |")
    print("|---:|---|")
    for name, count in sorted(stages.items()):
        print(f"| {count} | {name} |")
    for name, count in sorted(errors.items()):
        print(f"| {count} | {name} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the RAF-DB 3DDFA frontalization benchmark.")
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.results_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
