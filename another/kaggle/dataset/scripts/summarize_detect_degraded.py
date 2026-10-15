from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


BRANCHES = ("yolo", "esrgan_yolo")
THRESHOLDS = (0.25, 0.5)


def iou_xyxy(first, second) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def exact_mcnemar(fixed: int, broken: int) -> float:
    discordant = fixed + broken
    if discordant == 0:
        return 1.0
    tail = min(fixed, broken)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def load_rows(results_dir: Path) -> list[dict]:
    latest: dict[tuple[str, int, str], dict] = {}
    files = sorted(results_dir.glob("shard_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No shard_*.jsonl files in {results_dir}")
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    latest[(str(row["image"]), int(row["size"]), str(row["degradation"]))] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    return sorted(latest.values(), key=lambda row: (str(row["degradation"]), int(row["size"]), str(row["image"])))


def branch_metrics(rows: list[dict], branch: str, threshold: float, hit_iou: float = 0.4,
                   secondary_iou: float = 0.5) -> dict:
    considered = [row for row in rows if row.get("gt_box") is not None]
    n = len(considered)
    hit = 0
    hit50 = 0
    matched_ious = []
    matched_confidences = []
    false_positives = 0
    esrgan_times = []
    yolo_times = []
    for row in considered:
        boxes = [box for box in ((row.get("branches") or {}).get(branch) or {}).get("boxes", [])
                 if float(box.get("conf", 0)) >= threshold]
        overlaps = [(iou_xyxy(box["xyxy"], row["gt_box"]), float(box.get("conf", 0))) for box in boxes]
        best = max(overlaps, default=(0.0, 0.0), key=lambda item: item[0])
        if best[0] >= hit_iou:
            hit += 1
            matched_ious.append(best[0])
            matched_confidences.append(best[1])
        if best[0] >= secondary_iou:
            hit50 += 1
        false_positives += sum(overlap < hit_iou for overlap, _ in overlaps)
        timing = ((row.get("branches") or {}).get(branch) or {}).get("timings_ms") or {}
        try:
            esrgan_times.append(float(timing.get("esrgan", 0.0)))
            yolo_times.append(float(timing["yolo"]))
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "n_scenes": n,
        "recall_iou_0_4": hit / n if n else None,
        "recall_iou_0_5": hit50 / n if n else None,
        "mean_iou_matched": statistics.mean(matched_ious) if matched_ious else None,
        "false_positives_per_scene": false_positives / n if n else None,
        "mean_matched_confidence": statistics.mean(matched_confidences) if matched_confidences else None,
        "mean_esrgan_ms": statistics.mean(esrgan_times) if esrgan_times else None,
        "median_esrgan_ms": statistics.median(esrgan_times) if esrgan_times else None,
        "mean_yolo_ms": statistics.mean(yolo_times) if yolo_times else None,
        "median_yolo_ms": statistics.median(yolo_times) if yolo_times else None,
    }


def paired_metrics(rows: list[dict], threshold: float = 0.5, hit_iou: float = 0.4) -> dict:
    fixed = broken = 0
    for row in rows:
        gt = row.get("gt_box")
        if gt is None:
            continue
        hits = {}
        for branch in BRANCHES:
            boxes = [box for box in ((row.get("branches") or {}).get(branch) or {}).get("boxes", [])
                     if float(box.get("conf", 0)) >= threshold]
            hits[branch] = any(iou_xyxy(box["xyxy"], gt) >= hit_iou for box in boxes)
        fixed += int(not hits["yolo"] and hits["esrgan_yolo"])
        broken += int(hits["yolo"] and not hits["esrgan_yolo"])
    return {"threshold": threshold, "fixed_miss_to_hit": fixed, "broken_hit_to_miss": broken,
            "discordant": fixed + broken, "mcnemar_exact_two_sided_p": exact_mcnemar(fixed, broken)}


def summarize(results_dir: Path, thresholds: tuple[float, ...] = THRESHOLDS,
              hit_iou: float = 0.4, secondary_iou: float = 0.5,
              paired_threshold: float = 0.5) -> dict:
    rows = load_rows(results_dir)
    keys = sorted({(str(r["degradation"]), int(r["size"])) for r in rows})
    metric_rows = []
    paired_rows = []
    for degradation, size in keys:
        subset = [r for r in rows if r["degradation"] == degradation and int(r["size"]) == size]
        for branch in BRANCHES:
            for threshold in thresholds:
                metric_rows.append({"scope": "size", "degradation": degradation, "size": size,
                                   "branch": branch, "threshold": threshold,
                                   **branch_metrics(subset, branch, threshold, hit_iou, secondary_iou)})
        paired_rows.append({"scope": "size", "degradation": degradation, "size": size,
                            **paired_metrics(subset, paired_threshold, hit_iou)})
    for degradation in sorted({str(r["degradation"]) for r in rows}):
        subset = [r for r in rows if r["degradation"] == degradation]
        for branch in BRANCHES:
            for threshold in thresholds:
                metric_rows.append({"scope": "all_sizes", "degradation": degradation, "size": "all",
                                   "branch": branch, "threshold": threshold,
                                   **branch_metrics(subset, branch, threshold, hit_iou, secondary_iou)})
        paired_rows.append({"scope": "all_sizes", "degradation": degradation, "size": "all",
                            **paired_metrics(subset, paired_threshold, hit_iou)})
    errors = sum(bool(row.get("error")) for row in rows)
    summary = {"dataset": "RAF-DB aligned test faces pasted into synthetic scenes",
               "n_scenes": len(rows), "n_errors": errors, "degradations": sorted({str(r["degradation"]) for r in rows}),
               "sizes": sorted({int(r["size"]) for r in rows}), "metric_rows": metric_rows,
               "paired_comparisons": paired_rows,
               "iou_thresholds": {"match": hit_iou, "secondary_recall": secondary_iou},
               "confidence_thresholds": list(thresholds),
               "note": f"Detection metrics use confidence >= threshold. A hit is any box at IoU >= {hit_iou}; IoU >= {secondary_iou} recall is also reported. Mean matched IoU/confidence use the highest-IoU box among hits."}
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(results_dir / "metrics.csv", metric_rows)
    write_scenes_csv(results_dir / "scenes.csv", rows)
    print_markdown(metric_rows, paired_rows)
    return summary


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    columns = ["scope", "degradation", "size", "branch", "threshold", "n_scenes", "recall_iou_0_4", "recall_iou_0_5",
               "mean_iou_matched", "false_positives_per_scene", "mean_matched_confidence", "mean_esrgan_ms",
               "median_esrgan_ms", "mean_yolo_ms", "median_yolo_ms"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def write_scenes_csv(path: Path, rows: list[dict]) -> None:
    columns = ["image", "class", "class_id", "size", "degradation", "gt_box", "yolo_boxes", "yolo_esrgan_ms",
               "yolo_yolo_ms", "esrgan_yolo_boxes", "esrgan_yolo_esrgan_ms", "esrgan_yolo_yolo_ms", "error"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in ("image", "class", "class_id", "size", "degradation", "error")}
            flat["gt_box"] = json.dumps(row.get("gt_box"), separators=(",", ":"))
            for branch in BRANCHES:
                data = (row.get("branches") or {}).get(branch) or {}
                flat[f"{branch}_boxes"] = json.dumps(data.get("boxes", []), separators=(",", ":"))
                timing = data.get("timings_ms") or {}
                flat[f"{branch}_esrgan_ms"] = timing.get("esrgan")
                flat[f"{branch}_yolo_ms"] = timing.get("yolo")
            writer.writerow(flat)


def print_markdown(metrics: list[dict], paired: list[dict]) -> None:
    print("| scope | degradation | size | branch | conf | n | recall .4 | recall .5 | FP/scene | mean IoU | mean conf | mean/median ESR ms | mean/median YOLO ms |")
    print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in metrics:
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        print(f"| {row['scope']} | {row['degradation']} | {row['size']} | {row['branch']} | {row['threshold']:.2f} | {row['n_scenes']} | {fmt(row['recall_iou_0_4'])} | {fmt(row['recall_iou_0_5'])} | {fmt(row['false_positives_per_scene'])} | {fmt(row['mean_iou_matched'])} | {fmt(row['mean_matched_confidence'])} | {fmt(row['mean_esrgan_ms'])}/{fmt(row['median_esrgan_ms'])} | {fmt(row['mean_yolo_ms'])}/{fmt(row['median_yolo_ms'])} |")
    print("\n| scope | degradation | size | fixed | broken | exact McNemar p |")
    print("|---|---|---:|---:|---:|---:|")
    for row in paired:
        print(f"| {row['scope']} | {row['degradation']} | {row['size']} | {row['fixed_miss_to_hit']} | {row['broken_hit_to_miss']} | {row['mcnemar_exact_two_sided_p']:.6g} |")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize degraded-scene YOLO detection results.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.25,0.5")
    parser.add_argument("--match-iou", type=float, default=0.4)
    parser.add_argument("--secondary-iou", type=float, default=0.5)
    parser.add_argument("--paired-threshold", type=float, default=0.5)
    args = parser.parse_args()
    try:
        thresholds = tuple(float(value) for value in args.thresholds.split(",") if value.strip())
    except ValueError as exc:
        parser.error(f"invalid thresholds: {exc}")
    summarize(args.results_dir.resolve(), thresholds, args.match_iou, args.secondary_iou, args.paired_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
