"""Summarize the ADR-028 roll-only sweep and choose the roll threshold post hoc.

Input: shard_*.jsonl written by run_frontalize_worker.py --mode roll2d. For each row the
`raw` branch is the rotated image and `frontal` is that image after 2D roll correction
(identical to raw when |roll| < --min-roll). For a threshold T the deployed rule is:
use `frontal` when |estimated roll| >= T, else `raw`. Failures (no_face, ...) count as wrong.
Writes summary.json and threshold_sweep.csv; prints Markdown tables. Numbers only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

CLASSES = ("surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral")
MODELS = ("poster_rafdb", "ma3d_rafdb")
DEFAULT_THRESHOLDS = (5, 10, 15, 20, 25, 30, 35, 40, 50, 60)


def load(results_dir: Path) -> list[dict]:
    latest = {}
    for path in sorted(results_dir.glob("shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                latest[(row["image"], int(row["angle"]))] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    if not latest:
        raise SystemExit(f"No shard_*.jsonl rows in {results_dir}")
    return list(latest.values())


def label(row: dict, branch: str, model: str) -> str | None:
    value = (row.get(branch) or {}).get(model) or {}
    return value.get("label") if value.get("label") in CLASSES else None


def roll(row: dict) -> float | None:
    pose = row.get("pose")
    return float(pose[2]) if pose else None


def chosen(row: dict, model: str, threshold: float | None) -> str | None:
    """threshold None = raw only; 0 = always corrected; else the deployed rule."""
    if threshold is None:
        return label(row, "raw", model)
    r = roll(row)
    if r is not None and abs(r) >= threshold:
        return label(row, "frontal", model)
    return label(row, "raw", model)


def macro_f1(rows: list[dict], model: str, threshold: float | None) -> float:
    f1s = []
    for name in CLASSES:
        tp = sum(1 for r in rows if r["label"] == name and chosen(r, model, threshold) == name)
        predicted = sum(1 for r in rows if chosen(r, model, threshold) == name)
        support = sum(1 for r in rows if r["label"] == name)
        p = tp / predicted if predicted else 0.0
        rc = tp / support if support else 0.0
        f1s.append(2 * p * rc / (p + rc) if p + rc else 0.0)
    return sum(f1s) / len(f1s)


def stats(rows: list[dict], model: str, threshold: float | None) -> dict:
    n = len(rows)
    preds = [chosen(r, model, threshold) for r in rows]
    valid = sum(p is not None for p in preds)
    correct = sum(p == r["label"] for p, r in zip(preds, rows))
    return {"n": n, "coverage": valid / n, "accuracy_all": correct / n, "macro_f1": macro_f1(rows, model, threshold)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    args = parser.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    rows = load(args.results_dir.resolve())
    angles = sorted({int(r["angle"]) for r in rows})
    by_angle = {a: [r for r in rows if int(r["angle"]) == a] for a in angles}

    # Roll estimator check: estimated roll vs applied rotation (upright baseline subtracted per image).
    upright = {r["image"]: roll(r) for r in by_angle.get(0, []) if roll(r) is not None}
    estimator = {}
    for a in angles:
        errs = [roll(r) - a - upright[r["image"]] for r in by_angle[a] if roll(r) is not None and r["image"] in upright]
        rolls = [roll(r) for r in by_angle[a] if roll(r) is not None]
        estimator[a] = {"median_roll": statistics.median(rolls) if rolls else None,
                        "median_abs_error_vs_true": statistics.median([abs(e) for e in errs]) if errs else None,
                        "share_abs_error_gt_15": (sum(abs(e) > 15 for e in errs) / len(errs)) if errs else None}

    policies = [("raw", None), ("always", 0.0)] + [(f"T={t:g}", t) for t in thresholds]
    table = []
    for model in MODELS:
        for name, t in policies:
            per_angle = {a: stats(by_angle[a], model, t) for a in angles}
            mean_acc = sum(s["accuracy_all"] for s in per_angle.values()) / len(per_angle)
            for a, s in per_angle.items():
                table.append({"model": model, "policy": name, "threshold": t, "angle": a, **s})
            table.append({"model": model, "policy": name, "threshold": t, "angle": "mean", "n": None,
                          "coverage": sum(s["coverage"] for s in per_angle.values()) / len(per_angle),
                          "accuracy_all": mean_acc,
                          "macro_f1": sum(s["macro_f1"] for s in per_angle.values()) / len(per_angle)})

    def acc(model, policy, angle):
        return next(r["accuracy_all"] for r in table if r["model"] == model and r["policy"] == policy and r["angle"] == angle)

    recommendation = {}
    for model in MODELS:
        raw0 = acc(model, "raw", 0) if 0 in angles else None
        candidates = []
        for name, t in policies[2:]:
            loss0 = (raw0 - acc(model, name, 0)) if raw0 is not None else 0.0
            candidates.append((acc(model, name, "mean"), -loss0, name, loss0))
        # Best mean accuracy over angles among thresholds that cost <= 0.2 points on upright images.
        ok = [c for c in candidates if c[3] <= 0.002] or candidates
        best = max(ok)
        recommendation[model] = {"policy": best[2], "mean_accuracy_all": best[0], "upright_loss_points": 100 * best[3]}

    out = args.results_dir.resolve()
    with (out / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "policy", "threshold", "angle", "n", "coverage", "accuracy_all", "macro_f1"])
        writer.writeheader()
        writer.writerows(table)
    summary = {"n_rows": len(rows), "angles": angles, "estimator": estimator, "table": table,
               "recommendation": recommendation,
               "note": "Failures count as wrong. Rule: corrected if |estimated roll| >= T else raw. "
                       "Recommendation = best mean accuracy over angles with <= 0.2 pt loss at 0 deg."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n### Bộ ước lượng roll (3DDFA-V2)")
    print("| góc xoay | roll trung vị | sai số tuyệt đối trung vị | % sai > 15° |")
    print("|---|---|---|---|")
    for a, e in estimator.items():
        f = lambda v, pct=False: "-" if v is None else (f"{100 * v:.1f} %" if pct else f"{v:.1f}°")
        print(f"| {a}° | {f(e['median_roll'])} | {f(e['median_abs_error_vs_true'])} | {f(e['share_abs_error_gt_15'], True)} |")
    for model in MODELS:
        print(f"\n### {model}: Acc(all) theo chính sách × góc")
        print("| chính sách | " + " | ".join(f"{a}°" for a in angles) + " | trung bình | macro F1 TB |")
        print("|---" * (len(angles) + 3) + "|")
        for name, _ in policies:
            cells = [f"{100 * acc(model, name, a):.1f}" for a in angles]
            mean_row = next(r for r in table if r["model"] == model and r["policy"] == name and r["angle"] == "mean")
            print(f"| {name} | " + " | ".join(cells) + f" | {100 * mean_row['accuracy_all']:.1f} | {100 * mean_row['macro_f1']:.1f} |")
        rec = recommendation[model]
        print(f"\nĐề xuất {model}: {rec['policy']} (Acc TB {100 * rec['mean_accuracy_all']:.1f} %, mất ở 0°: {rec['upright_loss_points']:.2f} điểm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
