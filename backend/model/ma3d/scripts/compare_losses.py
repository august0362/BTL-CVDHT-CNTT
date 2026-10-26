"""
Compare --loss mace with --loss bmace on CheoFaMo, validating on a held-out part of the training set.

Run from backend/model/ma3d:
    python -m scripts.compare_losses --check     # check data, weights and environment, estimate the time
    python -m scripts.compare_losses             # train every configuration that has not finished yet
    python -m scripts.compare_losses --summary   # table of the finished runs
"""
import os
import re
import sys
import argparse
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Datasets", "CheoFamo")
WEIGHTS = ["checkpoints/ir50_o1.pth", "checkpoints/mobilefacenet_model_best.pth_o1.tar"]
FEATURES = {"exp": 50, "detail": 128, "pose": 6, "shape": 100, "tex": 50}
PAPER_COUNTS = {"Happiness": (2658, 665), "Sadness": (1249, 313), "Fear": (266, 66),
                "Love": (217, 54), "Anger": (256, 64), "Hatred": (29, 7)}
SECONDS_PER_STEP = 0.72  # MA3D + SAM, batch 8, measured on a GTX 1660 Ti
SECONDS_PER_EVAL_BATCH = 0.074


def get_args():
    parser = argparse.ArgumentParser("Compare MACE and BalancedMarginAwareCELoss on CheoFaMo")
    parser.add_argument("--check", action="store_true", help="only check data, weights and environment")
    parser.add_argument("--summary", action="store_true", help="only print the table of finished runs")
    parser.add_argument("--compute_stats", action="store_true",
                        help="write Datasets/CheoFamo/cheofamo_stats.npz if it is missing")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.25])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default="compare")
    return parser.parse_args()


def configurations(args):
    runs = [("mace", ["--loss", "mace"])]
    runs += [(f"bmace_tau{tau:g}", ["--loss", "bmace", "--tau", str(tau)]) for tau in args.taus]
    return [(f"{args.tag}_{name}_s{args.seed}_e{args.epochs}", extra) for name, extra in runs]


def log_path(run_name):
    return os.path.join(ROOT, "log", run_name + ".txt")


def is_finished(run_name):
    path = log_path(run_name)
    return os.path.exists(path) and "Best validation macro-F1" in open(path).read()


# ---------------------------------------------------------------- check
def check(args):
    problems, warnings = [], []

    try:
        import torch
        import torchvision  # noqa: F401
        if not torch.cuda.is_available():
            problems.append("CUDA is not available to PyTorch")
    except ImportError as e:
        problems.append(f"missing package: {e.name}")

    for w in WEIGHTS:
        if not os.path.isfile(os.path.join(ROOT, w)):
            problems.append(f"missing backbone weights: {w}")

    if not os.path.isfile(os.path.join(DATA_DIR, "Metadata.json")) or not os.path.isdir(os.path.join(DATA_DIR, "Frames")):
        problems.append(f"CheoFaMo not found: expected {DATA_DIR}\\Metadata.json and {DATA_DIR}\\Frames\\")
        return report(problems, warnings, None)

    sys.path.insert(0, ROOT)
    from Read_dataset import CheoFaMo

    train = CheoFaMo(DATA_DIR, split="train")
    test = CheoFaMo(DATA_DIR, split="test")
    print(f"Segments: train {len(train.samples)}, test {len(test.samples)} (paper: 4,675 / 1,169)")
    for name, (n_train, n_test) in PAPER_COUNTS.items():
        got_train = sum(s["emotion"] == name for s in train.samples)
        got_test = sum(s["emotion"] == name for s in test.samples)
        if (got_train, got_test) != (n_train, n_test):
            warnings.append(f"{name}: train/test {got_train}/{got_test}, paper {n_train}/{n_test}")
    unknown = set(train.label2idx) - set(PAPER_COUNTS)
    if unknown:
        warnings.append(f"emotion names not in the paper: {sorted(unknown)}")

    multi_image, bad_features = 0, []
    for s in train.samples + test.samples:
        files = os.listdir(s["folder"])
        multi_image += sum(f.lower().endswith((".jpg", ".png", ".jpeg")) for f in files) > 1
        for key, dim in FEATURES.items():
            path = os.path.join(s["folder"], key + ".npy")
            if not os.path.isfile(path) or np.load(path).shape != (dim,):
                bad_features.append(f"{os.path.basename(s['folder'])}/{key}.npy")
    if bad_features:
        problems.append(f"{len(bad_features)} missing or wrong-sized feature files, e.g. {bad_features[:3]}")
    if multi_image:
        warnings.append(f"{multi_image} segment folders hold several images; the loader uses the first one "
                        f"os.listdir returns, which may not be the frame the features came from")

    stats_path = os.path.join(DATA_DIR, "cheofamo_stats.npz")
    if not os.path.isfile(stats_path):
        problems.append("cheofamo_stats.npz is missing, so features would not be normalised "
                        "(run with --compute_stats)")
    else:
        stats = np.load(stats_path)
        missing = [k for k in FEATURES for part in ("mean", "std") if f"{k}_{part}" not in stats]
        if missing:
            problems.append(f"cheofamo_stats.npz lacks statistics for {sorted(set(missing))}")

    steps = int(len(train.samples) * (1 - args.val_split)) // args.batch_size
    eval_batches = int(len(train.samples) * args.val_split) / args.batch_size
    hours = args.epochs * (steps * SECONDS_PER_STEP + eval_batches * SECONDS_PER_EVAL_BATCH) / 3600
    todo = [name for name, _ in configurations(args) if not is_finished(name)]
    return report(problems, warnings, (hours, len(todo)))


def report(problems, warnings, estimate):
    for w in warnings:
        print(f"[WARN] {w}")
    for p in problems:
        print(f"[MISSING] {p}")
    if estimate:
        hours, n = estimate
        print(f"Estimated GPU time: {hours:.1f} h per run, {hours * n:.1f} h for the {n} unfinished run(s)")
    print("Ready to train." if not problems else "Not ready.")
    return not problems


# ---------------------------------------------------------------- run
def run(args):
    for name, extra in configurations(args):
        if is_finished(name):
            print(f"[skip] {name} already finished")
            continue
        if os.path.exists(log_path(name)):
            os.remove(log_path(name))  # an interrupted run is restarted from scratch
        cmd = [sys.executable, "train.py", "--data_type", "CheoFaMo", "--num_classes", "6",
               "--epochs", str(args.epochs), "--seed", str(args.seed), "--val_split", str(args.val_split),
               "--batch_size", str(args.batch_size), "--num_workers", str(args.num_workers),
               "--log_file", name + ".txt", "--resume_name", name + ".pth"] + extra
        print(f"[run] {name}: {' '.join(cmd[1:])}", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print(f"[fail] {name}; see {log_path(name)}")
            return False
    return True


# ---------------------------------------------------------------- summary
ROW = re.compile(r"^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$")


def summary(args):
    print(f"\nHeld-out validation ({args.val_split:.0%} of the training set), seed {args.seed}, {args.epochs} epochs")
    print("| Run | Best acc (epoch) | Best bal. acc | Best macro-F1 (epoch) | Mean of last 5 epochs: acc / bal. acc / macro-F1 |")
    print("|---|---|---|---|---|")
    for name, _ in configurations(args):
        if not os.path.exists(log_path(name)):
            print(f"| {name} | not run | | | |")
            continue
        rows = [tuple(map(float, m.groups())) for m in map(ROW.match, open(log_path(name))) if m]
        if not rows:
            print(f"| {name} | no epochs logged | | | |")
            continue
        rows = np.array(rows)
        best_acc, best_f1 = rows[rows[:, 1].argmax()], rows[rows[:, 3].argmax()]
        last = rows[-5:, 1:].mean(axis=0)
        status = "" if is_finished(name) else " (unfinished)"
        print(f"| {name}{status} | {best_acc[1]:.2f} ({int(best_acc[0])}) | {rows[:, 2].max():.2f} | "
              f"{best_f1[3]:.2f} ({int(best_f1[0])}) | {last[0]:.2f} / {last[1]:.2f} / {last[2]:.2f} |")


def main():
    args = get_args()
    if args.summary:
        return summary(args)
    if args.compute_stats and not os.path.isfile(os.path.join(DATA_DIR, "cheofamo_stats.npz")):
        sys.path.insert(0, ROOT)
        from Read_dataset.Cheo_FaMo import compute_cheofamo_stats
        compute_cheofamo_stats(DATA_DIR)
    ready = check(args)
    if args.check or not ready:
        return
    if run(args):
        summary(args)


if __name__ == "__main__":
    main()
