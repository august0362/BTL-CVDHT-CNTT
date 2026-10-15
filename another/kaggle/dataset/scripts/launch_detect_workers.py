from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch resumable degraded-scene detection shards on visible GPUs.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working/rafdb5"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sizes", default="24,48,96")
    parser.add_argument("--degradations", default="clean,blur,down4,jpeg10,noise20,combo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--canvas", default="640,480")
    parser.add_argument("--background-field", default="10,8")
    parser.add_argument("--background-rectangles", type=int, default=5)
    parser.add_argument("--background-color-range", default="24,232")
    parser.add_argument("--background-gradient-x", type=float, default=22.0)
    parser.add_argument("--background-gradient-y", type=float, default=18.0)
    parser.add_argument("--background-tint-range", type=int, default=12)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--down4-factor", type=int, default=4)
    parser.add_argument("--jpeg10-quality", type=int, default=10)
    parser.add_argument("--noise-sigma", type=float, default=20.0)
    parser.add_argument("--combo-downscale", type=int, default=2)
    parser.add_argument("--combo-blur-sigma", type=float, default=1.0)
    parser.add_argument("--combo-jpeg-quality", type=int, default=20)
    parser.add_argument("--confidence-min", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-pad", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()
    if args.gpus < 1:
        raise SystemExit("--gpus must be at least 1")
    bundle, workdir = args.bundle_root.resolve(), args.workdir.resolve()
    paths_file = workdir / "runtime_paths.json"
    if not paths_file.is_file():
        raise SystemExit(f"Run prepare_runtime.py first; missing {paths_file}")
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    project = Path(paths["project_root"]).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_script = Path(__file__).resolve().parent / "run_detect_degraded_worker.py"
    workers: list[subprocess.Popen] = []
    logs = []
    try:
        for gpu in range(args.gpus):
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            log = (output_dir / f"worker_gpu{gpu}.log").open("a", encoding="utf-8")
            logs.append(log)
            command = [sys.executable, str(worker_script), "--bundle-root", str(bundle), "--project-root", str(project),
                       "--out-jsonl", str(output_dir / f"shard_{gpu}.jsonl"), "--shard-index", str(gpu),
                       "--shards", str(args.gpus), "--images", str(args.images), "--limit", str(args.limit),
                       "--sizes", args.sizes, "--degradations", args.degradations, "--seed", str(args.seed),
                       "--canvas", args.canvas, "--background-field", args.background_field,
                       "--background-rectangles", str(args.background_rectangles),
                       "--background-color-range", args.background_color_range,
                       "--background-gradient-x", str(args.background_gradient_x),
                       "--background-gradient-y", str(args.background_gradient_y),
                       "--background-tint-range", str(args.background_tint_range), "--blur-sigma", str(args.blur_sigma),
                       "--down4-factor", str(args.down4_factor), "--jpeg10-quality", str(args.jpeg10_quality),
                       "--noise-sigma", str(args.noise_sigma), "--combo-downscale", str(args.combo_downscale),
                       "--combo-blur-sigma", str(args.combo_blur_sigma), "--combo-jpeg-quality", str(args.combo_jpeg_quality),
                       "--confidence-min", str(args.confidence_min), "--tile-size", str(args.tile_size), "--tile-pad", str(args.tile_pad)]
            workers.append(subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT))
        print(f"Started {args.gpus} detection worker(s). Output directory: {output_dir}", flush=True)
        while any(worker.poll() is None for worker in workers):
            time.sleep(5)
        statuses = [worker.returncode for worker in workers]
        for gpu, status in enumerate(statuses):
            if status:
                print((output_dir / f"worker_gpu{gpu}.log").read_text(encoding="utf-8", errors="replace")[-5000:])
        print(f"Worker exit codes: {statuses}", flush=True)
        return 0 if all(code == 0 for code in statuses) else 1
    except KeyboardInterrupt:
        print("Interrupt received; JSONL rows are flushed. Stopping workers.", flush=True)
        return 130
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
        for log in logs:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
