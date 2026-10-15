from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


MODEL_NAME = "EMOCA_v2_lr_mse_20"
MAX_START_SECONDS = 900


def check_health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/v1/health", timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run disjoint RAF-DB shards across the requested GPUs.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working/rafdb5"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Small smoke test; 0 means all 3068 images.")
    parser.add_argument("--gpus", type=int, default=2, help="Number of visible GPUs/workers (default: 2).")
    args = parser.parse_args()
    if args.gpus < 1:
        raise SystemExit("--gpus must be at least 1")

    bundle = args.bundle_root.resolve()
    workdir = args.workdir.resolve()
    paths_file = workdir / "runtime_paths.json"
    if not paths_file.is_file():
        raise SystemExit(f"Run prepare_runtime.py first; missing {paths_file}")
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    project = Path(paths["project_root"])
    emoca_source = Path(paths["emoca_source_root"])
    model_dir = emoca_source / "assets" / "EMOCA" / "models"
    emoca_env_file = bundle / "environment" / "emoca-work38.yml"
    if not (emoca_source / ".git").is_dir():
        raise SystemExit("EMOCA source is missing; run setup_emoca_source.sh")
    if not (model_dir / MODEL_NAME).is_dir():
        raise SystemExit(f"EMOCA assets are missing; run download_emoca_assets.sh: {model_dir / MODEL_NAME}")
    if not (project / "model" / "ma3d" / "checkpoints" / "raf_MA3D.pth").is_file():
        raise SystemExit("MA3D checkpoint link is missing; rerun prepare_runtime.py")
    if not (project / "model" / "yolo26_face" / "yolo26_face.pt").is_file():
        raise SystemExit("YOLO26-face checkpoint link is missing; rerun prepare_runtime.py")
    if not emoca_env_file.is_file():
        raise SystemExit(f"Missing EMOCA environment file: {emoca_env_file}")

    conda = os.environ.get("CONDA_EXE") or "conda"
    output_dir = (args.output_dir or Path(paths["results_root"]) / ("smoke" if args.limit else "full")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    service_logs = []
    services: list[subprocess.Popen] = []
    workers: list[subprocess.Popen] = []
    try:
        for gpu in range(args.gpus):
            port = 8765 + gpu
            service_env = os.environ.copy()
            service_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            # Keep the Python 3.8 EMOCA process isolated from Kaggle's Python 3.12 site-packages.
            service_env["PYTHONPATH"] = str(emoca_source)
            service_env["PYTHONUNBUFFERED"] = "1"
            service_env["EMOCA_WARMUP"] = service_env.get("EMOCA_WARMUP", "1")
            service_log = (output_dir / f"emoca_gpu{gpu}.log").open("a", encoding="utf-8")
            service_logs.append(service_log)
            command = [
                conda, "run", "--no-capture-output", "-n", "work38", "python", "-u",
                str(project / "services" / "emoca_svc" / "server.py"),
                "--config", str(project / "configs" / "services.yaml"),
                "--models-dir", str(model_dir), "--model-name", MODEL_NAME, "--port", str(port),
            ]
            services.append(
                subprocess.Popen(
                    command,
                    cwd=project,
                    env=service_env,
                    stdout=service_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )

        for gpu in range(args.gpus):
            url = f"http://127.0.0.1:{8765 + gpu}"
            deadline = time.monotonic() + MAX_START_SECONDS
            while time.monotonic() < deadline and not check_health(url):
                if services[gpu].poll() is not None:
                    log_path = output_dir / f"emoca_gpu{gpu}.log"
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                    raise RuntimeError(f"EMOCA server GPU {gpu} exited ({services[gpu].returncode}):\n{tail}")
                time.sleep(3)
            if not check_health(url):
                raise TimeoutError(f"EMOCA server GPU {gpu} did not become healthy within {MAX_START_SECONDS}s")
            print(f"EMOCA GPU {gpu} healthy at {url}", flush=True)

        worker_script = Path(__file__).resolve().parent / "run_worker.py"
        for gpu in range(args.gpus):
            worker_env = os.environ.copy()
            worker_env.pop("PYTHONPATH", None)
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            worker_env["MA3D_DEVICE"] = "cuda:0"
            worker_env["EMOCA_URL"] = f"http://127.0.0.1:{8765 + gpu}"
            worker_env["YOLO_CONFIG_DIR"] = str(workdir / "ultralytics" / f"gpu{gpu}")
            Path(worker_env["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
            worker_output = output_dir / f"shard_{gpu}.jsonl"
            command = [
                sys.executable, str(worker_script),
                "--bundle-root", str(bundle), "--project-root", str(project),
                "--emoca-url", worker_env["EMOCA_URL"], "--out-jsonl", str(worker_output),
                "--shard-index", str(gpu), "--shards", str(args.gpus), "--limit", str(args.limit),
            ]
            workers.append(subprocess.Popen(command, cwd=project, env=worker_env))
        print(f"Started {args.gpus} worker(s). Output directory: {output_dir}", flush=True)

        while any(worker.poll() is None for worker in workers):
            time.sleep(5)
        statuses = [worker.returncode for worker in workers]
        print(f"Worker exit codes: {statuses}", flush=True)
        return 0 if all(status == 0 for status in statuses) else 1
    except KeyboardInterrupt:
        print("Interrupt received; checkpoint rows are flushed. Stopping child processes.", flush=True)
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
        for service in services:
            stop_process(service)
        for log in service_logs:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
