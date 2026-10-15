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
    parser = argparse.ArgumentParser(description="Run frontalization benchmark shards with one EMOCA server per GPU.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working/rafdb5"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--angles", default="0,45,90")
    parser.add_argument("--mode", choices=("frontal3d", "roll2d"), default="frontal3d")
    parser.add_argument("--min-roll", type=float, default=5.0)
    args = parser.parse_args()
    if args.gpus < 1:
        raise SystemExit("--gpus must be at least 1")
    bundle, workdir = args.bundle_root.resolve(), args.workdir.resolve()
    paths_file = workdir / "runtime_paths.json"
    if not paths_file.is_file():
        raise SystemExit(f"Run prepare_runtime.py first; missing {paths_file}")
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    project, emoca_source = Path(paths["project_root"]), Path(paths["emoca_source_root"])
    model_dir = emoca_source / "assets" / "EMOCA" / "models"
    env_file = bundle / "environment" / "emoca-work38.yml"
    if not (emoca_source / ".git").is_dir():
        raise SystemExit("EMOCA source missing; run setup_emoca_source.sh")
    if not (model_dir / MODEL_NAME).is_dir():
        raise SystemExit(f"EMOCA assets missing; run the license-gated asset download: {model_dir / MODEL_NAME}")
    if not (project / "model" / "ma3d" / "checkpoints" / "raf_MA3D.pth").is_file():
        raise SystemExit("MA3D checkpoint link missing; rerun prepare_runtime.py")
    if not env_file.is_file():
        raise SystemExit(f"Missing EMOCA environment file: {env_file}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conda = os.environ.get("CONDA_EXE") or "conda"
    worker_script = Path(__file__).resolve().parent / "run_frontalize_worker.py"
    service_logs = []
    services: list[subprocess.Popen] = []
    workers: list[subprocess.Popen] = []
    try:
        for gpu in range(args.gpus):
            port = 8765 + gpu
            service_env = os.environ.copy()
            service_env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(emoca_source), "PYTHONUNBUFFERED": "1"})
            service_env.setdefault("EMOCA_WARMUP", "1")
            log = (output_dir / f"emoca_gpu{gpu}.log").open("a", encoding="utf-8")
            service_logs.append(log)
            command = [conda, "run", "--no-capture-output", "-n", "work38", "python", "-u",
                       str(project / "services" / "emoca_svc" / "server.py"),
                       "--config", str(project / "configs" / "services.yaml"),
                       "--models-dir", str(model_dir), "--model-name", MODEL_NAME, "--port", str(port)]
            services.append(subprocess.Popen(command, cwd=project, env=service_env, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True))

        for gpu in range(args.gpus):
            url = f"http://127.0.0.1:{8765 + gpu}"
            deadline = time.monotonic() + MAX_START_SECONDS
            while time.monotonic() < deadline and not check_health(url):
                if services[gpu].poll() is not None:
                    log_path = output_dir / f"emoca_gpu{gpu}.log"
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                    raise RuntimeError(f"EMOCA GPU {gpu} exited ({services[gpu].returncode}):\n{tail}")
                time.sleep(3)
            if not check_health(url):
                raise TimeoutError(f"EMOCA GPU {gpu} not healthy in {MAX_START_SECONDS}s")
            print(f"EMOCA GPU {gpu} healthy at {url}", flush=True)

        for gpu in range(args.gpus):
            worker_env = os.environ.copy()
            worker_env.pop("PYTHONPATH", None)
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            worker_env["MA3D_DEVICE"] = "cuda:0"
            worker_env["EMOCA_URL"] = f"http://127.0.0.1:{8765 + gpu}"
            command = [sys.executable, str(worker_script), "--bundle-root", str(bundle), "--project-root", str(project),
                       "--emoca-url", worker_env["EMOCA_URL"], "--out-jsonl", str(output_dir / f"shard_{gpu}.jsonl"),
                       "--shard-index", str(gpu), "--shards", str(args.gpus), "--limit", str(args.limit),
                       "--angles", args.angles, "--mode", args.mode, "--min-roll", str(args.min_roll)]
            workers.append(subprocess.Popen(command, cwd=project, env=worker_env))
        print(f"Started {args.gpus} frontalization worker(s). Output directory: {output_dir}", flush=True)
        while any(worker.poll() is None for worker in workers):
            time.sleep(5)
        statuses = [worker.returncode for worker in workers]
        print(f"Worker exit codes: {statuses}", flush=True)
        return 0 if all(code == 0 for code in statuses) else 1
    except KeyboardInterrupt:
        print("Interrupt received; JSONL rows are flushed. Stopping children.", flush=True)
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
