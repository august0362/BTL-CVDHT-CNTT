from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
BASE_NOTEBOOK = ROOT / "notebook" / "run_rafdb_5model.ipynb"
OUT_DIR = ROOT / "notebook_frontalize"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def generate() -> tuple[Path, Path]:
    # The existing generator's notebook contains the already-maintained config, package,
    # runtime staging, EMOCA environment, and license-gated asset setup cells. Reuse those
    # generated cells as data, without importing/executing its side-effecting generator.
    base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    setup_cells = base["cells"][1:6]
    config_source = "".join(setup_cells[0]["source"])
    config_source = config_source.replace(
        "RUN_FULL_BENCHMARK = True",
        "RUN_FULL_BENCHMARK = True\nANGLES = [0, 45, 90]  # Change angles here; rows = 3068 x len(ANGLES)\n"
        "RUN_DETECTION_BENCHMARK = True\nDET_IMAGES = 1000\nDET_FACE_SIZES = [24, 48, 96]\n"
        "DET_DEGRADATIONS = ['clean', 'blur', 'down4', 'jpeg10', 'noise20', 'combo']\n"
        "DET_SEED = 2026\nDET_CANVAS = [640, 480]\nDET_BACKGROUND_FIELD = [10, 8]\n"
        "DET_BACKGROUND_COLOR_RANGE = [24, 232]\nDET_BACKGROUND_GRADIENT_X = 22.0\n"
        "DET_BACKGROUND_GRADIENT_Y = 18.0\nDET_BACKGROUND_TINT_RANGE = 12\n"
        "DET_BACKGROUND_RECTANGLES = 5\nDET_BLUR_SIGMA = 2.0\nDET_DOWN4_FACTOR = 4\n"
        "DET_JPEG10_QUALITY = 10\nDET_NOISE_SIGMA = 20.0\nDET_COMBO_DOWNSCALE = 2\n"
        "DET_COMBO_BLUR_SIGMA = 1.0\nDET_COMBO_JPEG_QUALITY = 20\nDET_CONFIDENCE_MIN = 0.25\n"
        "DET_METRIC_THRESHOLDS = [0.25, 0.5]\nDET_MATCH_IOU = 0.4\nDET_SECONDARY_IOU = 0.5\n"
        "DET_PAIRED_THRESHOLD = 0.5\nDET_SMOKE_IMAGES = 2\nDET_GPU_WORKERS = 2\n"
        "DET_REALESRGAN_TILE_SIZE = 384\nDET_REALESRGAN_TILE_PAD = 32",
    )
    setup_cells[0] = code(config_source)

    asset_paths = {
        "frontalization_init.py": REPO_ROOT / "services" / "frontalization" / "__init__.py",
        "tdfa_v2_wrapper.py": REPO_ROOT / "services" / "frontalization" / "tdfa_v2_wrapper.py",
        "run_frontalize_worker.py": ROOT / "dataset" / "scripts" / "run_frontalize_worker.py",
        "summarize_frontalize.py": ROOT / "dataset" / "scripts" / "summarize_frontalize.py",
        "launch_frontalize_workers.py": ROOT / "dataset" / "scripts" / "launch_frontalize_workers.py",
    }
    encoded = {name: b64(path) for name, path in asset_paths.items()}
    detect_assets = {
        "run_detect_degraded_worker.py": ROOT / "dataset" / "scripts" / "run_detect_degraded_worker.py",
        "summarize_detect_degraded.py": ROOT / "dataset" / "scripts" / "summarize_detect_degraded.py",
        "launch_detect_workers.py": ROOT / "dataset" / "scripts" / "launch_detect_workers.py",
    }
    detect_encoded = {name: b64(path) for name, path in detect_assets.items()}
    sha256 = {
        "mb1_120x120.pth": "a45a946c6e9b16f8d3cf2e69376da9560a7cf9afae671bebceb7e437a405ea79",
        "bfm_noneck_v3.pkl": "89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca",
        "param_mean_std_62d_120x120.pkl": "090d7150f77cb66c29ddae21e4508fbde59123dcd1dea7facc24a7ed06d1c795",
        "tri.pkl": "0562a594d8369f7d1c86306522ff76fd3a2c81cf90ba9443893c1191ad0abbb8",
    }

    staging_and_3ddfa = f'''# Stage the new frontalization modules and fetch the exact ADR-026 3DDFA-V2 source/artifacts.
import base64, hashlib, shutil
from pathlib import Path

PROJECT = WORKDIR / 'project'
FRONTALIZATION_PACKAGE = PROJECT / 'services' / 'frontalization'
FRONTALIZATION_PACKAGE.mkdir(parents=True, exist_ok=True)
(FRONTALIZATION_PACKAGE / '__init__.py').write_bytes(base64.b64decode({encoded['frontalization_init.py']!r}))
(FRONTALIZATION_PACKAGE / 'tdfa_v2_wrapper.py').write_bytes(base64.b64decode({encoded['tdfa_v2_wrapper.py']!r}))
FRONTALIZE_SCRIPTS = WORKDIR / 'frontalize_scripts'
FRONTALIZE_SCRIPTS.mkdir(parents=True, exist_ok=True)
(FRONTALIZE_SCRIPTS / 'run_frontalize_worker.py').write_bytes(base64.b64decode({encoded['run_frontalize_worker.py']!r}))
(FRONTALIZE_SCRIPTS / 'summarize_frontalize.py').write_bytes(base64.b64decode({encoded['summarize_frontalize.py']!r}))
(FRONTALIZE_SCRIPTS / 'launch_frontalize_workers.py').write_bytes(base64.b64decode({encoded['launch_frontalize_workers.py']!r}))

UPSTREAM = PROJECT / 'third_party' / '3DDFA_V2'
UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
PINNED_COMMIT = '1b6c67601abffc1e9f248b291708aef0e43b55ae'
if not (UPSTREAM / '.git').is_dir():
    run_quiet(['git', 'clone', 'https://github.com/cleardusk/3DDFA_V2', str(UPSTREAM)], label='3DDFA-V2 clone')
run_quiet(['git', '-C', str(UPSTREAM), 'checkout', PINNED_COMMIT], label='3DDFA-V2 pinned checkout')
MODEL_DIR = PROJECT / 'model' / '3ddfa_v2'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_SHA256 = {sha256!r}
UPSTREAM_PATHS = {{
    'mb1_120x120.pth': 'weights/mb1_120x120.pth',
    'bfm_noneck_v3.pkl': 'configs/bfm_noneck_v3.pkl',
    'param_mean_std_62d_120x120.pkl': 'configs/param_mean_std_62d_120x120.pkl',
    'tri.pkl': 'configs/tri.pkl',
}}
for filename, relative in UPSTREAM_PATHS.items():
    source = UPSTREAM / relative
    destination = MODEL_DIR / filename
    shutil.copy2(source, destination)
    actual = hashlib.sha256(destination.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256[filename]:
        raise RuntimeError(f'3DDFA-V2 SHA-256 mismatch for {{filename}}: expected {{EXPECTED_SHA256[filename]}}, got {{actual}}')
    print(f'Verified {{filename}} sha256={{actual}}')

# Load the model and frontalize the first sorted RAF-DB test image before EMOCA setup.
import csv, sys
sys.path.insert(0, str(PROJECT))
from PIL import Image
from services.frontalization import get_tddfa_v2_frontalizer
with (BUNDLE_ROOT / 'dataset' / 'test_labels.csv').open(encoding='utf-8-sig', newline='') as stream:
    first = sorted(csv.DictReader(stream), key=lambda row: (int(row['label']), row['image']))[0]
sample_path = BUNDLE_ROOT / 'dataset' / 'DATASET' / 'test' / str(int(first['label'])) / Path(first['image']).name
with Image.open(sample_path) as image:
    frontalized = get_tddfa_v2_frontalizer().frontalize(image.convert('RGB'))
print('3DDFA-V2 load check passed; pose yaw/pitch/roll:', frontalized.yaw_deg, frontalized.pitch_deg, frontalized.roll_deg)
'''

    run_cells = [
        code("""# Smoke run over two images at every configured angle, then summarize it.
RESULTS = WORKDIR / 'results' / 'frontalize_3ddfa'
SMOKE = RESULTS / 'smoke'
ANGLES_CSV = ','.join(str(angle) for angle in ANGLES)
# Workers resume by (image, angle) and would skip rows from an earlier failed smoke: start clean.
shutil.rmtree(SMOKE, ignore_errors=True)
run_quiet([sys.executable, str(FRONTALIZE_SCRIPTS / 'launch_frontalize_workers.py'),
           '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
           '--output-dir', str(SMOKE), '--limit', '2', '--angles', ANGLES_CSV],
          label='3DDFA frontalization smoke inference')
subprocess.run([sys.executable, str(FRONTALIZE_SCRIPTS / 'summarize_frontalize.py'),
                '--results-dir', str(SMOKE)], check=True)
smoke = json.loads((SMOKE / 'summary.json').read_text(encoding='utf-8'))
if not smoke['metric_rows'] or not any(row['n_valid'] for row in smoke['metric_rows'] if row['group_type'] == 'angle'):
    # Show the real cause instead of only the verdict.
    for shard in sorted(SMOKE.glob('shard_*.jsonl')):
        for line in shard.read_text(encoding='utf-8').splitlines()[:3]:
            row = json.loads(line)
            print(row['image'], row['angle'], {k: row[k] for k in row if k.endswith('_error')},
                  {mid: v for mid, v in (row.get('raw') or {}).items()})
    raise RuntimeError('3DDFA smoke produced no valid model predictions; full test was not started.')
print('3DDFA frontalization smoke passed.')
"""),
        code("""# Full RAF-DB test: 3,068 images × len(ANGLES), with raw and frontal branches per row.
FULL = RESULTS / 'full'
if RUN_FULL_BENCHMARK:
    run_quiet([sys.executable, str(FRONTALIZE_SCRIPTS / 'launch_frontalize_workers.py'),
               '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
               '--output-dir', str(FULL), '--angles', ANGLES_CSV],
              label='Full RAF-DB 3DDFA frontalization benchmark')
    subprocess.run([sys.executable, str(FRONTALIZE_SCRIPTS / 'summarize_frontalize.py'),
                    '--results-dir', str(FULL)], check=True)
"""),
    ]
    detection_stage = f'''# Install the independent synthetic degraded-scene detection experiment into writable storage.
import base64
DETECT_SCRIPTS = WORKDIR / 'detect_degraded_scripts'
DETECT_SCRIPTS.mkdir(parents=True, exist_ok=True)
(DETECT_SCRIPTS / 'run_detect_degraded_worker.py').write_bytes(base64.b64decode({detect_encoded['run_detect_degraded_worker.py']!r}))
(DETECT_SCRIPTS / 'summarize_detect_degraded.py').write_bytes(base64.b64decode({detect_encoded['summarize_detect_degraded.py']!r}))
(DETECT_SCRIPTS / 'launch_detect_workers.py').write_bytes(base64.b64decode({detect_encoded['launch_detect_workers.py']!r}))
'''
    detection_run_cells = [
        code("""# Experiment 2 is independent of EMOCA. Run these cells after config, package setup, and runtime staging;
# skip the EMOCA/3DDFA cells below when running detection only.
if RUN_DETECTION_BENCHMARK:
    DETECT_RESULTS = WORKDIR / 'results' / 'detect_degraded'
    DETECT_SIZES_CSV = ','.join(str(size) for size in DET_FACE_SIZES)
    DETECT_DEGRADATIONS_CSV = ','.join(DET_DEGRADATIONS)
    DETECT_THRESHOLDS_CSV = ','.join(str(value) for value in DET_METRIC_THRESHOLDS)
    DETECT_COMMON_ARGS = ['--sizes', DETECT_SIZES_CSV, '--degradations', DETECT_DEGRADATIONS_CSV,
        '--seed', str(DET_SEED), '--canvas', ','.join(map(str, DET_CANVAS)),
        '--background-field', ','.join(map(str, DET_BACKGROUND_FIELD)),
        '--background-rectangles', str(DET_BACKGROUND_RECTANGLES),
        '--background-color-range', ','.join(map(str, DET_BACKGROUND_COLOR_RANGE)),
        '--background-gradient-x', str(DET_BACKGROUND_GRADIENT_X),
        '--background-gradient-y', str(DET_BACKGROUND_GRADIENT_Y),
        '--background-tint-range', str(DET_BACKGROUND_TINT_RANGE), '--blur-sigma', str(DET_BLUR_SIGMA),
        '--down4-factor', str(DET_DOWN4_FACTOR), '--jpeg10-quality', str(DET_JPEG10_QUALITY),
        '--noise-sigma', str(DET_NOISE_SIGMA), '--combo-downscale', str(DET_COMBO_DOWNSCALE),
        '--combo-blur-sigma', str(DET_COMBO_BLUR_SIGMA), '--combo-jpeg-quality', str(DET_COMBO_JPEG_QUALITY),
        '--confidence-min', str(DET_CONFIDENCE_MIN), '--tile-size', str(DET_REALESRGAN_TILE_SIZE),
        '--tile-pad', str(DET_REALESRGAN_TILE_PAD), '--gpus', str(DET_GPU_WORKERS)]
    DETECT_SMOKE = DETECT_RESULTS / 'smoke'
    shutil.rmtree(DETECT_SMOKE, ignore_errors=True)  # resumable workers would skip stale smoke rows
    run_quiet([sys.executable, str(DETECT_SCRIPTS / 'launch_detect_workers.py'),
               '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
               '--output-dir', str(DETECT_SMOKE), '--limit', str(DET_SMOKE_IMAGES), *DETECT_COMMON_ARGS],
              label='Degraded-scene detection smoke inference')
    subprocess.run([sys.executable, str(DETECT_SCRIPTS / 'summarize_detect_degraded.py'),
                    '--results-dir', str(DETECT_SMOKE), '--thresholds', DETECT_THRESHOLDS_CSV,
                    '--match-iou', str(DET_MATCH_IOU), '--secondary-iou', str(DET_SECONDARY_IOU),
                    '--paired-threshold', str(DET_PAIRED_THRESHOLD)], check=True)
    detection_smoke = json.loads((DETECT_SMOKE / 'summary.json').read_text(encoding='utf-8'))
    smoke_rows = [json.loads(line) for shard in sorted(DETECT_SMOKE.glob('shard_*.jsonl'))
                  for line in shard.read_text(encoding='utf-8').splitlines()]
    failed_rows = [row for row in smoke_rows if row.get('error')]
    if failed_rows:
        print('Example scene errors:', [row['error'][:300] for row in failed_rows[:3]])
    if (not detection_smoke['metric_rows'] or detection_smoke['n_scenes'] == 0
            or len(failed_rows) == len(smoke_rows)):
        raise RuntimeError('Detection smoke produced no successful scenes; full test was not started.')
    print('Detection smoke passed.')
"""),
        code("""# Full synthetic-scene detection benchmark. Each scene runs direct YOLO and RealESRGAN + YOLO.
DETECT_FULL = DETECT_RESULTS / 'full'
if RUN_DETECTION_BENCHMARK:
    run_quiet([sys.executable, str(DETECT_SCRIPTS / 'launch_detect_workers.py'),
               '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
               '--output-dir', str(DETECT_FULL), '--images', str(DET_IMAGES), *DETECT_COMMON_ARGS],
              label='Full degraded-scene detection benchmark')
    subprocess.run([sys.executable, str(DETECT_SCRIPTS / 'summarize_detect_degraded.py'),
                    '--results-dir', str(DETECT_FULL), '--thresholds', DETECT_THRESHOLDS_CSV,
                    '--match-iou', str(DET_MATCH_IOU), '--secondary-iou', str(DET_SECONDARY_IOU),
                    '--paired-threshold', str(DET_PAIRED_THRESHOLD)], check=True)
"""),
    ]
    detection_intro = md("""## Experiment 2 — Does RealESRGAN help YOLO26-face detect degraded faces?

This independent detection benchmark pastes a seeded, class-stratified sample of RAF-DB test faces onto deterministic 640×480 synthetic backgrounds. It compares YOLO26-face on each degraded scene with RealESRGAN → YOLO26-face on that same scene. The target is the pasted square; boxes are retained from confidence 0.25 upward, with metrics at 0.25 and 0.5. This tests detection only: if a downstream model were used, its crop would still come from the original image, consistent with ADR-027.

Backgrounds use low-frequency colour noise, smooth gradients, and a few filled rectangles; they contain no faces. Faces are placed at sizes 24, 48, and 96 by default. Whole-scene degradation defaults to clean, blur, down4, jpeg10, noise20, and combo. Change `DET_IMAGES`, `DET_FACE_SIZES`, `DET_DEGRADATIONS`, or `RUN_DETECTION_BENCHMARK` in the configuration cell. The smoke pass uses two images. This experiment needs the staged project and detector/restorer checkpoints, but not EMOCA; run these cells after runtime staging and before or after the 3DDFA benchmark.

The results are limited by synthetic backgrounds and pasted aligned faces. They do not measure performance on natural photos or naturally degraded faces.

Outputs are written to `/kaggle/working/rafdb5/results/detect_degraded/{smoke,full}/`: resumable shard JSONL, `summary.json`, `metrics.csv`, and `scenes.csv`. No images are saved.""")
    intro = md("""# RAF-DB 3DDFA-V2 frontalization benchmark

This notebook benchmarks the ADR-026 3DDFA-V2 frontalization step on every image in the aligned 100x100 RAF-DB test split. At each configured in-plane rotation angle, it compares `raw` (rotated image) with `frontal` (3DDFA-V2 output), then runs real EMOCA and both RAF-DB models: POSTER and MA3D-Net. It does not run RealESRGAN, YOLO, or CodeFormer.

Set `ANGLES` in the configuration cell (default `[0, 45, 90]`). The smoke run processes two images at all configured angles before the full benchmark, which runs when `RUN_FULL_BENCHMARK` is true. The worker shards across two T4 GPUs and resumes from existing JSONL rows. It writes numeric results only.

Attach the existing private Kaggle Dataset containing RAF-DB labels/images, MA3D checkpoints, and bundle scripts. The Dataset does not need a new version: this notebook clones 3DDFA-V2 at the ADR-026 pinned commit, verifies the four model artifact hashes from `model_info.json`, and performs a model load/frontalization check before EMOCA setup. Keep the Dataset and Kaggle notebook private. Personally accept applicable FLAME/EMOCA terms before the gated asset download.

Outputs are saved under `/kaggle/working/rafdb5/results/frontalize_3ddfa/{smoke,full}/`. The summary includes coverage and accuracy, macro and per-class F1, yaw/pitch strata at angle zero, paired changes with exact McNemar p-values, and stage/error counts. No benchmark image files are saved.
""")
    cells = [intro, *setup_cells[:3], detection_intro, code(detection_stage), *detection_run_cells,
             code(staging_and_3ddfa), *setup_cells[3:], *run_cells]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook_path = OUT_DIR / "run_rafdb_frontalize.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "id": "your-kaggle-username/rafdb-frontalize-3ddfa",
        "title": "RAF-DB 3DDFA-V2 Frontalization Benchmark",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = OUT_DIR / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return notebook_path, metadata_path


if __name__ == "__main__":
    notebook_path, metadata_path = generate()
    print(f"Notebook generated: {notebook_path}")
    print(f"Kaggle metadata generated: {metadata_path}")
