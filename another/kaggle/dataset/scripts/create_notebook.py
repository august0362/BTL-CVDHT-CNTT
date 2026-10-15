from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSET_SCRIPT_B64 = base64.b64encode(
    (ROOT / "dataset" / "bundle" / "environment" / "download_emoca_assets.sh").read_bytes()
).decode("ascii")
M6_PIPELINE_B64 = base64.b64encode(
    (ROOT.parent / "services" / "image_preprocessing" / "pipeline.py").read_bytes()
).decode("ascii")
PREPROCESS_WORKER_B64 = base64.b64encode(
    (ROOT / "dataset" / "scripts" / "run_predata_worker.py").read_bytes()
).decode("ascii")
PREPROCESS_SUMMARIZER_B64 = base64.b64encode(
    (ROOT / "dataset" / "scripts" / "summarize_predata.py").read_bytes()
).decode("ascii")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = [
    md("""# RAF-DB test: three no-YOLO restoration paths

This notebook runs only three new MA3D paths on the full 3,068-image RAF-DB test split: RealESRGAN + CodeFormer, RealESRGAN only, and CodeFormer only. The RAF-DB inputs are already aligned 100x100 face crops, so these paths do not run YOLO or crop faces. Existing POSTER, MA3D baseline, and older YOLO-based results are left untouched and are not rerun.

It partitions the test images across both Kaggle T4 GPUs, reports errors by processing stage, counts incorrect predictions, and exports up to 10 high-confidence misclassified original images per path into three separate folders. Results are written to a new `results/predata_no_yolo/` directory.

Before running:

1. Upload the contents of `kaggle/dataset/` as a **private Kaggle Dataset**. Keep `bundle/` and `scripts/` together at its root, then attach that Dataset as notebook input.
2. Import this notebook file separately into Kaggle (`File > Import Notebook`) and select **GPU T4 x2**. Enable Internet for package/source and gated EMOCA asset setup.
3. Register and personally accept FLAME and EMOCA terms before running the asset-download cell. The consent flag now defaults to `True` as you requested.
4. Run from top to bottom. The first pass installs the pinned EMOCA environment and can take a while. A two-image smoke run exercises only the three new paths before the full test starts.

The notebook calls EMOCA on the same Kaggle runtime over loopback. The bundle does not include FLAME/EMOCA assets because those must be obtained by a registered user after accepting their terms. Kaggle usage sends the RAF-DB test data and included model artifacts to Kaggle's cloud storage/runtime; keep the dataset private. The notebook embeds the two new benchmark scripts, so the existing large Kaggle Dataset does not need a new version.

The environment setup accepts the Anaconda `pkgs/main` and `pkgs/r` channel terms, as authorized by the project owner. The FLAME/EMOCA consent flag defaults to `True`; the owner must still personally register and accept those terms before running the asset-download cell. The notebook downloads only the selected EMOCA checkpoint and runtime assets; it skips alternate checkpoints and example videos to reduce Kaggle disk usage.
"""),
    code("""from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

def run_quiet(command, *, label, env=None):
    completed = subprocess.run(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors='replace',
    )
    if completed.returncode:
        tail = (completed.stdout or '').strip()[-4000:]
        raise RuntimeError(f'{label} failed (exit {completed.returncode}): {tail}')

# The project owner has already registered and accepted FLAME/EMOCA terms for this research run.
ACCEPT_FLAME_EMOCA_LICENSES = True
RUN_FULL_BENCHMARK = True
BUNDLE_ROOT_OVERRIDE = None  # Optional: exact path such as /kaggle/input/<dataset-slug>/bundle
WORKDIR = Path('/kaggle/working/rafdb5')

import torch
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator first.'
gpu_count = torch.cuda.device_count()
assert gpu_count == 2, f'This run is configured for T4 x2; found {gpu_count} visible GPU(s).'

if BUNDLE_ROOT_OVERRIDE:
    BUNDLE_ROOT = Path(BUNDLE_ROOT_OVERRIDE).resolve()
else:
    candidates = [p.parent for p in Path('/kaggle/input').rglob('BUNDLE_MANIFEST.json')
                  if (p.parent / 'dataset' / 'test_labels.csv').is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one attached Kaggle bundle; found {candidates}. Set BUNDLE_ROOT_OVERRIDE.')
    BUNDLE_ROOT = candidates[0]
KAGGLE_DIR = BUNDLE_ROOT.parent
SCRIPTS = KAGGLE_DIR / 'scripts'
assert (SCRIPTS / 'prepare_runtime.py').is_file(), f'Bundle scripts missing beside {BUNDLE_ROOT}'
"""),
    code("""# Keep Kaggle's preinstalled NumPy and CUDA PyTorch; the bundle pin is for isolated local environments.
requirements = BUNDLE_ROOT / 'environment' / 'requirements-worker.txt'
runtime_requirements = WORKDIR / 'requirements-worker-kaggle.txt'
WORKDIR.mkdir(parents=True, exist_ok=True)
requirements_lines = requirements.read_text(encoding='utf-8').splitlines()
runtime_requirements.write_text(
    '\\n'.join(line for line in requirements_lines
                if not line.strip().lower().startswith('numpy==')) + '\\n',
    encoding='utf-8',
)
run_quiet([sys.executable, '-m', 'pip', 'install', '-r', str(runtime_requirements)],
          label='Worker package installation')
import numpy as np
import cv2
run_quiet([sys.executable, str(SCRIPTS / 'verify_bundle.py'), str(BUNDLE_ROOT)],
          label='Bundle verification')
"""),
    code(f"""# Stage project code and make sure the optional M6 package is complete in older uploaded Datasets.
run_quiet([sys.executable, str(SCRIPTS / 'prepare_runtime.py'),
           '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR)],
          label='Runtime staging')
import base64
m6_pipeline = WORKDIR / 'project' / 'services' / 'image_preprocessing' / 'pipeline.py'
m6_pipeline.write_bytes(base64.b64decode({M6_PIPELINE_B64!r}))
source_setup = BUNDLE_ROOT / 'environment' / 'setup_emoca_source.sh'
run_quiet(['bash', str(source_setup), str(BUNDLE_ROOT), str(WORKDIR)],
          label='EMOCA source setup')
"""),
    code("""# Build the project's pinned Python 3.8 / Torch 1.11 / PyTorch3D EMOCA environment.
# Reuse Miniconda if a previous setup attempt already created it under /kaggle/working.
WORKDIR.mkdir(parents=True, exist_ok=True)
# Clear large unused assets from a previous upstream download before setup touches the disk.
asset_root = WORKDIR / 'project' / 'model' / 'emoca' / 'assets'
for relative in (
    'EMOCA/models/EMOCA.zip', 'EMOCA/models/EMOCA_v2_mp.zip',
    'EMOCA/models/EMOCA_v2_lr_cos_1.5.zip', 'data/EMOCA_test_example_data.zip',
    'DECA.zip', 'FaceRecognition.zip', 'FLAME.zip',
    'EMOCA/models/EMOCA', 'EMOCA/models/EMOCA_v2_mp',
    'EMOCA/models/EMOCA_v2_lr_cos_1.5', 'data/EMOCA_test_example_data',
):
    old_asset = asset_root / relative
    if old_asset.is_dir():
        shutil.rmtree(old_asset)
    elif old_asset.exists():
        old_asset.unlink()
try:
    (asset_root / 'data').rmdir()
except OSError:
    pass
miniconda = WORKDIR / 'miniconda'
miniconda_conda = miniconda / 'bin' / 'conda'
if miniconda_conda.is_file():
    os.environ['PATH'] = str(miniconda / 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['CONDA_EXE'] = str(miniconda_conda)
elif shutil.which('conda') is None:
    installer = WORKDIR / 'miniconda.sh'
    subprocess.run(['wget', '-q', 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh', '-O', str(installer)], check=True)
    subprocess.run(['bash', str(installer), '-b', '-p', str(miniconda)], check=True)
    os.environ['PATH'] = str(miniconda / 'bin') + os.pathsep + os.environ.get('PATH', '')
    os.environ['CONDA_EXE'] = str(miniconda_conda)
else:
    os.environ.setdefault('CONDA_EXE', shutil.which('conda'))
env_setup = BUNDLE_ROOT / 'environment' / 'setup_emoca_env.sh'
env_setup_text = env_setup.read_text(encoding='utf-8')
# Avoid Kaggle's unrelated Python mamba test-runner command and cross-version PYTHONPATH.
env_setup_text = env_setup_text.replace(
    'eval "$(conda shell.bash hook)"',
    'CONDA_EXE="$(command -v conda)"\\n'
    'CONDA_BIN="$(dirname "$CONDA_EXE")"\\n'
    'export PATH="$CONDA_BIN:$PATH"\\n'
    'unset PYTHONPATH\\n'
    'eval "$("$CONDA_EXE" shell.bash hook)"',
)
env_setup_text = env_setup_text.replace(
    'if ! command -v mamba >/dev/null; then',
    'if [[ ! -x "$CONDA_BIN/mamba" ]]; then',
)
env_setup_text = env_setup_text.replace(
    '  conda install -n base -c conda-forge mamba=2.3.3 -y',
    '  "$CONDA_EXE" install -n base -c conda-forge mamba=2.3.3 -y',
)
env_setup_text = env_setup_text.replace(
    '  mamba create -n work38 python=3.8 -y',
    '  "$CONDA_BIN/mamba" create -n work38 python=3.8 -y',
)
env_setup_text = env_setup_text.replace(
    'mamba env update -n work38 --file',
    '"$CONDA_BIN/mamba" env update -n work38 --file',
)
# Apply package fixes even if this notebook is attached to an older uploaded Dataset.
setup_lines = []
has_numpy_pin = 'python -m pip install numpy==1.23.5' in env_setup_text
found_pip_upgrade = False
for line in env_setup_text.splitlines():
    if 'python -m pip install --upgrade pip==' in line:
        found_pip_upgrade = True
        setup_lines.append(line)
        if not has_numpy_pin:
            setup_lines.append('conda run -n work38 python -m pip install numpy==1.23.5')
        continue
    if line.rstrip().endswith('python -m pip install Cython==0.29'):
        setup_lines.append(line.replace('Cython==0.29', 'Cython==0.29.37'))
        continue
    if 'python -m pip install onnx==1.10.2 onnxruntime-gpu' in line:
        setup_lines.append(line.replace(
            'onnx==1.10.2 onnxruntime-gpu ', ''
        ).replace('mediapipe ', 'mediapipe==0.10.9 '))
    elif '^(onnx|youtube-dl|mediapipe|mmcv-full|pyzmq)' in line:
        setup_lines.append(line.replace(
            '^(onnx|youtube-dl|mediapipe|mmcv-full|pyzmq)',
            '^(onnx|onnxruntime-gpu|youtube-dl|mediapipe|mmcv-full|pyzmq|opencv-python|opencv-python-headless|opencv-contrib-python|opencv-contrib-python-headless|insightface|deepspeech-gpu)',
        ))
    elif 'python -m pip install mmcv-full -f ' in line:
        continue  # Not imported anywhere in this benchmark's inference path.
    else:
        setup_lines.append(line)
if not found_pip_upgrade and not has_numpy_pin:
    raise RuntimeError('Could not place the pinned NumPy before installing EMOCA requirements.')
env_setup_text = '\\n'.join(setup_lines) + '\\n'
if 'opencv-python-headless==4.5.5.64' not in env_setup_text:
    editable_install = 'conda run -n work38 python -m pip install -e "$EMOCA_ROOT"'
    if editable_install not in env_setup_text:
        raise RuntimeError('Could not find EMOCA editable-install line for OpenCV normalization.')
    opencv_normalization = (
        editable_install + '\\n'
        'conda run -n work38 python -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless || true\\n'
        'conda run -n work38 python -m pip install opencv-python-headless==4.5.5.64\\n'
        'conda run -n work38 python -c "import cv2, mediapipe; print(cv2.__version__, mediapipe.__version__)"\\n'
    )
    env_setup_text = env_setup_text.replace(editable_install, opencv_normalization, 1)
if 'tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main' not in env_setup_text:
    conda_hook = 'eval "$("$CONDA_EXE" shell.bash hook)"'
    if conda_hook not in env_setup_text:
        raise RuntimeError('Could not locate Conda hook to add the authorized Anaconda ToS acceptance.')
    tos_acceptance = (
        conda_hook + '\\n'
        '# User-authorized acceptance for the Anaconda channels listed by EMOCA.\\n'
        '"$CONDA_EXE" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main\\n'
        '"$CONDA_EXE" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'
    )
    env_setup_text = env_setup_text.replace(conda_hook, tos_acceptance, 1)
if 'mamba create -n work38' in env_setup_text:
    raise RuntimeError('Could not patch the bundled Mamba command for Kaggle.')
env_setup_runtime = WORKDIR / 'setup_emoca_env_kaggle.sh'
env_setup_runtime.write_text(env_setup_text, encoding='utf-8')
setup_env = os.environ.copy()
setup_env['CONDA_EXE'] = setup_env.get('CONDA_EXE') or shutil.which('conda') or ''
if not setup_env['CONDA_EXE']:
    raise RuntimeError('Conda executable not found after Miniconda setup.')
setup_env['PATH'] = str(Path(setup_env['CONDA_EXE']).parent) + os.pathsep + setup_env.get('PATH', '')
setup_env.pop('PYTHONPATH', None)
os.environ.pop('PYTHONPATH', None)
run_quiet([setup_env['CONDA_EXE'], 'clean', '--all', '--yes'],
          env=setup_env, label='Conda cache cleanup')
run_quiet(['bash', str(env_setup_runtime), str(BUNDLE_ROOT), str(WORKDIR)],
          env=setup_env, label='EMOCA environment setup')
"""),
    code(f"""# License-gated download: fetch only benchmark assets, not upstream demos/alternate checkpoints.
if not ACCEPT_FLAME_EMOCA_LICENSES:
    raise RuntimeError('Asset download is disabled. Enable it only after personally accepting FLAME and EMOCA terms.')
import base64
asset_env = os.environ.copy()
asset_env['ACCEPT_FLAME_EMOCA_LICENSES'] = '1'
asset_setup = WORKDIR / 'download_emoca_assets_minimal.sh'
asset_setup.write_bytes(base64.b64decode({ASSET_SCRIPT_B64!r}))
run_quiet(['bash', str(asset_setup), str(WORKDIR)], env=asset_env,
          label='EMOCA/FLAME asset download')
"""),
    code(f"""# Install the new worker/summarizer into writable Kaggle storage; do not alter the attached Dataset.
import base64
PREPROCESS_SCRIPTS = WORKDIR / 'predata_no_yolo_scripts'
PREPROCESS_SCRIPTS.mkdir(parents=True, exist_ok=True)
(PREPROCESS_SCRIPTS / 'run_predata_worker.py').write_bytes(base64.b64decode({PREPROCESS_WORKER_B64!r}))
(PREPROCESS_SCRIPTS / 'summarize_predata.py').write_bytes(base64.b64decode({PREPROCESS_SUMMARIZER_B64!r}))

# Reuse the existing EMOCA service launcher, but point it at the new no-YOLO worker.
launcher_source = (SCRIPTS / 'launch_workers.py').read_text(encoding='utf-8')
old_worker_line = 'worker_script = Path(__file__).resolve().parent / "run_worker.py"'
if old_worker_line not in launcher_source:
    raise RuntimeError('Attached Dataset launch_workers.py has an unexpected version.')
launcher_source = launcher_source.replace(
    old_worker_line,
    'worker_script = Path(__file__).resolve().parent / "run_predata_worker.py"',
)
PREPROCESS_LAUNCHER = PREPROCESS_SCRIPTS / 'launch_predata_workers.py'
PREPROCESS_LAUNCHER.write_text(launcher_source, encoding='utf-8')
RESULTS = WORKDIR / 'results' / 'predata_no_yolo'
SMOKE = RESULTS / 'smoke'
run_quiet([sys.executable, str(PREPROCESS_LAUNCHER),
           '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
           '--output-dir', str(SMOKE), '--limit', '2'],
          label='Two-image no-YOLO smoke inference')
subprocess.run([sys.executable, str(PREPROCESS_SCRIPTS / 'summarize_predata.py'),
                '--bundle-root', str(BUNDLE_ROOT), '--results-dir', str(SMOKE),
                '--examples-per-model', '0', '--expected-images', '2'], check=True)
smoke_summary = json.loads((SMOKE / 'summary.json').read_text(encoding='utf-8'))
unavailable = [name for name, result in smoke_summary['metrics'].items() if result['n_valid'] == 0]
if unavailable:
    raise RuntimeError('No-YOLO smoke produced no valid predictions for: ' + str(unavailable) + '; full test was not started.')
print('No-YOLO smoke passed; continuing to the full test.')
"""),
    code("""# Full test: only the three new branches run; previous benchmark outputs are not overwritten.
FULL = RESULTS / 'full'
if RUN_FULL_BENCHMARK:
    run_quiet([sys.executable, str(PREPROCESS_LAUNCHER),
               '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
               '--output-dir', str(FULL)],
              label='Full RAF-DB no-YOLO inference')
    subprocess.run([sys.executable, str(PREPROCESS_SCRIPTS / 'summarize_predata.py'),
                    '--bundle-root', str(BUNDLE_ROOT), '--results-dir', str(FULL),
                    '--examples-per-model', '10'], check=True)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
(NOTEBOOK_DIR / "run_rafdb_5model.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print("Notebook generated: notebook/run_rafdb_5model.ipynb")
