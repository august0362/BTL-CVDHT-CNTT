from __future__ import annotations

import json
import base64
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "run_rafdb_5model_colab.ipynb"
ASSET_SCRIPT_B64 = base64.b64encode(
    (ROOT / "bundle" / "environment" / "download_emoca_assets.sh").read_bytes()
).decode("ascii")
M6_PIPELINE_B64 = base64.b64encode(
    (ROOT.parent / "services" / "image_preprocessing" / "pipeline.py").read_bytes()
).decode("ascii")


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).lstrip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).lstrip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # RAF-DB full test: five model paths (Google Colab)

        This notebook evaluates POSTER, MA3DNRe, MA3DNCo, raw MA3D, and MA3DReCo on all 3,068 RAF-DB test images. It uses the number of Colab GPUs available (up to two), keeps only concise cell output, and writes resumable numeric result files to Google Drive.

        ## Before running

        1. Upload the complete `ggcolab/` folder to `MyDrive/ggcolab/`. Keep `bundle/` and `scripts/` together. The private bundle contains RAF-DB images and model weights; do not make it public.
        2. Open this notebook in Colab. In **Runtime → Change runtime type**, select a GPU, then run the cells from top to bottom. Internet access is needed for package installation and the pinned EMOCA source/assets.
        3. The first setup downloads and builds the legacy EMOCA environment. Register and personally accept FLAME/EMOCA terms before the asset-download cell runs. The consent flag defaults to `True` as requested.
        4. The smoke pass runs two images before the full test. Results are saved under `MyDrive/ggcolab/results/`; rerunning can resume completed rows.

        The Colab runtime and its local disk are temporary. A runtime reset requires repeating setup and the gated asset download; completed result rows remain in Drive. Cell output is quiet on success; the final summary shows metrics, and a failure shows only the tail of its log.

        The worker keeps Colab's preinstalled CUDA PyTorch and NumPy. EMOCA runs in a separate Python 3.8 / Torch 1.11 environment. The benchmark writes predictions and metrics only, not input, cropped, or restored images.
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import shutil
        import subprocess
        import sys

        from google.colab import drive
        drive.mount('/content/drive')

        def run_quiet(command, *, label, env=None):
            completed = subprocess.run(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace',
            )
            if completed.returncode:
                tail = (completed.stdout or '').strip()[-4000:]
                raise RuntimeError(f'{label} failed (exit {completed.returncode}): {tail}')

        # Change this if you placed the uploaded folder elsewhere in Drive.
        DRIVE_ROOT = Path('/content/drive/MyDrive/ggcolab')
        # The project owner has already registered and accepted FLAME/EMOCA terms for this research run.
        ACCEPT_FLAME_EMOCA_LICENSES = True
        RUN_FULL_BENCHMARK = True
        MAX_GPUS = 2

        ROOT = Path('/content/rafdb5')
        WORKDIR = ROOT / 'runtime'
        LOCAL_BUNDLE = ROOT / 'bundle'
        SCRIPTS = ROOT / 'scripts'
        DRIVE_RESULTS = DRIVE_ROOT / 'results'
        WORKDIR.mkdir(parents=True, exist_ok=True)
        DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)

        import torch
        assert torch.cuda.is_available(), 'Select a GPU runtime in Colab first.'
        GPU_COUNT = min(torch.cuda.device_count(), MAX_GPUS)
        print(f"GPU workers: {GPU_COUNT} ({torch.cuda.get_device_name(0)})")
        assert (DRIVE_ROOT / 'bundle' / 'BUNDLE_MANIFEST.json').is_file(), (
            f"Could not find {DRIVE_ROOT / 'bundle'}. Upload the complete ggcolab folder to Drive."
        )
        """
    ),
    code(
        """
        # Copy the ~1.2 GB bundle from Drive to Colab's local disk for faster image/model access.
        shutil.copytree(DRIVE_ROOT / 'bundle', LOCAL_BUNDLE, dirs_exist_ok=True)
        shutil.copytree(DRIVE_ROOT / 'scripts', SCRIPTS, dirs_exist_ok=True)
        assert (SCRIPTS / 'prepare_runtime.py').is_file(), 'Missing benchmark scripts beside ggcolab/scripts.'

        # Install only the worker packages; do not replace Colab's CUDA-enabled torch or NumPy.
        requirements = LOCAL_BUNDLE / 'environment' / 'requirements-worker.txt'
        runtime_requirements = WORKDIR / 'requirements-worker-colab.txt'
        requirements_lines = requirements.read_text(encoding='utf-8').splitlines()
        runtime_requirements.write_text(
            '\\n'.join(line for line in requirements_lines
                        if not line.strip().lower().startswith('numpy==')) + '\\n',
            encoding='utf-8',
        )
        run_quiet([sys.executable, '-m', 'pip', 'install', '-r', str(runtime_requirements)],
                  label='Worker package installation')
        run_quiet([sys.executable, str(SCRIPTS / 'verify_bundle.py'), str(LOCAL_BUNDLE)],
                  label='Bundle verification')

        run_quiet([sys.executable, str(SCRIPTS / 'prepare_runtime.py'),
                   '--bundle-root', str(LOCAL_BUNDLE), '--workdir', str(WORKDIR)],
                  label='Runtime staging')
        # Older Drive bundles may predate M6 and omit this module, while the
        # packaged image-preprocessing __init__ imports it eagerly.
        import base64
        m6_pipeline = WORKDIR / 'project' / 'services' / 'image_preprocessing' / 'pipeline.py'
        m6_pipeline.parent.mkdir(parents=True, exist_ok=True)
        m6_pipeline.write_bytes(base64.b64decode('__EMBEDDED_M6_PIPELINE__'))
        source_setup = LOCAL_BUNDLE / 'environment' / 'setup_emoca_source.sh'
        run_quiet(['bash', str(source_setup), str(LOCAL_BUNDLE), str(WORKDIR)],
                  label='Pinned EMOCA source setup')
        print('Bundle verified; project and pinned EMOCA source are ready.')
        """.replace('__EMBEDDED_M6_PIPELINE__', M6_PIPELINE_B64)
    ),
    code(
        """
        # Build the isolated legacy EMOCA environment. Success logs are suppressed.
        if not ACCEPT_FLAME_EMOCA_LICENSES:
            raise RuntimeError(
                'Asset download is disabled. Enable it only after personally accepting FLAME and EMOCA terms.'
            )

        miniconda = WORKDIR / 'miniconda'
        miniconda_conda = miniconda / 'bin' / 'conda'
        if miniconda_conda.is_file():
            os.environ['PATH'] = str(miniconda / 'bin') + os.pathsep + os.environ.get('PATH', '')
            os.environ['CONDA_EXE'] = str(miniconda_conda)
        elif shutil.which('conda') is None:
            installer = WORKDIR / 'miniconda.sh'
            run_quiet(['wget', '-q', 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh', '-O', str(installer)],
                      label='Miniconda download')
            run_quiet(['bash', str(installer), '-b', '-p', str(miniconda)],
                      label='Miniconda installation')
            os.environ['PATH'] = str(miniconda / 'bin') + os.pathsep + os.environ.get('PATH', '')
            os.environ['CONDA_EXE'] = str(miniconda_conda)
        else:
            os.environ.setdefault('CONDA_EXE', shutil.which('conda'))

        setup_env = os.environ.copy()
        setup_env['CONDA_EXE'] = setup_env.get('CONDA_EXE') or shutil.which('conda') or ''
        if not setup_env['CONDA_EXE']:
            raise RuntimeError('Conda executable not found after Miniconda setup.')
        setup_env['PATH'] = str(Path(setup_env['CONDA_EXE']).parent) + os.pathsep + setup_env.get('PATH', '')
        setup_env.pop('PYTHONPATH', None)
        env_setup = LOCAL_BUNDLE / 'environment' / 'setup_emoca_env.sh'
        run_quiet(['bash', str(env_setup), str(LOCAL_BUNDLE), str(WORKDIR)],
                  env=setup_env, label='EMOCA Python 3.8 / Torch 1.11 environment setup')

        # Use the current minimal downloader embedded in this notebook. This avoids
        # accidentally running the older, full downloader from a previously uploaded bundle.
        import base64
        asset_env = setup_env.copy()
        asset_env['ACCEPT_FLAME_EMOCA_LICENSES'] = '1'
        asset_setup = WORKDIR / 'download_emoca_assets_minimal.sh'
        asset_setup.write_bytes(base64.b64decode('__EMBEDDED_ASSET_SCRIPT__'))
        run_quiet(['bash', str(asset_setup), str(WORKDIR)], env=asset_env,
                  label='FLAME/EMOCA asset download')
        print('EMOCA environment and required assets are ready.')
        """.replace('__EMBEDDED_ASSET_SCRIPT__', ASSET_SCRIPT_B64)
    ),
    code(
        """
        # Two-image smoke pass. The output contains only the metric summary.
        SMOKE = DRIVE_RESULTS / 'smoke'
        run_quiet([sys.executable, str(SCRIPTS / 'launch_workers.py'),
                   '--bundle-root', str(LOCAL_BUNDLE), '--workdir', str(WORKDIR),
                   '--output-dir', str(SMOKE), '--limit', '2', '--gpus', str(GPU_COUNT)],
                  label='Two-image smoke inference')
        subprocess.run([sys.executable, str(SCRIPTS / 'summarize.py'),
                        '--bundle-root', str(LOCAL_BUNDLE), '--results-dir', str(SMOKE),
                        '--sample-name', 'smoke-2-images', '--expected-images', '2'], check=True)
        """
    ),
    code(
        """
        # Full test (3,068 images); JSONL rows and summaries are written directly to Drive.
        FULL = DRIVE_RESULTS / 'full'
        if RUN_FULL_BENCHMARK:
            run_quiet([sys.executable, str(SCRIPTS / 'launch_workers.py'),
                       '--bundle-root', str(LOCAL_BUNDLE), '--workdir', str(WORKDIR),
                       '--output-dir', str(FULL), '--gpus', str(GPU_COUNT)],
                      label='Full RAF-DB test inference')
            subprocess.run([sys.executable, str(SCRIPTS / 'summarize.py'),
                            '--bundle-root', str(LOCAL_BUNDLE), '--results-dir', str(FULL),
                            '--sample-name', 'full-test-3068'], check=True)
            print(f'Results saved in: {FULL}')
        else:
            print(f'Full run skipped. Existing results: {FULL}')
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Wrote the Colab notebook.")
