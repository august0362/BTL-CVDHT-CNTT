"""Generate kaggle/notebook_roll/run_rafdb_roll_sweep.ipynb (ADR-028 roll-threshold sweep).

Reuses the maintained setup cells of run_rafdb_5model.ipynb and the 3DDFA staging cell of the
frontalization notebook (run create_notebook_frontalize.py first so it embeds current scripts).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_NOTEBOOK = ROOT / "notebook" / "run_rafdb_5model.ipynb"
FRONTALIZE_NOTEBOOK = ROOT / "notebook_frontalize" / "run_rafdb_frontalize.ipynb"
OUT_DIR = ROOT / "notebook_roll"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def source(cell: dict) -> str:
    return "".join(cell["source"])


def generate() -> Path:
    base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))["cells"][1:6]
    frontalize = json.loads(FRONTALIZE_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    staging = next(c for c in frontalize if "Stage the new frontalization modules" in source(c))
    config = source(base[0]).replace(
        "RUN_FULL_BENCHMARK = True",
        "RUN_FULL_BENCHMARK = True\n"
        "ANGLES = [0, 15, 30, 45, 60, 90]  # in-plane rotations; rows = 3068 x len(ANGLES)\n"
        "MIN_ROLL = 5.0  # below this |roll| the corrected image equals the input (raw result reused)\n"
        "ROLL_THRESHOLDS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60]  # evaluated post hoc")
    if "ANGLES = " not in config:
        raise SystemExit("config patch failed: 'RUN_FULL_BENCHMARK = True' not found in the base config cell")
    base[0] = code(config)
    sweep_b64 = base64.b64encode((ROOT / "dataset" / "scripts" / "summarize_roll_sweep.py").read_bytes()).decode("ascii")

    intro = md("""# RAF-DB roll-only correction sweep (ADR-028)

Measures the decision taken after the 3DDFA-V2 benchmark: estimate the face **roll** with 3DDFA-V2 and undo it with a plain **2D rotation**; yaw/pitch are not corrected and nothing is warped.

For every RAF-DB test image (3,068) and every angle in `ANGLES` (default 0/15/30/45/60/90°): `raw` = rotated image → EMOCA → POSTER / MA3D-Net; `frontal` = rotated image → 3DDFA-V2 roll estimate → rotate by −roll → EMOCA → POSTER / MA3D-Net. When |roll| < `MIN_ROLL` the corrected image equals the input, so the raw result is reused. The summary then evaluates every threshold T in `ROLL_THRESHOLDS` post hoc (use the corrected image only when |roll| ≥ T) and recommends the T with the best mean accuracy over angles that loses ≤ 0.2 points on upright images.

Run with the existing private Kaggle Dataset attached, **GPU T4 x2**, Internet on. Progress while the full run is going: open the Output panel → `rafdb5/results/roll_sweep/full/progress_shard0.json` and `progress_shard1.json` (percent, seconds per row, ETA). The last cell zips the results to `/kaggle/working/roll_results.zip` for download. Numbers only; no images are saved.""")

    embed = code(f"""# Embed the roll-sweep summarizer next to the frontalization scripts.
import base64
(FRONTALIZE_SCRIPTS / 'summarize_roll_sweep.py').write_bytes(base64.b64decode({sweep_b64!r}))
print('Roll-sweep summarizer installed.')
""")
    smoke = code("""# Smoke: two images at every angle, roll-only mode. Stale smoke rows are removed first.
RESULTS = WORKDIR / 'results' / 'roll_sweep'
SMOKE = RESULTS / 'smoke'
ANGLES_CSV = ','.join(str(angle) for angle in ANGLES)
THRESHOLDS_CSV = ','.join(str(t) for t in ROLL_THRESHOLDS)
ROLL_ARGS = ['--angles', ANGLES_CSV, '--mode', 'roll2d', '--min-roll', str(MIN_ROLL)]
shutil.rmtree(SMOKE, ignore_errors=True)
run_quiet([sys.executable, str(FRONTALIZE_SCRIPTS / 'launch_frontalize_workers.py'),
           '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
           '--output-dir', str(SMOKE), '--limit', '2', *ROLL_ARGS],
          label='Roll sweep smoke inference')
rows = [json.loads(line) for shard in sorted(SMOKE.glob('shard_*.jsonl'))
        for line in shard.read_text(encoding='utf-8').splitlines()]
valid = [r for r in rows if any('label' in (r.get('raw') or {}).get(m, {}) for m in ('poster_rafdb', 'ma3d_rafdb'))]
if not valid:
    for r in rows[:3]:
        print(r.get('image'), r.get('angle'), {k: r[k] for k in r if k.endswith('_error')}, r.get('raw'))
    raise RuntimeError('Roll sweep smoke produced no valid predictions; full run not started.')
subprocess.run([sys.executable, str(FRONTALIZE_SCRIPTS / 'summarize_roll_sweep.py'),
                '--results-dir', str(SMOKE), '--thresholds', THRESHOLDS_CSV], check=True)
print(f'Roll sweep smoke passed ({len(valid)}/{len(rows)} rows with predictions).')
""")
    full = code("""# Full run: 3,068 images x len(ANGLES). Resumable: re-running this cell continues where it stopped.
FULL = RESULTS / 'full'
if RUN_FULL_BENCHMARK:
    run_quiet([sys.executable, str(FRONTALIZE_SCRIPTS / 'launch_frontalize_workers.py'),
               '--bundle-root', str(BUNDLE_ROOT), '--workdir', str(WORKDIR),
               '--output-dir', str(FULL), *ROLL_ARGS],
              label='Full RAF-DB roll sweep')
    subprocess.run([sys.executable, str(FRONTALIZE_SCRIPTS / 'summarize_roll_sweep.py'),
                    '--results-dir', str(FULL), '--thresholds', THRESHOLDS_CSV], check=True)
""")
    pack = code("""# Zip the numeric results (no EMOCA logs) for download from the Output panel.
OUT = WORKDIR / 'roll_results'
shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(RESULTS / 'full', OUT / 'roll_sweep', ignore=shutil.ignore_patterns('emoca_gpu*.log'))
print('Download:', shutil.make_archive('/kaggle/working/roll_results', 'zip', OUT))
""")
    cells = [intro, *base[:3], staging, embed, *base[3:], smoke, full, pack]
    # Every name the run cells use must be defined by an earlier cell (caught a missing config cell).
    defined = "".join(source(c) for c in cells[:-3])
    for name in ("ANGLES", "MIN_ROLL", "ROLL_THRESHOLDS", "FRONTALIZE_SCRIPTS", "BUNDLE_ROOT", "WORKDIR", "run_quiet"):
        if f"{name} =" not in defined and f"def {name}" not in defined:
            raise SystemExit(f"generated notebook would not define {name} before the run cells")
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}},
                "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "run_rafdb_roll_sweep.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"id": "your-kaggle-username/rafdb-roll-sweep", "title": "RAF-DB Roll-only Correction Sweep",
                "code_file": path.name, "language": "python", "kernel_type": "notebook", "is_private": "true",
                "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print("Notebook generated:", generate())
