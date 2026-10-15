# Kaggle benchmark: RAF-DB no-YOLO preprocessing ablations

This Kaggle notebook runs only three new MA3D preprocessing paths on all 3,068 RAF-DB test images. The test images are already aligned 100x100 face crops, so these experiment paths do not call YOLO or crop faces. Previous POSTER, MA3D baseline, and YOLO-based results are not rerun or overwritten.

## Compared paths

| Output name | Path |
|---|---|
| `MA3DReCo_noYOLO` | Original aligned image → RealESRGAN Compact → CodeFormer → EMOCA → MA3D |
| `MA3DRe_noYOLO` | Original aligned image → RealESRGAN Compact → EMOCA → MA3D |
| `MA3DCo_noYOLO` | Original aligned image → CodeFormer → EMOCA → MA3D |

## Run

1. Keep using the existing private Kaggle Dataset with the RAF-DB test images, labels, checkpoints, and project source.
2. Import the updated `notebook/run_rafdb_5model.ipynb` and attach that existing Dataset. Select **GPU T4 x2** and enable Internet.
3. Register and personally accept FLAME/EMOCA terms before running the asset-download cell. The consent flag defaults to `True` as previously requested.
4. Run the notebook top to bottom. It runs a two-image smoke pass, then only the three new branches on the full test set. The new worker and summarizer are embedded in the notebook, so the Dataset does not need a new version.

The notebook reuses the existing EMOCA setup/assets when they remain in the Kaggle working session. It does not re-run POSTER, raw MA3D, or the previous YOLO-based paths. Previous outputs under `results/full/` remain untouched.

## Outputs

New results are written to `/another/kaggle/working/rafdb5/results/predata_no_yolo/full/`:

- `shard_0.jsonl`, `shard_1.jsonl`: resumable per-image predictions and stage statuses.
- `summary.json`, `metrics.csv`, `predictions.csv`: coverage, accuracy, macro F1, confidence, and per-image predictions.
- `stage_errors.csv`: stage status/error counts for RealESRGAN, CodeFormer, and each EMOCA/MA3D inference path.
- `misclassified/RealESRGAN_CodeFormer_MA3D/`, `misclassified/RealESRGAN_MA3D/`, and `misclassified/CodeFormer_MA3D/`: up to 10 original misclassified test images per path, chosen from the highest-confidence wrong predictions. Each folder includes `metadata.csv` with true/predicted labels and confidence.

The output also reports wrong valid predictions separately from failed predictions. Read accuracy together with coverage. Download the new result directory before a Kaggle runtime reset if you need to keep it.

## Privacy and licensing

The private Dataset contains RAF-DB images and model checkpoints; keep it private. The notebook sends images to EMOCA processes on the same Kaggle runtime over loopback. It saves numeric results and the requested misclassified source examples, but does not save enhanced or restored intermediate images. Internet access is needed for package installation and the pinned EMOCA source/assets.

## 3DDFA-V2 frontalization benchmark

`notebook_frontalize/run_rafdb_frontalize.ipynb` benchmarks the ADR-026 3DDFA-V2 frontalization step on the full 3,068-image RAF-DB test split. It compares each rotated image in a `raw` branch with the same image after 3DDFA-V2 frontalization in a `frontal` branch, then runs real EMOCA and the POSTER and MA3D RAF-DB models. The default `ANGLES = [0, 45, 90]` is configurable in the notebook configuration cell. No RealESRGAN, YOLO, or CodeFormer stage runs.

Use the existing private Kaggle Dataset and import the new notebook separately. Select GPU T4 x2 and enable Internet. The notebook stages the frontalization package and benchmark scripts into writable working storage, clones 3DDFA-V2 at the pinned ADR-026 commit, verifies all four asset SHA-256 hashes, and performs a frontalizer load check before EMOCA environment setup. The Dataset does not need a new version because 3DDFA-V2 and its assets are fetched from GitHub at that pinned commit.

The workload is `rows = 3068 x len(ANGLES)`. Every row makes two EMOCA + engine calls, one raw and one frontal, as well as one 3DDFA-V2 frontalization. Runtime therefore scales with `3068 x len(ANGLES) x measured seconds per row` plus setup and model startup; the worker prints its measured row rate while running. The notebook runs a two-image smoke pass at every configured angle, then runs the full benchmark when `RUN_FULL_BENCHMARK` is enabled.

Outputs are written under `/another/kaggle/working/rafdb5/results/frontalize_3ddfa/{smoke,full}/`: resumable `shard_*.jsonl`, `summary.json`, `metrics.csv`, and `predictions.csv`. The summary includes per-angle and angle-zero yaw/pitch bins, per-class F1, paired raw-to-frontal transitions with exact McNemar p-values, and stage/error counts. Only numeric predictions and metadata are saved; benchmark images are not written to disk. `notebook_frontalize/kernel-metadata.json` contains the private T4/Internet Kaggle settings and placeholder owner id `your-kaggle-username/rafdb-frontalize-3ddfa`.

## RealESRGAN + YOLO26-face degraded-scene detection experiment

The second independent experiment in `notebook_frontalize/run_rafdb_frontalize.ipynb` asks whether the app's RealESRGAN sharpening stage helps YOLO26-face detect faces in degraded input. It measures detection only: both branches see the same degraded scene, and `esrgan_yolo` restores that scene before YOLO. In the app's ADR-027 flow, RealESRGAN is used only for detection; any model crop is still taken from the original image.

The benchmark draws a seeded, class-stratified sample of `DET_IMAGES` RAF-DB test faces (default 1,000; seed 2026), resizes each face to `DET_FACE_SIZES` (24, 48, 96 by default), and pastes it at a seeded position inside a 640×480 scene. Backgrounds are generated deterministically from low-frequency colour noise, a smooth gradient, and a few filled rectangles, with no faces. It applies whole-scene `clean`, Gaussian blur (sigma 2), `down4`, JPEG quality 10, Gaussian noise (sigma 20), or `combo` (down2 + blur sigma 1 + JPEG quality 20), as selected by `DET_DEGRADATIONS`. It records all YOLO boxes at confidence 0.25 or higher and reports metrics at thresholds 0.25 and 0.5, including IoU 0.4 and 0.5 recall and paired exact two-sided McNemar comparisons at 0.5.

This experiment needs only the staged project, YOLO and RealESRGAN checkpoints, and the worker dependencies. It does not need EMOCA and can be run independently before or after the 3DDFA benchmark; skip the EMOCA/3DDFA cells for detection-only runs. The notebook does a two-image smoke pass before the full run and shards jobs across both T4 GPUs. Runtime workload is `scenes = DET_IMAGES × number of DET_FACE_SIZES × number of DET_DEGRADATIONS`; every scene makes one RealESRGAN call and two YOLO calls. Estimate runtime by multiplying that scene count by measured average scene time from the smoke or worker logs, plus model startup.

Results are written to `/another/kaggle/working/rafdb5/results/detect_degraded/{smoke,full}/`: `shard_*.jsonl` can resume by `(image, size, degradation)`, `summary.json` contains aggregate metrics and paired comparisons, `metrics.csv` contains metric rows, and `scenes.csv` contains per-scene boxes/timings. No images are saved. The experiment uses pasted aligned faces and synthetic backgrounds, so it does not establish performance on natural photos or naturally degraded faces.

## Roll-only threshold sweep (ADR-028)

`notebook_roll/run_rafdb_roll_sweep.ipynb` evaluates 2D roll correction on the 3,068 RAF-DB test images at 0/15/30/45/60/90° rotations (18,408 image-angle rows). It records raw and corrected predictions, then summarizes candidate thresholds 5/10/15/20/25/30/35/40/50/60°. The recommendation maximizes mean Acc(all) across the six angles, subject to no more than 0.2 percentage-point loss at 0° for each model. The completed Kaggle run recommended 15° for both POSTER and MA3D-Net; see [`documents/docs/qa/M8_KAGGLE_RESULTS_2026-10-03.md`](../../documents/docs/qa/M8_KAGGLE_RESULTS_2026-10-03.md) §3.

Use the existing private Kaggle Dataset, GPU T4 x2, and Internet. The notebook runs the full sweep across two resumable shards and packages the results as `/another/kaggle/working/roll_results.zip`, containing `threshold_sweep.csv`, `summary.json`, progress files, and per-row shard JSONL. It saves numeric results only; the sweep does not validate performance on natural user photos.

## Note — CodeFormer removed (ADR-027, 2026-10-03)

CodeFormer was deleted from the project, including `bundle/models/codeformer`, the CodeFormer source in `bundle/source`, and the matching entries in the local `dataset.zip` (now about 732 MB). `BUNDLE_MANIFEST.json` still lists CodeFormer, so `verify_bundle.py` and the older notebooks (`run_rafdb_5model.ipynb`, the no-YOLO restoration ablation) cannot be reproduced from this local copy without re-downloading CodeFormer (URL and SHA-256 are in ADR-027). The new `notebook_frontalize/` notebook does not use CodeFormer and works with the private Kaggle Dataset that is already uploaded. **Do not create a new Dataset version** from the trimmed local zip just to run it.
