# RAF-DB five-path benchmark on Google Colab

This folder is a self-contained Colab upload package for the 3,068-image RAF-DB test split. It contains the notebook, benchmark scripts, the private dataset/model bundle, and no FLAME/EMOCA licensed assets.

## Run

1. Upload the complete `ggcolab/` folder to `/MyDrive/ggcolab/` in Google Drive, keeping `bundle/` and `scripts/` beside the notebook. Keep it private.
2. Open `run_rafdb_5model_colab.ipynb` in Google Colab, select a GPU runtime, and run all cells.
3. The notebook detects one or two available GPUs, preserves Colab's existing CUDA PyTorch/NumPy for the workers, and creates a separate pinned Python 3.8 EMOCA environment.
4. Register and personally accept FLAME and EMOCA terms before running the asset-download cell. `ACCEPT_FLAME_EMOCA_LICENSES` now defaults to `True`; set it to `False` if you need to block downloads.
5. The two-image smoke pass runs before the full benchmark. Numeric outputs are written to `MyDrive/ggcolab/results/`; rerunning the full cell resumes completed image rows.

Cell output is quiet on success. The smoke/full summaries print the metrics, and a failed command displays only the last 4,000 characters of its log. Inputs, crops, and restored images are not copied to the results folder.

The bundle is about 1.21 GB and includes RAF-DB images and model weights. Keep it private. The notebook embeds the minimal asset downloader and the M6 pipeline module, so replacing the notebook also updates those files when an older bundle copy is already in Drive. The downloader fetches only the selected EMOCA checkpoint and runtime support assets, skips alternate checkpoints/example videos, and removes each archive after extraction. Colab runtime storage is temporary: after a runtime reset, setup and gated asset downloads need to run again, while completed results in Drive remain available. The notebook uses Internet for package installation and the pinned EMOCA source/asset downloads.

## Compared paths

| Output | Preprocessing path |
|---|---|
| `POSTER` | Original image → EMOCA alignment → POSTER |
| `MA3DNRe` | Original → YOLO26-face crop (1.3× padding) → CodeFormer → EMOCA → MA3D |
| `MA3DNCo` | Original → RealESRGAN Compact → YOLO26-face crop (1.3× padding) → EMOCA → MA3D |
| `MA3D` | Original → EMOCA → MA3D baseline |
| `MA3DReCo` | Original → RealESRGAN Compact → YOLO26-face crop (1.3× padding) → CodeFormer → EMOCA → MA3D |

The summary reports accuracy, mean class accuracy, macro F1, coverage, confidence/calibration metrics, per-class metrics, confusion matrices, and paired comparisons where available.

## Files

- `run_rafdb_5model_colab.ipynb`: Colab notebook.
- `bundle/`: test images, labels, model checkpoints, source snapshot, and environment setup files.
- `scripts/`: bundle verification, runtime staging, one/two-GPU launcher, inference workers, and metric summarizer.
- `scripts/build_colab_notebook.py`: reproducible notebook generator.
