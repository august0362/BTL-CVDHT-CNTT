# M6 general-restoration model switch — 2026-09-30

## Decision and files

At the user's direction, replace RealESRGAN_x4plus with the smaller upstream
general model `realesr-general-x4v3.pth`. Keep YOLO26-face for face detection and
CodeFormer for face-crop restoration. The standard M6 order remains:

`realesr-general-x4v3 → YOLO26-face → padded face crop → CodeFormer`.

The old `backend/model/realesrgan_x4plus/` directory and its contents have been removed.
The supplied checkpoint is located at
`backend/model/realesr-general-x4v3/realesr-general-x4v3.pth`.

## Artifact record

- Size: **4,885,111 bytes**.
- Locally measured SHA-256: **`8DC7EDB9AC80CCDC30C3A5DCA6616509367F05FBC184AD95B731F05BECE96292`**.
- Official release asset: [Real-ESRGAN v0.2.5.0 x4v3 checkpoint](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth).
- The digest is recorded from the local file; it has not been compared to an
  independent upstream checksum. The project repository declares BSD-3-Clause;
  preserve attribution and license notices. The user has specified research-only,
  non-commercial use.
- Metadata: `backend/model/realesr-general-x4v3/model_info.json`.

## Pipeline changes

- `backend/services/image_preprocessing/cli.py` defaults to the new checkpoint path and
  identifies x4v3 in its model metadata.
- `backend/web/predata_experiment.py` loads the same checkpoint for the optional visual
  comparison. The standard POSTER/MA3D baseline route is unchanged.
- `another/tests/test_m6_real_integration.py` now points to the x4v3 artifact so a future
  real-model integration run tests the selected model.
- Streamlit stage labels now say “RealESRGAN Compact”. The adapter remains generic
  and Spandrel selects the checkpoint architecture from the file.

## Verification status

File presence, byte size and local SHA-256 were checked. No model-load or inference
smoke was run in this update. Therefore x4v3 compatibility with the installed
Spandrel build, runtime, and actual speed on `angry.png` remain unverified. Earlier
M6 smoke and 21-image detection benchmark results used x4plus and must not be
attributed to x4v3. M6-01 and M6-03 are reopened in `ROADMAP_BACKLOG.md`; after a
smoke, record load status, per-image timing, memory, and the same paired detector
metrics before comparing speed or detection effect.

The earlier complete `angry.png` A/B/C/D comparison, also with x4plus, is recorded
in `ANGRY_PREPROCESS_COMPARISON_2026-09-30.md`.
