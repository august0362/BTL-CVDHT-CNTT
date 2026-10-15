# M6 — Milestone planning report

**Status:** milestone scope and acceptance criteria drafted only. No implementation was requested or added; M6 is **not complete or accepted**.

## Planned scope

- Sequence: RGB input → general-image restoration → face-tuned YOLO detection → padded/clamped crop per face → CodeFormer (preferred) or GFPGAN restoration → dimension cap at 4096 px → structured output and optional batch export.
- Proposed controls: confidence 0.5, crop padding ratio 0.15, CodeFormer fidelity `w=0.7`, configurable no-face handling, and auto device CUDA/MPS/CPU.
- Proposed outputs: enhanced full image, restored face crops, and JSON with source, boxes, confidence, sizes, per-stage duration, and runtime/model provenance.
- `documents/docs/context/ROADMAP_BACKLOG.md` now records the M6 tasks and acceptance criteria. The former MA3D improvement idea remains listed separately as deferred.

## Environment and blockers

Inspected default interpreter: Python 3.11.9. NumPy and Pillow are present; PyTorch, Spandrel, Ultralytics, and GFPGAN are absent. No packages or model weights were installed/downloaded. There is no implementation, real-model inference, output image, bbox/confidence sample, or face crop. Model licenses/provenance are unverified and no checkpoint is selected.

The milestone plan prefers CodeFormer and allows GFPGAN as fallback. The eventual `w`/weight control is not a guarantee that identity is preserved. A face-tuned YOLO checkpoint is required; generic COCO YOLO weights are not a face detector.

## Acceptance still required

1. Choose and document compatible local checkpoints, immutable source/revision, SHA-256, and applicable weight licenses.
2. Install optional dependencies in a dedicated environment and run CPU plus available accelerator smoke tests.
3. Inspect visual results and verify sequence, RGB/BGR conversion, padded/clamped boxes, no-face behavior, crop dimensions, and the 4096 cap on permitted fixtures.
4. Exercise a directory containing valid and invalid images and verify saved PNG/JSON outputs and continuation after per-image failures.
5. Decide whether restored crops are suitable for downstream training/inference using a representative evaluation set. Until then this module is standalone and the default app pipeline is unchanged.

No code, dependency files, or test suite were added. No tests or model runs were performed. This report records planning only, not implementation or acceptance evidence.

## Upstream implementation references

- Spandrel model loader and descriptor usage: <https://github.com/chaiNNer-org/spandrel/blob/main/README.md>
- Real-ESRGAN model and inference options: <https://github.com/xinntao/Real-ESRGAN>
- Ultralytics Python inference: <https://docs.ultralytics.com/modes/predict/>
- GFPGAN restoration API: <https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py>
- CodeFormer reference for future adapter: <https://github.com/sczhou/CodeFormer>
