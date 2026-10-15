# Angry image preprocessing trial — 2026-09-30

> Preliminary x4plus-only attempt. Its statement that the enhanced A/B/D
> comparison was not completed was true at that stage, but was superseded by the
> later complete run in `ANGRY_PREPROCESS_COMPARISON_2026-09-30.md`.

## Image and environment

- Input: `database/image/angry.png`, 2390×1760, RGBA (4.21 MP).
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM; `nvidia-smi` reported about 3965 MiB free.
- Web Python: PyTorch `2.14.0+cpu`; `torch.version.cuda` is `None` and `torch.cuda.is_available()` is false. The web FER engine is configured for CPU. The GPU exists, but this venv cannot use it until it has a compatible CUDA-enabled PyTorch build; 4 GB VRAM still requires tiled inputs.
- In this execution environment, TCP probes to `127.0.0.1:8765` (EMOCA) and `127.0.0.1:8599` (web) timed out. This does not prove that a server in the user's separate session is stopped; it means this runner could not call it.

## Results

- Raw YOLO26-face completed in 340 ms: bbox `(704, 187, 1693, 1428)`, confidence `0.8844`, crop 1287×1615.
- An initial full-frame RealESRGAN attempt without tiling used about 10.3 GB process working set; available system memory fell to about 0.7 GB. The process was stopped before the machine exhausted memory.
- RealESRGAN was then changed to overlapping 384 px tiles with 32 px padding, and the preprocessing process was limited to four logical CPUs on Windows. A second attempt held about 1.5–1.8 GB process working set and left 8.3–8.8 GB system memory available, but RealESRGAN had not finished after more than nine minutes. That attempt was stopped to restore the CPU for interactive use.
- Because the enhanced-image stage did not finish, enhanced YOLO, CodeFormer, and MA3D/EMOCA comparisons were not run. No conclusion can be made about whether the class changes from sad to angry after preprocessing.
- No input, crop, or restored image was written to disk by the trial.

## Follow-up

- The comparison UI now reports progress and distinguishes completed execution from a valid MA3D prediction; each failed branch displays its actual contract error/warnings.
- Tiled RealESRGAN bounds peak memory, but full-resolution restoration of this 4.21 MP image remains impractically slow on CPU. To complete a faithful A/B/D class comparison, run on a CUDA-enabled M6 environment or choose an explicitly documented lower-resolution/face-crop variant. Changing resolution/order can change the model input and must be labeled as a different experiment.
- Before claiming accuracy improvement, compare the raw and processed prediction against the image's known emotion label. GAN restoration can change facial details and can change a prediction without making it more correct.
