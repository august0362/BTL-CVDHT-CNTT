# M6 RealESRGAN → YOLO benchmark — 2026-09-30

> Historical benchmark: this run used RealESRGAN_x4plus. The current selected
> model is `realesr-general-x4v3`; the numbers below are not measurements of it.

## Method

Ran both the original RAF-DB 100×100 RGB image and its RealESRGAN_x4plus-restored version through the same local YOLO26-face checkpoint at `conf=0.5`. The RealESRGAN x4 output is resized back to 100×100 before detection. A detection means at least one face box was returned. The sample is deterministic: first, middle, and last filenames after sorting in each label folder `1..7` (21 images total).

Also ran the complete M6 chain (RealESRGAN → YOLO → centered 1.3× padded crop → CodeFormer) on `test_0002_aligned.jpg`. It completed with status `ok`, one face, and a 100×100 crop. No images or crops were saved; the test stores only numeric results and filenames in `%TEMP%`.

## Results

| Metric | Original image | RealESRGAN then YOLO |
|---|---:|---:|
| Images with a detection | 12/21 (57.1%) | 15/21 (71.4%) |
| Mean confidence among detected images | 0.7202 | 0.7442 |
| Newly detected after restoration | — | 3 |
| Detections lost after restoration | — | 0 |

Newly detected: `test_0007_aligned.jpg`, `test_2309_aligned.jpg`, `test_1382_aligned.jpg`.

| RAF-DB label | Original | Restored |
|---:|---:|---:|
| 1 | 3/3 | 3/3 |
| 2 | 2/3 | 2/3 |
| 3 | 1/3 | 3/3 |
| 4 | 1/3 | 1/3 |
| 5 | 2/3 | 3/3 |
| 6 | 2/3 | 2/3 |
| 7 | 1/3 | 1/3 |

## Interpretation and limits

On this paired sample, RealESRGAN preprocessing increased the face-detection recall proxy by **3 images (14.3 percentage points)** at the same confidence threshold, with no lost detections. This is not bbox accuracy: RAF-DB here has no face-box ground truth, so IoU and localization accuracy were not measured. The sample also contains only face images, so false positives on no-face images were not measured.

The earlier M5 report gives 16/21 at confidence 0.5 but does not retain the 21 filenames. Its number cannot be reconciled as an exact same-image baseline from the available artifact. This report therefore uses a paired original/restored result on its explicit manifest and does not claim a direct 16/21-to-15/21 comparison.

Full repeatable test: `another/tests/test_m6_real_integration.py` (heavy group in `another/tests/run_all.py`). Detailed per-image data is written to `%TEMP%/M6_REAL_BENCHMARK_2026-09-30.json`; no image outputs are written.
