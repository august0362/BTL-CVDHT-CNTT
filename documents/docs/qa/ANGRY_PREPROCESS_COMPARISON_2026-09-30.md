# Angry image preprocessing comparison — 2026-09-30

## Scope

This complete comparison used the previous general restorer, RealESRGAN_x4plus,
not the newly selected `realesr-general-x4v3`. The input is
`database/image/angry.png` (2390×1760, 4.21 MP). The user identifies its correct label
as **Anger**. The tested MA3D model is trained for RAF-DB.

The normal baseline and A/B/D pre-data branches all ran through real EMOCA and
real MA3D. Results are for this one image and do not establish dataset accuracy.

## MA3D predictions

| Branch | Input path | MA3D top-1 | Result against known label |
|---|---|---:|---|
| C · baseline | Original → EMOCA → MA3D | Sadness, 87.324136% | Wrong |
| A · no general restoration | Original → YOLO crop → CodeFormer → EMOCA → MA3D | Sadness, 88.662130% | Wrong |
| B · no CodeFormer | Original → RealESRGAN_x4plus → YOLO crop → EMOCA → MA3D | Sadness, 85.792774% | Wrong |
| D · full pre-data | Original → RealESRGAN_x4plus → YOLO crop → CodeFormer → EMOCA → MA3D | Sadness, 89.444387% | Wrong |

All branches returned valid MA3D predictions, but none changed the predicted
class to Anger. Branch D's confidence is higher than baseline while still wrong;
confidence is not a measure of accuracy. This result does not isolate which
individual preprocessing stage caused a score shift because A/B/D change more
than one input operation relative to baseline.

## Latency and UI stall

| Stage | Measured time |
|---|---:|
| Raw YOLO26-face | ~473 ms |
| RealESRGAN_x4plus on full image, CPU, tiled | **1,876,861 ms (~31m 17s)** |
| YOLO on restored full image | ~282 ms |
| CodeFormer on raw/enhanced crop | ~7.5 s each |
| Full comparison click, including baseline/preprocessing and branches | ~1,915.93 s (~31m 56s) |

The comparison call waits synchronously for the full-frame restoration, which
accounts for nearly all the apparent freeze. The Windows inference runtime used
PyTorch `2.14.0+cpu`; this is not a GPU timing or evidence of swap/OOM. The current
x4v3 replacement has not yet been benchmarked, so its speedup is unknown.

## Conclusion and limits

- RealESRGAN_x4plus changed confidence scores but did not correct this known Anger
  example; the unprocessed baseline was already wrong.
- Do not report an accuracy improvement from this one example or from increased
  confidence. Validate on a labeled, representative real-world set.
- The x4plus latency above is historical and cannot be attributed to x4v3.
- The same-size restoration adapter downsamples the 4× result to input dimensions
  before detection. Restoration may alter/hallucinate detail; it does not add
  original image information.
