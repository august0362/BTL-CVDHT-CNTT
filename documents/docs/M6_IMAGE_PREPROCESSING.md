# M6 — Image preprocessing CLI

Updated 2026-10-03. The standalone offline CLI uses `realesr-general-x4v3` via
Spandrel and the local YOLO26-face checkpoint. It writes an enhanced image,
one crop per detected face, and JSON metadata for each input. Face crops are
always cut from the EXIF-corrected original image using the recorded crop bbox.
There is no crop restoration stage.

## CLI and application behavior

The CLI always runs x4v3 before detection. Its job is to produce an enhanced
image as well as face crops and metadata, so the enhanced output is a required
artifact even when the original could already yield detections. YOLO runs on the
enhanced image; x4 output is resized back to source dimensions so bbox
coordinates remain in original-image pixel space. Crops are cut from the
original, not the enhanced image.

The app follows ADR-028 separately: YOLO checks the original first and invokes
x4v3 followed by YOLO only when the original yields zero faces. A one-face result
or detector error follows the app's baseline path. The CLI does not implement
that app routing rule because it must create an enhanced image for every input.

## Processing and outputs

1. Open the input, apply EXIF orientation, and convert to RGB.
2. Restore with `realesr-general-x4v3` through Spandrel. The 4× output is
   resampled to source dimensions before detection. Inputs over 1 MP use
   overlapping 384 px tiles with 32 px context padding.
3. Run YOLO26-face on the restored image (default confidence 0.5).
4. Clamp each detection box, expand it by 15% on each side, and crop that region
   from the original RGB image.
5. Limit the enhanced output and each crop to a 4096 px long side without
   upscaling. The normal inference path already returns the enhanced image at
   source dimensions.

The default no-face policy is `keep`: write the enhanced image and metadata with
an empty face list. `--no-face-policy error` instead marks that input as failed.
Each file is processed independently, so decoding or inference errors do not
stop later files. Output writes use temporary files followed by atomic replace.

```powershell
$env:PYTHONPATH='.'
python -m services.image_preprocessing.cli INPUT_DIR `
  --output OUTPUT_DIR `
  --general-weights backend/model/realesr-general-x4v3/realesr-general-x4v3.pth `
  --yolo-weights backend/model/yolo26_face/yolo26_face.pt `
  --device auto --confidence 0.5 --padding 0.15 --max-side 4096
```

Directory inputs are searched recursively. Output paths mirror relative input
paths:

```text
OUTPUT_DIR/
  enhanced_images/<relative-path>__<source-ext>.png
  faces/<relative-path>__<source-ext>_face_01.png
  metadata/<relative-path>__<source-ext>.json
```

Metadata includes input/output dimensions, status, detector settings, selected
device, model IDs, checkpoint filenames/sizes/SHA-256, face count, detector and
crop bboxes, confidence, crop dimensions, per-stage timings, and total time.
`PipelineConfig.numpy_color` defaults to RGB for library NumPy inputs; callers
with BGR arrays must select `BGR` explicitly.

## Local checkpoints and measured x4v3 status

Weights are read from disk and are never downloaded by the CLI:

- General restoration: `backend/model/realesr-general-x4v3/realesr-general-x4v3.pth`
  (4,885,111 bytes; SHA-256
  `8DC7EDB9AC80CCDC30C3A5DCA6616509367F05FBC184AD95B731F05BECE96292`). The
  local digest matches the value recorded in the roadmap and prior M6 docs.
- Face detector: `backend/model/yolo26_face/yolo26_face.pt` (20,289,541 bytes; SHA-256
  `9FB5C9E36D3257C2979842E7573082BC05E4E78309D30A7333971C47B4B725B4`).

Measured 2026-10-03 with Python 3.13.5, PyTorch 2.14.0+cpu, Spandrel 0.4.1,
Ultralytics 8.4.166, Pillow 12.3.0, NumPy 2.5.3. `auto` selected CPU; loaded
scale was 4. Each smoke row used a fresh process. Peak working set is the
Windows process peak after load and inference; timings vary with CPU load.

| Input | Size | Model load | x4v3 inference | Peak working set |
|---|---:|---:|---:|---:|
| `afraid.jpg` | 606×404 | 6.403 s | 4.803 s | 494.15 MiB |
| `angry.jpg` | 525×350 | 4.675 s | 2.336 s | 489.36 MiB |
| `base64-…-crop-…webp` | 1200×630 | 4.217 s | 10.709 s | 1085.16 MiB |
| `cong_dong_la_gi_5_c0a7fabc23.jpg` | 800×586 | 6.006 s | 7.052 s | 787.09 MiB |

The real recursive CLI acceptance run processed 4/4 images successfully,
produced 4 enhanced images, 4 metadata JSON files, and 21 original-image crops.
All 21 saved crops byte-matched the corresponding original-image region at the
recorded crop bbox. CLI wall time was 61.172 s; summed per-image processing time
was 52.206 s (mean 13.052 s). Per-file details and acceptance verdicts are in
`documents/docs/qa/M6_CLI_ACCEPTANCE_2026-10-03.md`.

The checkpoint paths and model names are local identifiers. Checkpoint hash
verification establishes file identity against the recorded value, not a
separate upstream signature. The measured run verifies load and inference on
this CPU environment; it does not establish output quality for recognition or
training. Review a representative A/B set before using enhanced imagery for
those purposes.
