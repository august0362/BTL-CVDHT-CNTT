# M6 independent image preprocessing — QA report

> **Historical checkpoint note (2026-09-30):** this report's real-model smoke and
> detector benchmark used RealESRGAN_x4plus. The user later replaced it with
> `realesr-general-x4v3`; see `M6_COMPACT_SWITCH_2026-09-30.md`. The old results
> remain valid only for x4plus and do not accept the compact checkpoint.

Date: 2026-09-30. Scope is only `backend/services/image_preprocessing/`; it does not call Streamlit, `frontend/app.py`, `backend/web/orchestrator.py`, EMOCA, POSTER, or MA3D-Net, and does not alter their default pipeline.

## Model selection and verified artifact state

- User supplied both checkpoints at the documented paths. RealESRGAN_x4plus is 67,040,989 bytes, SHA-256 `4FA0D38905F75AC06EB49A7951B426670021BE3018265FD191D2125DF9D682F1`; CodeFormer is 376,637,898 bytes, SHA-256 `1009E537E0C2A07D4CABCE6355F53CB66767CD4B4297EC7A4A64CA4B8A5684B7`. Both local digests exactly match immutable LFS/Xet pointers for matching release assets: [RealESRGAN model revision](https://huggingface.co/amd/realesrgan-x4plus/commit/9af1b12e679374d2f0b0ccdb2488cee92e2525e1) and [CodeFormer weight revision](https://huggingface.co/wwy/codeformer/commit/0ddc98c9396c4ec56d2ccf08dfe355208406694b).
- General restoration selected by the user: **RealESRGAN_x4plus**, loaded through Spandrel's ESRGAN/RRDB architecture. Official release asset: `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth`. The matching mirror identifies the checkpoint as unchanged and states BSD 3-Clause; preserve attribution/license notice. The 4× output is resized to source dimensions before detection, preserving bbox coordinates.
- Face restoration selected by the user: **CodeFormer**. Official source v0.1.0 is checked out at `backend/third_party/CodeFormer`, commit `fbc4a807ead48b2281aa65fd99cde220dbcf1398`. Its pinned README declares CC BY-NC-SA 4.0. User confirms research-only, non-commercial use; attribution/share-alike terms apply. The CLI uses this source checkout by default.
- The user confirmed the project is **research only, with no commercial use**. Keep both checkpoints and derived outputs within that scope and follow each stated attribution/license condition.
- The previous SwinIR/GFPGAN candidate choice is superseded by the user's final model selection. CodeFormer `w=0.7` is a quality/fidelity control, not a guarantee of identity preservation.
- YOLO: uses only existing `backend/model/yolo26_face/yolo26_face.pt`; 20,289,541 bytes; SHA-256 `9FB5C9E36D3257C2979842E7573082BC05E4E78309D30A7333971C47B4B725B4`; local metadata `backend/model/yolo26_face/model_info.json`. It was loaded/inferred successfully in M5 with Ultralytics 8.4.166. Upstream revision and independent weights-license verification remain unresolved. On the M5 balanced RAF-DB **100×100** sample, confidence 0.5 detected **16/21**. This is not evidence for high-resolution M6 input performance.
- Full real-model CLI smoke on `database/image/archive/DATASET/test/2/test_2303_aligned.jpg` (100×100 RGB), CPU, confidence 0.5, padding 0.15, CodeFormer w=0.7: **PASS**. RealESRGAN kept output at 100×100; YOLO found one face (conf 0.790718); padded/clamped crop was 100×100; CodeFormer returned 100×100. CLI wrote enhanced image, face crop, and metadata under `%TEMP%/m6-real-smoke/`; no test outputs were saved in the repository. Human visual review confirmed both files open and show the input face; this is a smoke check, not an image-quality A/B evaluation.
- M6 dependencies are listed in `environments/m6/requirements-m6.txt`; Spandrel and CodeFormer runtime packages were installed into the existing `ma3d-web` venv for this smoke. Torch CPU was already available; no web requirements changed.

## Implementation/acceptance evidence

Implemented: RGB path/PIL/NumPy input checks; RealESRGAN_x4plus through Spandrel → YOLO adapter → one centered 1.3× padded crop per bbox → CodeFormer adapter → 4096 long-side cap. The x4 output is resized to source dimensions before detection. No-face policies `keep` and `error`. CLI recurses input directories, writes enhanced images, face crops, and JSON metadata separately; checkpoint size/SHA-256 are included; per-image processing failures do not stop later inputs.

Automated offline tests: `another/tests/test_m6_pipeline.py` covers stage order, RGB, padding, all four edge clamps, bbox reporting, confidence filtering, no-face policies, 4096/no-upscale, dimension preservation, invalid config/arrays/empty bbox, path input, CLI recursive export/metadata/checkpoint hashes, and continuation after a bad file.

Command: `& 'C:\Users\admin\venvs\ma3d-web\Scripts\python.exe' -B -m pytest another/tests/test_m6_pipeline.py -q -p no:cacheprovider --basetemp $base` (PowerShell `$base` was created under the workspace and removed after pytest; its name was `.pytest_m6_verify_tmp`).

After the final model selection, CodeFormer CLI/adapter wiring, RealESRGAN dimension-resampling check, and device metadata update: **20 passed** in 0.89 seconds. Environment was the existing project venv (Python 3.13.5, Pillow 12.3.0, NumPy 2.5.3, pytest 9.1.1). CLI `--help` also passes and shows the prepared default checkpoint/source paths.

## Limits / future review

- M6-01 accepted for the declared non-commercial research scope. Each local SHA-256 matches an immutable mirror pointer for the named release asset; the CodeFormer source is pinned. Re-review the license before any commercial use or broader distribution.
- The YOLO adapter invocation is checked on one 100×100 sample only. The 16/21 figure is only the separate balanced 21-image 100×100 RAF-DB result.
- Product integration, face-recognition fidelity and image quality A/B are outside this independent package task and remain future work.

## Full-suite run and paired 21-image test (2026-09-30)

Added `another/tests/test_m6_real_integration.py` to the heavy group in `another/tests/run_all.py`. It runs the actual RealESRGAN_x4plus -> YOLO26-face -> centered 1.3x crop -> CodeFormer pipeline on one RAF-DB image, then compares original and RealESRGAN-restored inputs using the same YOLO checkpoint and `conf=0.5` on 21 deterministic images (first, middle, and last sorted file in each label folder 1-7). It writes numeric metrics and filenames only to `%TEMP%`; no source, enhanced, or crop images are stored in the repository.

Paired result: original images yielded detections on **12/21 (57.1%)**; after RealESRGAN they yielded **15/21 (71.4%)**, a gain of 3 images / 14.3 percentage points. No baseline detections were lost. Mean confidence among detected images was 0.7202 before and 0.7442 after. Newly detected: `test_0007_aligned.jpg`, `test_2309_aligned.jpg`, and `test_1382_aligned.jpg`. This is an image-level face-detection recall proxy, **not detector accuracy**: the sample has no face-box ground truth, so bbox IoU/localization accuracy and false-positive rate were not measured. The earlier M5 value of 16/21 lacks its filename manifest, so it is not a directly comparable baseline.

The single-image full M6 flow passed with status `ok`, one detected face, and a 100x100 crop. Full suite command: `python -B another/tests/run_all.py --heavy --timeout 300` in the existing `ma3d-web` venv. Result: **17 PASS, 2 FAIL / 19 tests** in 168.4 seconds. The M6 integration test passed in 58.1 seconds; app UI, orchestrator, M6 offline pipeline, and face-preprocessor tests also passed. The two failures are environment-specific: `test_colab_bundle.py` cannot clone `backend/model/emoca` because Git flags its Windows ownership as dubious for the sandbox account; `test_wsl_scripts.py` cannot write through Git Bash to the Windows profile path (`/c/Users/admin/...`), so its stage timing file is absent. Neither indicates a failure in the M6 pipeline. These failures remain open and must not be counted as a fully green suite.

Detailed paired data is generated at `%TEMP%/M6_REAL_BENCHMARK_2026-09-30.json`; methodology and interpretation are also recorded in `documents/docs/qa/M6_REAL_BENCHMARK_2026-09-30.md`.

Full restoration smoke: **PASS** on one 100×100 RAF-DB sample with all three models and saved CLI metadata. This does not establish high-resolution behavior or quality across the dataset; broader A/B and fidelity review remains future work.
