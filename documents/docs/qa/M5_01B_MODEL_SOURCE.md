# M5-01b — YOLO26-face model source verification

## Current status (2026-09-30)

The checkpoint is present at `backend/model/yolo26_face/yolo26_face.pt`; the local file has been SHA-256 fingerprinted and loaded successfully with Ultralytics. The user authorized this candidate for coursework and considers the source trusted. Its exact upstream revision/origin and an independent upstream weights-license statement remain unverified; the local hash identifies only the file currently present. Do not interpret the historical initial-review notes below as the current artifact or runtime status. Current details are in `backend/model/yolo26_face/model_info.json` and `documents/docs/qa/M5_STATUS_2026-09-30.md`.

## Artifact update (2026-09-30)

A local `backend/model/yolo26_face/yolo26_face.pt` is now present. Read-only inspection found:

- Size: **20,289,541 bytes**.
- SHA-256: **`9FB5C9E36D3257C2979842E7573082BC05E4E78309D30A7333971C47B4B725B4`**.
- The file does not begin with the Git LFS pointer signature; its initial bytes have ZIP/PyTorch-archive form. This does not establish that it is the upstream checkpoint or that it can load.
- Source revision, actual download date, and origin are **unverified**. The previously failed GitHub clone does not explain how this local file arrived.
- At this point in the review, inference had not yet been run; the subsequent smoke-test result is recorded below.
- User subsequently approved this candidate for coursework use and stated that the source is trusted. This project authorization is recorded separately from independent source/license verification; the model remains isolated from the default pipeline.

The file size differs from the previously supplied estimate of 20,237,568 bytes, so do not use that estimate as an integrity check. Metadata records the observed size and digest; source revision remains unpinned, and the user-approved coursework use is not represented as an independent legal review.

## Smoke test update (2026-09-30)

The project venv `C:\Users\admin\venvs\ma3d-web` already had Python 3.13.5, `torch 2.14.0+cpu`, and `torchvision 0.29.0+cpu`. Installed `ultralytics 8.4.166` and later `pytest 9.1.1` into that venv; Torch was not reinstalled or upgraded. Pip also installed Ultralytics dependencies including OpenCV, Polars, and supporting packages. Both runtime/test versions are now pinned in `environments/web/requirements-real.txt`.

1. Loaded `backend/model/yolo26_face/yolo26_face.pt` using `ultralytics.YOLO`; the model reported task `detect` and class map `{0: 'face'}`.
2. A synthetic 640×640 black image produced a valid empty detection collection, as expected for a no-face input.
3. Ran `FaceDetectionPreprocessor` on the existing local RAF-DB test image `database/image/archive/DATASET/test/2/test_2303_aligned.jpg`, without saving or displaying image-derived output. Result: **one face**, bbox `(3, 2, 95, 100)`, confidence `0.807876`, input/annotation `100×100`, RGB crop `92×98`, inference `4931.494 ms` on CPU. The annotated PIL image and crop were returned in memory and were non-empty.
4. Ultralytics logged a non-fatal permission error while attempting to update `%APPDATA%\Ultralytics\settings.json`; model loading and inference still succeeded.

**Smoke verdict:** load, forward pass, detection, annotation, and crop returned successfully on the available local sample. The user approved use of this candidate for coursework and stated the source is trusted. The local file's upstream revision is not independently identified. ADR-021 default pipeline is unchanged.

## Historical source review and initial download attempt (2026-09-30)

The requested download was attempted before inference. Environment inspection found Python 3.11.9, Pillow available, and `ultralytics`, `torch`, and OpenCV unavailable. Git LFS 3.7.1 is installed. No dependencies were installed.

- Command: `git clone --filter=blob:none --no-checkout https://github.com/marceloeatworld/yolo26-training.git %TEMP%\yolo26-training-m5-01b`
- Result: failed with `fatal: unable to access 'https://github.com/marceloeatworld/yolo26-training.git/': Failed to connect to github.com port 443 after 35 ms: Could not connect to server`.
- At the time of this initial attempt, Git LFS weights were not obtained and the local checkpoint did not exist. That status is superseded by the artifact update above; the later file's origin is unknown.
- Inference and bbox/annotation/crop processing were not run: the current Python environment lacks Ultralytics and PyTorch. There was no suitable approved test image identified; no face image or derived output was written to the repository.
- At the time of that initial review, training-weight rights had not been independently established and the default pipeline was not connected to YOLO. The later user authorization and smoke-test results are recorded above.

The previous source review below records candidate claims and outstanding provenance/rights questions; it does not constitute verification of the unavailable artifact.

**Initial review date:** 2026-09-30  
**Initial verdict:** BLOCKED at that time — no checkpoint downloaded; no preparation script or Ultralytics dependency added. The artifact update above records later local file discovery.

## Historical candidate and technical findings (initial review)

Candidate: `marceloeatworld/yolo26-training`, advertised as `yolo26_face`, a single-class face detector based on YOLO26s. The repository describes 640-pixel input, end-to-end detection output, and training on WIDER FACE. It names Ultralytics 8.4.33 and PyTorch 2.8.0 as the training stack. These establish the advertised task and architecture, not that the candidate file itself has been independently inspected.

Ultralytics' current documentation supports YOLO26 detection through `from ultralytics import YOLO`, loading `.pt` weights with `YOLO(path)`, and inference via `.predict(...)`. API compatibility is plausible, but this checkpoint has not been loaded or smoke-tested. No runtime version is pinned because the artifact is not approved.

## Historical provenance and rights review (initial review)

- Mutable candidate URL: <https://github.com/marceloeatworld/yolo26-training/blob/main/checkpoints/yolo26_face.pt>. Repository README says the file is 19.3 MB and indicates `.pt` files are Git LFS tracked. No immutable commit or release asset was verified; GitHub API inspection was unavailable during this review. **Revision: unverified.**
- **SHA-256: unverified.** No checkpoint was fetched or hashed. The README's size is not an integrity check.
- The README lists MIT for training code and AGPL-3.0 or Enterprise terms for the Ultralytics base model, and points to WIDER FACE terms. It does not state a license, grant, or attribution requirement for the trained face weights themselves. A repository README/license label is insufficient to establish those rights. **Weights license/attribution: unresolved.**
- The official WIDER FACE project page labels the dataset Creative Commons BY-NC-ND; the CUHK-CSE Hugging Face dataset metadata says CC BY-NC-ND 4.0. The candidate README reports training on WIDER FACE through a CUHK-CSE mirror. The effect of those dataset terms on training and redistribution/use of derived weights has not been resolved, and therefore cannot be treated as permission to use these weights.

## Historical initial decision and blockers (superseded)

Do not download or enable the candidate. Do not invent a digest, revision, or license. To unblock, obtain an immutable upstream revision/object identity and independently computed SHA-256, and obtain an explicit rights statement covering the trained weights plus their Ultralytics base and WIDER FACE training data terms. Once those are verified, pin the compatible Ultralytics version and implement the atomic temporary-download-and-verify script.

`backend/model/yolo26_face/model_info.json` remains `pending-verification` / `candidate-not-downloaded`; its technical fields are retained and its notes record the unresolved source and compatibility status.

## Sources checked

- Candidate repository and backend/model/training/license claims: <https://github.com/marceloeatworld/yolo26-training>
- Ultralytics YOLO26 model and Python inference docs: <https://docs.ultralytics.com/models/yolo26>
- Ultralytics detection task/API docs: <https://docs.ultralytics.com/tasks/detect>
- Official WIDER FACE project/license page: <https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/>
- CUHK-CSE dataset card and stated CC BY-NC-ND 4.0 terms: <https://huggingface.co/datasets/CUHK-CSE/wider_face>

## Initial review files and checks (historical)

- Changed: `backend/model/yolo26_face/model_info.json`, `documents/docs/qa/M5_01B_MODEL_SOURCE.md`.
- No checkpoint, script, or dependency file was added because verification is incomplete.
- Manual checks: read ADR-021, M5-01 backlog, current `model_info.json`; web review of candidate repository, Ultralytics docs, and WIDER FACE terms. No model-download or inference command was run.
