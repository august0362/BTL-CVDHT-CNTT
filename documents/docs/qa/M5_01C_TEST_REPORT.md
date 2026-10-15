# M5-01c — Face preprocessor test report

## Scope

Updated `backend/web/face_preprocessor.py` to reject malformed detector output with `RuntimeError`: `None`, a prediction without `boxes`, malformed box/confidence tensors, count/shape mismatch, and invalid confidence values. `NoFaceDetectedError` is reserved for an empty result list or structurally valid prediction with zero boxes. Tests use a detector fake with the tensor-like `detach()`, `cpu()`, and `numpy()` interface. No checkpoint, network, or Ultralytics import is needed.

Coverage includes valid no-face outputs versus malformed outputs; highest-confidence selection and stable ties; bbox rounding, clamping, and non-empty crops; path/PIL/NumPy BGR/NumPy RGB inputs and RGB output; invalid arrays, input type/path/color order, and constructor settings; annotation and metadata; malformed result/box/confidence structures; and non-finite or out-of-range values.

## Verification

Command run in the project venv after installing `pytest==9.1.1`:

```powershell
& $PY -m pytest another/tests/test_face_preprocessor.py -q -p no:cacheprovider --basetemp C:\tmp\pytest_m5_01_20260930_r2
```

Result: **35 passed** in 0.23 seconds; one Pillow `Image.getdata()` deprecation warning. The first run had 33 passes and two `tmp_path` setup errors because the sandbox denied writes to the default Temp directory. Rerunning with a dedicated temporary directory under `C:\tmp` completed successfully; the temporary directory was removed afterward.

Syntax check attempted:

```powershell
& 'C:\Users\admin\venvs\ma3d-web\Scripts\python.exe' -m py_compile backend/web/face_preprocessor.py another/tests/test_face_preprocessor.py
```

Earlier syntax-check attempt: blocked by a sandbox `PermissionError` writing `web\\__pycache__`; the read-only AST check below passed. Current venv versions: Python 3.13.5, Pillow 12.3.0, NumPy 2.5.3, pytest 9.1.1. `pytest` was installed into the project venv; no project requirements changed in this test step.

Read-only syntax check:

```powershell
& 'C:\Users\admin\venvs\ma3d-web\Scripts\python.exe' -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8'), filename=p) for p in ['backend/web/face_preprocessor.py','another/tests/test_face_preprocessor.py']]; print('AST syntax: PASS')"
```

Result: `AST syntax: PASS`.

## Integration note

The offline unit suite is verified. Production deployment remains opt-in under ADR-021. A balanced 21-image RAF-DB A/B ran through real EMOCA and both real FER models: YOLO detected 16/21 at confidence 0.5, and each FER model had one top-1 change among the 16 matched pairs. The measured result and limits are in `documents/docs/qa/M5_01_AB_2026-09-30.md`. Improve/evaluate detection on the target 100×100 RAF-DB images before forwarding the crop in the application. The checkpoint is user-approved for this coursework; its exact upstream commit is not recorded locally.
