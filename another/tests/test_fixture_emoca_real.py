"""Kiểm fixture vàng RÚT GỌN của EMOCA thật: tests/fixtures/emoca_analyze_real_sample.min.json (M1-01 AC2, M0-04 AC5).

Chạy:  python tests/test_fixture_emoca_real.py        (cần jsonschema; không cần torch/GPU)

Fixture là TẬP CON của một phản hồi `POST /v1/analyze` thật (EMOCA_v2_lr_mse_20, e0be0db+patches, WSL2, 2026-09-25):
đã bỏ `faces[].renders` và `faces[].crop_image` (ảnh dựng từ FLAME có giấy phép; ảnh gốc AffectNet không được phát
lại) — xem tests/fixtures/README.md. Vì contract BẮT BUỘC `renders`, fixture KHÔNG qua schema đầy đủ; test này
dùng schema con suy ra từ chính contracts/emoca_analyze.schema.json (chỉ bỏ `renders` khỏi `required` của face,
không sửa contract) rồi kiểm thêm: contract_version 0.3.0 (đo lúc 0.2.0; ADR-019 chỉ thêm trường tuỳ chọn cho web_analyze, B1 không đổi hình dạng), codes 100/50/50/6/128, 68 landmark ở cả crop224 và
original_px, crop_transform · original_px ≈ crop224 (≤ 1 px), số hữu hạn.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("SKIP thiếu jsonschema (pip install jsonschema)")
    sys.exit(0)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
FIXTURE = ROOT / "another" / "tests" / "fixtures" / "emoca_analyze_real_sample.min.json"
SCHEMA = ROOT / "backend" / "contracts" / "emoca_analyze.schema.json"
CODE_DIMS = {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}  # PROJECT_CONTEXT §3
N_LANDMARKS = 68
TOL_PX = 1.0  # như tests/e2e_wsl_emoca.py LANDMARK_TOL_PX
REMOVED = ("renders", "crop_image")
META_KEY = "_fixture"  # khối mô tả nguồn của fixture, không thuộc contract

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def subset_schema(full: dict) -> dict:
    """Schema con: y hệt contract, chỉ bỏ các trường ảnh đã gỡ khỏi `required` của face."""
    s = copy.deepcopy(full)
    face = s["$defs"]["face"]
    face["required"] = [k for k in face["required"] if k not in REMOVED]
    return s


def apply(m, x, y):
    w = m[2][0] * x + m[2][1] * y + m[2][2]
    return (m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w


def main() -> int:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    full = json.loads(SCHEMA.read_text(encoding="utf-8"))
    body = {k: v for k, v in data.items() if k != META_KEY}

    meta = data.get(META_KEY, {})
    check(meta.get("kind") == "subset_of_emoca_analyze_response", f"có khối {META_KEY} mô tả đây là tập con")
    faces = body.get("faces") or []
    check(all(k not in f for f in faces for k in REMOVED), "không còn ảnh (renders, crop_image) trong fixture")
    check(len(FIXTURE.read_bytes()) < 100_000, f"fixture nhỏ ({len(FIXTURE.read_bytes())} byte) — không lọt base64 ảnh")

    try:
        jsonschema.validate(body, full)
        check(False, "schema ĐẦY ĐỦ phải từ chối fixture (thiếu renders) — nếu qua thì README/test này đã lỗi thời")
    except jsonschema.ValidationError as e:
        check("renders" in e.message, f"schema đầy đủ từ chối đúng vì thiếu renders ({e.message[:60]})")

    try:
        jsonschema.validate(body, subset_schema(full))
        check(True, "đúng schema con (contract, bỏ renders khỏi required của face)")
    except jsonschema.ValidationError as e:
        where = "/".join(str(p) for p in e.absolute_path) or "(gốc)"
        check(False, f"sai schema con tại {where}: {e.message[:120]}")

    check(body.get("contract_version") == "0.3.0", f"contract_version = {body.get('contract_version')!r} (cần '0.3.0')")
    model = body.get("model", {})
    check(model.get("name") == "EMOCA_v2_lr_mse_20" and model.get("upstream_commit", "").startswith("e0be0db"),
          f"model = {model.get('name')} @ {model.get('upstream_commit')}")
    check(len(faces) == 1, f"{len(faces)} mặt (cần 1)")
    if not faces:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    f = faces[0]

    dims = {k: len(f["codes"].get(k) or []) for k in CODE_DIMS}
    check(dims == CODE_DIMS, f"codes dài shape/exp/tex/pose/detail = {dims}")
    check(all(math.isfinite(v) for k in CODE_DIMS for v in f["codes"][k]), "codes toàn số hữu hạn")

    lm = f["landmarks"]
    c224, orig = lm.get("crop224") or [], lm.get("original_px") or []
    check(lm.get("n_points") == N_LANDMARKS and len(c224) == N_LANDMARKS and len(orig) == N_LANDMARKS,
          f"{N_LANDMARKS} landmark: n_points={lm.get('n_points')}, crop224={len(c224)}, original_px={len(orig)}")
    check(all(0 <= u <= 224 and 0 <= v <= 224 for u, v in c224), "landmark crop224 nằm trong khung 224×224")

    tf = f["crop_transform"]
    check(tf.get("from") == "original_px" and tf.get("to") == "crop224", f"crop_transform {tf.get('from')} → {tf.get('to')}")
    m = tf["matrix_3x3"]
    err = max(max(abs(apply(m, x, y)[0] - u), abs(apply(m, x, y)[1] - v)) for (x, y), (u, v) in zip(orig, c224))
    check(err <= TOL_PX, f"crop_transform · original_px ≈ crop224: sai lệch lớn nhất {err:.4f} px (≤ {TOL_PX})")

    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
