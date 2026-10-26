"""Kiểm file thống kê chuẩn hoá 3DMM `*_emoca_stats.npz` TRƯỚC khi đưa vào registry — không cần torch.

Chạy:
  python tests/check_emoca_stats.py <đường_dẫn.npz>   # kiểm một file thật; mã thoát 0 = PASS, 1 = FAIL
  python tests/check_emoca_stats.py --selftest        # tự sinh .npz giả trong thư mục tạm, kiểm đường PASS
                                                     # và từng kiểu FAIL (thiếu khoá, sai chiều, NaN, std=0)

Định dạng đúng (suy từ mã gốc, không đoán):
- Một file `np.savez` gồm ĐÚNG 10 khoá `{exp,detail,pose,shape,tex}_{mean,std}`; mỗi mảng là vector số thực
  có số phần tử: exp 50, detail 128, pose 6, shape 100, tex 50 (model/ma3d/Read_dataset/RAF_DB.py:30-31).
  Tổng 334 = chiều `x_3d` ghép theo thứ tự [exp, detail, pose, shape, tex] (model/ma3d/engine.py:33;
  web/engine.py `_X3D_ORDER`).
- Cách dùng: x = (x − `<key>_mean`) / `<key>_std` cho từng nhóm (RAF_DB.py:76-77; Caers.py:90-91;
  DatasetCheo.py:98-99; web/engine.py `_x3d`, đọc file bằng `_load_stats` rồi reshape(-1)).
- Khoá thừa là lỗi: `compute_stats` sinh khoá từ TÊN FILE .npy trong thư mục mẫu (RAF_DB.py:132), nên
  một file .npy lạ (vd. cam/light) sẽ sinh khoá lạ ⇒ dấu hiệu dữ liệu đầu vào không đúng bộ 5 nhóm.
- std đã được mã gốc thay bằng 1.0 khi < 1e-6 (RAF_DB.py:150), nên std ≤ 0 là file hỏng; std < 1e-8 chỉ cảnh báo.

Cách sinh file theo mã gốc:
- Repo gốc CÓ `compute_stats` cho RAF-DB: model/ma3d/Read_dataset/RAF_DB.py:102-160, gọi mẫu ở :163-168
  (`compute_stats(root_dir="Datasets/RAF-DB_lmk", split="train", output_name="rafdb_emoca_stats.npz")`).
  Nó duyệt `<root>/train/<nhãn>/<mẫu>/*.npy` (bỏ `*_lmk.npy`), reshape(-1), bỏ khoá có < 10 mẫu
  (`min_samples`, :144-146), lấy mean/std theo trục mẫu (std quần thể, ddof=0), ghi `<root>/rafdb_emoca_stats.npz`.
  Hướng dẫn chung: model/ma3d/README.md:24. Bản tương tự: Caers.py:101-145 (ghi `caers_stats.npz`),
  DatasetCheo.py:109-167 (`compute_cheo_stats` → `cheo_emoca_stats.npz`), VKIST_3087.py:95.
- Đầu vào của nó là mã 3DMM EMOCA của từng ảnh TRAIN RAF-DB (các file exp/detail/pose/shape/tex.npy) —
  phải chạy EMOCA (Colab, ADR-015) trên toàn bộ tập train; dự án hiện CHƯA có dữ liệu này.
- Lưu ý: `min_samples` âm thầm bỏ khoá ⇒ file có thể thiếu nhóm; công cụ này bắt lỗi đó.

Bước tiếp theo để đưa vào registry (CHỈ mô tả — việc của CONFIG, cần người dùng cung cấp file):
1. `python tests/check_emoca_stats.py <file>` → PASS.
2. Đặt file dưới `model/ma3d/checkpoints/` (gitignored, không commit), đo `sha256sum` và số byte.
3. configs/models.yaml: thêm mục `artifacts.<id>` (`path`, `sha256`, `bytes`, nguồn/người cung cấp), rồi
   đặt `stats: <id>` cho mô hình MA3D tương ứng (vd. `ma3d_rafdb`).
4. `python tests/test_configs.py --sha256` → ALL PASS.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Số chiều mỗi nhóm — model/ma3d/Read_dataset/RAF_DB.py:31. Thứ tự = web/engine.py `_X3D_ORDER`.
DIMS = {"exp": 50, "detail": 128, "pose": 6, "shape": 100, "tex": 50}
EXPECTED_KEYS = {f"{g}_{s}" for g in DIMS for s in ("mean", "std")}
STD_WARN = 1e-8


def check_file(path: Path, quiet: bool = False) -> tuple[list[str], list[str]]:
    """Trả (errors, warnings). In tóm tắt nếu không quiet."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        npz = np.load(path, allow_pickle=False)
    except Exception as exc:  # file hỏng/không phải npz: báo lỗi, không nuốt
        return [f"không đọc được npz: {type(exc).__name__}: {exc}"], warnings
    with npz:
        keys = set(npz.files)
        for k in sorted(EXPECTED_KEYS - keys):
            errors.append(f"thiếu khoá '{k}'")
        for k in sorted(keys - EXPECTED_KEYS):
            errors.append(f"khoá thừa '{k}'")
        arrays = {k: npz[k] for k in sorted(keys & EXPECTED_KEYS)}

    for k, a in arrays.items():
        dim = DIMS[k.rsplit("_", 1)[0]]
        if not np.issubdtype(a.dtype, np.floating):
            errors.append(f"'{k}': dtype {a.dtype} không phải số thực")
            continue
        if a.size != dim:
            errors.append(f"'{k}': {a.size} phần tử (shape {a.shape}), cần {dim}")
            continue
        if a.shape != (dim,):
            warnings.append(f"'{k}': shape {a.shape} ≠ ({dim},) — web/engine.py reshape(-1) nên vẫn dùng được")
        n_bad = int((~np.isfinite(a)).sum())
        if n_bad:
            errors.append(f"'{k}': {n_bad} giá trị không hữu hạn (NaN/Inf)")
            continue
        if k.endswith("_std"):
            n_nonpos = int((a <= 0).sum())
            if n_nonpos:
                errors.append(f"'{k}': {n_nonpos} phần tử std <= 0")
            elif (a < STD_WARN).any():
                warnings.append(f"'{k}': {int((a < STD_WARN).sum())} phần tử std < {STD_WARN:g}")

    if not quiet:
        print(f"== {path}")
        print(f"  {'nhóm':<7}{'chiều':>6}  {'mean min':>11}{'mean max':>11}  {'std min':>11}{'std max':>11}")
        for g, dim in DIMS.items():
            m, s = arrays.get(f"{g}_mean"), arrays.get(f"{g}_std")

            def rng(a):
                ok = a is not None and np.issubdtype(a.dtype, np.floating) and a.size and np.isfinite(a).all()
                return (f"{a.min():>11.4g}{a.max():>11.4g}") if ok else f"{'—':>11}{'—':>11}"
            print(f"  {g:<7}{dim:>6}  {rng(m)}  {rng(s)}")
        for w in warnings:
            print("  WARN " + w)
        for e in errors:
            print("  FAIL " + e)
        print("PASS" if not errors else f"FAIL — {len(errors)} lỗi")
    return errors, warnings


def _good_stats(rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for g, dim in DIMS.items():
        out[f"{g}_mean"] = rng.normal(size=dim).astype(np.float32)
        out[f"{g}_std"] = rng.uniform(0.1, 2.0, size=dim).astype(np.float32)
    return out


def selftest() -> int:
    rng = np.random.default_rng(0)
    cases: list[tuple[str, dict, str | None]] = []  # (tên, dữ liệu, chuỗi phải có trong lỗi; None = PASS)
    cases.append(("hợp lệ", _good_stats(rng), None))
    d = _good_stats(rng); del d["pose_std"]
    cases.append(("thiếu khoá", d, "thiếu khoá 'pose_std'"))
    d = _good_stats(rng); d["cam_mean"] = np.zeros(3, np.float32)
    cases.append(("khoá thừa", d, "khoá thừa 'cam_mean'"))
    d = _good_stats(rng); d["exp_mean"] = np.zeros(49, np.float32)
    cases.append(("sai chiều", d, "'exp_mean': 49 phần tử"))
    d = _good_stats(rng); d["detail_mean"][3] = np.nan
    cases.append(("NaN", d, "'detail_mean': 1 giá trị không hữu hạn"))
    d = _good_stats(rng); d["shape_std"][0] = 0.0
    cases.append(("std=0", d, "'shape_std': 1 phần tử std <= 0"))
    d = _good_stats(rng); d["tex_mean"] = np.arange(50, dtype=np.int64)
    cases.append(("dtype nguyên", d, "'tex_mean': dtype int64"))

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, data, expect) in enumerate(cases):
            p = Path(tmp) / f"case{i}.npz"
            np.savez(p, **data)
            errors, _ = check_file(p, quiet=True)
            ok = (not errors) if expect is None else any(expect in e for e in errors)
            print(("  ok   " if ok else "  FAIL ") + f"{name}: "
                  + ("PASS như mong đợi" if expect is None and ok else "; ".join(errors) or "không phát hiện lỗi"))
            failed += not ok
        # Cảnh báo std rất nhỏ nhưng dương: vẫn PASS
        d = _good_stats(rng); d["pose_std"][0] = 1e-9
        p = Path(tmp) / "warn.npz"; np.savez(p, **d)
        errors, warnings = check_file(p, quiet=True)
        ok = not errors and any("pose_std" in w for w in warnings)
        print(("  ok   " if ok else "  FAIL ") + f"std < {STD_WARN:g}: PASS kèm cảnh báo")
        failed += not ok
    print()
    print("PASS" if not failed else f"FAIL — {failed} ca sai")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        return selftest()
    if len(argv) != 1 or argv[0].startswith("-"):
        print(__doc__.split("\n\n")[1])
        return 1
    p = Path(argv[0])
    if not p.is_file():
        print(f"FAIL — không thấy file: {p}")
        return 1
    errors, _ = check_file(p)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
