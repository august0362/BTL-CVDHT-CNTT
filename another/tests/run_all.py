"""Bộ chạy kiểm chứng chung của dự án (CLAUDE.md §6) — chỉ dùng thư viện chuẩn, Python >= 3.10.

Chạy:  python tests/run_all.py [--heavy] [--list] [--only MẪU] [--timeout N] [--dir THƯ_MỤC]

  (mặc định)    chạy nhóm NHẸ, tuần tự, rồi in bảng: tên | system/venv | PASS/FAIL/SKIP | giây | số dòng ok/PASS
  --heavy       thêm nhóm NẶNG (test_engine_poster_real.py, test_engine_registry_extend.py: nạp torch + checkpoint, ~1.5 GB RAM)
  --list        chỉ in kế hoạch (test nào, nhóm nào, trình thông dịch nào), không chạy gì
  --only MẪU    chỉ giữ test có tên chứa MẪU (hoặc khớp glob nếu MẪU có * ? [); lặp lại được
  --timeout N   giới hạn mỗi test N giây (mặc định 300); quá hạn ⇒ FAIL
  --dir DIR     tìm test trong DIR thay vì tests/ (dùng để thử chính bộ chạy trên test giả)

Ví dụ:
  python tests/run_all.py                       # nhóm nhẹ
  python tests/run_all.py --only test_configs   # một test
  python tests/run_all.py --heavy --list        # xem kế hoạch đầy đủ, không chạy

Tìm test: mọi tests/test_*.py và tests/check_*.py (tự nhặt test mới). KHÔNG chạy audit_*.py (cần trình duyệt)
và file bắt đầu bằng `_` (module phụ trợ, ví dụ _torchfree_ckpt.py).

Trình thông dịch:
  - mặc định: chính Python đang chạy file này (sys.executable) — "system";
  - "venv": test mà docstring đầu file nhắc tới "venv web" (tức cần streamlit/torch; hiện là test_app_ui.py,
    test_engine_poster_real.py) chạy bằng python của venv web: biến môi trường MA3D_VENV_PYTHON, mặc định
    <thư mục người dùng>\\venvs\\ma3d-web\\Scripts\\python.exe (Path.home()). Không có file đó ⇒ SKIP có lý do (không chạy bằng python khác).

Phân loại (đọc đầu ra từng test):
  FAIL  mã thoát khác 0, hoặc quá hạn, hoặc có dòng bắt đầu bằng "FAIL" (kể cả thụt lề) — in 15 dòng cuối đầu ra;
  SKIP  mã 0, có dòng bắt đầu bằng "SKIP" ở đầu dòng (không thụt lề), và không có "ALL PASS";
  PASS  mã 0 và có "ALL PASS", hoặc mã 0 với test không in dòng đó.
  Dòng "  SKIP ..." thụt lề là bỏ qua một phần bên trong test — vẫn tính PASS, cột ghi chú báo số phần bị bỏ qua.
Mã thoát của bộ chạy: 0 nếu không có FAIL, 1 nếu có.

Thêm test: tạo tests/test_<tên>.py, in "  ok ..."/"PASS ..." cho từng kiểm, "ALL PASS" khi xong, thoát mã khác 0 khi hỏng;
thiếu thư viện ⇒ in "SKIP <lý do>" và thoát 0. Muốn test chạy bằng venv web: ghi "venv web" trong docstring đầu file.
Đổi nhóm: thêm/bớt tên file trong HEAVY bên dưới (nhóm NẶNG = nạp torch/checkpoint lớn, chỉ chạy khi có --heavy).
Test cần tham số khi chạy trong bộ (vd. check_emoca_stats.py --selftest): thêm vào ARGS bên dưới.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
HEAVY = {"test_engine_poster_real.py", "test_engine_registry_extend.py", "test_m6_real_integration.py",
         "test_m7_real_acceptance.py", "test_m7_real_detection_cap.py", "test_frontalization.py"}
# Tham số dòng lệnh riêng cho test không tự chạy được khi gọi trần (xem docstring của từng file).
ARGS = {"check_emoca_stats.py": ["--selftest"]}
VENV_MARKER = "venv web"
# Theo thư mục người dùng hiện tại (RUNBOOK 1.1: "$HOME\venvs\ma3d-web"), không ghi cứng tên người dùng
# (docs/qa/RUNBOOK_DRYRUN_2026-09-25.md L3). MA3D_VENV_PYTHON vẫn thắng nếu đặt.
DEFAULT_VENV = str(Path.home() / "venvs" / "ma3d-web" / "Scripts" / "python.exe")
CHILD_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "2",
}


def discover(test_dir: Path) -> list[Path]:
    found = set(test_dir.glob("test_*.py")) | set(test_dir.glob("check_*.py"))
    return sorted(found, key=lambda p: p.name)


def needs_venv(path: Path) -> bool:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    except (SyntaxError, UnicodeDecodeError):
        return False  # để chạy thật và FAIL với lỗi cú pháp, thay vì đoán
    return VENV_MARKER in doc


def matches(name: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if any(c in pat for c in "*?[") and fnmatch.fnmatch(name, pat):
            return True
        if pat in name:
            return True
    return False


def plan(args) -> list[dict]:
    venv_python = os.environ.get("MA3D_VENV_PYTHON", DEFAULT_VENV)
    items = []
    for path in discover(args.dir):
        heavy = path.name in HEAVY
        venv = needs_venv(path)
        item = {"path": path, "name": path.name, "group": "NẶNG" if heavy else "nhẹ",
                "env": "venv" if venv else "system",
                "python": venv_python if venv else sys.executable, "args": ARGS.get(path.name, []), "skip": None}
        if args.only and not matches(path.name, args.only):
            continue
        if heavy and not args.heavy:
            item["skip"] = "nhóm NẶNG (thêm --heavy)"
        elif venv and not Path(venv_python).is_file():
            item["skip"] = f"không thấy python venv web: {venv_python} (đặt MA3D_VENV_PYTHON)"
        items.append(item)
    return items


def classify(rc: int | None, out: str) -> tuple[str, str]:
    lines = out.splitlines()
    fail_lines = [ln for ln in lines if ln.lstrip().startswith("FAIL")]
    sub_skips = [ln for ln in lines if ln.startswith((" ", "\t")) and ln.lstrip().startswith("SKIP")]
    has_all_pass = any(ln.strip() == "ALL PASS" for ln in lines)
    if rc is None:
        return "FAIL", "quá hạn"
    if rc != 0:
        return "FAIL", f"mã thoát {rc}"
    if fail_lines:
        return "FAIL", "mã 0 nhưng có dòng FAIL"
    if not has_all_pass and any(ln.startswith("SKIP") for ln in lines):
        return "SKIP", next(ln for ln in lines if ln.startswith("SKIP"))[:90]
    note = "" if has_all_pass else "không in ALL PASS"
    if sub_skips:
        note = (note + "; " if note else "") + f"{len(sub_skips)} phần SKIP bên trong"
    return "PASS", note


def count_checks(out: str) -> int:
    n = 0
    for ln in out.splitlines():
        s = ln.lstrip()
        if s.startswith("ok ") or (s.startswith("PASS ") and s.strip() != "ALL PASS"):
            n += 1
    return n


def run_one(item: dict, timeout: int) -> dict:
    env = dict(os.environ, **CHILD_ENV)
    # Running an absolute tests/<file>.py puts tests/ (not the repository root)
    # at sys.path[0]. Keep project-package imports working for direct scripts.
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT / "backend") + (os.pathsep + python_path if python_path else "")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([item["python"], str(item["path"]), *item["args"]], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout)
        rc, raw = proc.returncode, proc.stdout
    except subprocess.TimeoutExpired as exc:
        rc, raw = None, exc.stdout or b""
    out = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    status, note = classify(rc, out)
    return {"status": status, "note": note, "secs": time.perf_counter() - t0, "checks": count_checks(out), "out": out}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Chạy tuần tự các test của dự án (xem docstring).")
    ap.add_argument("--heavy", action="store_true", help="thêm nhóm NẶNG")
    ap.add_argument("--list", action="store_true", help="chỉ in kế hoạch")
    ap.add_argument("--only", action="append", default=[], metavar="MẪU", help="lọc theo tên (lặp lại được)")
    ap.add_argument("--timeout", type=int, default=300, metavar="N", help="giây/test (mặc định 300)")
    ap.add_argument("--dir", type=Path, default=ROOT / "another" / "tests", help="thư mục chứa test")
    args = ap.parse_args()
    args.dir = args.dir.resolve()

    items = plan(args)
    if not items:
        print(f"Không có test nào khớp trong {args.dir}")
        return 1

    if args.list:
        print(f"Kế hoạch ({len(items)} test, thư mục {args.dir}):")
        for it in items:
            what = f"BỎ QUA — {it['skip']}" if it["skip"] else " ".join([f"chạy bằng {it['python']}", *it["args"]])
            print(f"  {it['name']:<32} {it['group']:<5} {it['env']:<7} {what}")
        return 0

    results = []
    t_all = time.perf_counter()
    for it in items:
        if it["skip"]:
            res = {"status": "SKIP", "note": it["skip"], "secs": 0.0, "checks": 0, "out": ""}
        else:
            print(f"… {it['name']} ({it['env']})", flush=True)
            res = run_one(it, args.timeout)
            if res["status"] == "FAIL":
                tail = res["out"].splitlines()[-15:]
                print(f"  FAIL {it['name']} ({res['note']}) — 15 dòng cuối:")
                for ln in tail:
                    print(f"    | {ln}")
        results.append((it, res))

    w = max(len(it["name"]) for it, _ in results)
    print()
    print(f"{'test':<{w}} | env    | kết quả | giây   | ok/PASS | ghi chú")
    print(f"{'-' * w}-+--------+---------+--------+---------+--------")
    for it, res in results:
        print(f"{it['name']:<{w}} | {it['env']:<6} | {res['status']:<7} | {res['secs']:6.1f} | {res['checks']:>7} | {res['note']}")
    counts = {s: sum(1 for _, r in results if r["status"] == s) for s in ("PASS", "SKIP", "FAIL")}
    print(f"\nTỔNG: {counts['PASS']} PASS, {counts['SKIP']} SKIP, {counts['FAIL']} FAIL / {len(results)} test "
          f"— {time.perf_counter() - t_all:.1f} giây")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())

