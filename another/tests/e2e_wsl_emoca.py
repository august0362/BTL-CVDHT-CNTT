"""Kiểm đầu-cuối từ phía Windows: web/emoca_client.py (chế độ real) → emoca_svc thật đang chạy trong WSL2 (ADR-018).

Chạy (venv web hoặc python hệ thống có pillow, pyyaml, jsonschema; KHÔNG cần torch):
    python tests/e2e_wsl_emoca.py --image <ảnh-khuôn-mặt.jpg>            # mặc định host/port ở configs/services.yaml
    python tests/e2e_wsl_emoca.py --image <ảnh> --url http://127.0.0.1:8765 --runs 3
    python tests/e2e_wsl_emoca.py --selftest                              # kiểm CHÍNH script này bằng mock_server

Token (nếu server bật Bearer): biến môi trường EMOCA_TOKEN — không nhận qua tham số dòng lệnh để không lộ trong lịch sử.
Làm gì: GET /v1/health (device, load_seconds, VRAM/GPU) → gọi analyze `--runs` lần, đo độ trễ khứ hồi → kiểm phản hồi
theo contracts/emoca_analyze.schema.json, codes dài 100/50/50/6/128, 68 landmark ở cả `crop224` và `original_px`,
và hai hệ khớp nhau qua `crop_transform`. Thoát 0 = PASS, 1 = FAIL, 2 = sai tham số.
Phục vụ M0-06 AC(4) và M1-03 từ phía web. Tên KHÔNG bắt đầu bằng `test_` để tests/run_all.py không tự chạy.

Quyền riêng tư: ảnh do người dùng chọn, chỉ đọc vào bộ nhớ và gửi tới địa chỉ `--url` (mặc định loopback); script không
tải ảnh từ mạng, không ghi ảnh hay phản hồi ra đĩa. Địa chỉ không phải loopback ⇒ phải thêm --allow-remote.
`--selftest` chạy với services/emoca_svc/mock_server.py: dữ liệu GIẢ, KHÔNG phải kết quả EMOCA thật — chỉ chứng minh
script kiểm đúng, không chứng minh gì về EMOCA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import jsonschema
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))
import emoca_client  # noqa: E402

SCHEMA = json.loads((ROOT / "backend" / "contracts" / "emoca_analyze.schema.json").read_text(encoding="utf-8"))
CODE_DIMS = {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}  # PROJECT_CONTEXT §3
N_LANDMARKS = 68
HEALTH_KEYS = ("mode", "device", "torch", "cuda", "load_seconds", "vram_after_load_mb", "vram_peak_mb",
               "vram_reserved_mb", "vram_reserved_peak_mb", "gpu_free_mb", "gpu_total_mb", "requests_served")
LANDMARK_TOL_PX = 1.0  # sai khác cho phép khi đổi original_px → crop224 bằng matrix_3x3 (làm tròn số ở server)


def default_url() -> str:
    cfg = yaml.safe_load((ROOT / "backend" / "configs" / "services.yaml").read_text(encoding="utf-8"))["emoca_svc"]
    return f"http://{cfg['host']}:{cfg['port']}"


def is_loopback(url: str) -> bool:
    return (urllib.parse.urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, cond: bool, msg: str) -> bool:
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            self.failures.append(msg)
        return cond


def check_face(rep: Report, face: dict, run: int) -> None:
    codes = face.get("codes", {})
    dims = {k: len(codes.get(k) or []) for k in CODE_DIMS}
    rep.check(dims == CODE_DIMS, f"lần {run}: codes dài shape/exp/tex/pose/detail = {dims}")
    rep.check(all(math.isfinite(v) for k in CODE_DIMS for v in (codes.get(k) or [])), f"lần {run}: codes toàn số hữu hạn")
    lm = face.get("landmarks", {})
    c224, orig = lm.get("crop224") or [], lm.get("original_px") or []
    rep.check(lm.get("n_points") == N_LANDMARKS and len(c224) == N_LANDMARKS and len(orig) == N_LANDMARKS,
              f"lần {run}: {N_LANDMARKS} landmark ở crop224 ({len(c224)}) và original_px ({len(orig)})")
    tf = face.get("crop_transform", {})
    m = tf.get("matrix_3x3")
    if tf.get("from") == "original_px" and tf.get("to") == "crop224" and m and len(c224) == len(orig) > 0:
        err = 0.0
        for (x, y), (u, v) in zip(orig, c224):
            w = m[2][0] * x + m[2][1] * y + m[2][2]
            err = max(err, abs((m[0][0] * x + m[0][1] * y + m[0][2]) / w - u), abs((m[1][0] * x + m[1][1] * y + m[1][2]) / w - v))
        rep.check(err <= LANDMARK_TOL_PX, f"lần {run}: crop_transform·original_px ≈ crop224 (sai lệch lớn nhất {err:.3f} px)")
    else:
        rep.check(False, f"lần {run}: thiếu crop_transform original_px → crop224")


def run_e2e(url: str, token: str | None, image: Image.Image, runs: int, timeout_s: float) -> int:
    endpoint = {"url": url, "timeout_s": timeout_s}
    if token:
        endpoint["token"] = token
    rep = Report()
    print(f"== health: GET {url}/v1/health" + (" (có token)" if token else ""))
    hp = emoca_client.check_service(endpoint)
    rep.check(hp["ok"], f"check_service: {hp['message']} ({hp['latency_ms']} ms)")
    info = hp.get("info") or {}
    for k in HEALTH_KEYS:
        if k in info:
            print(f"       {k:<22} {info[k]}")
    if "gpu_free_mb" not in info and info.get("mode") == "real":
        print("       (server chưa báo gpu_free_mb/gpu_total_mb — bản emoca_runner cũ hoặc đo cudaMemGetInfo lỗi)")
    if info.get("mode") == "mock":
        print("  CHÚ Ý: server đang chạy MOCK — kết quả dưới đây KHÔNG phải EMOCA thật.")
    if not hp["ok"]:
        print("\nFAIL — không nối được emoca_svc; dừng trước khi gửi ảnh.")
        return 1

    print(f"== analyze: {runs} lần, ảnh {image.width}×{image.height} px")
    latencies = []
    for i in range(1, runs + 1):
        t0 = time.perf_counter()
        out = emoca_client.get_analysis(image, mode="real", endpoint=endpoint)
        ms = (time.perf_counter() - t0) * 1000
        if "error" in out:
            rep.check(False, f"lần {i}: lỗi {out['error'].get('code')}: {out['error'].get('message')} ({ms:.0f} ms)")
            continue
        latencies.append(ms)
        try:
            jsonschema.validate(out, SCHEMA)
            rep.check(True, f"lần {i}: đúng emoca_analyze.schema.json — {ms:.0f} ms khứ hồi, "
                            f"timing server {out.get('timing_ms', {})}")
        except jsonschema.ValidationError as e:
            where = "/".join(str(p) for p in e.absolute_path) or "(gốc)"
            rep.check(False, f"lần {i}: sai schema tại {where}: {e.message[:120]}")
        faces = out.get("faces") or []
        if rep.check(len(faces) >= 1, f"lần {i}: {len(faces)} mặt (model {out.get('model', {}).get('name')})"):
            check_face(rep, faces[0], i)
        for w in out.get("warnings", []) + [w for f in faces for w in f.get("warnings", [])]:
            print(f"       cảnh báo: {w}")
    if latencies:
        print(f"== độ trễ khứ hồi (ms): {', '.join(f'{v:.0f}' for v in latencies)}; "
              f"trung vị {statistics.median(latencies):.0f}" + (" — lần 1 thường gồm khởi động CUDA/cuDNN" if len(latencies) > 1 else ""))
    print()
    if rep.failures:
        print(f"FAIL — {len(rep.failures)} lỗi")
        return 1
    print("PASS")
    return 0


def selftest() -> int:
    """Chạy run_e2e với mock_server (dữ liệu GIẢ, không phải EMOCA thật) — chỉ kiểm script này: đúng thì PASS,
    sai token thì FAIL, server tắt thì FAIL."""
    sys.path.insert(0, str(ROOT / "backend" / "services" / "emoca_svc"))
    import mock_server  # noqa: WPS433

    print("#### SELFTEST với mock_server — dữ liệu GIẢ, KHÔNG phải kết quả EMOCA thật ####")
    token = "selftest-token-0123456789"
    srv = mock_server.make_server(dict(mock_server.DEFAULTS), port=0, token=token)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    img = Image.new("RGB", (640, 480), (150, 120, 100))  # ảnh tổng hợp trong bộ nhớ; mock không cần mặt thật
    results = []
    try:
        print("\n## đúng token → kỳ vọng PASS")
        results.append(("đúng token → 0", run_e2e(url, token, img, 2, 10) == 0))
        print("\n## sai token → kỳ vọng FAIL")
        results.append(("sai token → 1", run_e2e(url, "sai", img, 1, 10) == 1))
    finally:
        srv.shutdown()
        srv.server_close()

    def broken(data, cfg, max_faces=1):  # phản hồi hỏng có chủ đích: expcode 49 chiều, landmark lệch khỏi crop_transform
        status, body = mock_server.analyze_bytes(data, cfg, max_faces)
        if status == 200:
            face = body["faces"][0]
            face["codes"]["exp"] = face["codes"]["exp"][:49]
            face["landmarks"]["original_px"] = [[x + 5.0, y] for x, y in face["landmarks"]["original_px"]]
        return status, body

    bad = mock_server.make_server(dict(mock_server.DEFAULTS), port=0, analyzer=broken)
    threading.Thread(target=bad.serve_forever, daemon=True).start()
    try:
        print("\n## phản hồi hỏng (exp 49 chiều, landmark lệch 5 px) → kỳ vọng FAIL")
        results.append(("phản hồi hỏng → 1", run_e2e(f"http://127.0.0.1:{bad.server_address[1]}", None, img, 1, 10) == 1))
    finally:
        bad.shutdown()
        bad.server_close()
    print("\n## server đã tắt → kỳ vọng FAIL")
    results.append(("server tắt → 1", run_e2e(url, token, img, 1, 3) == 1))
    print("\n#### kết quả selftest")
    for name, ok in results:
        print(("  ok   " if ok else "  FAIL ") + name)
    ok = all(r for _, r in results)
    print("ALL PASS (selftest với MOCK)" if ok else "FAIL (selftest)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Kiểm đầu-cuối emoca_svc (WSL2) từ Windows qua web/emoca_client.py")
    ap.add_argument("--image", help="ảnh khuôn mặt do người dùng chọn (JPEG/PNG/WEBP)")
    ap.add_argument("--url", default=None, help="địa chỉ emoca_svc (mặc định từ configs/services.yaml)")
    ap.add_argument("--runs", type=int, default=3, help="số lần gọi analyze (mặc định 3)")
    ap.add_argument("--timeout", type=float, default=120.0, help="giây cho mỗi request (mặc định 120)")
    ap.add_argument("--allow-remote", action="store_true", help="cho phép gửi ảnh tới địa chỉ không phải loopback")
    ap.add_argument("--selftest", action="store_true", help="kiểm script bằng mock_server (không cần EMOCA)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.image or args.runs < 1:
        ap.print_usage()
        print("cần --image <đường dẫn> và --runs ≥ 1")
        return 2
    url = (args.url or default_url()).rstrip("/")
    if not is_loopback(url) and not args.allow_remote:
        print(f"{url} không phải loopback — ảnh sẽ rời khỏi máy. Thêm --allow-remote nếu thật sự muốn.")
        return 2
    try:
        with Image.open(args.image) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")  # original_px tính trên ảnh đã xoay EXIF, như web
    except (OSError, ValueError) as e:
        print(f"Không đọc được ảnh: {type(e).__name__}: {e}")
        return 2
    return run_e2e(url, os.environ.get("EMOCA_TOKEN") or None, image, args.runs, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
