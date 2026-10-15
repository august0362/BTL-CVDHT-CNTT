"""Kiểm phần logic của emoca_svc THẬT (services/emoca_svc/{server,emoca_runner}.py) bằng backend GIẢ — không GPU.

Chạy:  python tests/test_emoca_svc_real.py

Phạm vi (đo được ở máy thường): chọn mặt, đổi hệ toạ độ, dựng response đúng schema, các nhánh lỗi, xác thực Bearer,
tuần tự hoá GPU, không ghi đĩa, tương thích cú pháp Python 3.8; thân `Transfer-Encoding: chunked`, trần max_faces,
mặt phụ lỗi không làm hỏng request, hook giải phóng VRAM sau OOM, nhãn `+patches` (review ADR-015 D2, L1–L3, L5);
trường VRAM/GPU của health (`EmocaBackend.info()` với torch GIẢ — finding H1); công tắc cuDNN `EMOCA_CUDNN`
(`resolve_cudnn`, `load()` với torch/gdl GIẢ: cờ đặt trước khi dựng EMOCA/FAN, health báo `cudnn_enabled`).
KHÔNG phủ: lời gọi EMOCA/torch thật (`EmocaBackend`) — chỉ chạy được trên Colab/WSL2 (notebooks/emoca_colab.ipynb).
"""

from __future__ import annotations

import ast
import base64
import http.client as httpclient  # alias: tên `http` đã dùng cho hàm trợ giúp bên dưới
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
SVC = ROOT / "backend" / "services" / "emoca_svc"
sys.path.insert(0, str(SVC))
import emoca_runner  # noqa: E402
import mock_server  # noqa: E402
import server  # noqa: E402

SCHEMA = json.loads((ROOT / "backend" / "contracts" / "emoca_analyze.schema.json").read_text(encoding="utf-8"))
TOKEN = "token-thu-nghiem-0123456789"
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def valid(body) -> bool:
    try:
        jsonschema.validate(body, SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print("       schema:", e.message[:200])
        return False


def img_bytes(size=(640, 480), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (150, 120, 100)).save(buf, format=fmt)
    return buf.getvalue()


class FakeBackend:
    """Giả lập EMOCA: hình học đúng công thức của TestData (kpt68, scale 1.25, đồng dạng → 223 px) trên numpy,
    và trả landmark chuẩn hoá SUY RA từ landmark gốc đã biết ⇒ kiểm được chuỗi đổi toạ độ khép kín."""

    def __init__(self, boxes=None, sleep=0.0, **overrides):
        self.boxes = boxes if boxes is not None else [[220.0, 140.0, 420.0, 340.0]]
        self.sleep = sleep
        self.overrides = overrides
        self.truth = {}          # face -> landmark gốc đã biết (68×2, original_px)
        self._matrix = None
        self._bbox = None
        self.active = 0
        self.max_active = 0
        self.released = 0        # số lần server gọi release_memory() (sau lỗi hết VRAM)
        self._lock = threading.Lock()

    def release_memory(self):
        self.released += 1

    def detect(self, image):
        return [list(b) for b in self.boxes], "kpt68", None

    def crop(self, image, bbox, bbox_type):
        left, top, right, bottom = bbox
        old_size = (right - left + bottom - top) / 2 * 1.1
        cx, cy = right - (right - left) / 2.0, bottom - (bottom - top) / 2.0
        size = int(old_size * 1.25)
        s = (emoca_runner.CROP_SIZE - 1) / size
        x0, y0 = cx - size / 2, cy - size / 2
        self._matrix = [[s, 0.0, -s * x0], [0.0, s, -s * y0], [0.0, 0.0, 1.0]]
        self._bbox = bbox
        return np.full((224, 224, 3), 0.5), self._matrix

    def infer(self, crop01):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.sleep:
                time.sleep(self.sleep)
            left, top, right, bottom = self._bbox
            u, v = np.meshgrid(np.linspace(0.1, 0.9, 17), np.linspace(0.1, 0.9, 4))[:2]
            pts = np.stack([left + u.ravel() * (right - left), top + v.ravel() * (bottom - top)], axis=1)[:68]
            truth = np.concatenate([pts, pts[: 68 - len(pts)]]) if len(pts) < 68 else pts
            self.truth[tuple(self._bbox)] = truth
            crop_px = emoca_runner.apply_homography(self._matrix, truth)
            lm_norm = (crop_px - 112.0) / 112.0
            rng = np.random.RandomState(7)
            codes = {k: rng.randn(n) for k, n in emoca_runner.CODE_DIMS.items()}
            res = {
                "codes": codes,
                "landmarks_norm": lm_norm,
                "renders": {k: np.full((224, 224, 3), 40 + 30 * i, dtype=np.uint8)
                            for i, k in enumerate(emoca_runner.RENDER_SOURCES)},
                "timing": {"encode": 0.010, "decode": 0.020},
            }
            res.update(self.overrides)
            return res
        finally:
            with self._lock:
                self.active -= 1

    def info(self):
        return {"model": {"name": "FAKE-EMOCA", "mode": "detail", "upstream_commit": "e0be0db+patches"},
                "loaded": True, "device": "fake-gpu"}


def make_runner(**kw):
    backend = FakeBackend(**{k: v for k, v in kw.items() if k in ("boxes", "sleep")},
                          **{k: v for k, v in kw.items() if k not in ("boxes", "sleep")})
    return backend, emoca_runner.EmocaRunner(backend, mock_server.CONTRACT_VERSION, min_face_px=32)


def raises(code, fn):
    try:
        fn()
    except emoca_runner.RunnerError as e:
        return e.code == code, e
    return False, None


def http(base, method, path, body=None, ctype=None, token=TOKEN, extra_headers=None):
    req = urllib.request.Request(base + path, data=body, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for k, v in (extra_headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, dict(r.headers), json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read().decode("utf-8"))


def post_image(base, data, options=None, **kw):
    req = {"image_b64": base64.b64encode(data).decode("ascii")}
    if options is not None:
        req["options"] = options
    return http(base, "POST", "/v1/analyze", json.dumps(req).encode(), "application/json", **kw)


def post_chunked(port, payload=None, token=TOKEN, ctype="application/json", chunk=7000, raw=None):
    """POST `Transfer-Encoding: chunked` bằng http.client (putrequest + gửi từng chunk). Trả (status, body_json)."""
    conn = httpclient.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        conn.putrequest("POST", "/v1/analyze")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.putheader("Content-Type", ctype)
        if token:
            conn.putheader("Authorization", f"Bearer {token}")
        conn.endheaders()
        if raw is not None:
            conn.send(raw)
        else:
            for i in range(0, len(payload), chunk):
                part = payload[i:i + chunk]
                conn.send(b"%x\r\n" % len(part) + part + b"\r\n")
            conn.send(b"0\r\n\r\n")
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def serve(runner, token=TOKEN, **cfg_over):
    srv = server.build_server(dict(mock_server.DEFAULTS, **cfg_over), runner, token, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_runner_logic():
    print("== EmocaRunner: logic thuần")
    backend, runner = make_runner()
    body = runner.analyze(Image.new("RGB", (640, 480), (1, 2, 3)), max_faces=1)
    check(valid(body), "response thành công đúng emoca_analyze.schema.json")
    check(emoca_runner.check_body(body) == [], "check_body: không vi phạm")
    f = body["faces"][0]
    check(body["image"] == {"width": 640, "height": 480} and f["face_index"] == 0, "image + face_index")
    check(body["model"]["upstream_commit"] == "e0be0db+patches", "model.upstream_commit lấy từ backend")
    truth = backend.truth[(220.0, 140.0, 420.0, 340.0)]
    got = np.asarray(f["landmarks"]["original_px"])
    check(float(np.abs(got - truth).max()) < 0.01, f"original_px khép kín với landmark gốc (sai số {np.abs(got - truth).max():.4f} px)")
    M = f["crop_transform"]["matrix_3x3"]
    back = emoca_runner.apply_homography(M, got)
    check(float(np.abs(back - np.asarray(f["landmarks"]["crop224"])).max()) < 0.01, "matrix_3x3 · original_px ≈ crop224")
    check(float(np.abs(np.asarray(f["landmarks"]["crop224"]) - 112.0).max()) < 112.5, "crop224 nằm trong khung ~224×224")
    center = emoca_runner.apply_homography(M, [[320.0, 240.0]])[0]
    check(abs(center[0] - 111.5) < 0.6 and abs(center[1] - 111.5) < 0.6,
          f"tâm hộp mặt ↔ tâm crop (kỳ vọng ≈ 111.5; được {center.round(2).tolist()})")
    crop = Image.open(io.BytesIO(base64.b64decode(f["crop_image"]["b64"])))
    check(crop.size == (224, 224) and crop.mode == "RGB", "crop_image PNG 224×224 RGB")
    check(all(Image.open(io.BytesIO(base64.b64decode(v["b64"]))).size == (224, 224) for v in f["renders"].values()),
          "4 ảnh render giải mã được")
    check({k: len(v) for k, v in f["codes"].items()} == emoca_runner.CODE_DIMS, "codes 100/50/50/6/128")
    check(set(body["timing_ms"]) == {"detect", "encode", "decode", "total"} and body["timing_ms"]["encode"] == 10.0
          and body["timing_ms"]["decode"] == 20.0, f"timing_ms có 4 khoá, encode/decode lấy từ backend ({body['timing_ms']})")
    lr = runner.info()["landmark_norm_range_seen"]
    check(lr is not None and -1.5 < lr[0] < lr[1] < 1.5, f"info() ghi miền landmark chuẩn hoá đã thấy {lr}")
    check(runner.info()["requests_served"] == 1, "requests_served = 1")

    print("== EmocaRunner: chọn mặt")
    boxes = [[10.0, 10.0, 60.0, 60.0], [200.0, 100.0, 420.0, 330.0], [500.0, 300.0, 600.0, 400.0]]
    backend, runner = make_runner(boxes=boxes)
    body = runner.analyze(Image.new("RGB", (640, 480)), max_faces=1)
    check(len(body["faces"]) == 1 and body["faces"][0]["bbox_original_px"] == [200.0, 100.0, 420.0, 330.0],
          "max_faces=1 → mặt LỚN NHẤT")
    check(any("phát hiện 3 mặt" in w for w in body["faces"][0]["warnings"]), "cảnh báo có nhiều mặt hơn max_faces")
    body = runner.analyze(Image.new("RGB", (640, 480)), max_faces=2)
    check([f["face_index"] for f in body["faces"]] == [0, 1] and body["faces"][1]["bbox_original_px"][0] == 500.0,
          "max_faces=2 → hai mặt theo thứ tự diện tích, face_index 0,1")
    check(valid(body), "hai mặt vẫn đúng schema")

    print("== EmocaRunner: mặt phụ lỗi không làm hỏng cả request (review D5/L2)")
    backend, runner = make_runner(boxes=boxes)
    real_crop = backend.crop

    def crop_failing(bad_left, exc):
        def crop(image, bbox, bbox_type):
            if bbox[0] == bad_left:
                raise exc
            return real_crop(image, bbox, bbox_type)
        return crop

    small = emoca_runner.RunnerError("no_face", "Mặt quá nhỏ để cắt (giả lập).", False, 422)
    backend.crop = crop_failing(500.0, small)  # mặt thứ 2 theo diện tích (100×100)
    body = runner.analyze(Image.new("RGB", (640, 480)), max_faces=3)
    check([f["bbox_original_px"][0] for f in body["faces"]] == [200.0, 10.0] and [f["face_index"] for f in body["faces"]] == [0, 1],
          "mặt phụ lỗi → bỏ qua, các mặt còn lại vẫn trả, face_index liền 0,1")
    check(any("bỏ qua mặt thứ 2" in w and "no_face" in w for w in body["faces"][0]["warnings"]) and valid(body),
          "cảnh báo bỏ qua mặt phụ ghi ở mặt 0, vẫn đúng schema")
    backend.crop = crop_failing(200.0, small)  # mặt LỚN NHẤT lỗi
    ok, _ = raises("no_face", lambda: runner.analyze(Image.new("RGB", (640, 480)), max_faces=3))
    check(ok, "mặt lớn nhất lỗi → cả request lỗi như cũ (no_face)")
    backend.crop = crop_failing(500.0, RuntimeError("lỗi bất ngờ"))
    try:
        runner.analyze(Image.new("RGB", (640, 480)), max_faces=3)
        check(False, "lỗi bất ngờ ở mặt phụ phải lan ra (không bị nuốt)")
    except RuntimeError:
        check(True, "lỗi bất ngờ (không phải RunnerError) ở mặt phụ vẫn lan ra — không che lỗi thật")

    print("== EmocaRunner: nhánh lỗi / cảnh báo")
    _, runner = make_runner(boxes=[])
    ok, e = raises("no_face", lambda: runner.analyze(Image.new("RGB", (64, 64))))
    check(ok and e.status == 422 and not e.retryable, "không phát hiện mặt → no_face 422 (không rơi về ảnh nguyên)")
    codes_nan = {k: np.zeros(n) for k, n in emoca_runner.CODE_DIMS.items()}
    codes_nan["exp"][3] = float("nan")
    _, runner = make_runner(codes=codes_nan)
    ok, e = raises("contract_mismatch", lambda: runner.analyze(Image.new("RGB", (640, 480))))
    check(ok and "NaN" in e.message, "codes có NaN → contract_mismatch (không gửi dữ liệu hỏng)")
    _, runner = make_runner(landmarks_norm=np.zeros((51, 2)))
    ok, _ = raises("contract_mismatch", lambda: runner.analyze(Image.new("RGB", (640, 480))))
    check(ok, "landmark sai hình dạng (51×2) → contract_mismatch")
    partial = {k: np.zeros(n) for k, n in emoca_runner.CODE_DIMS.items() if k != "detail"}
    _, runner = make_runner(codes=partial)
    ok, _ = raises("codes_missing", lambda: runner.analyze(Image.new("RGB", (640, 480))))
    check(ok, "thiếu detailcode → codes_missing")
    _, runner = make_runner(landmarks_norm=np.full((68, 2), 3.0))
    body = runner.analyze(Image.new("RGB", (640, 480)))
    check(any("ngoài miền" in w for w in body["faces"][0]["warnings"]) and valid(body),
          "landmark ngoài [-1,1] → cảnh báo hệ toạ độ (M0-05), không chặn")
    _, runner = make_runner(boxes=[[100.0, 100.0, 120.0, 118.0]])
    body = runner.analyze(Image.new("RGB", (640, 480)))
    check(any("mặt nhỏ" in w for w in body["faces"][0]["warnings"]), "mặt < 32 px → cảnh báo kém tin cậy")

    print("== hàm hình học")
    lm = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    check(np.allclose(emoca_runner.denormalize_landmarks(lm), [[0, 0], [112, 112], [224, 224]]),
          "denormalize: -1→0, 0→112, 1→224 (công thức tensor_vis_landmarks)")
    m = [[2.0, 0.0, 5.0], [0.0, 2.0, -3.0], [0.0, 0.0, 1.0]]
    pts = np.array([[1.0, 2.0], [10.0, 20.0]])
    check(np.allclose(emoca_runner.apply_homography(np.linalg.inv(m), emoca_runner.apply_homography(m, pts)), pts),
          "apply_homography khả nghịch")
    check(emoca_runner.check_body({"faces": [{"codes": {}, "landmarks": {}, "crop_transform": {}, "renders": {}}]}) != [],
          "check_body bắt được face rỗng")


def test_http_layer():
    print("== HTTP: xác thực Bearer, ánh xạ lỗi, health")
    backend, runner = make_runner()
    cfg = dict(mock_server.DEFAULTS, max_image_bytes=2_000_000, max_image_side_px=1000)
    srv = server.build_server(cfg, runner, TOKEN, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        png = img_bytes()
        st, hd, body = post_image(base, png)
        check(st == 200 and valid(body) and hd.get("X-Emoca-Mode") == "real", f"đúng token → 200, schema hợp lệ, X-Emoca-Mode real ({st})")
        for label, tok in (("không token", None), ("sai token", "sai-token")):
            st, _, body = post_image(base, png, token=tok)
            check(st == 401 and valid(body) and body["error"]["code"] == "service_unavailable" and not body["error"]["retryable"],
                  f"{label} → 401, envelope hợp lệ")
        st, _, body = http(base, "GET", "/v1/health", token=None)
        check(st == 401, "GET /v1/health không token → 401 (không lộ thông tin GPU)")
        st, _, body = http(base, "GET", "/v1/health")
        check(st == 200 and body["mode"] == "real" and body["device"] == "fake-gpu" and body["requests_served"] == 1
              and body["contract_version"] == mock_server.CONTRACT_VERSION, f"health: mode real, thông tin runner, requests_served=1 ({body})")
        check(TOKEN not in json.dumps(body), "health không lộ token")

        cases = [
            ("ảnh rác → image_invalid 400", post_image(base, b"khong phai anh"), 400, "image_invalid"),
            ("cạnh dài 1200 > 1000 → image_too_large 413", post_image(base, img_bytes((1200, 300))), 413, "image_too_large"),
            ("thân JSON hỏng → contract_mismatch 400", http(base, "POST", "/v1/analyze", b"{}", "application/json"), 400, "contract_mismatch"),
        ]
        for name, (st, _, body), want_st, want_code in cases:
            check(st == want_st and body.get("error", {}).get("code") == want_code and valid(body), f"{name} (được {st})")

        backend.boxes = []
        st, _, body = post_image(base, png)
        check(st == 422 and body["error"]["code"] == "no_face" and valid(body), "runner no_face → 422 + envelope hợp lệ")
        backend.boxes = [[220.0, 140.0, 420.0, 340.0]]

        real_detect = backend.detect
        backend.detect = lambda image: (_ for _ in ()).throw(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        old_stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            st, _, body = post_image(base, png)
        finally:
            sys.stderr = old_stderr
        check(st == 503 and body["error"]["code"] == "service_unavailable" and body["error"]["retryable"] and "VRAM" in body["error"]["message"]
              and valid(body), "lỗi bất ngờ (CUDA OOM) → 503 service_unavailable, retryable, không sập tiến trình")
        check(backend.released == 1, f"sau OOM server gọi backend.release_memory() đúng 1 lần (được {backend.released})")
        backend.detect = lambda image: (_ for _ in ()).throw(RuntimeError("lỗi khác"))
        old_stderr, sys.stderr = sys.stderr, io.StringIO()
        try:
            st, _, body = post_image(base, png)
        finally:
            sys.stderr = old_stderr
        check(st == 503 and backend.released == 1, "lỗi không phải OOM → 503 nhưng không gọi release_memory()")
        backend.detect = real_detect
        st, _, body = post_image(base, png)
        check(st == 200, "sau lỗi, dịch vụ vẫn phục vụ tiếp")

        print("== HTTP: thân Transfer-Encoding: chunked (review D2)")
        port = srv.server_address[1]
        payload = json.dumps({"image_b64": base64.b64encode(png).decode("ascii")}).encode()
        st, body = post_chunked(port, payload)
        check(st == 200 and valid(body), f"JSON gửi chunked + đúng token → 200, đúng schema (được {st})")
        served = runner.requests_served
        st, body = post_chunked(port, payload, token=None)
        check(st == 401 and body["error"]["code"] == "service_unavailable" and runner.requests_served == served,
              f"chunked không token → 401 (kiểm token trước khi xử lý thân, đọc bỏ thân) (được {st})")
        limit = cfg["max_image_bytes"] * 4 // 3 + 4096
        st, body = post_chunked(port, b"\xff" * (limit + 10_000), ctype="image/jpeg", chunk=65536)
        check(st == 413 and body["error"]["code"] == "image_too_large" and valid(body),
              f"chunked vượt giới hạn thân → 413 image_too_large (được {st})")
        st, body = post_chunked(port, raw=b"zz\r\nabc\r\n0\r\n\r\n")
        check(st == 400 and body["error"]["code"] == "contract_mismatch" and valid(body),
              f"chunked hỏng → 400 contract_mismatch (được {st})")
    finally:
        srv.shutdown()
        srv.server_close()

    print("== HTTP: trần max_faces theo limits.max_faces (review L1)")
    boxes = [[200.0, 100.0, 420.0, 330.0], [500.0, 300.0, 600.0, 400.0], [10.0, 10.0, 60.0, 60.0]]
    for cap, want in ((1, 1), (2, 2)):
        backend, runner = make_runner(boxes=boxes)
        srv, base = serve(runner, max_faces=cap)
        try:
            st, _, body = post_image(base, img_bytes(), options={"max_faces": 3})
            check(st == 200 and len(body["faces"]) == want and valid(body),
                  f"yêu cầu max_faces=3, config max_faces={cap} → trả {want} mặt (được {len(body.get('faces', []))})")
        finally:
            srv.shutdown()
            srv.server_close()

    print("== an toàn cấu hình")
    try:
        server.build_server(dict(mock_server.DEFAULTS, host="0.0.0.0"), runner, None, port=0)
        check(False, "host 0.0.0.0 không token phải bị từ chối")
    except SystemExit as e:
        check("từ chối" in str(e), "nghe ngoài loopback mà không token → SystemExit")
    srv2 = server.build_server(dict(mock_server.DEFAULTS), runner, None, port=0)
    srv2.server_close()
    check(True, "loopback không token vẫn chạy (dev cục bộ)")


def test_serialization_and_disk():
    print("== GPU tuần tự + không ghi đĩa")
    backend, runner = make_runner(sleep=0.15)
    cfg = dict(mock_server.DEFAULTS)
    srv = server.build_server(cfg, runner, TOKEN, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    png = img_bytes()

    def snapshot():
        return set((ROOT / "backend" / "services").rglob("*")) - set((ROOT / "backend" / "services").rglob("__pycache__/*")) | set(Path.cwd().iterdir())
    try:
        before = snapshot()
        results = []
        ts = [threading.Thread(target=lambda: results.append(post_image(base, png)[0])) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        after = snapshot()
        check(results == [200] * 4, f"4 request đồng thời đều 200 ({results})")
        check(backend.max_active == 1, f"tối đa 1 ảnh qua backend cùng lúc (đo được {backend.max_active})")
        check(after == before, f"không tạo file mới ({len(after - before)} file mới)")
    finally:
        srv.shutdown()
        srv.server_close()


def test_static():
    print("== tĩnh: cú pháp Python 3.8, không nạp torch")
    for name in ("server.py", "emoca_runner.py", "mock_server.py"):
        src = (SVC / name).read_text(encoding="utf-8")
        try:
            ast.parse(src, filename=name, feature_version=(3, 8))
            ok = True
        except SyntaxError as e:
            ok = False
            print("       ", e)
        check(ok, f"{name} là cú pháp hợp lệ của Python 3.8")
    check("torch" not in sys.modules, "nạp emoca_runner/server không kéo theo torch (chỉ EmocaBackend.load() mới nạp)")
    try:
        emoca_runner.EmocaBackend("x", mode="coarse")
        check(False, "mode coarse phải bị từ chối")
    except ValueError:
        check(True, "EmocaBackend từ chối mode coarse (contract cần codes.detail)")
    b = emoca_runner.EmocaBackend("khong-co", mode="detail")
    ok, e = raises("model_unavailable", b.load)
    check(ok and not e.retryable, f"load() thiếu torch/EMOCA → model_unavailable rõ ràng ({(e.message[:60] + '…') if e else ''})")
    check(b.info()["loaded"] is False, "info() trước khi nạp không lỗi")
    src = (SVC / "emoca_runner.py").read_text(encoding="utf-8")
    check("open(" not in src and ".save(" not in src.replace("Image.fromarray(a).save(buf", ""),
          "emoca_runner không ghi ảnh ra đĩa (chỉ nén PNG vào bộ nhớ)")


def fake_torch(mem_get_info="ok", cudart="ok", reserved="ok"):
    """torch giả cho EmocaBackend.info(): chỉ các hàm đo bộ nhớ. "ok" = trả số, "missing" = không có, "raise" = ném lỗi."""
    MiB = 2**20

    def boom(*_a, **_k):
        raise RuntimeError("CUDA error: giả lập")

    cuda = types.SimpleNamespace(max_memory_allocated=lambda: 900 * MiB)
    if reserved == "ok":
        cuda.memory_reserved, cuda.max_memory_reserved = (lambda: 1024 * MiB), (lambda: 1536.44 * MiB)
    elif reserved == "raise":
        cuda.memory_reserved = cuda.max_memory_reserved = boom
    if mem_get_info == "ok":
        cuda.mem_get_info = lambda: (3292 * MiB, 4095.5 * MiB)
    elif mem_get_info == "raise":
        cuda.mem_get_info = boom
    if cudart == "ok":  # binding kiểu (mã lỗi, free, total)
        cuda.cudart = lambda: types.SimpleNamespace(cudaMemGetInfo=lambda dev: (0, 2000 * MiB, 4095 * MiB))
    elif cudart == "raise":
        cuda.cudart = boom
    return types.SimpleNamespace(__version__="1.11.0+fake", version=types.SimpleNamespace(cuda="11.5"), cuda=cuda,
                                 backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=True)))


def loaded_backend(torch_mod):
    b = emoca_runner.EmocaBackend("khong-co", mode="detail", upstream_commit="e0be0db+patches")
    b._loaded, b._torch, b._device_name, b._vram_after_load_mb, b.load_seconds = True, torch_mod, "fake-gpu", 800.04, 12.3
    return b


def test_vram_fields():
    print("== health: VRAM reserved + cudaMemGetInfo (EmocaBackend.info() với torch giả, finding H1)")
    gpu_keys = ("gpu_free_mb", "gpu_total_mb")
    res_keys = ("vram_reserved_mb", "vram_reserved_peak_mb")
    info = loaded_backend(fake_torch()).info()
    check(info["gpu_free_mb"] == 3292.0 and info["gpu_total_mb"] == 4095.5
          and info["vram_reserved_mb"] == 1024.0 and info["vram_reserved_peak_mb"] == 1536.4
          and info["vram_after_load_mb"] == 800.0 and info["vram_peak_mb"] == 900.0,
          f"đủ trường, MiB làm tròn 1 chữ số ({ {k: info.get(k) for k in gpu_keys + res_keys} })")
    info = loaded_backend(fake_torch(mem_get_info="missing")).info()
    check((info.get("gpu_free_mb"), info.get("gpu_total_mb")) == (2000.0, 4095.0),
          f"không có mem_get_info → dùng cudart().cudaMemGetInfo(0) ({info.get('gpu_free_mb')}/{info.get('gpu_total_mb')})")
    info = loaded_backend(fake_torch(mem_get_info="raise")).info()
    check(info.get("gpu_free_mb") == 2000.0, "mem_get_info ném lỗi → thử cudart")
    info = loaded_backend(fake_torch(mem_get_info="raise", cudart="raise", reserved="raise")).info()
    check(not any(k in info for k in gpu_keys + res_keys) and info["vram_peak_mb"] == 900.0 and info["loaded"] is True,
          "mọi hàm đo phụ ném lỗi → bỏ trường, info() vẫn trả phần còn lại")
    info = loaded_backend(fake_torch(mem_get_info="missing", cudart="missing", reserved="missing")).info()
    check(not any(k in info for k in gpu_keys + res_keys), "torch thiếu các hàm đo → bỏ trường, không lỗi")

    for label, tm in (("đủ trường", fake_torch()),
                      ("thiếu trường", fake_torch(mem_get_info="raise", cudart="raise", reserved="raise"))):
        runner = emoca_runner.EmocaRunner(loaded_backend(tm), mock_server.CONTRACT_VERSION, min_face_px=32)
        srv, base = serve(runner)
        try:
            st, _, body = http(base, "GET", "/v1/health")
            has = all(k in body for k in gpu_keys + res_keys)
            check(st == 200 and body["mode"] == "real" and body["device"] == "fake-gpu"
                  and has == (label == "đủ trường"), f"GET /v1/health {label} → 200 ({st}, có trường GPU: {has})")
        finally:
            srv.shutdown()
            srv.server_close()


def test_cudnn_switch():
    print("== cuDNN: EMOCA_CUDNN auto|0|1 + tệp probe (dự phòng khi cuDNN lỗi/tự huỷ trên WSL2)")
    wa = emoca_runner.wsl_ldpath_active
    check(wa({"LD_LIBRARY_PATH": "/usr/lib/wsl/lib:/opt/x"}) and wa({"LD_LIBRARY_PATH": "/opt/x:/usr/lib/wsl/lib/"})
          and not wa({"LD_LIBRARY_PATH": "/opt/x:/usr/lib/wsl/lib2"}) and not wa({}) and not wa({"LD_LIBRARY_PATH": ""})
          and wa({"LD_LIBRARY_PATH": "/khac/lib", "EMOCA_WSL_LIB_DIR": "/khac/lib"}),
          "wsl_ldpath_active: nhận /usr/lib/wsl/lib (hoặc EMOCA_WSL_LIB_DIR) ở bất kỳ vị trí; không nhầm tiền tố")
    saved_ld = {k: os.environ.get(k) for k in ("LD_LIBRARY_PATH", "EMOCA_WSL_LIB_DIR")}
    try:
        os.environ.pop("EMOCA_WSL_LIB_DIR", None)
        for ld, want in (("/usr/lib/wsl/lib:/opt/cuda", True), ("/opt/cuda", False)):
            os.environ["LD_LIBRARY_PATH"] = ld
            got = {"trước nạp": emoca_runner.EmocaBackend("x", mode="detail").info().get("ld_library_path_wsl"),
                   "sau nạp": loaded_backend(fake_torch()).info().get("ld_library_path_wsl")}
            check(got == {"trước nạp": want, "sau nạp": want}, f"health.ld_library_path_wsl = {want} khi LD_LIBRARY_PATH={ld} ({got})")
    finally:
        for k, v in saved_ld.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rc = emoca_runner.resolve_cudnn
    check(rc({"EMOCA_CUDNN": "0"})[0] is False and rc({"EMOCA_CUDNN": " 0 "})[0] is False, "EMOCA_CUDNN=0 → tắt")
    check(rc({"EMOCA_CUDNN": "1"})[0] is True, "EMOCA_CUDNN=1 → bật")
    check(rc({})[0] is None and rc({"EMOCA_CUDNN": "auto"})[0] is None,
          "auto (mặc định), không có EMOCA_CUDNN_PROBE → giữ mặc định torch")
    for bad in ("yes", "off", "2"):
        try:
            rc({"EMOCA_CUDNN": bad})
            check(False, f"EMOCA_CUDNN={bad} phải bị từ chối")
        except ValueError:
            check(True, f"EMOCA_CUDNN={bad} → ValueError (không đoán)")
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "cudnn_probe.txt"
        env = {"EMOCA_CUDNN": "auto", "EMOCA_CUDNN_PROBE": str(probe)}
        check(rc(env)[0] is False, f"auto + probe không đọc được → tắt cho an toàn ({rc(env)[1]})")
        probe.write_text("# phep thu\ncudnn_ok=0\nrc_cudnn_on=134\n", encoding="utf-8")
        check(rc(env)[0] is False and "hỏng" in rc(env)[1], "auto + cudnn_ok=0 → tắt")
        probe.write_text("cudnn_ok=1\n", encoding="utf-8")
        check(rc(env)[0] is True, "auto + cudnn_ok=1 → bật")
        probe.write_text("rc_cudnn_on=0\n", encoding="utf-8")
        check(rc(env)[0] is False, "auto + probe thiếu cudnn_ok → tắt cho an toàn")
        check(rc({"EMOCA_CUDNN": "1", "EMOCA_CUDNN_PROBE": str(probe)})[0] is True, "đặt tay 1 thắng probe")

        # load() với torch/gdl GIẢ: cờ cuDNN phải được đặt TRƯỚC khi dựng EMOCA và FAN; health báo trạng thái thật.
        tm = fake_torch()
        seen = {}
        tm.cuda.is_available = lambda: True
        tm.cuda.synchronize = lambda: None
        tm.cuda.memory_allocated = lambda: 700 * 2**20
        tm.cuda.reset_peak_memory_stats = lambda: None
        tm.cuda.get_device_name = lambda i: "fake-gpu"

        class FakeEmoca:
            def cuda(self):
                return self

            def eval(self):
                return self

        def load_model(models_dir, name, mode):
            seen["cudnn_at_load_model"] = tm.backends.cudnn.enabled
            return FakeEmoca(), {}

        class FAN:
            def __init__(self):
                seen["cudnn_at_fan"] = tm.backends.cudnn.enabled

        mods = {"torch": tm}
        for name, attrs in {
            "gdl": {}, "gdl.datasets": {}, "gdl.utils": {}, "gdl_apps": {}, "gdl_apps.EMOCA": {}, "gdl_apps.EMOCA.utils": {},
            "gdl.datasets.ImageDatasetHelpers": {"bbox2point": None}, "gdl.utils.FaceDetector": {"FAN": FAN},
            "gdl.utils.lightning_logging": {"_fix_image": None},
            "gdl_apps.EMOCA.utils.io": {"decode": None, "torch_img_to_np": None},
            "gdl_apps.EMOCA.utils.load": {"load_model": load_model},
            "skimage": {}, "skimage.transform": {"estimate_transform": None, "warp": None},
        }.items():
            m = types.ModuleType(name)
            m.__dict__.update(attrs)
            mods[name] = m
        saved_mods = {k: sys.modules.get(k) for k in mods}
        saved_env = {k: os.environ.get(k) for k in ("EMOCA_CUDNN", "EMOCA_CUDNN_PROBE")}
        try:
            sys.modules.update(mods)
            for cudnn_env, want in (("0", False), ("1", True)):
                tm.backends.cudnn.enabled = True if cudnn_env == "0" else False  # bắt đầu NGƯỢC để thấy load() đặt lại
                os.environ["EMOCA_CUDNN"] = cudnn_env
                os.environ.pop("EMOCA_CUDNN_PROBE", None)
                seen.clear()
                b = emoca_runner.EmocaBackend("fake-models", mode="detail", upstream_commit="e0be0db+patches")
                b.load()
                info = b.info()
                check(seen == {"cudnn_at_load_model": want, "cudnn_at_fan": want},
                      f"EMOCA_CUDNN={cudnn_env}: cờ = {want} trước load_model và FAN ({seen})")
                check(info["cudnn_enabled"] is want and info["cudnn_reason"] == f"EMOCA_CUDNN={cudnn_env}",
                      f"info(): cudnn_enabled={info['cudnn_enabled']}, cudnn_reason={info['cudnn_reason']!r}")
            os.environ["EMOCA_CUDNN"] = "auto"
            os.environ["EMOCA_CUDNN_PROBE"] = str(probe)
            probe.write_text("cudnn_ok=0\n", encoding="utf-8")
            tm.backends.cudnn.enabled = True
            b = emoca_runner.EmocaBackend("fake-models", mode="detail", upstream_commit="e0be0db+patches")
            b.load()
            check(b.info()["cudnn_enabled"] is False, "auto + probe cudnn_ok=0 → load() tắt cuDNN, health báo False")
            os.environ["EMOCA_CUDNN"] = "sai"
            ok, e = raises("model_unavailable", emoca_runner.EmocaBackend("fake-models", mode="detail").load)
            check(ok and "EMOCA_CUDNN" in e.message, "EMOCA_CUDNN sai → model_unavailable rõ ràng, không nạp")
            runner = emoca_runner.EmocaRunner(b, mock_server.CONTRACT_VERSION, min_face_px=32)
            srv, base = serve(runner)
            try:
                st, _, body = http(base, "GET", "/v1/health")
                check(st == 200 and body.get("cudnn_enabled") is False and "cudnn_reason" in body,
                      f"GET /v1/health có cudnn_enabled/cudnn_reason ({st}, {body.get('cudnn_enabled')})")
            finally:
                srv.shutdown()
                srv.server_close()
        finally:
            for k, v in saved_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def test_cudnn_runtime_fallback():
    print("== cuDNN lỗi chập chờn lúc suy luận ⇒ tắt cuDNN, thử lại đúng 1 lần (torch giả)")
    flaky = "Unable to find a valid cuDNN algorithm to run convolution"

    def seq(*outcomes):
        """Hàm giả: mỗi lần gọi lấy kết quả kế tiếp; Exception thì ném. Ghi trạng thái cờ cuDNN lúc được gọi."""
        calls = []

        def fn(*_a, **_k):
            calls.append(b.backends_flag())
            o = outcomes[len(calls) - 1]
            if isinstance(o, Exception):
                raise o
            return o
        return fn, calls

    tm = fake_torch()
    b = loaded_backend(tm)
    b._cudnn_reason = "EMOCA_CUDNN=1"
    b.backends_flag = lambda: tm.backends.cudnn.enabled
    fan_ok = ([[10.0, 10.0, 90.0, 90.0]], "kpt68", None)
    b._fan = types.SimpleNamespace()
    b._fan.run, calls = seq(RuntimeError(flaky), fan_ok)
    got = b.detect(np.zeros((8, 8, 3), np.uint8))
    info = b.info()
    check(got[0] == fan_ok[0] and calls == [True, False] and tm.backends.cudnn.enabled is False,
          f"FAN: lỗi cuDNN khi đang bật ⇒ tắt rồi thử lại 1 lần, thành công (cờ lúc gọi {calls})")
    check(info["cudnn_fallbacks"] == 1 and info["cudnn_enabled"] is False and "TẮT lúc chạy" in info["cudnn_reason"]
          and flaky in info["cudnn_reason"], f"health: cudnn_fallbacks=1, cudnn_enabled=False, lý do ghi lỗi ({info['cudnn_reason']!r})")
    # cuDNN đã tắt ⇒ lỗi cuDNN sau đó ném thẳng (không lặp, không đếm thêm).
    b._infer_impl, calls = seq(RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"))
    try:
        b.infer(np.zeros((224, 224, 3)))
        check(False, "cuDNN đã tắt: lỗi phải được ném ra")
    except RuntimeError:
        check(calls == [False] and b.cudnn_fallbacks == 1, "cuDNN đã tắt ⇒ lỗi cuDNN ném thẳng, chỉ 1 lần gọi, không đếm thêm")

    tm = fake_torch()
    b = loaded_backend(tm)
    b._cudnn_reason = "EMOCA_CUDNN=1"
    b.backends_flag = lambda: tm.backends.cudnn.enabled
    b._infer_impl, calls = seq(RuntimeError(flaky), {"codes": {}})
    check(b.infer(np.zeros((224, 224, 3))) == {"codes": {}} and calls == [True, False] and b.cudnn_fallbacks == 1,
          "encode/decode: lỗi cuDNN ⇒ tắt, thử lại 1 lần, thành công")
    tm = fake_torch()
    b = loaded_backend(tm)
    b.backends_flag = lambda: tm.backends.cudnn.enabled
    b._infer_impl, calls = seq(RuntimeError(flaky), RuntimeError(flaky))
    try:
        b.infer(np.zeros((224, 224, 3)))
        check(False, "thử lại vẫn lỗi ⇒ phải ném")
    except RuntimeError:
        check(calls == [True, False] and b.cudnn_fallbacks == 1, "thử lại vẫn lỗi ⇒ ném lỗi, đúng 2 lần gọi (không lặp vô hạn)")
    tm = fake_torch()
    b = loaded_backend(tm)
    b.backends_flag = lambda: tm.backends.cudnn.enabled
    for exc in (RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"), ValueError("cudnn nhưng không phải RuntimeError")):
        b._infer_impl, calls = seq(exc)
        try:
            b.infer(np.zeros((224, 224, 3)))
        except type(exc):
            pass
        check(calls == [True] and tm.backends.cudnn.enabled is True and b.cudnn_fallbacks == 0,
              f"lỗi khác ({type(exc).__name__}: {str(exc)[:30]}…) ⇒ không đổi gì, ném thẳng")


def test_git_commit_label():
    print("== nhãn upstream_commit: '+patches' chỉ khi gdl/ hoặc gdl_apps/ đổi (review L3)")
    if shutil.which("git") is None:
        print("  SKIP không có git — bỏ qua")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "gdl").mkdir()
        (root / "gdl_apps").mkdir()
        (root / "gdl" / "__init__.py").write_text("x = 1\n", encoding="utf-8")
        (root / "gdl_apps" / "__init__.py").write_text("", encoding="utf-8")
        (root / "requirements38.txt").write_text("numpy\n", encoding="utf-8")

        def git(*args):
            subprocess.run(["git", "-C", str(root), "-c", "user.email=t@example.invalid", "-c", "user.name=test",
                            "-c", "core.autocrlf=false", *args], check=True, capture_output=True)

        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "init")
        fake = types.ModuleType("gdl")
        fake.__file__ = str(root / "gdl" / "__init__.py")
        saved = sys.modules.get("gdl")
        sys.modules["gdl"] = fake
        try:
            clean = emoca_runner.EmocaBackend._git_commit()
            (root / "requirements38.txt").write_text("numpy\n# notebook ô 16 sửa\n", encoding="utf-8")
            req_only = emoca_runner.EmocaBackend._git_commit()
            (root / "gdl" / "__init__.py").write_text("x = 2  # như một bản vá\n", encoding="utf-8")
            patched = emoca_runner.EmocaBackend._git_commit()
        finally:
            if saved is None:
                sys.modules.pop("gdl", None)
            else:
                sys.modules["gdl"] = saved
    check(len(clean) == 7 and "+" not in clean, f"cây sạch → mã commit 7 ký tự, không hậu tố ({clean})")
    check(req_only == clean, f"chỉ sửa requirements38.txt → KHÔNG có '+patches' ({req_only})")
    check(patched == clean + "+patches", f"sửa gdl/ → '+patches' ({patched})")


def test_detect_downscale():
    print("== dò mặt trên bản thu nhỏ (cạnh dài ≤ detect_max_side_px), khung đổi về original_px, crop trên ảnh gốc")
    d = emoca_runner.downscale_for_detect
    img = np.zeros((2944, 2208, 3), np.uint8)
    small, sx, sy = d(img, 1024)
    check(small.shape[:2] == (1024, 768) and abs(sx - 768 / 2208) < 1e-12 and abs(sy - 1024 / 2944) < 1e-12,
          f"2208×2944 → 768×1024, sx/sy = rộng_mới/rộng_gốc ({small.shape[:2]}, {sx:.4f}, {sy:.4f})")
    same, sx1, sy1 = d(np.zeros((600, 450, 3), np.uint8), 1024)
    check(same.shape == (600, 450, 3) and (sx1, sy1) == (1.0, 1.0), "ảnh nhỏ hơn ngưỡng ⇒ giữ nguyên, hệ số 1")
    check(d(img, 0)[1:] == (1.0, 1.0), "detect_max_side_px = 0 ⇒ không thu")

    truth_box = [600.0, 1500.0, 1700.0, 2600.0]  # original_px

    class ScaledDetect(FakeBackend):
        def detect(self, image):
            self.seen_shape = image.shape
            k = image.shape[1] / 2208.0
            return [[v * k for v in truth_box]], "kpt68", None

    b = ScaledDetect()
    runner = emoca_runner.EmocaRunner(b, mock_server.CONTRACT_VERSION, min_face_px=32, detect_max_side_px=1024)
    out = runner.analyze(Image.fromarray(np.full((2944, 2208, 3), 128, np.uint8)))
    got = out["faces"][0]["bbox_original_px"]
    check(b.seen_shape[:2] == (1024, 768), f"backend.detect nhận bản 768×1024 ({b.seen_shape[:2]})")
    check(max(abs(g - t) for g, t in zip(got, truth_box)) < 1e-6 and b._bbox is not None
          and max(abs(g - t) for g, t in zip(b._bbox, truth_box)) < 1e-6,
          f"bbox_original_px và khung đưa vào crop = khung gốc ({[round(v, 3) for v in got]})")
    check(out["image"] == {"width": 2208, "height": 2944}, "image trong response = kích thước GỐC (hệ original_px)")
    b2 = ScaledDetect()
    emoca_runner.EmocaRunner(b2, mock_server.CONTRACT_VERSION, 32, detect_max_side_px=0).analyze(
        Image.fromarray(np.full((2944, 2208, 3), 128, np.uint8)))
    check(b2.seen_shape[:2] == (2944, 2208), "detect_max_side_px = 0 ⇒ detect trên ảnh gốc")
    cfg = mock_server.load_config(ROOT / "backend" / "configs" / "services.yaml") if hasattr(mock_server, "load_config") else None
    check(cfg is None or cfg.get("detect_max_side_px") == 1024, f"configs/services.yaml limits.detect_max_side_px = 1024 ({cfg and cfg.get('detect_max_side_px')})")


def test_warmup():
    print("== làm nóng sau khi nạp (EMOCA_WARMUP): ảnh tổng hợp, không ghi đĩa, health báo warmed/warmup_seconds")
    rw = emoca_runner.resolve_warmup
    check(rw({}) == (True, [224, 512, 1024]) and rw({"EMOCA_WARMUP": "1"})[0] is True, "mặc định bật, cạnh 224/512/1024")
    check(all(rw({"EMOCA_WARMUP": v}) == (False, []) for v in ("0", "false", "OFF", " no ")), "EMOCA_WARMUP=0/false/off/no → tắt")
    check(rw({"EMOCA_WARMUP_SIZES": "256, 800"}) == (True, [256, 800]), "EMOCA_WARMUP_SIZES chọn cạnh")
    for env in ({"EMOCA_WARMUP": "maybe"}, {"EMOCA_WARMUP_SIZES": "a,b"}, {"EMOCA_WARMUP_SIZES": "32"},
                {"EMOCA_WARMUP_SIZES": "224,9000"}, {"EMOCA_WARMUP_SIZES": ","}):
        try:
            rw(env)
            check(False, f"{env} phải bị từ chối")
        except ValueError:
            check(True, f"{env} → ValueError (không đoán)")
    img = emoca_runner.synthetic_image(64)
    check(img.shape == (64, 64, 3) and img.dtype == np.uint8 and (emoca_runner.synthetic_image(64) == img).all()
          and float(img.std()) > 10, "synthetic_image: RGB uint8 đúng cạnh, tất định, không phải ảnh phẳng")

    def warm_backend(fan_run, infer_impl=None):
        tm = fake_torch()
        tm.cuda.synchronize = lambda: None
        tm.cuda.empty_cache = lambda: calls.append("empty_cache")
        b = loaded_backend(tm)
        b._fan = types.SimpleNamespace(run=fan_run)
        b._infer_impl = infer_impl or (lambda crop: calls.append(("infer", crop.shape, str(crop.dtype),
                                                                  float(crop.min()), float(crop.max()))) or {"codes": {}})
        b._cudnn_reason = "EMOCA_CUDNN=1"
        return b

    calls = []

    def fan_ok(image, with_landmarks=False, detected_faces=None):
        calls.append(("fan", image.shape, None if detected_faces is None else [round(v) for v in detected_faces[0]]))
        return [], "kpt68", []

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            b = warm_backend(fan_ok)
            b.warmup([224, 512])
            written = list(Path(tmp).rglob("*"))
        finally:
            os.chdir(cwd)
    fans = [c for c in calls if c[0] == "fan"]
    check(fans == [("fan", (224, 224, 3), None), ("fan", (224, 224, 3), [56, 45, 168, 179]),
                   ("fan", (512, 512, 3), None), ("fan", (512, 512, 3), [128, 102, 384, 410])],
          f"mỗi cạnh: FAN dò mặt + FAN 68 điểm trên khung giả ({fans})")
    inf = [c for c in calls if c[0] == "infer"]
    check(len(inf) == 1 and inf[0][1] == (224, 224, 3) and inf[0][2] == "float32" and 0.0 <= inf[0][3] and inf[0][4] <= 1.0,
          f"encode/decode đúng 1 lần trên crop 224 float [0,1] ({inf})")
    info = b.info()
    check(b.warmed is True and info["warmed"] is True and info["warmup_sizes"] == [224, 512]
          and isinstance(info["warmup_seconds"], float) and "warmup_error" not in info,
          f"health: warmed=True, warmup_seconds, warmup_sizes ({ {k: info.get(k) for k in ('warmed', 'warmup_seconds', 'warmup_sizes')} })")
    check(not written, f"không ghi tệp nào ra đĩa trong lúc làm nóng ({written[:3]})")

    calls = []

    def fan_oom(image, with_landmarks=False, detected_faces=None):
        if image.shape[0] == 1024:
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
        return [], "kpt68", []

    b = warm_backend(fan_oom)
    b.warmup([224, 1024])
    info = b.info()
    check(b.warmed is False and "out of memory" in info.get("warmup_error", "") and info["warmed"] is False
          and "empty_cache" in calls, f"lỗi lúc làm nóng: không ném ra, warmed=False, health có warmup_error, đã empty_cache ({info.get('warmup_error')!r})")

    # load(): EMOCA_WARMUP=0 ⇒ không làm nóng; mặc định ⇒ làm nóng đúng cạnh; giá trị sai ⇒ model_unavailable.
    tm = fake_torch()
    for attr, val in (("is_available", lambda: True), ("synchronize", lambda: None), ("reset_peak_memory_stats", lambda: None),
                      ("memory_allocated", lambda: 700 * 2**20), ("get_device_name", lambda i: "fake-gpu"),
                      ("empty_cache", lambda: None)):
        setattr(tm.cuda, attr, val)

    class FakeEmoca:
        def cuda(self):
            return self

        def eval(self):
            return self

    mods = {"torch": tm}
    for name, attrs in {
        "gdl": {}, "gdl.datasets": {}, "gdl.utils": {}, "gdl_apps": {}, "gdl_apps.EMOCA": {}, "gdl_apps.EMOCA.utils": {},
        "gdl.datasets.ImageDatasetHelpers": {"bbox2point": None},
        "gdl.utils.FaceDetector": {"FAN": lambda: types.SimpleNamespace(run=fan_ok)},
        "gdl.utils.lightning_logging": {"_fix_image": None},
        "gdl_apps.EMOCA.utils.io": {"decode": None, "torch_img_to_np": None},
        "gdl_apps.EMOCA.utils.load": {"load_model": lambda d, n, m: (FakeEmoca(), {})},
        "skimage": {}, "skimage.transform": {"estimate_transform": None, "warp": None},
    }.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        mods[name] = m
    keys = ("EMOCA_CUDNN", "EMOCA_CUDNN_PROBE", "EMOCA_WARMUP", "EMOCA_WARMUP_SIZES")
    saved_mods = {k: sys.modules.get(k) for k in mods}
    saved_env = {k: os.environ.get(k) for k in keys}
    try:
        sys.modules.update(mods)
        os.environ["EMOCA_CUDNN"] = "1"
        os.environ.pop("EMOCA_CUDNN_PROBE", None)
        for env, want_warmed, want_sizes in (({"EMOCA_WARMUP": "0"}, False, []), ({"EMOCA_WARMUP_SIZES": "256"}, True, [256])):
            for k in ("EMOCA_WARMUP", "EMOCA_WARMUP_SIZES"):
                os.environ.pop(k, None)
            os.environ.update(env)
            b = emoca_runner.EmocaBackend("fake-models", mode="detail", upstream_commit="e0be0db+patches")
            b._infer_impl = lambda crop: {"codes": {}}
            b.load()
            info = b.info()
            check(info["loaded"] and info["warmed"] is want_warmed and info["warmup_sizes"] == want_sizes
                  and (info["warmup_seconds"] is None) == (not want_warmed),
                  f"load() với {env}: warmed={info['warmed']}, sizes={info['warmup_sizes']}, seconds={info['warmup_seconds']}")
        os.environ["EMOCA_WARMUP"] = "sai"
        ok, e = raises("model_unavailable", emoca_runner.EmocaBackend("fake-models", mode="detail").load)
        check(ok and "EMOCA_WARMUP" in e.message, "EMOCA_WARMUP sai → model_unavailable rõ ràng, không nạp")
    finally:
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main() -> int:
    test_runner_logic()
    test_http_layer()
    test_serialization_and_disk()
    test_vram_fields()
    test_cudnn_switch()
    test_cudnn_runtime_fallback()
    test_warmup()
    test_detect_downscale()
    test_git_commit_label()
    test_static()
    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
