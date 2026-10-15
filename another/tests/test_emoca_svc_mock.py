"""Kiểm M1-02: services/emoca_svc/mock_server.py qua HTTP thật (cổng tạm), không GPU.

Chạy:  python tests/test_emoca_svc_mock.py
AC M1-02: trả dữ liệu mẫu đúng schema (contracts/emoca_analyze.schema.json), chạy không cần GPU.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "services" / "emoca_svc"))
import mock_server  # noqa: E402

SCHEMA = json.loads((ROOT / "backend" / "contracts" / "emoca_analyze.schema.json").read_text(encoding="utf-8"))
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def img_bytes(fmt: str, size=(640, 480), color=(180, 140, 120), exif_orientation: int | None = None) -> bytes:
    im = Image.new("RGB", size, color)
    buf = io.BytesIO()
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        im.save(buf, format=fmt, exif=exif)
    else:
        im.save(buf, format=fmt)
    return buf.getvalue()


def request(base: str, method: str, path: str, body: bytes | None = None, ctype: str | None = None):
    req = urllib.request.Request(base + path, data=body, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read().decode("utf-8"))


def post_json(base, data: bytes, **options):
    payload = {"image_b64": base64.b64encode(data).decode("ascii")}
    if options:
        payload["options"] = options
    return request(base, "POST", "/v1/analyze", json.dumps(payload).encode(), "application/json")


def post_chunked(port: int, payload: bytes | None = None, ctype: str = "application/json", chunk: int = 7000,
                 raw: bytes | None = None, headers: dict | None = None):
    """POST với `Transfer-Encoding: chunked` (như proxy/tunnel có thể chuyển tiếp). `raw` = gửi nguyên văn byte đã
    mã hoá chunk (để thử thân hỏng). Trả (status, body_json)."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        conn.putrequest("POST", "/v1/analyze")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.putheader("Content-Type", ctype)
        for k, v in (headers or {}).items():
            conn.putheader(k, v)
        conn.endheaders()
        if raw is not None:
            conn.send(raw)
        else:
            for i in range(0, len(payload), chunk):
                part = payload[i:i + chunk]
                conn.send(b"%x;ext=1\r\n" % len(part) + part + b"\r\n" if i == 0 else b"%x\r\n" % len(part) + part + b"\r\n")
            conn.send(b"0\r\n\r\n")
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def valid(body) -> bool:
    try:
        jsonschema.validate(body, SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print("       schema:", e.message[:200])
        return False


def main() -> int:
    cfg = dict(mock_server.DEFAULTS, max_image_bytes=2_000_000)
    srv = mock_server.make_server(cfg, port=0)  # cổng tạm do hệ điều hành cấp
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        print("== health")
        st, hd, body = request(base, "GET", "/v1/health")
        check(st == 200 and body.get("mode") == "mock" and body.get("contract_version") == mock_server.CONTRACT_VERSION,
              f"GET /v1/health → 200, mode mock ({st}, {body})")

        print("== thành công")
        jpg = img_bytes("JPEG")
        st, hd, body = post_json(base, jpg)
        check(st == 200, f"JSON image_b64 JPEG → 200 (được {st})")
        check(valid(body), "đúng emoca_analyze.schema.json (success)")
        check(hd.get("X-Emoca-Mode") == "mock", "header X-Emoca-Mode: mock")
        check(body["image"] == {"width": 640, "height": 480}, "image.width/height = kích thước thật")
        f = body["faces"][0]
        check(any("mock" in w for w in f["warnings"]), "face.warnings ghi rõ là mock")
        # original_px → crop224 bằng matrix_3x3 phải ra đúng landmarks.crop224 (bất biến §7).
        M = f["crop_transform"]["matrix_3x3"]
        err = max(abs(M[0][0] * x + M[0][1] * y + M[0][2] - u) + abs(M[1][0] * x + M[1][1] * y + M[1][2] - v)
                  for (x, y), (u, v) in zip(f["landmarks"]["original_px"], f["landmarks"]["crop224"]))
        check(err < 0.05, f"matrix_3x3 · original_px ≈ crop224 (sai số tối đa {err:.4f} px)")
        crop = Image.open(io.BytesIO(base64.b64decode(f["crop_image"]["b64"])))
        check(crop.size == (224, 224), f"crop_image 224×224 (được {crop.size})")
        check({k: len(v) for k, v in f["codes"].items()} == mock_server.CODE_DIMS, "codes đúng 100/50/50/6/128")
        check(any(x != 0 for v in f["codes"].values() for x in v), "codes không phải toàn số 0")

        st2, _, body2 = post_json(base, jpg)
        check(body2["faces"][0]["codes"] == f["codes"], "cùng ảnh → cùng codes (tất định)")
        st3, _, body3 = post_json(base, img_bytes("JPEG", color=(20, 90, 200)))
        check(body3["faces"][0]["codes"] != f["codes"], "ảnh khác → codes khác")

        st, _, body = request(base, "POST", "/v1/analyze", img_bytes("PNG", (300, 400)), "image/png")
        check(st == 200 and valid(body) and body["image"] == {"width": 300, "height": 400},
              "bytes PNG thô (image/png) → 200, đúng schema")
        st, _, body = post_json(base, img_bytes("WEBP"))
        check(st == 200 and valid(body), "WEBP → 200, đúng schema")
        st, _, body = post_json(base, img_bytes("JPEG", (64, 48), exif_orientation=6))
        check(st == 200 and body["image"] == {"width": 48, "height": 64}, f"xoay EXIF 6: 64×48 → 48×64 ({body.get('image')})")

        print("== lỗi (error envelope, đúng schema)")
        cases = [
            ("ảnh 20×20 → no_face 422", post_json(base, img_bytes("PNG", (20, 20))), 422, "no_face"),
            ("rác → image_invalid 400", post_json(base, b"khong phai anh"), 400, "image_invalid"),
            ("rỗng → image_invalid 400", request(base, "POST", "/v1/analyze", b"", "image/jpeg"), 400, "image_invalid"),
            ("JPEG cụt → image_invalid 400", post_json(base, jpg[: len(jpg) // 3]), 400, "image_invalid"),
            ("GIF → image_invalid 400", post_json(base, img_bytes("GIF")), 400, "image_invalid"),
            ("quá cỡ → image_too_large 413", request(base, "POST", "/v1/analyze", b"\xff" * 2_000_001, "image/jpeg"),
             413, "image_too_large"),
            ("JSON thiếu image_b64 → contract_mismatch", request(base, "POST", "/v1/analyze", b"{}", "application/json"),
             400, "contract_mismatch"),
            ("base64 hỏng → contract_mismatch", request(base, "POST", "/v1/analyze", b'{"image_b64":"@@@"}',
                                                         "application/json"), 400, "contract_mismatch"),
            ("max_faces=0 → contract_mismatch", post_json(base, jpg, max_faces=0), 400, "contract_mismatch"),
            ("Content-Type lạ → contract_mismatch", request(base, "POST", "/v1/analyze", jpg, "text/plain"),
             400, "contract_mismatch"),
            ("đường dẫn lạ → 404", request(base, "POST", "/v1/predict", jpg, "image/jpeg"), 404, "contract_mismatch"),
        ]
        for name, (st, _, body), want_st, want_code in cases:
            check(st == want_st and body.get("error", {}).get("code") == want_code and valid(body),
                  f"{name} (được {st}, {body.get('error', {}).get('code')})")

        print("== thân Transfer-Encoding: chunked (review ADR-015 D2)")
        port = srv.server_address[1]
        payload = json.dumps({"image_b64": base64.b64encode(jpg).decode("ascii")}).encode()
        st, body = post_chunked(port, payload)
        check(st == 200 and valid(body) and body["faces"][0]["codes"] == f["codes"],
              f"JSON gửi chunked (nhiều chunk, có chunk-extension) → 200, cùng kết quả như Content-Length (được {st})")
        st, body = post_chunked(port, img_bytes("PNG", (300, 400)), ctype="image/png", chunk=1000)
        check(st == 200 and valid(body) and body["image"] == {"width": 300, "height": 400}, f"PNG thô chunked → 200 ({st})")
        limit = cfg["max_image_bytes"] * 4 // 3 + 4096
        st, body = post_chunked(port, b"\xff" * (limit + 10_000), ctype="image/jpeg", chunk=65536)
        check(st == 413 and body["error"]["code"] == "image_too_large" and valid(body),
              f"chunked vượt giới hạn thân ({limit} byte) → 413 image_too_large (được {st})")
        for name, raw in (("kích thước chunk không phải hex", b"zz\r\nabc\r\n0\r\n\r\n"),
                          ("thiếu CRLF sau dữ liệu chunk", b"3\r\nabcXY0\r\n\r\n"),
                          ("kích thước âm", b"-5\r\nabc\r\n0\r\n\r\n")):
            st, body = post_chunked(port, raw=raw)
            check(st == 400 and body["error"]["code"] == "contract_mismatch" and valid(body),
                  f"chunked hỏng ({name}) → 400 contract_mismatch (được {st})")

        print("== không ghi đĩa")
        # Chỉ quét thư mục dịch vụ + thư mục làm việc: các session khác có thể đang ghi nơi khác trong dự án.
        def snapshot():
            return set((ROOT / "backend" / "services").rglob("*")) | set(Path.cwd().iterdir())
        before = snapshot()
        post_json(base, jpg)
        after = snapshot()
        check(after == before, f"không tạo file mới ở services/ và cwd ({len(after - before)} file mới)")
    finally:
        srv.shutdown()
        srv.server_close()

    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
