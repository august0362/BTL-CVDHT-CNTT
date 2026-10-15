"""emoca_svc — bản MOCK của dịch vụ HTTP EMOCA (M1-02, contract B1 `POST /v1/analyze`).

Trả dữ liệu GIẢ đúng `contracts/emoca_analyze.schema.json` (contract_version xem CONTRACT_VERSION), không cần GPU,
torch hay EMOCA. Dùng để phát triển/kiểm phía gọi (`web/emoca_client.py` chế độ `real`) trước khi có
EMOCA thật (M1-03). Không import gì từ `web/` (bất biến module hoá) — chỉ stdlib + Pillow (+ PyYAML
nếu có, để đọc configs/services.yaml). Chạy được trên Python 3.8 (env EMOCA) lẫn 3.13.

Chạy:
    python services/emoca_svc/mock_server.py [--port 8765] [--config configs/services.yaml]

API:
    GET  /v1/health   → {"service","mode":"mock","status":"up","contract_version"}
    POST /v1/analyze  → body là
        - application/json: {"image_b64": "<base64>", "options": {"max_faces": 1}}
        - hoặc image/jpeg | image/png | image/webp | application/octet-stream: bytes ảnh thô
      Thành công 200 + success; lỗi → error envelope với HTTP 400/404/413/422.
      Thân request theo `Content-Length` hoặc `Transfer-Encoding: chunked` (proxy/tunnel có thể chuyển sang chunked);
      giới hạn kích thước áp ngay khi đọc. `options.max_faces` bị kẹp theo `limits.max_faces` của config.

Riêng tư (ADR-009): ảnh chỉ nằm trong bộ nhớ trong một request; không ghi đĩa; log chỉ có
method/đường dẫn/mã trạng thái, không có nội dung ảnh.

Tầng HTTP (`make_handler`) và kiểm ảnh vào (`check_request`) dùng chung với `server.py` (EMOCA THẬT,
ADR-015): truyền `analyzer=` khác, `mode="real"`, `token=` để bật xác thực Bearer.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import io
import json
import math
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

CONTRACT_VERSION = "0.3.0"  # ADR-019 (B1 không đổi hình dạng, chỉ đổi số phiên bản chung; 0.2.0: ADR-013)
ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {"host": "127.0.0.1", "port": 8765, "max_image_bytes": 10485760, "max_faces": 30,
            "min_face_side_px": 32, "max_image_side_px": 4096, "detect_max_side_px": 1024}

# Tham số hình học của mock — cùng quy ước với crop EMOCA (CONTRACTS_ADR Phần A: bbox × 1.25 → 224).
CROP_SIZE = 224
CROP_SCALE = 1.25
FACE_FRACTION = 0.6          # cạnh bbox giả = 0.6 × cạnh ngắn ảnh, đặt giữa ảnh
CODE_DIMS = {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}  # PROJECT_CONTEXT §3
RENDER_KEYS = ("geometry_coarse", "geometry_detail", "output_coarse", "output_detail")
MOCK_MODEL = {"name": "MOCK-emoca_svc", "mode": "detail", "upstream_commit": "mock"}
MOCK_WARNING = "mock: emoca_svc/mock_server.py — dữ liệu giả, không phải EMOCA thật"
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


# ---------------------------------------------------------------------------- cấu hình

def load_config(path: Path | None) -> dict:
    cfg = dict(DEFAULTS)
    if path is None or not path.is_file():
        return cfg
    try:
        import yaml  # tuỳ chọn
    except ImportError:
        print(f"[emoca_svc mock] Không có PyYAML — bỏ qua {path}, dùng mặc định.", file=sys.stderr)
        return cfg
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    svc, lim = data.get("emoca_svc", {}), data.get("limits", {})
    for k in ("host", "port"):
        if k in svc:
            cfg[k] = svc[k]
    for k in ("max_image_bytes", "max_faces", "min_face_side_px", "max_image_side_px", "detect_max_side_px"):
        if k in lim:
            cfg[k] = lim[k]
    cfg["max_faces"] = max(1, min(int(cfg["max_faces"]), 30))
    return cfg


# ---------------------------------------------------------------------------- dữ liệu giả

def _error(code: str, message: str, retryable: bool = False) -> dict:
    return {"error": {"code": code, "message": message, "retryable": retryable}}


def _png_b64(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"format": "png", "b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def _landmark_template_crop224() -> list:
    """68 điểm kiểu iBUG cách điệu trong hệ crop224 (KHÔNG phải landmark thật)."""
    pts = []
    for i in range(17):  # 0-16 đường hàm
        t = math.pi - i * math.pi / 16
        pts.append((0.5 + 0.5 * math.cos(t), 0.35 + 0.62 * math.sin(t)))
    for u0 in (0.15, 0.58):  # 17-26 lông mày
        for k in range(5):
            pts.append((u0 + k * 0.0675, 0.28 - 0.05 * math.sin(math.pi * k / 4)))
    for k in range(4):  # 27-30 sống mũi
        pts.append((0.5, 0.35 + k * 0.0767))
    for k in range(5):  # 31-35 cánh mũi
        pts.append((0.40 + k * 0.05, 0.63 + 0.02 * math.sin(math.pi * k / 4)))
    for cx in (0.30, 0.70):  # 36-47 mắt
        rx, ry, cy = 0.08, 0.035, 0.40
        pts += [(cx - rx, cy), (cx - rx / 2, cy - ry), (cx + rx / 2, cy - ry),
                (cx + rx, cy), (cx + rx / 2, cy + ry), (cx - rx / 2, cy + ry)]
    for n, rx, ry in ((12, 0.18, 0.07), (8, 0.10, 0.03)):  # 48-67 môi
        for k in range(n):
            a = math.pi + k * 2 * math.pi / n
            pts.append((0.5 + rx * math.cos(a), 0.78 + ry * math.sin(a)))
    assert len(pts) == 68
    margin = CROP_SIZE * (1 - 1 / CROP_SCALE) / 2
    side = CROP_SIZE / CROP_SCALE
    return [[round(margin + u * side, 3), round(margin + v * side, 3)] for u, v in pts]


LANDMARKS_CROP224 = _landmark_template_crop224()


def _render_placeholder(name: str) -> dict:
    img = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (60, 64, 72))
    d = ImageDraw.Draw(img)
    d.rectangle((4, 4, CROP_SIZE - 5, CROP_SIZE - 5), outline=(230, 60, 60), width=3)
    d.text((20, 90), "MOCK emoca_svc", fill=(255, 255, 255))
    d.text((20, 110), name, fill=(200, 200, 200))
    return _png_b64(img)


RENDERS = {k: _render_placeholder(k) for k in RENDER_KEYS}  # không phụ thuộc ảnh → dựng một lần


def _mock_face(img: Image.Image, digest: bytes) -> dict:
    w, h = img.size
    side = FACE_FRACTION * min(w, h)
    cx, cy = w / 2, h / 2
    bbox = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    crop_side = side * CROP_SCALE
    x0, y0 = cx - crop_side / 2, cy - crop_side / 2
    s = CROP_SIZE / crop_side
    matrix = [[s, 0.0, -s * x0], [0.0, s, -s * y0], [0.0, 0.0, 1.0]]  # original_px → crop224
    crop = img.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR, box=(x0, y0, x0 + crop_side, y0 + crop_side))
    lm_orig = [[round(x0 + x / s, 3), round(y0 + y / s, 3)] for x, y in LANDMARKS_CROP224]
    rng = random.Random(digest)  # cùng ảnh → cùng mã (tất định)
    codes = {k: [round(rng.gauss(0.0, 0.5), 6) for _ in range(n)] for k, n in CODE_DIMS.items()}
    return {
        "face_index": 0,
        "bbox_original_px": [round(v, 3) for v in bbox],
        "crop_transform": {"from": "original_px", "to": "crop224", "matrix_3x3": matrix},
        "crop_image": _png_b64(crop),
        "landmarks": {"n_points": 68, "crop224": [list(p) for p in LANDMARKS_CROP224], "original_px": lm_orig},
        "codes": codes,
        "renders": {k: dict(v) for k, v in RENDERS.items()},
        "warnings": [MOCK_WARNING],
    }


def check_request(data: bytes, cfg: dict, max_faces=1) -> tuple:
    """Kiểm đầu vào dùng chung cho mock và EMOCA thật. Trả (img RGB đã xoay EXIF, None)
    hoặc (None, (http_status, body_lỗi)). Không raise với đầu vào xấu."""
    if not data:
        return None, (400, _error("image_invalid", "Ảnh rỗng."))
    if len(data) > cfg["max_image_bytes"]:
        return None, (413, _error("image_too_large", f"Ảnh {len(data)} byte > giới hạn {cfg['max_image_bytes']}."))
    if not isinstance(max_faces, int) or isinstance(max_faces, bool) or max_faces < 1:
        return None, (400, _error("contract_mismatch", "options.max_faces phải là số nguyên ≥ 1."))
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            size = probe.size
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            return None, (400, _error("image_invalid", f"Định dạng {fmt} không hỗ trợ (chỉ JPEG/PNG/WEBP)."))
        max_side = cfg.get("max_image_side_px")
        if max_side and max(size) > max_side:  # kiểm theo tiêu đề ảnh, trước khi giải mã cả ảnh vào RAM
            return None, (413, _error("image_too_large",
                                      f"Ảnh {size[0]}×{size[1]} px có cạnh dài hơn giới hạn {max_side} px."))
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Image.DecompressionBombError:
        return None, (413, _error("image_too_large", "Ảnh có quá nhiều điểm ảnh."))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return None, (400, _error("image_invalid", f"Không đọc được ảnh: {type(e).__name__}."))
    return img, None


def analyze_bytes(data: bytes, cfg: dict, max_faces: int = 1) -> tuple:
    """Trả (http_status, body_dict). Không raise với đầu vào xấu."""
    t0 = time.perf_counter()
    img, err = check_request(data, cfg, max_faces)
    if err is not None:
        return err
    t_detect = time.perf_counter()
    if min(img.size) < cfg["min_face_side_px"]:
        return 422, _error("no_face", f"Không thấy khuôn mặt (mock: cạnh ngắn < {cfg['min_face_side_px']} px).")
    digest = hashlib.sha256(data).digest()
    faces = [_mock_face(img, digest)][:max_faces]  # mock luôn 1 mặt
    t_end = time.perf_counter()
    ms = lambda a, b: round((b - a) * 1000, 3)  # noqa: E731
    return 200, {
        "contract_version": CONTRACT_VERSION,
        "service": "emoca_svc",
        "model": dict(MOCK_MODEL),
        "image": {"width": img.width, "height": img.height},
        "faces": faces,
        "timing_ms": {"detect": ms(t0, t_detect), "encode": ms(t_detect, t_end), "decode": 0.0,
                      "total": ms(t0, t_end)},
    }


# ---------------------------------------------------------------------------- HTTP

def make_handler(cfg: dict, analyzer=None, mode: str = "mock", token: str | None = None, health_info=None):
    """Tạo lớp xử lý HTTP. `analyzer(data, cfg, max_faces) -> (status, body)` mặc định là bản mock.
    `token` khác None ⇒ mọi đường dẫn đòi `Authorization: Bearer <token>` (dùng khi mở qua tunnel công khai).
    `health_info() -> dict` bổ sung trường cho /v1/health (vd GPU, thời gian nạp)."""
    analyze = analyzer or analyze_bytes
    tag = f"emoca_svc {mode}"

    class Handler(BaseHTTPRequestHandler):
        server_version = f"emoca_svc-{mode}/0.1"
        timeout = 120  # giây; ngắt kết nối treo (slowloris) thay vì giữ luồng mãi

        def _send(self, status: int, body: dict) -> None:
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-Emoca-Mode", mode)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt, *args):  # không log nội dung request, không log header (có token)
            sys.stderr.write(f"[{tag}] {self.command} {self.path} → {args[1] if len(args) > 1 else ''}\n")

        def _body_limit(self) -> int:
            # Base64 phình ~4/3: cho phép thân JSON lớn hơn giới hạn ảnh một chút rồi kiểm lại sau giải mã.
            return cfg["max_image_bytes"] * 4 // 3 + 4096

        def _is_chunked(self) -> bool:
            return "chunked" in (self.headers.get("Transfer-Encoding") or "").lower()

        def _read_chunked(self, limit: int) -> tuple:
            """Đọc thân `Transfer-Encoding: chunked`. Trả (bytes, None) hoặc (None, (http_status, body_lỗi)).
            Tổng vượt `limit` ⇒ 413, nhưng vẫn đọc bỏ phần còn lại tới 2×limit để máy khách nhận được mã lỗi thay vì
            RST; quá mức đó, hoặc chunk hỏng/thiếu ⇒ đóng kết nối. Không giữ phần vượt giới hạn trong bộ nhớ."""
            parts, total, too_large = [], 0, False
            try:
                while True:
                    line = self.rfile.readline(1026)
                    if not line.endswith(b"\n"):
                        raise ValueError("dòng kích thước chunk hỏng")
                    size = int(line.split(b";", 1)[0].strip(), 16)
                    if size < 0:
                        raise ValueError("kích thước chunk âm")
                    if size == 0:
                        while True:  # trailer (thường rỗng) kết thúc bằng một dòng trống
                            trailer = self.rfile.readline(1026)
                            if trailer in (b"\r\n", b"\n", b""):
                                break
                            if not trailer.endswith(b"\n"):
                                raise ValueError("trailer hỏng")
                        break
                    total += size
                    if total > limit:
                        too_large = True
                        if total > 2 * limit:
                            self.close_connection = True
                            return None, (413, _error("image_too_large", f"Thân request > {limit} byte."))
                    chunk = self.rfile.read(size)
                    if len(chunk) != size:
                        raise ValueError("thiếu dữ liệu chunk")
                    if not too_large:
                        parts.append(chunk)
                    if self.rfile.readline(3) not in (b"\r\n", b"\n"):
                        raise ValueError("thiếu CRLF sau chunk")
            except (ValueError, OSError) as e:  # OSError gồm cả socket.timeout
                self.close_connection = True
                return None, (400, _error("contract_mismatch", f"Thân chunked không hợp lệ ({e})."))
            if too_large:
                return None, (413, _error("image_too_large", f"Thân request {total} byte > giới hạn {limit} byte."))
            return b"".join(parts), None

        def _drain(self) -> None:
            """Đọc bỏ thân request khi từ chối SỚM. Không đọc thì đóng socket lúc còn dữ liệu chưa đọc sẽ gửi RST,
            máy khách thấy ConnectionReset thay vì mã lỗi (đã gặp trên Windows ở test_emoca_client_remote)."""
            if self._is_chunked():
                self._read_chunked(self._body_limit())  # lỗi/quá lớn ⇒ tự đặt close_connection
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if 0 < n <= self._body_limit():
                self.rfile.read(n)
            elif n != 0:
                self.close_connection = True  # thân quá lớn/không rõ: không tốn công đọc, đóng kết nối

        def _authorized(self) -> bool:
            if not token:
                return True
            got = self.headers.get("Authorization") or ""
            ok = got.startswith("Bearer ") and hmac.compare_digest(got[7:].encode("utf-8"), token.encode("utf-8"))
            if not ok:
                self._drain()
                # Bộ mã lỗi của contract 0.2.0 không có `unauthorized`; dùng service_unavailable (không thử lại được).
                self._send(401, _error("service_unavailable", "Thiếu hoặc sai token truy cập (HTTP 401)."))
            return ok

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/v1/health":
                body = {"service": "emoca_svc", "mode": mode, "status": "up", "contract_version": CONTRACT_VERSION}
                if health_info is not None:
                    body.update(health_info())
                self._send(200, body)
            else:
                self._send(404, _error("contract_mismatch", f"Không có đường dẫn {self.path}."))

        def do_POST(self):
            if not self._authorized():  # kiểm token TRƯỚC khi đọc thân request
                return
            if self.path != "/v1/analyze":
                self._drain()
                self._send(404, _error("contract_mismatch", f"Không có đường dẫn {self.path}."))
                return
            limit = self._body_limit()
            if self._is_chunked():  # Transfer-Encoding thắng Content-Length (RFC 9112 §6.3)
                body, err = self._read_chunked(limit)
                if err is not None:
                    self._send(*err)
                    return
            else:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if length < 0:
                    self._send(400, _error("contract_mismatch", "Content-Length không hợp lệ."))
                    return
                if length > limit:
                    self._send(413, _error("image_too_large", f"Thân request {length} byte quá lớn."))
                    return
                body = self.rfile.read(length)
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            max_faces = 1
            if ctype == "application/json":
                try:
                    req = json.loads(body.decode("utf-8"))
                    data = base64.b64decode(req["image_b64"], validate=True)
                    max_faces = (req.get("options") or {}).get("max_faces", 1)
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, binascii.Error) as e:
                    self._send(400, _error("contract_mismatch", f"JSON không đúng B1 ({type(e).__name__})."))
                    return
            elif ctype.startswith("image/") or ctype == "application/octet-stream":
                data = body
            else:
                self._send(400, _error("contract_mismatch", f"Content-Type {ctype or '(trống)'} không hỗ trợ."))
                return
            cap_value = cfg.get("max_faces", 30)
            cap = max(1, min(30, int(cap_value))) if isinstance(cap_value, int) and not isinstance(cap_value, bool) else 30
            if (isinstance(max_faces, int) and not isinstance(max_faces, bool) and isinstance(cap, int)
                    and not isinstance(cap, bool) and cap >= 1):
                max_faces = min(max_faces, cap)  # trần theo limits.max_faces (giữ GPU không bị một request chiếm)
            status, out = analyze(data, cfg, max_faces)
            self._send(status, out)

    return Handler


def make_server(cfg: dict, port: int | None = None, **handler_kwargs) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((cfg["host"], cfg["port"] if port is None else port),
                               make_handler(cfg, **handler_kwargs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emoca_svc MOCK (contract B1, không GPU)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "services.yaml")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    srv = make_server(cfg, args.port)
    host, port = srv.server_address[:2]
    print(f"[emoca_svc mock] http://{host}:{port}  (contract {CONTRACT_VERSION}, KHÔNG phải EMOCA thật)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
