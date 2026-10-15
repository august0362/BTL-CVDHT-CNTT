"""AI — phía gọi EMOCA bên trong `web/` (ADR-012).

`get_analysis(image)` trả dict đúng `contracts/emoca_analyze.schema.json` hoặc
`{"error": {...}}`. Không import torch, không import Streamlit.

Hai chế độ:
* `mock` — dữ liệu giả, không mạng, không GPU.
* `real` — POST ảnh tới `emoca_svc` (services/emoca_svc/server.py) qua HTTP; máy chủ có thể là Colab qua tunnel
  (ADR-015). Cần `endpoint` (URL [+ token]). Thiếu URL ⇒ `model_unavailable`; KHÔNG âm thầm rơi về mock.

Quyền riêng tư (ADR-009): ảnh chỉ xử lý trong bộ nhớ, không ghi đĩa, không log. Ở chế độ `real` ảnh được GỬI tới
máy chủ ở `endpoint["url"]` — nếu máy chủ không phải loopback thì đó là dịch vụ bên thứ ba (ngoại lệ có ghi ở
ADR-015; giao diện phải xin xác nhận trước khi gọi). Token chỉ đi trong header, không bao giờ vào thông báo lỗi/log.
"""

from __future__ import annotations

import base64
import binascii
import functools
import io
import ipaddress
import json
import math
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from PIL import Image, ImageDraw

try:
    from . import mock_data
except ImportError:  # chạy với web/ trên sys.path (streamlit run frontend/app.py)
    import mock_data

_MODES = ("mock", "real")
MAX_FACES_PER_IMAGE = 30
_RENDER_KEYS = ("geometry_coarse", "geometry_detail", "output_coarse", "output_detail")


def _error(code: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "retryable": retryable}}


def _png_b64(img: Image.Image) -> dict[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"format": "png", "b64": base64.b64encode(buf.getvalue()).decode("ascii")}


@functools.lru_cache(maxsize=None)
def _mock_render(name: str) -> str:
    """Ảnh giữ chỗ 224×224 có chữ "MOCK RENDER" (không phụ thuộc ảnh người dùng → cache được)."""
    img = Image.new("RGB", (mock_data.CROP_SIZE, mock_data.CROP_SIZE), (70, 70, 80))
    draw = ImageDraw.Draw(img)
    draw.rectangle((4, 4, mock_data.CROP_SIZE - 5, mock_data.CROP_SIZE - 5), outline=(230, 60, 60), width=3)
    draw.text((20, 90), "MOCK RENDER", fill=(255, 255, 255))
    draw.text((20, 110), name, fill=(200, 200, 200))
    return _png_b64(img)["b64"]


def _landmark_template_crop224() -> list[list[float]]:
    """68 điểm kiểu iBUG giả trong hệ crop224 (hình học cách điệu, KHÔNG phải landmark thật).

    Toạ độ chuẩn hoá (u, v) ∈ [0,1] theo bbox mặt; bbox chiếm phần giữa 1/scale của crop.
    """
    pts: list[tuple[float, float]] = []
    for i in range(17):  # 0-16 hàm
        t = math.pi - i * math.pi / 16
        pts.append((0.5 + 0.5 * math.cos(t), 0.35 + 0.62 * math.sin(t)))
    for u0 in (0.15, 0.58):  # 17-21, 22-26 lông mày
        for k in range(5):
            pts.append((u0 + k * 0.0675, 0.28 - 0.05 * math.sin(math.pi * k / 4)))
    for k in range(4):  # 27-30 sống mũi
        pts.append((0.5, 0.35 + k * 0.0767))
    for k in range(5):  # 31-35 cánh mũi
        pts.append((0.40 + k * 0.05, 0.63 + 0.02 * math.sin(math.pi * k / 4)))
    for cx in (0.30, 0.70):  # 36-41, 42-47 mắt
        rx, ry, cy = 0.08, 0.035, 0.40
        pts += [(cx - rx, cy), (cx - rx / 2, cy - ry), (cx + rx / 2, cy - ry),
                (cx + rx, cy), (cx + rx / 2, cy + ry), (cx - rx / 2, cy + ry)]
    for n, rx, ry in ((12, 0.18, 0.07), (8, 0.10, 0.03)):  # 48-59 môi ngoài, 60-67 môi trong
        for k in range(n):
            a = math.pi + k * 2 * math.pi / n
            pts.append((0.5 + rx * math.cos(a), 0.78 + ry * math.sin(a)))
    assert len(pts) == 68

    margin = mock_data.CROP_SIZE * (1 - 1 / mock_data.MOCK_CROP_SCALE) / 2
    side = mock_data.CROP_SIZE / mock_data.MOCK_CROP_SCALE
    return [[round(margin + u * side, 3), round(margin + v * side, 3)] for u, v in pts]


_LANDMARKS_CROP224 = _landmark_template_crop224()


def _mock_face(img: Image.Image, digest: bytes) -> dict[str, Any]:
    w, h = img.size
    side = mock_data.MOCK_FACE_FRACTION * min(w, h)
    cx, cy = w / 2, h / 2
    bbox = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]

    crop_side = side * mock_data.MOCK_CROP_SCALE
    x0, y0 = cx - crop_side / 2, cy - crop_side / 2
    s = mock_data.CROP_SIZE / crop_side
    # original_px → crop224: [x', y', 1] = M · [x, y, 1]
    matrix = [[s, 0.0, -s * x0], [0.0, s, -s * y0], [0.0, 0.0, 1.0]]
    crop = img.resize((mock_data.CROP_SIZE, mock_data.CROP_SIZE), Image.BILINEAR,
                      box=(x0, y0, x0 + crop_side, y0 + crop_side))
    # crop224 → original_px bằng ma trận nghịch đảo (chỉ tỉ lệ + tịnh tiến).
    lm_orig = [[round(x0 + x / s, 3), round(y0 + y / s, 3)] for x, y in _LANDMARKS_CROP224]

    rng = random.Random(digest)
    codes = {k: [round(rng.gauss(0.0, 0.5), 6) for _ in range(n)] for k, n in mock_data.CODE_DIMS.items()}

    return {
        "face_index": 0,
        "bbox_original_px": [round(v, 3) for v in bbox],
        "crop_transform": {"from": "original_px", "to": "crop224", "matrix_3x3": matrix},
        "crop_image": _png_b64(crop),
        "landmarks": {
            "n_points": 68,
            "crop224": [list(p) for p in _LANDMARKS_CROP224],
            "original_px": lm_orig,
        },
        "codes": codes,
        "renders": {k: {"format": "png", "b64": _mock_render(k)} for k in _RENDER_KEYS},
        "warnings": [mock_data.MOCK_WARNING],
    }


def _mock_analysis(image: Image.Image, max_faces: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    img = image if image.mode == "RGB" else image.convert("RGB")
    w, h = img.size
    if min(w, h) < mock_data.MOCK_MIN_SIDE_PX:
        return _error("no_face", f"mock: ảnh {w}×{h} quá nhỏ (cạnh ngắn < {mock_data.MOCK_MIN_SIDE_PX} px), "
                                 "giả lập không thấy mặt.")
    digest = mock_data.image_digest(img)
    t1 = time.perf_counter()
    faces = [_mock_face(img, digest)][:max_faces]  # mock chỉ "thấy" đúng 1 mặt ở giữa ảnh
    t2 = time.perf_counter()
    return {
        "contract_version": mock_data.CONTRACT_VERSION,
        "service": "emoca_svc",
        "model": dict(mock_data.MOCK_EMOCA_MODEL),
        "image": {"width": w, "height": h},
        "faces": faces,
        "timing_ms": {
            "detect": round((t1 - t0) * 1000, 3),
            "encode": round((t2 - t1) * 1000, 3),
            "decode": 0.0,
            "total": round((time.perf_counter() - t0) * 1000, 3),
        },
    }


# ---------------------------------------------------------------------------- chế độ real (HTTP)

_ERROR_CODES = ("image_invalid", "image_too_large", "no_face", "service_unavailable", "timeout",
                "model_unavailable", "codes_missing", "contract_mismatch")  # enum của emoca_analyze.schema.json
_USER_AGENT = "ma3d-web/1.0 (emoca_client)"  # có UA rõ ràng: một số proxy/tunnel chặn UA mặc định của urllib
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024


def _base_url(endpoint: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    """Chuẩn hoá URL gốc. Trả (url, None) hoặc (None, lỗi). Chỉ nhận http/https."""
    raw = ((endpoint or {}).get("url") or "").strip()
    if not raw:
        return None, _error("model_unavailable",
                            "Chưa cấu hình địa chỉ emoca_svc (configs/services.yaml `emoca_svc.url`, biến EMOCA_URL "
                            "hoặc ô \"Địa chỉ EMOCA\" ở bảng Cài đặt).")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None, _error("model_unavailable", "Địa chỉ emoca_svc phải bắt đầu bằng http:// hoặc https://.")
    return raw.rstrip("/"), None


def _headers(endpoint: dict[str, Any] | None, content_type: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    token = ((endpoint or {}).get("token") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _encode_upload(image: Image.Image, max_bytes: int) -> bytes | None:
    """PNG (không mất điểm ảnh); quá cỡ thì JPEG q95; vẫn quá cỡ → None. Chỉ trong bộ nhớ."""
    for kwargs in ({"format": "PNG"}, {"format": "JPEG", "quality": 95}):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, **kwargs)
        if buf.tell() <= max_bytes:
            return buf.getvalue()
    return None


def _unreachable_hint(url: str) -> str:
    """Gợi ý khi không nối được, theo nơi chạy máy chủ: loopback ⇒ server WSL2 trên máy này (ADR-018, đường chính);
    địa chỉ khác ⇒ Colab qua tunnel (ADR-015, dự phòng)."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if loopback:
        return "Máy chủ EMOCA trong WSL2 đã chạy chưa? bash ~/emoca_wsl/bin/run_emoca_server.sh"
    return "Phiên Colab còn chạy không? Địa chỉ tunnel có đổi không?"


def _http_error_hint(status: int) -> str:
    return {
        401: "token sai hoặc thiếu (HTTP 401) — kiểm tra ô Token",
        403: "bị chặn (HTTP 403) — tunnel/proxy từ chối yêu cầu",
        404: "không thấy dịch vụ (HTTP 404) — sai địa chỉ?",
        502: "tunnel không nối được tới dịch vụ (HTTP 502) — server.py đã chạy chưa?",
        503: "dịch vụ tạm thời không sẵn sàng (HTTP 503)",
        530: "tunnel đã tắt (HTTP 530) — phiên Colab có thể đã ngắt",
    }.get(status, f"HTTP {status}")


def _request(method: str, url: str, endpoint: dict[str, Any] | None, body: bytes | None = None,
             content_type: str | None = None) -> tuple[int | None, Any, dict[str, Any] | None]:
    """Gửi một request. Trả (http_status, json, None) hoặc (None, None, lỗi_kiểu_contract).
    Lỗi HTTP có thân JSON vẫn trả (status, json, None) để người gọi đọc error envelope."""
    timeout = float((endpoint or {}).get("timeout_s") or _DEFAULT_TIMEOUT_S)
    req = urllib.request.Request(url, data=body, method=method, headers=_headers(endpoint, content_type))
    status: int | None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            raw = exc.read(_MAX_RESPONSE_BYTES + 1)
        except OSError:
            raw = b""
    except (socket.timeout, TimeoutError):
        return None, None, _error("timeout", f"emoca_svc không trả lời trong {timeout:.0f} s.", retryable=True)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            return None, None, _error("timeout", f"emoca_svc không trả lời trong {timeout:.0f} s.", retryable=True)
        return None, None, _error("service_unavailable",
                                  f"Không nối được tới emoca_svc ({type(exc.reason).__name__}). "
                                  f"{_unreachable_hint(url)}", retryable=True)
    except (ConnectionError, OSError) as exc:
        return None, None, _error("service_unavailable", f"Lỗi kết nối tới emoca_svc ({type(exc).__name__}). "
                                  f"{_unreachable_hint(url)}", retryable=True)
    if len(raw) > _MAX_RESPONSE_BYTES:
        return None, None, _error("contract_mismatch", "Phản hồi của emoca_svc quá lớn.")
    try:
        return status, json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError):
        if status is not None and status >= 400:
            return None, None, _error("service_unavailable",
                                      f"emoca_svc/tunnel trả lỗi không phải JSON: {_http_error_hint(status)}.",
                                      retryable=status >= 500)
        return None, None, _error("contract_mismatch", "Phản hồi của emoca_svc không phải JSON.")


def _remote_analysis(image: Image.Image, max_faces: int, endpoint: dict[str, Any] | None) -> dict[str, Any]:
    base, err = _base_url(endpoint)
    if err:
        return err
    # Kiểm cạnh ảnh TRƯỚC khi mã hoá/gửi: emoca_svc sẽ từ chối ảnh vượt `max_image_side_px` (413), nên gửi đi chỉ làm
    # ảnh rời khỏi máy vô ích (review ADR-015 D3). Không tự thu nhỏ: sẽ đổi hệ `original_px` (CLAUDE.md §1.7).
    max_side = (endpoint or {}).get("max_image_side_px")
    if isinstance(max_side, (int, float)) and not isinstance(max_side, bool) and max_side > 0 \
            and max(image.size) > max_side:
        return _error("image_too_large", f"Ảnh {image.width}×{image.height} px có cạnh dài hơn giới hạn {int(max_side)} px "
                                         "của emoca_svc — ảnh CHƯA được gửi đi. Hãy thu nhỏ ảnh rồi thử lại.")
    max_bytes = int((endpoint or {}).get("max_image_bytes") or _DEFAULT_MAX_IMAGE_BYTES)
    data = _encode_upload(image, max_bytes)
    if data is None:
        return _error("image_too_large", f"Ảnh sau khi nén vẫn lớn hơn {max_bytes} byte.")
    payload = json.dumps({"image_b64": base64.b64encode(data).decode("ascii"),
                          "options": {"max_faces": max_faces}}).encode("utf-8")
    del data
    status, body, err = _request("POST", f"{base}/v1/analyze", endpoint, payload, "application/json")
    if err:
        return err
    if isinstance(body, dict) and isinstance(body.get("error"), dict):  # error envelope hợp lệ → chuyển tiếp nguyên vẹn
        e = body["error"]
        code = e.get("code") if e.get("code") in _ERROR_CODES else "contract_mismatch"
        msg = str(e.get("message") or _http_error_hint(status or 0))
        if status == 401:
            msg = f"{msg} — {_http_error_hint(401)}"
        return _error(code, msg, bool(e.get("retryable", False)))
    if (status != 200 or not isinstance(body, dict) or body.get("service") != "emoca_svc"
            or not isinstance(body.get("faces"), list)):
        return _error("contract_mismatch", f"emoca_svc trả dữ liệu không đúng contract ({_http_error_hint(status or 0)}).")
    if body.get("contract_version") != mock_data.CONTRACT_VERSION:
        return _error("contract_mismatch",
                      f"contract_version của emoca_svc là {body.get('contract_version')!r}, web cần "
                      f"{mock_data.CONTRACT_VERSION!r}.")
    return body


def check_service(endpoint: dict[str, Any] | None) -> dict[str, Any]:
    """Kiểm kết nối `GET /v1/health` (cho nút "Kiểm tra kết nối"). Không raise.
    Trả {"ok": bool, "message": str, "latency_ms": float | None, "info": dict}; không chứa token."""
    base, err = _base_url(endpoint)
    if err:
        return {"ok": False, "message": err["error"]["message"], "latency_ms": None, "info": {}}
    t0 = time.perf_counter()
    status, body, err = _request("GET", f"{base}/v1/health", endpoint)
    latency = round((time.perf_counter() - t0) * 1000, 1)
    if err:
        return {"ok": False, "message": err["error"]["message"], "latency_ms": latency, "info": {}}
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        msg = str(body["error"].get("message") or _http_error_hint(status or 0))
        return {"ok": False, "message": msg, "latency_ms": latency, "info": {}}
    if status != 200 or not isinstance(body, dict) or body.get("service") != "emoca_svc":
        return {"ok": False, "message": f"Không phải emoca_svc ({_http_error_hint(status or 0)}).",
                "latency_ms": latency, "info": {}}
    if body.get("contract_version") != mock_data.CONTRACT_VERSION:
        return {"ok": False, "latency_ms": latency, "info": body,
                "message": f"contract_version {body.get('contract_version')!r} ≠ {mock_data.CONTRACT_VERSION!r}."}
    mode = body.get("mode", "?")
    return {"ok": True, "latency_ms": latency, "info": body,
            "message": f"emoca_svc chạy chế độ {mode}" + (" (KHÔNG phải EMOCA thật)" if mode == "mock" else "")}


def get_analysis(image: Image.Image, *, max_faces: int = 1, mode: str = "mock",
                 endpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Landmark + mã 3DMM + render cho từng mặt; không raise với lỗi dự kiến.

    image: ảnh PIL đã xoay theo EXIF (toạ độ `original_px` tính trên ảnh này).
    max_faces: số mặt tối đa trả về (≥ 1).
    mode: "mock" (dữ liệu giả, không GPU, không mạng) | "real" (HTTP tới emoca_svc, cần `endpoint`).
    endpoint: {"url", "token"?, "timeout_s"?, "max_image_bytes"?, "max_image_side_px"?} — chỉ dùng khi mode="real";
              ảnh có cạnh dài hơn `max_image_side_px` bị từ chối (`image_too_large`) mà KHÔNG gửi đi.
    """
    if not isinstance(image, Image.Image):
        return _error("image_invalid", f"Cần PIL.Image.Image, nhận {type(image).__name__}.")
    if isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces < 1:
        return _error("contract_mismatch", f"max_faces phải là số nguyên ≥ 1, nhận {max_faces!r}.")
    max_faces = min(max_faces, MAX_FACES_PER_IMAGE)
    if mode not in _MODES:
        return _error("contract_mismatch", f"mode không hợp lệ: {mode!r}; chỉ nhận {_MODES}.")
    if mode == "real":
        return _remote_analysis(image, max_faces, endpoint)
    return _mock_analysis(image, max_faces)


def decode_crop(face: dict[str, Any]) -> Image.Image | None:
    """Giải mã `face["crop_image"]` (PNG/JPEG base64) thành ảnh PIL RGB, để đưa vào `engine.predict`
    (ADR-006: POSTER và MA3D nhận cùng crop 224 của EMOCA). Thiếu/hỏng → None.
    """
    try:
        raw = base64.b64decode(face["crop_image"]["b64"], validate=True)
        img = Image.open(io.BytesIO(raw), formats=("PNG", "JPEG"))
        img.load()
        return img.convert("RGB")
    except (KeyError, TypeError, binascii.Error, OSError, ValueError):
        return None
