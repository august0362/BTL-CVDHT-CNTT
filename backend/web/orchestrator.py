"""BACKEND — điều phối trong web/ (ADR-012).

`analyze(image)` nhận ảnh upload, kiểm tra hợp lệ, gọi AI và trả dict đúng
hình dạng `contracts/web_analyze.schema.json`. Module này không import torch,
không import Streamlit, không chứa logic mô hình.

Phần "gọi AI": `web/emoca_client.py` (landmark/3DMM/render) rồi `web/engine.py` (xác suất),
cả hai thuộc AI. `engine` có `mock` và `real` (POSTER/MA3D-Net trên CPU, M2-05; chọn bằng `resolve_engine_mode`:
    tham số `engine_mode=` > hằng `_ENGINE_MODE` > biến môi trường ENGINE_MODE > `engine.mode` trong configs/services.yaml
    > mock). `emoca_client` có `mock` và `real`
(HTTP tới emoca_svc, thường là Colab qua tunnel — ADR-015); chế độ EMOCA chọn theo thứ tự ưu tiên
    tham số `emoca=` của `analyze` (giao diện) > hằng `_EMOCA_MODE` (nếu đặt) > biến môi trường
    EMOCA_MODE/EMOCA_URL/EMOCA_TOKEN > configs/services.yaml `emoca_svc` > mock.

Quyền riêng tư (ADR-009): ảnh chỉ xử lý trong bộ nhớ, không ghi đĩa, không log. Khi EMOCA `real` trỏ tới
máy không phải loopback thì ảnh bị GỬI ra ngoài: `analyze` từ chối (`consent_required`) trừ khi người gọi
truyền `emoca={"remote_consent": True}` (giao diện chỉ đặt cờ này sau khi người dùng bấm xác nhận).
"""

from __future__ import annotations

import base64
import io
import ipaddress
import os
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, BinaryIO, Callable, Literal, Union

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from . import emoca_client, engine
except ImportError:  # chạy với web/ trên sys.path (streamlit run frontend/app.py)
    import emoca_client
    import engine
try:  # ADR-020: căn chỉnh ảnh vào POSTER/MA3D-Net kiểu RAF-DB — module AI (web/face_align.py); chưa có ⇒ None
    from . import face_align
except ImportError:
    try:
        import face_align
    except ImportError:
        face_align = None

CONTRACT_VERSION = "0.3.0"  # ADR-019 (0.2.0: ADR-013)

_ROOT = Path(__file__).resolve().parent.parent
_SERVICES_CONFIG = _ROOT / "configs" / "services.yaml"
_FRONTALIZATION_CONFIG = _ROOT / "configs" / "frontalization.yaml"


def roll_correction_settings() -> tuple[bool, float]:
    """Read the ADR-029 switch and threshold without importing the 3DDFA stack."""
    import yaml

    config = yaml.safe_load(_FRONTALIZATION_CONFIG.read_text(encoding="utf-8")) or {}
    enabled = config.get("enabled_in_baseline", False) is True
    threshold = float(config.get("roll_threshold_deg", 15))
    if not 0 < threshold <= 90:
        raise ValueError("roll_threshold_deg must be in (0, 90]")
    return enabled, threshold


def _correct_roll_crop(crop: Image.Image, threshold: float) -> tuple[Image.Image, dict[str, Any] | None]:
    """Run the optional 3DDFA roll estimate on a face-tight crop, failing open."""
    try:
        try:
            from services.frontalization import get_tddfa_v2_frontalizer
        except ImportError:
            from ..services.frontalization import get_tddfa_v2_frontalizer
        result = get_tddfa_v2_frontalizer().correct_roll(crop, threshold)
        info = {"roll_deg": float(result.roll_deg), "rotation_deg": float(result.rotation_deg),
                "threshold_deg": float(result.threshold_deg), "applied": bool(result.applied)}
        return (result.image if result.applied else crop), info
    except Exception as exc:
        return crop, {"error": f"{type(exc).__name__}: {exc}"}


def roll_correct_upload(image: Image.Image, views: Any) -> tuple[Image.Image, dict[str, Any] | None]:
    """ADR-029: use one YOLO crop to estimate roll, then rotate the original upload in memory."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    try:
        enabled, threshold = roll_correction_settings()
    except Exception as exc:
        return image, {"error": f"{type(exc).__name__}: {exc}"}
    faces = getattr(views, "faces", None)
    if (not enabled or getattr(views, "detection_complete", True) is False
            or not isinstance(faces, (list, tuple)) or len(faces) != 1):
        return image, None
    try:
        crop = faces[0].detection.cropped_face
    except Exception:
        return image, None
    corrected, info = _correct_roll_crop(crop, threshold)
    if info is None or info.get("error") or not info.get("applied"):
        if info and not info.get("error"):
            return image, {key: info[key] for key in ("roll_deg", "rotation_deg", "threshold_deg")}
        return image, info
    original = ImageOps.exif_transpose(image).convert("RGB")
    fill = tuple(int(value) for value in np.asarray(original, dtype=np.float32).mean(axis=(0, 1)))
    rotated = original.rotate(info["rotation_deg"], resample=Image.Resampling.BICUBIC,
                              expand=True, fillcolor=fill)
    return rotated, {key: info[key] for key in ("roll_deg", "rotation_deg", "threshold_deg")}

# EMOCA: None ⇒ suy từ biến môi trường EMOCA_MODE / configs/services.yaml (`emoca_svc.mode`) / "mock" (xem
# `resolve_emoca_settings`). Đặt "mock"/"real" ở đây (vd trong test) để ép, thắng cả biến môi trường và config.
_EMOCA_MODE: str | None = None
# Engine (POSTER/MA3D-Net): None ⇒ suy từ biến môi trường ENGINE_MODE / configs/services.yaml (`engine.mode`) / "mock".
# Đặt "mock"/"real" ở đây (vd trong test) để ép. `real` cần torch + checkpoint (environments/web/requirements-real.txt, M2-05).
_ENGINE_MODE: str | None = None

_MODES = ("mock", "real")
_DEFAULT_EMOCA_TIMEOUT_S = 60.0
MAX_FACES_PER_IMAGE = 30

# Chỉ nhận các định dạng này (03_BACKEND.md §2.2). Pillow chỉ thử đúng các decoder này.
_ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")

# TODO(CONFIG): giá trị tạm, chỉ dùng khi configs/services.yaml chưa tồn tại.
# Hình dạng khoá theo CONTRACTS_ADR.md Phần C: max_image_bytes, max_faces, max_concurrent.
_FALLBACK_LIMITS = {"max_image_bytes": 10 * 1024 * 1024, "max_faces": MAX_FACES_PER_IMAGE, "max_concurrent": 1, "max_image_side_px": 4096}

MockCase = Literal["ok", "partial"]  # "partial": giả lập emoca_svc không phản hồi
_MOCK_CASES = ("ok", "partial")

# EMOCA báo các mã này = vấn đề ở ảnh đầu vào → status "error" (không có kết quả một phần).
_INPUT_ERROR_CODES = ("no_face", "image_invalid", "image_too_large")

ImageInput = Union[bytes, bytearray, memoryview, BinaryIO, Any]

# Khoá toàn cục: không chạy hai ảnh cùng lúc qua AI (GPU nhỏ, 03_BACKEND.md §2.8).
# max_concurrent > 1 chưa hỗ trợ; khi có config thật sẽ đổi thành Semaphore.
_AI_LOCK = threading.Lock()


class AnalyzeError(Exception):
    """Lỗi có mã thống nhất (CONTRACTS_ADR.md Phần A, dòng 'Lỗi')."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def _load_limits() -> tuple[dict[str, int], list[str]]:
    """Đọc giới hạn từ configs/services.yaml; thiếu file → giá trị tạm + cảnh báo."""
    if not _SERVICES_CONFIG.is_file():
        return dict(_FALLBACK_LIMITS), ["config_missing: configs/services.yaml, dùng giới hạn tạm"]
    import yaml  # chỉ cần khi đã có file config

    with _SERVICES_CONFIG.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    section = cfg.get("limits", cfg)
    limits = dict(_FALLBACK_LIMITS)
    missing = []
    for key in _FALLBACK_LIMITS:
        if key in section:
            limits[key] = int(section[key])
        else:
            missing.append(key)
    limits["max_faces"] = max(1, min(int(limits["max_faces"]), MAX_FACES_PER_IMAGE))
    warnings = [f"config_missing_key: {k}, dùng giá trị tạm" for k in missing]
    return limits, warnings


def max_faces_limit() -> int:
    """Configured face cap, bounded by the application's hard maximum."""
    return _load_limits()[0]["max_faces"]


def _load_emoca_config() -> tuple[dict[str, Any], list[str]]:
    """Đọc mục `emoca_svc` của configs/services.yaml; lỗi/thiếu ⇒ ({}, cảnh báo) — không làm hỏng trang."""
    if not _SERVICES_CONFIG.is_file():
        return {}, []
    try:
        import yaml

        with _SERVICES_CONFIG.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 — config hỏng không được làm sập phân tích; báo bằng warnings
        return {}, [f"config_invalid: không đọc được {_SERVICES_CONFIG.name} ({type(exc).__name__})"]
    svc = cfg.get("emoca_svc") if isinstance(cfg, dict) else None
    return (svc if isinstance(svc, dict) else {}), []


def _host_of(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_remote_url(url: str) -> bool:
    """True nếu URL trỏ tới máy khác loopback — ảnh gửi đi sẽ rời khỏi máy này (ADR-009/ADR-015).
    Địa chỉ mạng LAN (192.168.x.x…) cũng tính là "từ xa": ảnh vẫn rời khỏi máy."""
    host = _host_of(url)
    if not host:
        return False
    if host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:  # không phải địa chỉ IP (tên miền) ⇒ từ xa
        return True


def resolve_emoca_settings(overrides: dict[str, Any] | None = None,
                           max_image_bytes: int | None = None,
                           max_image_side_px: int | None = None) -> tuple[dict[str, Any], list[str]]:
    """Gộp cài đặt EMOCA. Ưu tiên: `overrides` (giao diện) > `_EMOCA_MODE` (nếu đặt, chỉ cho mode) > biến môi trường
    EMOCA_MODE/EMOCA_URL/EMOCA_TOKEN > configs/services.yaml > mặc định. Trả (settings, warnings).
    `settings` có token — chỉ đưa cho `emoca_client`, không hiển thị/log."""
    overrides = overrides or {}
    svc, warnings = _load_emoca_config()
    mode = overrides.get("mode") or _EMOCA_MODE or os.environ.get("EMOCA_MODE") or svc.get("mode") or "mock"
    if mode not in _MODES:
        warnings.append(f"config_invalid: emoca_svc.mode={mode!r} không hợp lệ, dùng mock")
        mode = "mock"
    try:
        timeout_s = float(overrides.get("timeout_s") or svc.get("timeout_s") or _DEFAULT_EMOCA_TIMEOUT_S)
    except (TypeError, ValueError):
        timeout_s = _DEFAULT_EMOCA_TIMEOUT_S
    settings = {
        "mode": mode,
        "url": str(overrides.get("url") or os.environ.get("EMOCA_URL") or svc.get("url") or "").strip(),
        "token": str(overrides.get("token") or os.environ.get("EMOCA_TOKEN") or ""),
        "timeout_s": timeout_s,
        "max_image_bytes": max_image_bytes or _FALLBACK_LIMITS["max_image_bytes"],
        # Cạnh dài tối đa (px) mà máy chủ EMOCA nhận; client kiểm TRƯỚC khi gửi để ảnh quá lớn không rời máy vô ích.
        "max_image_side_px": max_image_side_px or _FALLBACK_LIMITS["max_image_side_px"],
    }
    return settings, warnings


def resolve_engine_mode(override: str | None = None) -> tuple[str, list[str]]:
    """Chế độ engine: `override` (giao diện) > hằng `_ENGINE_MODE` (nếu đặt) > biến môi trường ENGINE_MODE >
    configs/services.yaml `engine.mode` > "mock". Giá trị không hợp lệ ⇒ dùng mock kèm cảnh báo (không im lặng)."""
    warnings: list[str] = []
    cfg_mode = None
    if _SERVICES_CONFIG.is_file():
        try:
            import yaml

            data = yaml.safe_load(_SERVICES_CONFIG.read_text(encoding="utf-8")) or {}
            section = data.get("engine") if isinstance(data, dict) else None
            cfg_mode = section.get("mode") if isinstance(section, dict) else None
        except Exception as exc:  # noqa: BLE001 — config hỏng không được làm sập phân tích
            warnings.append(f"config_invalid: không đọc được {_SERVICES_CONFIG.name} ({type(exc).__name__})")
    mode = override or _ENGINE_MODE or os.environ.get("ENGINE_MODE") or cfg_mode or "mock"
    if mode not in _MODES:
        warnings.append(f"config_invalid: engine.mode={mode!r} không hợp lệ, dùng mock")
        mode = "mock"
    return mode, warnings


def emoca_status(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tóm tắt cài đặt EMOCA cho giao diện (KHÔNG có token): mode, url, has_token, remote, host, warnings."""
    settings, warnings = resolve_emoca_settings(overrides)
    return {"mode": settings["mode"], "url": settings["url"], "has_token": bool(settings["token"]),
            "remote": settings["mode"] == "real" and is_remote_url(settings["url"]),
            "host": _host_of(settings["url"]), "timeout_s": settings["timeout_s"], "warnings": warnings}


def check_emoca(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Kiểm kết nối `emoca_svc` (GET /v1/health) — không gửi ảnh. Trả {"ok", "message", "latency_ms", "info"}."""
    settings, _ = resolve_emoca_settings(overrides)
    return emoca_client.check_service(settings)


def _read_bytes(image: ImageInput) -> bytes:
    """Chấp nhận bytes hoặc đối tượng file-like (vd Streamlit UploadedFile)."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    if hasattr(image, "getvalue"):
        return bytes(image.getvalue())
    if hasattr(image, "read"):
        return bytes(image.read())
    raise AnalyzeError("image_invalid", f"Kiểu đầu vào không hỗ trợ: {type(image).__name__}.")


def _validate_image(data: bytes, max_bytes: int) -> Image.Image:
    """Kiểm tra kích thước, định dạng, giải mã được; xoay theo EXIF. Trả ảnh RGB trong bộ nhớ."""
    if not data:
        raise AnalyzeError("image_invalid", "Tệp rỗng.")
    if len(data) > max_bytes:
        raise AnalyzeError(
            "image_too_large",
            f"Ảnh {len(data)} byte vượt giới hạn {max_bytes} byte.",
        )
    try:
        img = Image.open(io.BytesIO(data), formats=_ALLOWED_FORMATS)
        img.load()
    except Image.DecompressionBombError as exc:
        raise AnalyzeError("image_too_large", f"Ảnh quá nhiều điểm ảnh: {exc}") from exc
    except UnidentifiedImageError as exc:
        raise AnalyzeError(
            "image_invalid",
            f"Không nhận dạng được ảnh; chỉ nhận {', '.join(_ALLOWED_FORMATS)}.",
        ) from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise AnalyzeError("image_invalid", f"Không giải mã được ảnh: {exc}") from exc

    img = ImageOps.exif_transpose(img)  # toạ độ original_px tính sau khi xoay EXIF
    return img.convert("RGB")


def _error_response(err: AnalyzeError, width: int = 0, height: int = 0,
                    warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "error",
        "image": {"width": width, "height": height},
        "faces": [],
        "models_info": [],
        "timing_ms": {},
        "warnings": list(warnings or []),
        "error": {"code": err.code, "message": err.message, "retryable": err.retryable},
    }


def _emoca_call(img: Image.Image, max_faces: int, mock_case: MockCase, settings: dict[str, Any]) -> dict[str, Any]:
    """Gọi EMOCA qua `emoca_client`. `mock_case="partial"` giả lập emoca_svc không phản hồi,
    rồi đi qua CÙNG đường xử lý lỗi như lỗi thật."""
    if mock_case == "partial":
        return {"error": {"code": "service_unavailable",
                          "message": "emoca_svc không phản hồi (giả lập bằng mock_case='partial').",
                          "retryable": True}}
    return emoca_client.get_analysis(img, max_faces=max_faces, mode=settings["mode"], endpoint=settings)


def _models_info(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Danh sách mô hình đã chạy được (gộp theo id), lấy từ chính kết quả dự đoán."""
    seen: set[str] = set()
    info = []
    for p in predictions:
        if p.get("status") == "ok" and p["model_id"] not in seen:
            seen.add(p["model_id"])
            info.append({"id": p["model_id"], "display_name": p["display_name"], "dataset": p["dataset"],
                         "num_classes": len(p["labels"]), "enabled": True})
    return info


def _codes_are_real(emoca_model: dict[str, Any] | None) -> bool:
    """Mã 3DMM chỉ dùng được cho MA3D-Net khi do EMOCA THẬT sinh ra. EMOCA giả (mock trong web hoặc
    services/emoca_svc/mock_server.py) luôn báo tên mô hình bắt đầu bằng 'MOCK'."""
    return not str((emoca_model or {}).get("name", "")).upper().startswith("MOCK")


def _png_b64(img: Image.Image) -> dict[str, str]:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return {"format": "png", "b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def _model_input(img: Image.Image, face: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """ADR-020: ảnh vào POSTER/MA3D-Net căn chỉnh kiểu RAF-DB từ landmark original_px (web/face_align.py).
    BẬT (configs/align.yaml `enabled: true`, mẫu đã đo 2026-09-29). Bật mà căn chỉnh lỗi ⇒ (None, "align_failed: …") và
    dùng crop EMOCA. Thành công ⇒ ({"image": PIL, "transform": {from, to, matrix_3x3}, "size": [w, h]}, None)."""
    if face_align is None:
        return None, None
    tpl, err = face_align.load_template()
    if err:
        return None, f"align_failed: {err} — POSTER/MA3D-Net dùng crop EMOCA"
    if not tpl.get("enabled"):
        return None, None
    res, err = face_align.align_face(img, face["landmarks"]["original_px"], tpl)
    if err or not res:
        return None, f"align_failed: {err or 'không căn chỉnh được'} — POSTER/MA3D-Net dùng crop EMOCA"
    return {"image": res["image"],
            "transform": {"from": "original_px", "to": "model_input", "matrix_3x3": res["matrix_3x3"]},
            "size": list(res["size"])}, None


def _run_ai(img: Image.Image, max_faces: int, mock_case: MockCase, settings: dict[str, Any], engine_mode: str,
            model_ids: list[str] | None = None,
            progress_callback: Callable[[str, str, str | None], None] | None = None):
    """Ảnh → EMOCA (landmark/3DMM/crop) → engine (xác suất) → faces[] đúng `web_analyze`.

    Trả (faces, models_info, timing_ms, error, emoca_model). `error` khác None ⇒ kết quả một phần (partial):
    hoặc EMOCA lỗi dịch vụ (faces rỗng), hoặc có mô hình lỗi (mục lỗi nằm trong predictions).
    Lỗi do ảnh đầu vào (no_face…) được raise thành AnalyzeError → status "error".
    `emoca_model`: {"name","mode","upstream_commit"} do EMOCA báo (None nếu EMOCA lỗi/không chạy).
    """
    def report(module: str, state: str, detail: str | None = None) -> None:
        if progress_callback is not None:
            progress_callback(module, state, detail)

    report("emoca", "running", "Đang dò mặt, tạo crop và mã 3DMM…")
    t_emoca = time.perf_counter()
    emoca = _emoca_call(img, max_faces, mock_case, settings)
    emoca_roundtrip_ms = round((time.perf_counter() - t_emoca) * 1000, 2)
    if "error" in emoca:
        err = emoca["error"]
        report("emoca", "error", err.get("message", err.get("code", "EMOCA lỗi")))
        if err["code"] in _INPUT_ERROR_CODES:
            raise AnalyzeError(err["code"], err["message"], err.get("retryable", False))
        return [], [], {}, dict(err), None

    emoca_faces = emoca.get("faces") or []
    if not emoca_faces:
        report("emoca", "error", "EMOCA không trả khuôn mặt nào.")
        raise AnalyzeError("no_face", "EMOCA không trả khuôn mặt nào.")
    report("emoca", "done", f"Đã trả {min(len(emoca_faces), max_faces)} crop/mặt.")

    codes_real = _codes_are_real(emoca.get("model"))
    faces: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    ma3d_ms = 0.0
    engine_wall_ms = 0.0
    for face in emoca_faces[:max_faces]:
        crop = emoca_client.decode_crop(face)
        if crop is None:
            raise AnalyzeError("contract_mismatch",
                               "EMOCA không trả crop_image hợp lệ (ADR-006: mọi mô hình dùng chung crop này).")
        report("alignment", "running", f"Căn chỉnh mặt #{face['face_index']}…")
        model_input, align_warning = _model_input(img, face)
        report("alignment", "done" if model_input else "skipped",
               align_warning or ("Ảnh model_input đã sẵn sàng." if model_input else "Dùng crop224 của EMOCA."))
        t_engine = time.perf_counter()
        report("classifier", "running", f"Đang gọi module phân loại cho mặt #{face['face_index']}…")
        predictions = engine.predict(model_input["image"] if model_input else crop, face.get("codes"),
                                     models=model_ids, attention=True, mode=engine_mode,
                                     codes_real=codes_real)["predictions"]
        prediction_states = [p.get("status") == "ok" for p in predictions]
        classifier_state = "done" if prediction_states and all(prediction_states) else (
            "warn" if any(prediction_states) else "error")
        report("classifier", classifier_state,
               f"{sum(prediction_states)}/{len(predictions)} model trả kết quả.")
        engine_wall_ms += (time.perf_counter() - t_engine) * 1000
        all_predictions += predictions
        ma3d_ms += sum(p.get("latency_ms", 0.0) for p in predictions if p.get("status") == "ok")
        out: dict[str, Any] = {"face_index": face["face_index"], "bbox_original_px": face["bbox_original_px"]}
        # ADR-019: trường tuỳ chọn, chép nguyên từ emoca_analyze — để giao diện phủ đúng khung (crop224), không đoán.
        if face.get("crop_transform"):
            out["crop_transform"] = face["crop_transform"]
        out["crop_image"] = face["crop_image"]  # đã giải mã được ở trên
        if face["landmarks"].get("crop224"):
            out["landmarks_crop224"] = face["landmarks"]["crop224"]
        if model_input:  # ADR-020: ảnh engine THẬT SỰ nhận; attention phủ lên ảnh này
            out["model_input"] = {"image": _png_b64(model_input["image"]), "transform": model_input["transform"],
                                  "size": model_input["size"]}
        out.update({
            "landmarks": {"space": "original_px", "n_points": face["landmarks"]["n_points"],
                          "points": face["landmarks"]["original_px"]},
            "renders": face["renders"],
            "predictions": predictions,
            "warnings": list(face.get("warnings", [])) + ([align_warning] if align_warning else []),
        })
        faces.append(out)

    failed = [p for p in all_predictions if p.get("status") != "ok"]
    error = dict(failed[0]["error"]) if failed else None
    timing = {"emoca": emoca.get("timing_ms", {}).get("total", 0.0), "ma3d": round(ma3d_ms, 3)}
    if settings["mode"] == "real":
        timing["emoca_roundtrip"] = emoca_roundtrip_ms  # gồm cả mạng/tunnel — để đo độ trễ thật (M4-01)
    if engine_mode == "real":
        # `ma3d` chỉ cộng latency suy luận từng mô hình; lần đầu còn nạp torch/checkpoint (vài giây) — đo thấy 7.4 s tổng
        # nhưng `ma3d` chỉ 641 ms (docs/qa/E2E_ENGINE_REAL_UI.md, U4). `engine_total` là thời gian thật của lời gọi engine.
        timing["engine_total"] = round(engine_wall_ms, 2)
    return faces, _models_info(all_predictions), timing, error, emoca.get("model")


# model_ids=None keeps the configured POSTER + MA3D comparison; callers may request a subset.
def analyze(image: ImageInput, *, mock_case: MockCase = "ok", emoca: dict[str, Any] | None = None,
            engine_mode: str | None = None, model_ids: list[str] | None = None,
            max_faces: int | None = None,
            progress_callback: Callable[[str, str, str | None], None] | None = None) -> dict[str, Any]:
    """Phân tích một ảnh upload; luôn trả dict đúng `web_analyze.schema.json`, không raise.

    image: bytes/bytearray/memoryview hoặc file-like có `getvalue()`/`read()`
           (vd `st.file_uploader(...)` trả về UploadedFile).
    mock_case: "ok" → chạy bình thường; "partial" → giả lập emoca_svc không phản hồi
               (đi qua cùng đường xử lý lỗi như lỗi thật).
    emoca: ghi đè cài đặt EMOCA cho lần gọi này — {"mode", "url", "token", "timeout_s", "remote_consent"}.
           Không truyền ⇒ theo `resolve_emoca_settings`. EMOCA `real` trỏ ra ngoài máy mà thiếu
           `remote_consent=True` ⇒ status "error", code `consent_required`, KHÔNG gửi ảnh đi đâu.
    engine_mode: "mock" | "real" cho POSTER/MA3D-Net (None ⇒ theo `resolve_engine_mode`). `real` chạy trên CPU trong
           tiến trình này; mô hình nạp một lần (lần đầu vài giây). MA3D-Net thiếu thống kê ⇒ `codes_missing` ⇒ `partial`.
    """
    t0 = time.perf_counter()
    warnings: list[str] = []
    width = height = 0
    try:
        limits, cfg_warnings = _load_limits()
        warnings.extend(cfg_warnings)

        configured_face_cap = limits["max_faces"]
        if max_faces is None:
            # Preserve the existing one-face baseline unless a caller opts into N faces.
            limits["max_faces"] = min(1, configured_face_cap)
        else:
            if isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces < 1:
                raise AnalyzeError("contract_mismatch", "max_faces phải là số nguyên dương.")
            limits["max_faces"] = min(max_faces, configured_face_cap, MAX_FACES_PER_IMAGE)

        data = _read_bytes(image)
        img = _validate_image(data, limits["max_image_bytes"])
        width, height = img.size
        del data  # không giữ bản thô lâu hơn cần thiết

        if mock_case not in _MOCK_CASES:
            raise AnalyzeError("contract_mismatch", f"mock_case không hợp lệ: {mock_case!r}.")

        settings, emoca_warnings = resolve_emoca_settings(emoca, limits["max_image_bytes"], limits["max_image_side_px"])
        warnings.extend(emoca_warnings)
        remote = settings["mode"] == "real" and is_remote_url(settings["url"])
        if remote and mock_case == "ok" and not (emoca or {}).get("remote_consent"):
            raise AnalyzeError("consent_required",
                               f"EMOCA đang trỏ tới máy chủ từ xa ({_host_of(settings['url'])}); ảnh sẽ rời khỏi máy này. "
                               "Chưa có xác nhận gửi ảnh nên không gửi.", False)

        eng_mode, eng_warnings = resolve_engine_mode(engine_mode)
        warnings.extend(eng_warnings)
        with _AI_LOCK:
            faces, models_info, timing, ai_error, emoca_model = _run_ai(
                img, limits["max_faces"], mock_case, settings, eng_mode, model_ids=model_ids,
                progress_callback=progress_callback)
        del img

        if ai_error is not None and not faces:
            warnings.append("emoca_svc_unavailable")
        mock_parts = []
        if settings["mode"] == "mock":
            mock_parts.append("emoca_client (landmark, 3DMM, ảnh dựng)")
        if eng_mode == "mock":
            mock_parts.append("engine (xác suất POSTER/MA3D-Net)")
        elif faces:
            ran = sorted({p["model_id"] for f in faces for p in f["predictions"] if p.get("status") == "ok"})
            warnings.append("engine_real: " + (", ".join(ran) if ran else "không mô hình nào chạy được")
                            + " chạy thật trên CPU trong tiến trình web")
        if mock_parts:
            warnings.append("mock_ai: " + " và ".join(mock_parts) + " chạy chế độ mock — dữ liệu giả, không phải suy luận thật")
        if settings["mode"] == "real" and emoca_model:
            warnings.append(f"emoca_real: {emoca_model.get('name', '?')} ({emoca_model.get('mode', '?')}), "
                            f"upstream {emoca_model.get('upstream_commit', '?')}"
                            + (f", chạy trên máy chủ từ xa {_host_of(settings['url'])}" if remote else ""))
        # `image` là kích thước thật của ảnh đã xoay EXIF (hệ `original_px` của landmark).
        return {
            "contract_version": CONTRACT_VERSION,
            "status": "ok" if ai_error is None else "partial",
            "image": {"width": width, "height": height},
            "faces": faces,
            "models_info": models_info,
            "timing_ms": {**timing, "total": round((time.perf_counter() - t0) * 1000, 2)},
            "warnings": warnings,
            "error": ai_error,
        }
    except AnalyzeError as err:
        return _error_response(err, width, height, warnings)
    except Exception as exc:  # lỗi không lường trước: vẫn trả dict hợp lệ, không làm hỏng trang
        err = AnalyzeError("internal_error", f"{type(exc).__name__}: {exc}", retryable=True)
        return _error_response(err, width, height, warnings)


def analyze_multiface(
    image: Image.Image,
    *,
    emoca: dict[str, Any] | None = None,
    engine_mode: str | None = None,
    max_faces: int = MAX_FACES_PER_IMAGE,
    progress_callback: Callable[[str], None] | None = None,
    views: Any = None,
) -> dict[str, Any]:
    """Run MA3D serially on original-image crops without changing web_analyze."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces < 1:
        raise ValueError("max_faces must be a positive integer")
    face_limit = min(max_faces, max_faces_limit(), MAX_FACES_PER_IMAGE)

    try:
        from . import predata_experiment
    except ImportError:  # Streamlit launched with web/ on sys.path
        import predata_experiment

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    if views is None:
        views = predata_experiment.build_multiface_views(
            image, max_faces=face_limit, progress_callback=report
        )
    elif len(getattr(views, "faces", ())) > face_limit:
        views.faces = views.faces[:face_limit]

    def run_ma3d(face_image: Image.Image) -> dict[str, Any]:
        buffer = io.BytesIO()
        face_image.convert("RGB").save(buffer, format="PNG")
        return analyze(buffer.getvalue(), emoca=emoca, engine_mode=engine_mode,
                       model_ids=["ma3d_rafdb"], max_faces=1)

    rows: list[dict[str, Any]] = []
    if len(views.faces) == 1:
        return {"kind": "single", "views": views, "rows": rows}

    if views.faces:
        for index, item in enumerate(views.faces, start=1):
            report(f"Đang phân tích mặt {index}/{len(views.faces)}…")
            model_image = item.detection.cropped_face
            roll_info = None
            try:
                roll_enabled, roll_threshold = roll_correction_settings()
                if roll_enabled:
                    model_image, roll_info = _correct_roll_crop(model_image, roll_threshold)
                    if roll_info and roll_info.get("error"):
                        views.errors.append(f"Face {item.face_id} 3DDFA roll: {roll_info['error']}")
            except Exception as exc:
                roll_info = {"error": f"{type(exc).__name__}: {exc}"}
                views.errors.append(f"Face {item.face_id} 3DDFA roll: {roll_info['error']}")
            try:
                prediction_result = run_ma3d(model_image)
            except Exception as exc:
                prediction_result = {"status": "error", "faces": [],
                                     "error": {"code": "internal_error",
                                               "message": f"{type(exc).__name__}: {exc}"}}
                views.errors.append(f"Face {item.face_id} MA3D: {type(exc).__name__}: {exc}")
            rows.append({
                "face_id": item.face_id,
                "reference": item.reference_image,
                "processed": model_image,
                "roll_info": roll_info,
                "result": prediction_result,
            })
        return {"kind": "faces", "views": views, "rows": rows}

    return {"kind": "baseline", "views": views, "rows": rows}


def prepare_multiface_views(
    image: Image.Image,
    *,
    max_faces: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Any:
    """Run the face-preparation/detection pass before choosing single or multi inference."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if max_faces is not None and (isinstance(max_faces, bool) or not isinstance(max_faces, int) or max_faces < 1):
        raise ValueError("max_faces must be a positive integer")
    face_limit = max_faces_limit() if max_faces is None else min(max_faces, max_faces_limit())
    try:
        from . import predata_experiment
    except ImportError:
        import predata_experiment
    return predata_experiment.build_multiface_views(
        image, max_faces=face_limit, progress_callback=progress_callback
    )


def run_frontalization_experiment(views: Any, *, emoca: dict[str, Any] | None = None,
                                  engine_mode: str | None = None,
                                  progress_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Thin package/script-compatible entry point for the opt-in M8 experiment."""
    try:
        from .frontalize_experiment import run_frontalization_experiment as run
    except ImportError:  # Streamlit launched with web/ on sys.path
        from frontalize_experiment import run_frontalization_experiment as run
    return run(views, emoca=emoca, engine_mode=engine_mode or "mock",
               progress_callback=progress_callback)
