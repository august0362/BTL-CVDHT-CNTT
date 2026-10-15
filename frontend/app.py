"""Giao diện Streamlit của web demo MA3D-Net (FRONTEND, ADR-012, ADR-032).

Chạy:  streamlit run frontend/app.py

Chỉ HIỂN THỊ dữ liệu `analyze(image)` của backend/web/orchestrator.py trả về (hình dạng
contracts/web_analyze.schema.json). Không có logic mô hình, không tự tính/đổi xác suất, không nhúng cứng nhãn hay tên
mô hình. Ảnh người dùng chỉ nằm trong bộ nhớ phiên (st.session_state), không ghi xuống đĩa.

Giao diện tối giản (ADR-032): tiêu đề, ô tải ảnh và khối "Kết quả trực quan" — mỗi khuôn mặt một hàng ảnh
Input | (Rotated Face) | Facial Landmark | 3DMM | POSTER | MA3D-Net. EMOCA và các mô hình luôn chạy thật; không có chế độ giả lập trên giao diện.
Địa chỉ và token EMOCA do orchestrator lấy từ biến môi trường hoặc backend/configs/services.yaml.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import html
import io
import re
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from background_jobs import session_jobs
from primary_pipeline import run_primary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
BACKEND_WEB = BACKEND_ROOT / "web"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_WEB))

# Không nạp được orchestrator (hoặc emoca_client/engine mà nó import) ⇒ báo lỗi thật trên trang,
# KHÔNG lặng lẽ thay bằng fixture contracts/ (trước đây che lỗi thành "mock", nợ Session 04–05).
try:
    from orchestrator import (
        analyze, analyze_multiface, max_faces_limit, prepare_multiface_views,
        roll_correct_upload, roll_correction_settings,
    )

    ORCHESTRATOR_ERROR = None
except ImportError as exc:
    analyze = analyze_multiface = max_faces_limit = prepare_multiface_views = None
    roll_correct_upload = roll_correction_settings = None
    ORCHESTRATOR_ERROR = f"{type(exc).__name__}: {exc}"


# --- Hằng số hiển thị (cần chuyển sang configs/ khi CONFIG tạo registry) ---
ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp"]
STYLE_CSS = Path(__file__).resolve().parent / "style.css"
LABELS_YAML = BACKEND_ROOT / "configs" / "labels.yaml"  # tên tiếng Việt (+ màu nếu CONFIG đã điền) của nhãn
SERVICES_YAML = BACKEND_ROOT / "configs" / "services.yaml"  # limits.max_image_bytes → dòng hướng dẫn trong ô tải ảnh
UPLOAD_LIMIT_FALLBACK = 10 * 1024 * 1024  # = _FALLBACK_LIMITS của backend/web/orchestrator.py khi thiếu config
MODELS_YAML = BACKEND_ROOT / "configs" / "models.yaml"  # registry: tên hiển thị/kiến trúc cho mô hình lỗi
UI_YAML = BACKEND_ROOT / "configs" / "ui.yaml"  # show_probability: hiện % cạnh nhãn dự đoán (mặc định tắt)
EMOCA_OVERRIDES: dict = {"mode": "real"}  # endpoint/token vẫn do orchestrator lấy từ env/config
ENGINE_MODE = "real"             # luôn chạy mô hình thật; UI không có lựa chọn mock
# Thông báo theo mã lỗi (CONTRACTS_ADR.md, bảng lỗi) — kèm việc cần làm tiếp.
ERROR_HINTS = {
    "image_invalid": "Ảnh không đọc được. Hãy thử tệp JPEG/PNG/WebP khác.",
    "image_too_large": "Ảnh quá lớn. Hãy thu nhỏ ảnh rồi tải lại.",
    "no_face": "Không tìm thấy khuôn mặt. Hãy dùng ảnh chính diện, đủ sáng, mặt không bị che.",
    "service_unavailable": "Một dịch vụ (EMOCA/mô hình) chưa sẵn sàng. Kiểm tra emoca_svc đã chạy chưa rồi thử lại.",
    "timeout": "Xử lý quá thời gian cho phép. Thử lại, hoặc dùng ảnh nhỏ hơn.",
    "model_unavailable": "Mô hình chưa nạp được (thiếu checkpoint?). Xem log của tiến trình.",
    "codes_missing": "MA3D-Net cần mã 3DMM do EMOCA THẬT sinh ra (EMOCA giả lập không dùng được) VÀ file thống kê chuẩn hoá "
                     "(*_emoca_stats.npz); thiếu một trong hai thì không chạy (cố ý không điền 0 hay dùng mã giả). "
                     "Đọc dòng lỗi phía trên để biết thiếu cái nào. POSTER không bị ảnh hưởng.",
    "contract_mismatch": "Dữ liệu trả về lệch contract. Báo cho nhóm BACKEND/GENERAL.",
    "consent_required": "Máy chủ EMOCA không nằm trên máy này nên ảnh không được gửi đi. "
                        "Đặt EMOCA_URL (hoặc emoca_svc.url) về máy chủ cục bộ, ví dụ http://127.0.0.1:8765.",
}

esc = html.escape


def md(markup: str) -> None:
    """Chèn HTML của chính ứng dụng (mọi chuỗi từ dữ liệu đã qua html.escape). Không xuống dòng trống
    để bộ phân tích Markdown coi cả khối là HTML thô."""
    st.markdown(markup, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- config (đọc, có cache)
@st.cache_data(show_spinner=False)
def _label_meta(signature: tuple):
    """Đọc labels.yaml để hiện tên tiếng Việt; lỗi đọc trả metadata rỗng như trước."""
    if not LABELS_YAML.is_file():
        return {}, [f"không thấy {LABELS_YAML.name}"]
    try:
        import yaml

        raw = yaml.safe_load(LABELS_YAML.read_text(encoding="utf-8")) or {}
    except ImportError:
        return {}, ["thiếu thư viện PyYAML"]
    except (OSError, yaml.YAMLError) as exc:
        return {}, [f"{LABELS_YAML.name}: {exc}"]
    meta, problems = {}, []
    for name, entry in (raw.get("labels") or {}).items():
        entry = entry or {}
        display = entry.get("display") or {}
        color = entry.get("color")
        if color is not None and not (isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color)):
            problems.append(f"nhãn {name}: color không phải #rrggbb ({color!r}), bỏ qua")
            color = None
        meta[str(name).strip().lower()] = {"vi": display.get("vi"), "en": display.get("en"), "color": color}
    return meta, problems


@st.cache_data(show_spinner=False)
def _model_registry(signature: tuple) -> dict:
    """{model_id: {display_name, arch}} từ configs/models.yaml. Lỗi đọc ⇒ {} (giao diện rơi về model_id)."""
    try:
        import yaml

        data = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — thiếu/hỏng registry chỉ làm mất tên đẹp, không được làm sập trang
        return {}
    return {m["id"]: {"display_name": m.get("display_name"), "arch": m.get("arch")}
            for m in data.get("models") or [] if isinstance(m, dict) and m.get("id")}


def model_registry() -> dict:
    return _model_registry(_signature(MODELS_YAML))


@st.cache_data(show_spinner=False)
def _show_probability(signature: tuple) -> bool:
    """configs/ui.yaml `show_probability`; thiếu/hỏng file ⇒ False (mặc định ẩn %)."""
    try:
        import yaml

        return (yaml.safe_load(UI_YAML.read_text(encoding="utf-8")) or {}).get("show_probability") is True
    except Exception:  # noqa: BLE001 — chỉ là tuỳ chọn hiển thị, không được làm sập trang
        return False


def show_probability() -> bool:
    return _show_probability(_signature(UI_YAML))


@st.cache_data(show_spinner=False)
def _upload_limit_bytes(signature: tuple) -> int:
    """limits.max_image_bytes của configs/services.yaml (giới hạn mà bộ điều phối áp); lỗi đọc ⇒ giá trị dự phòng."""
    try:
        import yaml

        v = int(((yaml.safe_load(SERVICES_YAML.read_text(encoding="utf-8")) or {}).get("limits") or {})["max_image_bytes"])
        return v if v > 0 else UPLOAD_LIMIT_FALLBACK
    except Exception:  # noqa: BLE001 — chỉ là dòng hướng dẫn, không được làm sập trang
        return UPLOAD_LIMIT_FALLBACK


def upload_hint_css() -> str:
    """Biến CSS cho dòng hướng dẫn của ô tải ảnh trong style.css."""
    mb = _upload_limit_bytes(_signature(SERVICES_YAML)) / (1024 * 1024)
    hint = f"JPEG · PNG · WebP · tối đa {mb:g} MB · ảnh chỉ nằm trong bộ nhớ, không lưu lại"
    return ':root { --mx-upload-hint: "' + hint.replace("\\", "").replace('"', "") + '"; }'


def _signature(folder: Path) -> tuple:
    if not folder.exists():
        return ()
    items = [folder] if folder.is_file() else sorted(folder.iterdir())
    return tuple((p.name, p.stat().st_mtime_ns) for p in items if p.is_file())


# ----------------------------------------------------------------------------- helpers
def load_upload(data: bytes) -> Image.Image | None:
    """Giải mã ảnh chỉ để kiểm tra hợp lệ; xoay theo EXIF để khớp hệ original_px."""
    try:
        img = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def decode_render(render) -> Image.Image | None:
    if not isinstance(render, dict) or not render.get("b64"):
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(render["b64"], validate=True))).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
        return None


def label_text(label: str, meta: dict) -> str:
    """Tên hiển thị (tiếng Việt nếu configs/labels.yaml có), nhãn thô nếu không."""
    return (meta.get(str(label).strip().lower()) or {}).get("vi") or str(label)


# ----------------------------------------------------------------------------- kết quả trực quan (slide 19 bài báo)
# Mỗi khuôn mặt một hàng ảnh cùng cỡ: Input | (Rotated Face) | Facial Landmark | 3DMM | POSTER | MA3D-Net (ours).
# Input = ảnh tải lên (một mặt) hoặc YOLO crop từ ảnh gốc (nhiều mặt); "Rotated Face" chỉ có khi có mặt được sửa roll
# (ADR-029). Attention (ADR-013, lưới 7×7 trên ảnh 224 mà engine nhận) phủ lên `face["model_input"]["image"]` — ảnh căn
# chỉnh kiểu RAF-DB mà engine THẬT SỰ nhận (ADR-020); thiếu thì phủ lên `face["crop_image"]` (crop EMOCA, ADR-019).
# Landmark crop từ `face["landmarks_crop224"]` (ADR-019). Mọi trường đều TUỲ CHỌN; thiếu ⇒ ô ghi lý do, không đoán khung.
VIS_TILE = 224                   # cạnh mỗi ô (px) = cạnh crop224
VIS_ALPHA = 0.45                 # độ phủ của bản đồ attention lên crop
VIS_LANDMARK_RGB = (230, 30, 30)  # chấm landmark đỏ như slide 19; viền trắng để thấy trên nền tối
VIS_INPUT, VIS_ROTATED, VIS_LANDMARK, VIS_GEOMETRY = "Input", "Rotated Face", "Facial Landmark", "3DMM"


def _turbo_lut() -> tuple[list[int], list[int], list[int]]:
    """Bảng màu Turbo (Google AI, xấp xỉ đa thức bậc 5) → 3 bảng tra 256 mức cho Image.point."""
    r_l, g_l, b_l = [], [], []
    for i in range(256):
        x = i / 255
        r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
        g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
        b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
        for lst, c in ((r_l, r), (g_l, g), (b_l, b)):
            lst.append(max(0, min(255, round(c * 255))))
    return r_l, g_l, b_l


TURBO_LUT = _turbo_lut()


def square_tile(img: Image.Image, size: int = VIS_TILE) -> Image.Image:
    """Đưa ảnh về ô vuông size×size, giữ tỉ lệ (viền đen) — để các ô cùng cỡ."""
    return ImageOps.pad(img.convert("RGB"), (size, size), method=Image.BICUBIC, color=(0, 0, 0))


def attention_overlay(crop: Image.Image, grid_map, alpha: float = VIS_ALPHA) -> tuple[Image.Image | None, str]:
    """Phủ attention (grid_map hệ crop224, vd 7×7, 0..1) lên crop: nội suy bicubic lên VIS_TILE, tô Turbo, trộn alpha.
    Trả (ảnh VIS_TILE×VIS_TILE, "") hoặc (None, lý do)."""
    if not isinstance(grid_map, dict):
        return None, "No attention returned by the model"
    if grid_map.get("space") != "crop224":
        return None, f"Attention is in {grid_map.get('space')!r}, not crop224 — not overlaid"
    grid, values = grid_map.get("grid") or [], grid_map.get("values") or []
    if len(grid) != 2 or len(values) != grid[0] * grid[1]:
        return None, "Attention has the wrong shape"
    rows, cols = grid
    small = Image.new("L", (cols, rows))
    small.putdata([round(min(max(float(v), 0.0), 1.0) * 255) for v in values])
    heat = small.resize((VIS_TILE, VIS_TILE), Image.BICUBIC)
    colored = Image.merge("RGB", [heat.point(lut) for lut in TURBO_LUT])
    base = crop.convert("RGB").resize((VIS_TILE, VIS_TILE), Image.BICUBIC)
    return Image.blend(base, colored, alpha), ""


def landmark_tile(img: Image.Image, points, ref_size: int | None = None) -> Image.Image:
    """Chấm đỏ viền trắng tại `points` (toạ độ theo ảnh `img`, hoặc theo khung ref_size×ref_size nếu có), rồi về ô vuông."""
    base = img.convert("RGB")
    if ref_size:  # điểm crop224 trên crop đã phóng về VIS_TILE
        base = base.resize((VIS_TILE, VIS_TILE), Image.BICUBIC)
        k = VIS_TILE / ref_size
        points = [(x * k, y * k) for x, y in points]
    draw = ImageDraw.Draw(base)
    r = max(1.6, max(base.size) / VIS_TILE * 1.8)
    for x, y in points:
        draw.ellipse([x - r - 0.8, y - r - 0.8, x + r + 0.8, y + r + 0.8], fill=(255, 255, 255))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=VIS_LANDMARK_RGB)
    return square_tile(base)


def _tile(title: str, img: Image.Image | None, missing: str = "", top: str = "") -> dict:
    """Một ô: {title, top (nhãn dự đoán ở trên), img (PIL|None), note (lý do khi thiếu ảnh)}."""
    return {"title": title, "top": top, "img": img, "note": "" if img is not None else missing}


def analysis_tiles(face: dict, image: Image.Image | None, result: dict, meta: dict,
                   show_prob: bool = False) -> list[dict]:
    """Facial Landmark | 3DMM | một ô cho mỗi mô hình, từ một mặt của `result`. `image` = ảnh đã gửi cho analyze();
    `show_prob` = thêm % top-1 cạnh nhãn dự đoán (configs/ui.yaml `show_probability`)."""
    size = result.get("image") or {}
    same_image = image is not None and image.size == (int(size.get("width", 0)), int(size.get("height", 0)))
    crop = decode_render(face.get("crop_image"))
    crop_lm = face.get("landmarks_crop224")
    model_in = decode_render((face.get("model_input") or {}).get("image"))
    att_base = model_in if model_in is not None else crop   # ảnh engine nhận (ADR-020), không có thì crop EMOCA

    lm = face.get("landmarks") or {}
    if crop is not None and isinstance(crop_lm, list) and crop_lm:
        tiles = [_tile(VIS_LANDMARK, landmark_tile(crop, crop_lm, ref_size=crop.width))]
    elif same_image and lm.get("space") == "original_px" and lm.get("points"):
        tiles = [_tile(VIS_LANDMARK, landmark_tile(image, lm["points"]))]
    else:
        tiles = [_tile(VIS_LANDMARK, None, "No landmarks to draw")]

    renders = face.get("renders") or {}
    geo = decode_render(renders.get("geometry_detail")) or decode_render(renders.get("geometry_coarse"))
    tiles.append(_tile(VIS_GEOMETRY, square_tile(geo) if geo is not None else None, "No 3D reconstruction from EMOCA"))

    reg = model_registry()
    for pred in face.get("predictions") or []:
        info = reg.get(pred.get("model_id"), {})
        title = pred.get("display_name") or info.get("display_name") or str(pred.get("model_id"))
        if str(info.get("arch", "")).upper() == "MA3D":
            title += " (ours)"
        if pred.get("status") != "ok":
            err = pred.get("error") or {}
            tiles.append(_tile(title, None, f"Model error [{err.get('code', pred.get('status'))}] {err.get('message', '')}".strip()))
            continue
        top1 = pred.get("top1") or {}
        top = label_text(top1.get("label", "?"), meta)
        if show_prob:
            top += f" · {float(top1.get('prob', 0)):.0%}"
        if att_base is None:
            tiles.append(_tile(title, None, "No model input image — attention not overlaid", top))
        else:
            overlay, reason = attention_overlay(att_base, pred.get("attention"))
            tiles.append(_tile(title, overlay, reason, top))
    return tiles


def visual_row(source: Image.Image | None, rotated: Image.Image | None, show_rotation: bool,
               face: dict, image: Image.Image | None, result: dict, meta: dict,
               show_prob: bool = False) -> list[dict]:
    """Một hàng của khối Kết quả trực quan. `source` = ảnh đầu vào ban đầu; `rotated` = ảnh đã sửa roll (None nếu không
    xoay); `show_rotation` = hàng có cột "Rotated Face" (các hàng của cùng một ảnh luôn cùng số cột)."""
    tiles = [_tile(VIS_INPUT, square_tile(source) if source is not None else None, "No input image")]
    if show_rotation:
        tiles.append(_tile(VIS_ROTATED, square_tile(rotated) if rotated is not None else None, "No rotation needed"))
    return tiles + analysis_tiles(face, image, result, meta, show_prob)


def render_visual(rows: list[list[dict]]) -> None:
    with st.container(border=True):
        section("Kết quả trực quan")
        for index, tiles in enumerate(rows):
            if index:
                st.divider()
            for col, t in zip(st.columns(len(tiles)), tiles):  # màn hẹp: Streamlit tự xếp dọc các cột
                with col:
                    md(f'<div class="mx-vis-top">{esc(t["top"]) or "&nbsp;"}</div>')
                    if t["img"] is not None:
                        st.image(t["img"], width="stretch")
                    else:
                        md(f'<div class="mx-vis-miss" role="img" aria-label="{esc(t["title"])}: {esc(t["note"])}">'
                           f'<span>{esc(t["note"])}</span></div>')
                    md(f'<div class="mx-vis-cap">{esc(t["title"])}</div>')


def _detected(result) -> bool:
    """Mặt được EMOCA nhận (ok/partial và có `faces`); mặt lỗi bị bỏ khỏi danh sách nhiều mặt."""
    return isinstance(result, dict) and result.get("status") in ("ok", "partial") and bool(result.get("faces"))


def render_single(value: dict, meta: dict) -> None:
    """Một mặt (hoặc YOLO không thấy mặt): ảnh tải lên → (xoay roll) → analyze()."""
    result = value["result"]
    status = result.get("status", "error")
    render_error(result.get("error"), status)
    faces = result.get("faces") or []
    if status == "error" or not faces:
        if status == "ok":
            st.error(f"**[no_face]** {ERROR_HINTS['no_face']}")
        return
    rotated = value["corrected"] if (value.get("roll_info") or {}).get("rotation_deg") else None
    render_visual([visual_row(value["image"], rotated, rotated is not None, faces[0], value["corrected"], result, meta,
                              show_probability())])


def render_multiface(value: dict, meta: dict) -> None:
    """Nhiều mặt: một hàng cho mỗi mặt EMOCA nhận được, theo thứ tự YOLO (trên xuống, trái sang phải)."""
    crops = {item.face_id: item.detection.cropped_face for item in getattr(value["views"], "faces", None) or []}
    rows = [row for row in (value["multi"] or {}).get("rows") or [] if _detected(row.get("result"))]
    if not rows:
        st.error(f"**[no_face]** {ERROR_HINTS['no_face']}")
        return
    show_rotation = any((row.get("roll_info") or {}).get("applied") for row in rows)
    show_prob = show_probability()
    tiles = []
    for row in rows:
        result = row["result"]
        rotated = row.get("processed") if (row.get("roll_info") or {}).get("applied") else None
        source = crops.get(row.get("face_id"), row.get("processed"))
        tiles.append(visual_row(source, rotated, show_rotation, result["faces"][0], row.get("processed"), result, meta,
                                show_prob))
    render_visual(tiles)


# ----------------------------------------------------------------------------- khối giao diện
def section(title: str) -> None:
    md(f'<div class="mx-sec"><div class="mx-sec-t">{esc(title)}</div></div>')


def page_header() -> None:
    md('<div class="mx-topbar"><h1 class="mx-title">MA3D-Net</h1></div>')


def render_error(err: dict | None, status: str):
    if not err:
        return
    code = err.get("code", "unknown")
    msg = f"**[{code}]** {err.get('message', '')}\n\n{ERROR_HINTS.get(code, '')}"
    (st.warning if status == "partial" else st.error)(msg)
    if err.get("retryable"):
        st.caption("Lỗi này có thể thử lại — bấm **Phân tích lại**.")


# ----------------------------------------------------------------------------- luồng chính (chạy nền)
_PRIMARY_RESULTS = ("result", "multiface_result", "_primary_applied_job")


def _clear_primary_results() -> None:
    for key in _PRIMARY_RESULTS:
        st.session_state.pop(key, None)
    session_jobs(st.session_state).discard("main")


def _save_primary_upload(name: str, data: bytes) -> None:
    old = st.session_state.get("_primary_upload")
    if not old or old["data"] != data:
        _clear_primary_results()
    st.session_state["_primary_upload"] = {"name": name, "data": data}


def _primary_upload_changed(widget_key: str) -> None:
    upload = st.session_state.get(widget_key)
    if upload is None:
        st.session_state.pop("_primary_upload", None)
        _clear_primary_results()
    else:
        _save_primary_upload(upload.name, upload.getvalue())


def _remove_primary_upload() -> None:
    st.session_state.pop("_primary_upload", None)
    st.session_state["_primary_picker_revision"] = st.session_state.get("_primary_picker_revision", 0) + 1
    _clear_primary_results()


def _render_primary_job(request_key: tuple, meta: dict, polling: bool) -> None:
    job = session_jobs(st.session_state).get("main")
    if job is None or job.key != request_key:
        return
    snapshot = job.snapshot()
    if snapshot.active:
        st.status(snapshot.message, state="running", expanded=False)
        return
    if snapshot.state == "error":
        if polling:
            st.rerun()
        st.error(snapshot.error or "Không hoàn tất phân tích ảnh.")
        return
    value = snapshot.result
    fresh = st.session_state.get("_primary_applied_job") != snapshot.id
    if fresh:
        if value["multi"] is not None:
            st.session_state["multiface_result"] = value["multi"]
            st.session_state.pop("result", None)
        else:
            st.session_state["result"] = (request_key, value["result"])
            st.session_state.pop("multiface_result", None)
        st.session_state["_primary_applied_job"] = snapshot.id
    if fresh or polling:
        st.rerun()  # Dừng polling và vẽ lại cả trang với kết quả vừa hoàn tất.
    if not getattr(value["views"], "detection_complete", True):
        st.warning("YOLO không hoàn tất; đã phân tích trên ảnh gốc.")
    if value["multi"] is not None:
        render_multiface(value, meta)
    else:
        render_single(value, meta)


# ----------------------------------------------------------------------------- trang
def main():
    st.set_page_config(page_title="MA3D-Net", page_icon="🎭", layout="wide", initial_sidebar_state="collapsed")

    meta, _ = _label_meta(_signature(LABELS_YAML))
    md(f"<style>{STYLE_CSS.read_text(encoding='utf-8')}\n{upload_hint_css()}</style>")

    page_header()
    if ORCHESTRATOR_ERROR:
        st.error(f"Không nạp được bộ điều phối: {ORCHESTRATOR_ERROR}")
        return

    with st.container(border=True):
        section("Tải ảnh khuôn mặt")
        picker_key = f"primary_upload_picker_{st.session_state.get('_primary_picker_revision', 0)}"
        up = st.file_uploader("Chọn ảnh khuôn mặt (JPEG/PNG/WebP)", type=ACCEPTED_TYPES,
                              label_visibility="collapsed", key=picker_key,
                              on_change=_primary_upload_changed, args=(picker_key,))
        if up is not None:
            _save_primary_upload(up.name, up.getvalue())
        saved = st.session_state.get("_primary_upload")
        if saved is None:
            return
        data = saved["data"]
        if load_upload(data) is None:
            st.error(f"**[image_invalid]** {ERROR_HINTS['image_invalid']}")
            return
        existing = session_jobs(st.session_state).get("main")
        active = bool(existing and existing.snapshot().active)
        retry_col, remove_col, _ = st.columns([1, 1, 4])
        rerun = retry_col.button("Phân tích lại", type="primary", key="primary_rerun", disabled=active)
        remove_col.button("Xóa ảnh", key="primary_remove", on_click=_remove_primary_upload)

    engine_mode = ENGINE_MODE
    face_limit = max_faces_limit()
    request_key = (hashlib.sha256(data).hexdigest(), engine_mode, face_limit, roll_correction_settings(),
                   "poster+ma3d-v2")
    if rerun:
        _clear_primary_results()

    def work(report):
        return run_primary(data, prepare_fn=prepare_multiface_views, roll_fn=roll_correct_upload,
                           analyze_fn=analyze, multiface_fn=analyze_multiface, emoca=dict(EMOCA_OVERRIDES),
                           engine_mode=engine_mode, face_limit=face_limit, report=report)

    job = session_jobs(st.session_state).submit("main", request_key, work, force=rerun)
    polling = job.snapshot().active
    st.fragment(run_every=.6 if polling else None)(_render_primary_job)(request_key, meta, polling)


if __name__ == "__main__":
    main()
