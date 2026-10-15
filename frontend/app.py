"""Giao diện Streamlit của web demo MA3D-Net (FRONTEND, ADR-012).

Chạy:  streamlit run frontend/app.py

Chỉ HIỂN THỊ dữ liệu `analyze(image)` của web/orchestrator.py trả về (hình dạng
contracts/web_analyze.schema.json, contract_version 0.1.0). Không có logic mô hình,
không tự tính/đổi xác suất, không nhúng cứng nhãn hay tên mô hình.
Ảnh người dùng chỉ nằm trong bộ nhớ phiên (st.session_state), không ghi xuống đĩa.

Thiết kế (frontend/theme.py + frontend/theme.css): bảng màu lấy từ ảnh trong frontend/assets/theme, có sáng/tối và
tương phản cao, bố cục thẻ kiểu Canva, bảng Cài đặt ở đầu trang. Không phụ thuộc màu nền/chế độ tối
của trình duyệt (color-scheme đặt tường minh), không tải font/CDN bên ngoài.
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
from types import SimpleNamespace

import streamlit as st
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

import theme as mx  # frontend/theme.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
BACKEND_WEB = BACKEND_ROOT / "web"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_WEB))
FACE_DETECTION_YAML = BACKEND_ROOT / "configs" / "face_detection.yaml"

# Không nạp được orchestrator (hoặc emoca_client/engine mà nó import) ⇒ báo lỗi thật trên trang,
# KHÔNG lặng lẽ thay bằng fixture contracts/ (trước đây che lỗi thành "mock", nợ Session 04–05).
try:
    from orchestrator import (
        analyze, analyze_multiface, check_emoca, emoca_status, max_faces_limit,
        prepare_multiface_views, resolve_engine_mode, roll_correct_upload, roll_correction_settings,
    )

    ORCHESTRATOR_ERROR = None
except ImportError as exc:
    analyze = analyze_multiface = check_emoca = emoca_status = max_faces_limit = None
    prepare_multiface_views = resolve_engine_mode = None
    roll_correct_upload = roll_correction_settings = None
    ORCHESTRATOR_ERROR = f"{type(exc).__name__}: {exc}"


# --- Hằng số hiển thị (cần chuyển sang configs/ khi CONFIG tạo registry) ---
ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp"]
LANDMARK_RADIUS_RATIO = 0.006   # bán kính điểm landmark theo cạnh dài của ảnh
BBOX_WIDTH_RATIO = 0.004
THEME_DIR = Path(__file__).resolve().parent / "assets" / "theme"  # ảnh bảng màu (mỗi ảnh = 4 màu)
LABELS_YAML = BACKEND_ROOT / "configs" / "labels.yaml"  # tên tiếng Việt (+ màu nếu CONFIG đã điền) của nhãn
SERVICES_YAML = BACKEND_ROOT / "configs" / "services.yaml"  # limits.max_image_bytes → dòng hướng dẫn trong ô tải ảnh
UPLOAD_LIMIT_FALLBACK = 10 * 1024 * 1024  # = _FALLBACK_LIMITS của backend/web/orchestrator.py khi thiếu config
MODELS_YAML = BACKEND_ROOT / "configs" / "models.yaml"  # registry: tên hiển thị/dataset cho mô hình lỗi

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
    "consent_required": "Ảnh chưa được gửi đi. Bật ô \"Tôi đồng ý gửi ảnh…\" ở bảng Cài đặt (mục Kết nối EMOCA), "
                        "hoặc chuyển về Giả lập.",
}

# Icon nét mảnh (kiểu Feather, MIT) — nhúng trực tiếp, không tải icon từ ngoài.
_ICONS = {
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>',
    "box": '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.27 6.96 12 12.01l8.73-5.05M12 22.08V12"/>',
    "columns": '<rect x="3" y="3" width="7" height="18" rx="1.5"/><rect x="14" y="3" width="7" height="18" rx="1.5"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"/>',
    "flask": '<path d="M9 3h6M10 3v6.5L4.5 19a1.5 1.5 0 0 0 1.3 2.2h12.4a1.5 1.5 0 0 0 1.3-2.2L14 9.5V3"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
}


def svg(name: str) -> str:
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


esc = html.escape


def md(markup: str) -> None:
    """Chèn HTML của chính ứng dụng (mọi chuỗi từ dữ liệu đã qua html.escape). Không xuống dòng trống
    để bộ phân tích Markdown coi cả khối là HTML thô."""
    st.markdown(markup, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cài đặt (đầu trang)
DEFAULT_PREFS = {"pref_mode": "light", "pref_level": "normal", "pref_font": "M", "pref_still": False,
                 "pref_lm": True, "pref_bb": True, "pref_att": True}
QP_KEYS = {"pref_theme": "t", "pref_mode": "m", "pref_level": "c", "pref_font": "f", "pref_still": "rm",
           "pref_lm": "lm", "pref_bb": "bb", "pref_att": "at"}
CHOICES = {"pref_mode": ("light", "dark"), "pref_level": ("normal", "high"), "pref_font": ("S", "M", "L")}
CHOICE_LABELS = {
    "pref_mode": {"light": "Sáng", "dark": "Tối"},
    "pref_level": {"normal": "Chuẩn", "high": "Cao"},
    "pref_font": {"S": "Nhỏ", "M": "Vừa", "L": "Lớn"},
}


@st.cache_data(show_spinner=False)
def _palettes(signature: tuple):
    """`signature` (tên + mtime từng ảnh) để cache tự làm mới khi bạn thêm/sửa ảnh bảng màu."""
    return mx.load_palettes(THEME_DIR)


@st.cache_data(show_spinner=False)
def _label_meta(signature: tuple):
    return mx.load_label_meta(LABELS_YAML)


@st.cache_data(show_spinner=False)
def _model_registry(signature: tuple) -> dict:
    """{model_id: {display_name, dataset, num_classes, checkpoint}} từ configs/models.yaml. Lỗi đọc ⇒ {} (giao diện rơi về
    model_id). `checkpoint` = tên tệp của artifact `ckpt` (web_analyze không có trường checkpoint nên tra registry, M3-05)."""
    try:
        import yaml

        data = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — thiếu/hỏng registry chỉ làm mất tên đẹp, không được làm sập trang
        return {}
    artifacts = data.get("artifacts") or {}

    def ckpt_name(m: dict) -> str | None:
        path = (artifacts.get(m.get("ckpt")) or {}).get("path") if isinstance(artifacts, dict) else None
        return Path(str(path)).name if path else None

    return {m["id"]: {"display_name": m.get("display_name"), "dataset": m.get("dataset"), "num_classes": m.get("num_classes"),
                      "checkpoint": ckpt_name(m), "arch": m.get("arch")}
            for m in data.get("models") or [] if isinstance(m, dict) and m.get("id")}


def model_registry() -> dict:
    return _model_registry(_signature(MODELS_YAML))


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
    """Biến CSS cho dòng hướng dẫn của ô tải ảnh (theme.css đọc qua var(--mx-upload-hint))."""
    mb = _upload_limit_bytes(_signature(SERVICES_YAML)) / (1024 * 1024)
    hint = f"JPEG · PNG · WebP · tối đa {mb:g} MB · ảnh chỉ nằm trong bộ nhớ, không lưu lại"
    return ':root { --mx-upload-hint: "' + hint.replace("\\", "").replace('"', "") + '"; }'


def _signature(folder: Path) -> tuple:
    if not folder.exists():
        return ()
    items = [folder] if folder.is_file() else sorted(folder.iterdir())
    return tuple((p.name, p.stat().st_mtime_ns) for p in items if p.is_file())


def init_prefs(theme_ids: list[str], default_theme: str) -> None:
    """Lần đầu của phiên: lấy cài đặt từ địa chỉ trang (?t=…&m=dark…) nếu hợp lệ, nếu không thì mặc định."""
    if st.session_state.get("_prefs_ready"):
        return
    qp = st.query_params
    t = qp.get(QP_KEYS["pref_theme"])
    st.session_state["pref_theme"] = t if t in theme_ids else default_theme
    for key, default in DEFAULT_PREFS.items():
        raw = qp.get(QP_KEYS[key])
        if isinstance(default, bool):
            st.session_state[key] = default if raw not in ("0", "1") else raw == "1"
        else:
            st.session_state[key] = raw if raw in CHOICES[key] else default
    st.session_state["_prefs_ready"] = True


def sync_query_params() -> None:
    """Ghi cài đặt hiện tại lên địa chỉ trang để tải lại/chia sẻ vẫn giữ nguyên giao diện."""
    want = {}
    for key, qk in QP_KEYS.items():
        val = st.session_state.get(key)
        want[qk] = ("1" if val else "0") if isinstance(val, bool) else str(val)
    for qk, v in want.items():
        if st.query_params.get(qk) != v:
            st.query_params[qk] = v


def _restore_choice(key: str) -> None:
    """Thanh chọn cho phép bỏ chọn (None); khôi phục giá trị trước đó để luôn có một lựa chọn."""
    if st.session_state.get(key) is None:
        st.session_state[key] = st.session_state.get(f"_last_{key}", DEFAULT_PREFS[key])


def _reset_prefs(default_theme: str) -> None:
    st.session_state["pref_theme"] = default_theme
    st.session_state.update(DEFAULT_PREFS)


def choice(label: str, key: str) -> str:
    opts = CHOICES[key]
    st.segmented_control(label, opts, format_func=lambda o: CHOICE_LABELS[key][o], key=key,
                         on_change=_restore_choice, args=(key,))
    st.session_state[f"_last_{key}"] = st.session_state.get(key) or DEFAULT_PREFS[key]
    return st.session_state[f"_last_{key}"]


# ----------------------------------------------------------------------------- kết nối EMOCA (ADR-015)
# Chỉ giữ trong st.session_state (bộ nhớ phiên): KHÔNG ghi lên địa chỉ trang, vì có token và địa chỉ tunnel.
EMOCA_MODES = ("mock", "real")
EMOCA_MODE_LABELS = {"mock": "Giả lập", "real": "Máy chủ EMOCA"}


ENGINE_MODES = ("mock", "real")
ENGINE_MODE_LABELS = {"mock": "Giả lập", "real": "Thật (CPU)"}


def _engine_mode_changed() -> None:
    if st.session_state.get("engine_mode") is None:  # thanh chọn cho phép bỏ chọn ⇒ khôi phục giá trị trước
        st.session_state["engine_mode"] = st.session_state.get("_last_engine_mode", "mock")


def render_engine_panel() -> None:
    """Chọn POSTER/MA3D-Net giả lập hay chạy thật trên CPU (web/engine.py, M2-05)."""
    md('<div class="mx-set-h">Mô hình POSTER / MA3D-Net</div>')
    if resolve_engine_mode is None:
        st.caption("Không có bộ điều phối nên không đặt được chế độ mô hình.")
        return
    st.segmented_control("Suy luận", ENGINE_MODES, format_func=lambda o: ENGINE_MODE_LABELS[o], key="engine_mode",
                         on_change=_engine_mode_changed)
    st.session_state["_last_engine_mode"] = st.session_state.get("engine_mode") or "mock"
    if st.session_state["_last_engine_mode"] == "real":
        st.caption("POSTER chạy thật trên CPU trong tiến trình web: lần đầu nạp mô hình vài giây (~1 GB RAM cho hai mô "
                   "hình), các lần sau nhanh hơn. MA3D-Net chỉ chạy khi mã 3DMM do EMOCA thật sinh ra; với EMOCA giả lập "
                   "nó báo lỗi thiếu mã — đó là hành vi đúng, không phải hỏng.")
    else:
        st.caption("Xác suất và attention là dữ liệu giả, không cần torch.")


def init_emoca_state() -> None:
    """Lần đầu của phiên: lấy mặc định từ bộ điều phối (biến môi trường EMOCA_* > configs/services.yaml).
    Xác nhận gửi ảnh luôn bắt đầu ở trạng thái CHƯA đồng ý. Cũng đặt mặc định chế độ engine (ENGINE_MODE/config)."""
    if "engine_mode" not in st.session_state:
        mode = resolve_engine_mode()[0] if resolve_engine_mode else "mock"
        st.session_state["engine_mode"] = mode if mode in ENGINE_MODES else "mock"
    if st.session_state.get("_emoca_ready"):
        for key in ("emoca_url", "emoca_token", "emoca_consent"):  # widget không được vẽ ở lần chạy này sẽ bị Streamlit xoá
            if key in st.session_state:                             # trạng thái; gán lại để giữ (cách chính thức)
                st.session_state[key] = st.session_state[key]
        return
    base = emoca_status() if emoca_status else {}
    st.session_state["emoca_mode"] = base.get("mode") if base.get("mode") in EMOCA_MODES else "mock"
    st.session_state["emoca_url"] = base.get("url", "")
    st.session_state["emoca_token"] = ""
    st.session_state["emoca_consent"] = False
    st.session_state["_emoca_ready"] = True


def emoca_overrides() -> dict:
    """Cài đặt EMOCA hiện tại của phiên → tham số `emoca=` của analyze()."""
    if (st.session_state.get("emoca_mode") or "mock") != "real":
        return {"mode": "mock"}
    return {"mode": "real", "url": (st.session_state.get("emoca_url") or "").strip(),
            "token": st.session_state.get("emoca_token") or "",
            "remote_consent": bool(st.session_state.get("emoca_consent"))}


def emoca_signature() -> tuple:
    """Đổi chế độ/địa chỉ/xác nhận ⇒ phân tích lại. Token cố ý KHÔNG nằm trong đây (sửa token → bấm "Phân tích lại")."""
    ov = emoca_overrides()
    return ov["mode"], ov.get("url", ""), bool(ov.get("remote_consent"))


def _emoca_settings_changed(restore_mode: bool = False) -> None:
    """Đổi chế độ hoặc địa chỉ ⇒ thu hồi xác nhận gửi ảnh (không thể đồng ý cho máy A rồi ngầm gửi sang máy B)."""
    if restore_mode and st.session_state.get("emoca_mode") is None:  # thanh chọn cho phép bỏ chọn
        st.session_state["emoca_mode"] = st.session_state.get("_last_emoca_mode", "mock")
    st.session_state["emoca_consent"] = False
    st.session_state.pop("emoca_check", None)


def render_emoca_panel() -> None:
    md('<div class="mx-set-h">Kết nối EMOCA</div>')
    if emoca_status is None:
        st.caption("Không có bộ điều phối nên không đặt được nguồn EMOCA.")
        return
    st.segmented_control("Nguồn EMOCA", EMOCA_MODES, format_func=lambda o: EMOCA_MODE_LABELS[o], key="emoca_mode",
                         on_change=_emoca_settings_changed, args=(True,))
    st.session_state["_last_emoca_mode"] = st.session_state.get("emoca_mode") or "mock"
    if st.session_state["_last_emoca_mode"] != "real":
        st.caption("Landmark, mã 3DMM và ảnh dựng đều là dữ liệu giả; ảnh không đi đâu cả.")
        return
    st.text_input("Địa chỉ EMOCA", key="emoca_url", placeholder="https://….trycloudflare.com",
                  on_change=_emoca_settings_changed,
                  help="In ra ở cuối notebook Colab (another/notebooks/emoca_colab.ipynb); đổi mỗi phiên Colab.")
    st.text_input("Token", key="emoca_token", type="password",
                  help="Chuỗi bí mật do notebook in ra. Chỉ giữ trong bộ nhớ phiên này, không ghi lên địa chỉ trang.")
    view = emoca_status(emoca_overrides())
    if view["remote"]:
        st.toggle("Tôi đồng ý gửi ảnh tới máy chủ này", key="emoca_consent",
                  help="Ảnh khuôn mặt là dữ liệu cá nhân. Chỉ bật khi đó là ảnh của bạn hoặc ảnh công khai.")
        st.caption(f"Ảnh sẽ RỜI khỏi máy bạn, tới **{view['host']}** (Colab của Google và đơn vị cung cấp tunnel). "
                   "Máy chủ EMOCA của dự án không lưu ảnh, nhưng hai bên kia nằm ngoài tầm kiểm soát của dự án.")
    elif view["url"]:
        st.caption("Máy chủ nằm trên máy này — ảnh không rời khỏi máy.")
    else:
        st.caption("Chưa có địa chỉ — dán địa chỉ do notebook Colab in ra.")
    if st.button("Kiểm tra kết nối", key="btn_emoca_check", disabled=not view["url"]):
        with st.spinner("Đang kiểm tra…"):
            st.session_state["emoca_check"] = check_emoca(emoca_overrides())
    res = st.session_state.get("emoca_check")
    if res:
        latency = f" · {res['latency_ms']:.0f} ms" if res.get("latency_ms") is not None else ""
        info = res.get("info") or {}
        gpu = f" · {info['device']}" if info.get("device") else ""
        (st.success if res["ok"] else st.error)(f"{res['message']}{latency}{gpu}")


def render_settings(palettes: list, default_theme: str, problems: list[str], label_problems: list[str]) -> None:
    """Bảng Cài đặt ở đầu trang, mở/đóng bằng nút "Cài đặt" (không dùng thanh bên: thu gọn rồi thì không mở lại được).
    Expander luôn vẽ các widget bên trong kể cả khi đóng ⇒ Streamlit không xoá trạng thái của chúng."""
    by_id = {p.id: p for p in palettes}
    with st.expander("Cài đặt", icon=":material/settings:", key="ui_settings_open"):
        c_look, c_view, c_link = st.columns(3, gap="large")
    with c_look:
        md('<div class="mx-set-h">Giao diện</div>')
        st.selectbox("Chủ đề màu", list(by_id), format_func=lambda i: by_id[i].name, key="pref_theme",
                     help="Mỗi chủ đề lấy 4 màu từ một ảnh trong frontend/assets/theme.")
        pal = by_id[st.session_state["pref_theme"]]
        swatch = "".join(f'<i style="background:{c}" title="{c}"></i>' for c in pal.colors)
        md(f'<div class="mx-swatch" role="img" aria-label="Bảng màu: {esc(", ".join(pal.colors))}">{swatch}</div>'
           f'<div class="mx-mood"><b>{esc(pal.family)}</b> · {esc(pal.mood)}</div>')
        choice("Chế độ", "pref_mode")
        choice("Độ tương phản", "pref_level")
        choice("Cỡ chữ", "pref_font")
        st.toggle("Giảm chuyển động", key="pref_still", help="Tắt hiệu ứng chuyển mượt.")
    with c_view:
        md('<div class="mx-set-h">Hiển thị</div>')
        st.toggle("Hiện landmark trên ảnh", key="pref_lm")
        st.toggle("Hiện khung mặt", key="pref_bb")
        st.toggle("Hiện ảnh trung gian & attention", key="pref_att")
        render_engine_panel()
        if st.button("✦ Pipeline Studio", key="open_pipeline_studio", help="Mở canvas trực quan hoá pipeline"):
            st.query_params["page"] = "pipeline-studio"
            st.rerun()
    with c_link:
        render_emoca_panel()
        md('<div class="mx-set-h">Khác</div>')
        st.button("Đặt lại cài đặt", on_click=_reset_prefs, args=(default_theme,), key="btn_reset")
        if problems or label_problems:  # Streamlit cấm expander lồng expander ⇒ liệt kê thẳng
            md(f'<div class="mx-set-h">Ghi chú nguồn màu ({len(problems) + len(label_problems)})</div>')
            for line in problems + label_problems:
                st.caption(line)
        md(f'<div class="mx-mood">{len(palettes)} chủ đề từ <b>frontend/assets/theme</b>. Màu chữ, nút, viền được tự chỉnh '
           'để đạt độ tương phản WCAG.</div>')


# ----------------------------------------------------------------------------- helpers
def load_upload(data: bytes) -> Image.Image | None:
    """Giải mã ảnh chỉ để HIỂN THỊ; xoay theo EXIF để khớp hệ original_px."""
    try:
        img = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def _get_predata_views(image: Image.Image, image_digest: str):
    """Lazily build and keep all M6 comparison pixels in this browser session only."""
    cached = st.session_state.get("predata_views")
    if cached and cached.get("digest") == image_digest:
        return cached["views"]
    from predata_experiment import build_predata_views

    views = build_predata_views(image)
    st.session_state["predata_views"] = {"digest": image_digest, "views": views}
    return views


def render_yolo_face_shadow(image: Image.Image, image_digest: str) -> None:
    """Show input, RealESRGAN detector image, and YOLO crop from the original upload."""
    with st.expander("Thử YOLO26-face + pre-data (tùy chọn)", expanded=False):
        st.caption("RealESRGAN chỉ hỗ trợ YOLO; crop được lấy từ ảnh gốc. Không đổi kết quả chính; ảnh chỉ giữ trong session.")
        if st.button("Chạy detector phụ", key="run_yolo26_shadow"):
            status = st.status("Đang chạy pipeline detector…", expanded=True)
            try:
                views = _get_predata_views(image, image_digest)
                st.session_state.pop("ma3d_predata_result", None)
                st.session_state["ma3d_predata_run_key"] = None
                status.update(label="Đã tạo xong các ảnh tiền xử lý.", state="complete", expanded=False)
            except Exception as exc:
                st.session_state.pop("predata_views", None)
                st.session_state.pop("frontalize_3ddfa", None)
                status.update(label="Pipeline tiền xử lý gặp lỗi.", state="error", expanded=True)
                st.error(f"Không chạy được pipeline pre-data: {type(exc).__name__}: {exc}")

        cached = st.session_state.get("predata_views")
        if not cached or cached.get("digest") != image_digest:
            return
        views = cached["views"]
        enhanced_face = views.enhanced_detection
        images = [
            ("Ảnh đầu vào", views.source_image, "Ảnh gốc"),
            ("RealESRGAN Compact", views.enhanced_image, "Ảnh chỉ dùng cho YOLO"),
            ("YOLO crop · ảnh gốc", enhanced_face.cropped_face if enhanced_face else None,
             "BBox từ ảnh RealESRGAN; pixel crop từ ảnh gốc"),
        ]
        cols = st.columns(3)
        for col, (title, img, note) in zip(cols, images):
            with col:
                if img is None:
                    reason = views.branch_errors.get("enhanced_detection")
                    st.info(f"{title}: {reason or 'Chưa có ảnh ở bước này.'}")
                else:
                    st.image(img, caption=title, width="stretch")
                    st.caption(note)

        if enhanced_face is not None:
            st.caption(
                f"YOLO trên ảnh RealESRGAN Compact; crop từ ảnh gốc: bbox {enhanced_face.bbox_xyxy}, "
                f"confidence {enhanced_face.confidence:.3f}, crop {enhanced_face.crop_size[0]}×{enhanced_face.crop_size[1]} px. "
                f"Thời gian các bước: {', '.join(f'{name} {value:.0f} ms' for name, value in views.stage_times_ms.items())}."
            )
        for branch, message in views.branch_errors.items():
            if message:
                st.caption(f"{branch}: {message}")


def _find_prediction(result: dict, model_id: str) -> tuple[dict | None, dict | None]:
    for face in result.get("faces") or []:
        for prediction in face.get("predictions") or []:
            if prediction.get("model_id") == model_id:
                return face, prediction
    return None, None


def _variant_error_message(result: dict | None, prediction: dict | None) -> str:
    error = (prediction or {}).get("error") or (result or {}).get("error") or {}
    if isinstance(error, dict) and error.get("message"):
        code = error.get("code")
        return f"[{code}] {error['message']}" if code else str(error["message"])
    warnings = (result or {}).get("warnings") or []
    if warnings:
        return "; ".join(str(item) for item in warnings)
    return "Nhánh đã chạy nhưng không trả về prediction MA3D."


def _render_ma3d_variant(col, title: str, note: str, face: dict | None, prediction: dict | None,
                         tokens: dict, meta: dict, result: dict | None = None) -> None:
    with col:
        st.markdown(f"**{title}**")
        st.caption(note)
        if not prediction or prediction.get("status") != "ok":
            st.error(_variant_error_message(result, prediction))
            return
        top1 = prediction.get("top1") or {}
        st.markdown(f"**{label_text(top1.get('label', '?'), meta)} · {float(top1.get('prob', 0)):.1%}**")
        base = decode_render((face.get("model_input") or {}).get("image")) if face else None
        if base is None and face:
            base = decode_render(face.get("crop_image"))
        overlay, reason = attention_overlay(base, prediction.get("attention")) if base else (None, "Thiếu ảnh model input.")
        if overlay is not None:
            st.image(overlay, caption="MA3D attention", width="stretch")
        else:
            st.caption(reason)
        st.markdown(bars_html(prediction, tokens, meta), unsafe_allow_html=True)


def render_ma3d_predata(image: Image.Image | None, image_digest: str, baseline_result: dict,
                        baseline_face: dict, tokens: dict, meta: dict) -> None:
    """Opt-in ablation row. Baseline and POSTER result are never altered."""
    baseline_face_ma3d, baseline_prediction = _find_prediction(baseline_result, "ma3d_rafdb")
    real_emoca = any(str(w).startswith("emoca_real:") for w in baseline_result.get("warnings") or [])
    real_engine = st.session_state.get("engine_mode") == "real"
    can_run = image is not None and real_emoca and real_engine
    with st.container(border=True):
        section("image", "Kết quả trực quan — MA3D + pre-data",
                "B′ chạy tuần tự; C dùng lại baseline. POSTER và hàng baseline phía trên giữ nguyên ảnh gốc.")
        if not can_run:
            st.info("Bật EMOCA thật và POSTER/MA3D thật để chạy các nhánh so sánh; MA3D cần mã 3DMM thật.")
        run = st.button("Chạy so sánh MA3D + pre-data", key="run_ma3d_predata", disabled=not can_run)
        run_key = (image_digest, emoca_signature(), st.session_state.get("engine_mode"))
        if run:
            status = st.status("Đang chuẩn bị so sánh MA3D + pre-data…", expanded=True)
            progress = st.progress(0, text="Chuẩn bị ảnh và model")
            try:
                views = _get_predata_views(image, image_digest)
                from_predata = {"c": baseline_result}

                def run_ma3d_only(source: Image.Image | None, branch: str) -> dict:
                    if source is None:
                        return {"status": "error", "message": views.branch_errors.get(branch) or "Nhánh không có crop hợp lệ."}
                    buf = io.BytesIO()
                    source.convert("RGB").save(buf, format="PNG")
                    return analyze(buf.getvalue(), emoca=emoca_overrides(), engine_mode="real",
                                   model_ids=["ma3d_rafdb"])

                enhanced_face = views.enhanced_detection
                branches = [
                    ("b_prime", "B′ · RealESRGAN chỉ cho YOLO", enhanced_face.cropped_face if enhanced_face else None,
                     "enhanced_detection"),
                ]
                for index, (key, label, source, error_key) in enumerate(branches, start=1):
                    status.update(label=f"Đang chạy {label} ({index}/{len(branches)})…", state="running", expanded=True)
                    progress.progress((index - 1) / len(branches), text=f"{label} · {index}/{len(branches)}")
                    from_predata[key] = run_ma3d_only(source, error_key)
                    st.session_state["ma3d_predata_result"] = dict(from_predata)
                    st.session_state["ma3d_predata_run_key"] = run_key
                    progress.progress(index / len(branches), text=f"Đã xong {label} · {index}/{len(branches)}")
                successes = sum(
                    1 for key in ("b_prime",)
                    if (_find_prediction(from_predata[key], "ma3d_rafdb")[1] or {}).get("status") == "ok"
                )
                if successes == 1:
                    status.update(label="Nhánh B′ đã trả kết quả MA3D.", state="complete", expanded=False)
                else:
                    status.update(label="Đã xử lý xong nhưng nhánh B′ không có kết quả MA3D. Xem lỗi bên dưới.",
                                  state="error", expanded=True)
            except Exception as exc:
                status.update(label="So sánh bị lỗi; xem thông báo bên dưới.", state="error", expanded=True)
                st.error(f"Không chạy được so sánh MA3D + pre-data: {type(exc).__name__}: {exc}")

        variants = st.session_state.get("ma3d_predata_result")
        if not variants or st.session_state.get("ma3d_predata_run_key") != run_key:
            return
        face_b, pred_b = _find_prediction(variants.get("b_prime", {}), "ma3d_rafdb")
        face_c, pred_c = _find_prediction(variants.get("c", {}), "ma3d_rafdb")
        cols = st.columns(2)
        _render_ma3d_variant(cols[0], "B′ · MA3D + pre-data", "Ảnh gốc → RealESRGAN (chỉ cho YOLO) → YOLO box → crop ảnh gốc → EMOCA → MA3D",
                             face_b, pred_b, tokens, meta, variants.get("b_prime"))
        _render_ma3d_variant(cols[1], "C · MA3D baseline", "Ảnh gốc → EMOCA → MA3D; dùng baseline hiện có",
                             face_c or baseline_face_ma3d, pred_c or baseline_prediction, tokens, meta,
                             variants.get("c"))


def decode_render(render) -> Image.Image | None:
    if not isinstance(render, dict) or not render.get("b64"):
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(render["b64"], validate=True))).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
        return None


def is_mock(result: dict) -> bool:
    """Contract 0.1.0 chưa có trường mode mock/real — suy từ tên hiển thị và cảnh báo."""
    names = [m.get("display_name", "") for m in result.get("models_info") or []]
    for face in result.get("faces") or []:
        names += [p.get("display_name", "") for p in face.get("predictions") or []]
    texts = names + list(result.get("warnings") or [])
    return any("mock" in str(t).lower() for t in texts)


def label_text(label: str, meta: dict) -> str:
    """Tên hiển thị (tiếng Việt nếu configs/labels.yaml có), nhãn thô nếu không."""
    return (meta.get(str(label).strip().lower()) or {}).get("vi") or str(label)


def draw_overlay(base: Image.Image, face: dict, show_landmarks: bool, show_bbox: bool, tokens: dict) -> Image.Image:
    """Landmark = chấm màu nhấn có viền trắng (đọc được trên cả ảnh sáng lẫn tối); khung = màu chính."""
    img = base.copy()
    draw = ImageDraw.Draw(img)
    long_side = max(img.size)
    if show_bbox and face.get("bbox_original_px"):
        x0, y0, x1, y1 = face["bbox_original_px"]
        w = max(2, int(long_side * BBOX_WIDTH_RATIO))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=w + 2)
        draw.rectangle([x0, y0, x1, y1], outline=mx.hex_to_rgb(tokens["brand"]), width=w)
    lm = face.get("landmarks") or {}
    if show_landmarks and lm.get("space") == "original_px":
        r = max(1.8, long_side * LANDMARK_RADIUS_RATIO)
        fill = mx.hex_to_rgb(tokens["accent"])
        for x, y in lm.get("points") or []:
            draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], fill=(255, 255, 255))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    return img


def _heat_color(v: float, tokens: dict) -> str:
    """Thang 3 điểm: nền nhạt → màu chính pha → màu chính. Độ chói đơn điệu để đọc được cả khi in đen trắng."""
    v = min(max(float(v), 0.0), 1.0)
    lo, mid, hi = tokens["surface2"], mx.mix(tokens["surface2"], tokens["brand"], 0.6), tokens["brand"]
    return mx.mix(lo, mid, v * 2) if v < 0.5 else mx.mix(mid, hi, (v - 0.5) * 2)


def heat_html(grid_map: dict | None, tokens: dict, caption: str, tip: str = "") -> str | None:
    """Một bản đồ nhiệt lưới (mặc định 7×7) bằng các ô HTML; None nếu dữ liệu không đúng hình dạng."""
    if not isinstance(grid_map, dict):
        return None
    grid, values = grid_map.get("grid") or [], grid_map.get("values") or []
    if len(grid) != 2 or len(values) != grid[0] * grid[1]:
        return None
    rows, cols = grid
    cells = "".join(f'<i style="background:{_heat_color(v, tokens)}" title="hàng {i // cols + 1}, cột {i % cols + 1}: {float(v):.2f}"></i>'
                    for i, v in enumerate(values))
    return (f'<figure class="mx-heat-fig" title="{esc(tip)}"><div class="mx-heat" role="img" '
            f'aria-label="{esc(caption)}: bản đồ {rows}×{cols}" style="grid-template-columns:repeat({cols},1fr)">{cells}</div>'
            f'<figcaption>{esc(caption)}</figcaption></figure>')


def heat_legend_html(tokens: dict) -> str:
    stops = ", ".join(_heat_color(v, tokens) for v in (0.0, 0.5, 1.0))
    return f'<div class="mx-legend"><span>thấp</span><i style="background:linear-gradient(90deg, {stops})"></i><span>cao</span></div>'


# Khoá ảnh trung gian theo contract 0.2.0 / ADR-013 → (tên hiển thị, gợi ý). Đây là nhãn giao diện cho các
# khoá của contract, không phải nhãn cảm xúc hay tên mô hình.
FEATURE_MAP_CAPTIONS = {
    "image_backbone": ("Nhánh ảnh", "Đặc trưng ngoại hình từ IR50 (ADR-013)"),
    "landmark_backbone": ("Nhánh landmark", "Đặc trưng điểm mốc 2D từ MobileFaceNet (ADR-013)"),
    "fusion_delta": ("Tác động của 3DMM", "Nơi mã 3DMM làm đổi đặc trưng ảnh — chỉ MA3D-Net (ADR-013)"),
}


def feature_maps_html(pred: dict, tokens: dict) -> str | None:
    """Ảnh trung gian + attention của một mô hình (mọi khoá đều tuỳ chọn). None nếu không có gì để vẽ."""
    figs = []
    fmaps = pred.get("feature_maps") or {}
    for key, (caption, tip) in FEATURE_MAP_CAPTIONS.items():
        fig = heat_html(fmaps.get(key), tokens, caption, tip)
        if fig:
            figs.append(fig)
    fig = heat_html(pred.get("attention"), tokens, "Attention", "Chú ý của transformer trên crop 224×224")
    if fig:
        figs.append(fig)
    if not figs:
        return None
    return f'<div class="mx-heats">{"".join(figs)}</div>{heat_legend_html(tokens)}'


# ----------------------------------------------------------------------------- kết quả trực quan (slide 19 bài báo)
# Hàng 5 ảnh cùng cỡ: Input | Facial Landmark | 3DMM | POSTER | MA3D-Net (ours). Ảnh dựng bằng PIL.
# Attention (ADR-013, lưới 7×7 trên ảnh 224 mà engine nhận) phủ lên `face["model_input"]["image"]` — ảnh căn chỉnh kiểu
# RAF-DB mà engine THẬT SỰ nhận (ADR-020); thiếu thì phủ lên `face["crop_image"]` (crop EMOCA, ADR-019) và ghi rõ.
# Landmark crop từ `face["landmarks_crop224"]` (ADR-019). Mọi trường đều TUỲ CHỌN; thiếu ⇒ ô ghi lý do, không đoán khung.
VIS_TILE = 224                   # cạnh mỗi ô (px) = cạnh crop224
VIS_ALPHA = 0.45                 # độ phủ của bản đồ attention lên crop
VIS_LANDMARK_RGB = (230, 30, 30)  # chấm landmark đỏ như slide 19; viền trắng để thấy trên nền tối
VIS_TITLES = ("Input", "Facial Landmark", "3DMM")


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
    """Đưa ảnh về ô vuông size×size, giữ tỉ lệ (viền đen) — để 5 ô cùng cỡ."""
    return ImageOps.pad(img.convert("RGB"), (size, size), method=Image.BICUBIC, color=(0, 0, 0))


def attention_overlay(crop: Image.Image, grid_map, alpha: float = VIS_ALPHA) -> tuple[Image.Image | None, str]:
    """Phủ attention (grid_map hệ crop224, vd 7×7, 0..1) lên crop: nội suy bicubic lên VIS_TILE, tô Turbo, trộn alpha.
    Trả (ảnh VIS_TILE×VIS_TILE, "") hoặc (None, lý do)."""
    if not isinstance(grid_map, dict):
        return None, "mô hình không trả attention"
    if grid_map.get("space") != "crop224":
        return None, f"attention ở hệ {grid_map.get('space')!r}, không phải crop224 — không phủ"
    grid, values = grid_map.get("grid") or [], grid_map.get("values") or []
    if len(grid) != 2 or len(values) != grid[0] * grid[1]:
        return None, "attention sai hình dạng"
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


def visual_tiles(face: dict, image: Image.Image | None, result: dict, meta: dict) -> list[dict]:
    """5 ô theo slide 19: {title, top (nhãn dự đoán ở trên), img (PIL|None), note (lý do khi thiếu / ghi chú nguồn)}."""
    warns = [str(w) for w in result.get("warnings") or []]
    emoca_real = any(w.startswith("emoca_real:") for w in warns)
    engine_real = any(w.startswith("engine_real:") for w in warns)
    size = result.get("image") or {}
    same_image = image is not None and image.size == (int(size.get("width", 0)), int(size.get("height", 0)))
    crop = decode_render(face.get("crop_image"))
    crop_lm = face.get("landmarks_crop224")
    model_in = decode_render((face.get("model_input") or {}).get("image"))
    att_base = model_in if model_in is not None else crop   # ảnh engine nhận (ADR-020), không có thì crop EMOCA
    att_src = ("trên ảnh căn chỉnh RAF-DB mà mô hình nhận" if model_in is not None
               else "trên crop EMOCA (ảnh mô hình nhận)")   # mặc định: căn chỉnh RAF-DB bật (configs/align.yaml)
    no_crop = "chưa có ảnh mặt mà mô hình nhận trong kết quả (thiếu model_input/crop_image) — không phủ đúng khung được"
    tiles = []

    if crop is not None:
        tiles.append({"title": VIS_TITLES[0], "img": square_tile(crop), "note": "crop 224 của EMOCA (ảnh mô hình nhận)"})
    elif same_image:
        tiles.append({"title": VIS_TITLES[0], "img": square_tile(image), "note": "ảnh gốc (chưa có crop 224)"})
    else:
        tiles.append({"title": VIS_TITLES[0], "img": None, "note": "không có ảnh đầu vào khớp kết quả"})

    lm = face.get("landmarks") or {}
    if crop is not None and isinstance(crop_lm, list) and crop_lm:
        tiles.append({"title": VIS_TITLES[1], "img": landmark_tile(crop, crop_lm, ref_size=crop.width),
                      "note": f"{len(crop_lm)} điểm, hệ crop224"})
    elif same_image and lm.get("space") == "original_px" and lm.get("points"):
        tiles.append({"title": VIS_TITLES[1], "img": landmark_tile(image, lm["points"]),
                      "note": f"{len(lm['points'])} điểm trên ảnh gốc (original_px)"})
    else:
        tiles.append({"title": VIS_TITLES[1], "img": None, "note": "không có landmark vẽ được lên ảnh"})

    renders = face.get("renders") or {}
    geo = decode_render(renders.get("geometry_detail")) or decode_render(renders.get("geometry_coarse"))
    geo_note = ("EMOCA dựng lại (geometry)" if emoca_real else "ảnh giữ chỗ — EMOCA giả lập, chưa dựng 3D thật")
    tiles.append({"title": VIS_TITLES[2], "img": square_tile(geo) if geo is not None else None,
                  "note": geo_note if geo is not None else "EMOCA không trả ảnh dựng geometry"})

    reg = model_registry()
    for pred in face.get("predictions") or []:
        info = reg.get(pred.get("model_id"), {})
        title = pred.get("display_name") or info.get("display_name") or str(pred.get("model_id"))
        if str(info.get("arch", "")).upper() == "MA3D":
            title += " (ours)"
        tile = {"title": title, "top": "", "img": None, "note": ""}
        if pred.get("status") != "ok":
            err = pred.get("error") or {}
            tile["note"] = f"mô hình lỗi [{err.get('code', pred.get('status'))}] {err.get('message', '')}".strip()
        else:
            top1 = pred.get("top1") or {}
            tile["top"] = f"{label_text(top1.get('label', '?'), meta)} · {float(top1.get('prob', 0)):.0%}"
            if att_base is None:
                tile["note"] = no_crop
            else:
                tile["img"], reason = attention_overlay(att_base, pred.get("attention"))
                kind = "attention giả lập" if not engine_real else "attention (độ lớn chú ý)"
                tile["note"] = reason or f"{kind}, {att_src}"
        tiles.append(tile)
    return tiles


def render_visual(face: dict, image: Image.Image | None, result: dict, meta: dict) -> None:
    tiles = visual_tiles(face, image, result, meta)
    with st.container(border=True):
        section("columns", "Kết quả trực quan")
        cols = st.columns(len(tiles))  # màn hẹp: Streamlit tự xếp dọc các cột
        for col, t in zip(cols, tiles):
            with col:
                md(f'<div class="mx-vis-top">{esc(t.get("top") or "") or "&nbsp;"}</div>')
                if t["img"] is not None:
                    st.image(t["img"], width="stretch")
                else:
                    md(f'<div class="mx-vis-miss" role="img" aria-label="{esc(t["title"])}: {esc(t["note"])}">'
                       f'<span>{esc(t["note"])}</span></div>')
                md(f'<div class="mx-vis-cap">{esc(t["title"])}</div>'
                   + (f'<div class="mx-vis-note">{esc(t["note"])}</div>' if t["img"] is not None and t["note"] else ""))


# ----------------------------------------------------------------------------- khối giao diện
def section(icon: str, title: str, sub: str = "") -> None:
    subline = f'<div class="mx-sec-s">{esc(sub)}</div>' if sub else ""
    md(f'<div class="mx-sec"><span class="mx-ico">{svg(icon)}</span><div><div class="mx-sec-t">{esc(title)}</div>{subline}</div></div>')


def page_header(mock: bool, palette, tokens: dict) -> None:
    mode = "Tối" if tokens["mode"] == "dark" else "Sáng"
    mock_pill = f'<span class="mx-pill warn">{svg("flask")}Dữ liệu MOCK</span>' if mock else ""
    md('<div class="mx-topbar"><div class="mx-brand"><div class="mx-logo" aria-hidden="true">3D</div><div>'
       '<h1 class="mx-title">MA3D-Net Lab</h1>'
        '<div class="mx-sub">So sánh POSTER và MA3D-Net cho một khuôn mặt; hỗ trợ phân tích riêng từng crop khi có nhiều người.</div>'
       f'</div></div><div class="mx-pills">{mock_pill}<span class="mx-pill">{esc(palette.name)} · {mode}</span></div></div>'
       '<div class="mx-stripe" aria-hidden="true"></div>')


def steps(current: int) -> None:
    names = ["Tải ảnh", "Hình học", "So sánh"]
    items = []
    for i, name in enumerate(names):
        cls = "done" if i < current else ("active" if i == current else "")
        mark = svg("check") if i < current else str(i + 1)
        items.append(f'<li class="{cls}"><b>{mark}</b>{name}</li>')
    md(f'<ol class="mx-steps" aria-label="Các bước">{"".join(items)}</ol>')


def empty_state() -> None:
    tiles = [("upload", "1. Tải ảnh", "Chọn một ảnh có khuôn mặt rõ, chính diện."),
             ("box", "2. EMOCA dựng 3D", "Lấy landmark, mã 3DMM và ảnh dựng lại khuôn mặt."),
             ("columns", "3. So sánh hai mô hình", "POSTER và MA3D-Net dự đoán trên cùng một khuôn mặt.")]
    body = "".join(f'<div class="mx-tile"><span class="mx-ico">{svg(i)}</span><b>{esc(t)}</b><span>{esc(s)}</span></div>'
                   for i, t, s in tiles)
    md(f'<div class="mx-empty">{body}</div>')


def render_error(err: dict | None, status: str):
    if not err:
        return
    code = err.get("code", "unknown")
    msg = f"**[{code}]** {err.get('message', '')}\n\n{ERROR_HINTS.get(code, '')}"
    (st.warning if status == "partial" else st.error)(msg)
    if err.get("retryable"):
        st.caption("Lỗi này có thể thử lại — bấm **Phân tích lại**.")


def render_geometry(face: dict, image: Image.Image | None, result_size: tuple[int, int], tokens: dict):
    with st.container(border=True):
        section("box", "Hình học khuôn mặt", "Landmark và khuôn mặt 3D do EMOCA dựng lại từ ảnh.")
        lm = face.get("landmarks") or {}
        if lm and lm.get("space") != "original_px":
            st.warning(f"Landmark ở hệ `{lm.get('space')}`, không phải `original_px` — không phủ lên ảnh gốc.")

        if image is not None and image.size == result_size:
            base = image
        else:
            # Ảnh tải lên và ảnh backend xử lý khác kích thước (vd dữ liệu mock cố định) → không phủ lên ảnh thật.
            if image is not None:
                st.info(
                    f"Kích thước ảnh trả về {result_size[0]}×{result_size[1]} khác ảnh tải lên "
                    f"{image.size[0]}×{image.size[1]} — landmark vẽ trên nền trống thay vì ảnh của bạn."
                )
            base = Image.new("RGB", result_size, mx.hex_to_rgb(tokens["surface2"]))

        cols = st.columns(3)
        cols[0].image(draw_overlay(base, face, st.session_state["pref_lm"], st.session_state["pref_bb"], tokens),
                      caption=f"Ảnh vào + {lm.get('n_points', 0)} landmark (original_px)", width="stretch")
        renders = face.get("renders") or {}
        for col, (key, title) in zip(cols[1:], [("geometry_coarse", "3DMM coarse"), ("output_detail", "3DMM detail")]):
            img = decode_render(renders.get(key))
            if img is not None:
                col.image(img, caption=f"{title} (EMOCA)", width="stretch")
            else:
                col.info(f"{title}: chưa có ảnh dựng" + (" (placeholder)" if key in renders else ""))


def bars_html(pred: dict, tokens: dict, meta: dict) -> str:
    labels, probs = pred.get("labels") or [], pred.get("probs") or []
    top_idx = (pred.get("top1") or {}).get("index")
    rows = []
    for i, (lab, p) in enumerate(zip(labels, probs)):
        p = min(max(float(p), 0.0), 1.0)
        color, emoji = mx.emotion_style(lab, i, tokens, (meta.get(str(lab).strip().lower()) or {}).get("color"))
        name = label_text(lab, meta)
        top = " top" if i == top_idx else ""
        star = " ★" if i == top_idx else ""
        rows.append(
            f'<div class="mx-bar{top}"><div class="mx-bar-l" title="{esc(str(lab))}"><span aria-hidden="true">{emoji}</span>'
            f'<span>{esc(name)}{star}</span></div>'
            f'<div class="mx-track" role="meter" aria-label="{esc(name)}" aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{p * 100:.1f}"><div class="mx-fill" style="width:{max(p * 100, 0.8):.1f}%;background:{color}"></div></div>'
            f'<div class="mx-bar-v">{p:.1%}</div></div>')
    return f'<div class="mx-bars">{"".join(rows)}</div>'


def render_model_column(col, pred: dict, index: int, disagree: bool, tokens: dict, meta: dict):
    with col:
        with st.container(border=True):
            reg = model_registry().get(pred.get("model_id"), {})  # dự đoán LỖI không kèm tên/dataset ⇒ tra registry
            name = pred.get("display_name") or reg.get("display_name") or pred.get("model_id", "?")
            color = mx.model_color(index, tokens)
            md(f'<div class="mx-model" style="--mc:{color}"><span class="mx-dot"></span>{esc(str(name))}</div>')
            if pred.get("status") != "ok":
                err = pred.get("error") or {}
                st.error(f"Mô hình lỗi **[{err.get('code', pred.get('status'))}]** {err.get('message', '')}")
                return
            top1 = pred.get("top1") or {}
            lab = top1.get("label", "?")
            _, emoji = mx.emotion_style(lab, top1.get("index", 0), tokens,
                                        (meta.get(str(lab).strip().lower()) or {}).get("color"))
            flag = f'<div class="mx-flag">{svg("alert")}Bất đồng với mô hình còn lại</div>' if disagree else ""
            md(f'{flag}<div class="mx-top1"><span class="mx-emo" aria-hidden="true">{emoji or "🙂"}</span><div>'
               f'<div class="mx-top1-l">{esc(label_text(lab, meta))}</div>'
               f'<div class="mx-top1-p">Top-1 · độ tin cậy {float(top1.get("prob", 0)):.1%}</div></div></div>')
            md(bars_html(pred, tokens, meta))
            if pred.get("latency_ms") is not None:
                st.caption(f"Thời gian suy luận: {pred['latency_ms']:.0f} ms")
            maps = feature_maps_html(pred, tokens) if st.session_state["pref_att"] else None
            if maps:
                with st.expander("Ảnh trung gian & attention (7×7)"):
                    md(maps)
                    st.caption("Lưới 7×7 trên ảnh 224×224 mà mô hình nhận (mỗi ô 32×32 px); bản phủ lên mặt nằm ở khối "
                               "\"Kết quả trực quan\". Đây là độ lớn kích hoạt (chuẩn L2 theo kênh, min-max 0..1 trong "
                               "từng bản đồ), không phải mức đóng góp vào dự đoán.")


def render_comparison(face: dict, tokens: dict, meta: dict):
    preds = face.get("predictions") or []
    ok = [p for p in preds if p.get("status") == "ok" and p.get("top1")]
    tops = {p["top1"].get("label") for p in ok}
    disagree = len(tops) > 1
    with st.container(border=True):
        section("columns", "So sánh biểu cảm", "Cùng một khuôn mặt, hai mô hình cho kết quả như thế nào.")
        if not preds:
            st.info("Không có dự đoán cho mặt này.")
            return
        if len(ok) >= 2:
            if disagree:
                parts = " · ".join(f'{esc(str(p.get("display_name", p.get("model_id"))))}: <b>{esc(label_text(p["top1"]["label"], meta))}</b>'
                                   for p in ok)
                md(f'<div class="mx-insight diff">{svg("alert")}<div>Hai mô hình <b>bất đồng</b> — {parts}</div></div>')
            else:
                only = label_text(next(iter(tops)), meta)
                md(f'<div class="mx-insight">{svg("check")}<div>Hai mô hình <b>cùng dự đoán</b>: {esc(only)}</div></div>')
        cols = st.columns(len(preds))
        for i, (col, p) in enumerate(zip(cols, preds)):
            render_model_column(col, p, i, disagree, tokens, meta)


def render_result(result: dict, image: Image.Image | None, tokens: dict, meta: dict, image_digest: str = ""):
    """Vẽ toàn bộ kết quả; tách riêng để kiểm thử được mà không cần upload."""
    status = result.get("status", "error")
    render_error(result.get("error"), status)

    faces = result.get("faces") or []
    if status != "error" and faces:
        if len(faces) > 1:
            st.info(f"Phát hiện {len(faces)} khuôn mặt — tối đa 30 mặt mỗi ảnh.")
            idx = st.selectbox("Chọn khuôn mặt", range(len(faces)),
                               format_func=lambda i: f"Mặt #{faces[i].get('face_index', i)}")
        else:
            idx = 0
        face = faces[idx]
        size = result.get("image") or {}
        render_visual(face, image, result, meta)  # khối 5 ảnh ngay sau khi phân tích, trên phần so sánh xác suất
        render_geometry(face, image, (int(size.get("width", 1)), int(size.get("height", 1))), tokens)
        render_comparison(face, tokens, meta)
        render_ma3d_predata(image, image_digest, result, face, tokens, meta)
    elif status == "ok" and not faces:
        st.error(f"**[no_face]** {ERROR_HINTS['no_face']}")


def _face_row(rows: list[dict], face_id: int) -> dict | None:
    return next((row for row in rows if row.get("face_id") == face_id), None)


def _render_face_prediction(row: dict | None, tokens: dict, meta: dict) -> None:
    if not row:
        st.caption("Chưa chạy Multi-Object Analysis cho khuôn mặt này.")
        return
    result = row["result"]
    _, prediction = _find_prediction(result, "ma3d_rafdb")
    if prediction and prediction.get("status") == "ok":
        top1 = prediction.get("top1") or {}
        st.markdown(f"**{label_text(top1.get('label', '?'), meta)} · {float(top1.get('prob', 0)):.1%}**")
        st.markdown(bars_html(prediction, tokens, meta), unsafe_allow_html=True)
    else:
        st.error(_variant_error_message(result, prediction))
def _render_face_prediction_summary(row: dict | None, meta: dict, failed: bool = False) -> None:
    """Compact top prediction shown directly below each detected face crop."""
    prediction = None
    if row:
        _, prediction = _find_prediction(row.get("result") or {}, "ma3d_rafdb")
    if prediction and prediction.get("status") == "ok":
        top1 = prediction.get("top1") or {}
        label = esc(label_text(top1.get("label", "?"), meta))
        probability = max(0.0, min(1.0, float(top1.get("prob", 0))))
        st.markdown(
            f'<div class="mx-face-prediction" aria-label="Dự đoán: {label}, {probability:.0%}">'
            f'<span>{label}</span><b>{probability:.0%}</b></div>',
            unsafe_allow_html=True,
        )
    elif row:
        _, prediction = _find_prediction(row.get("result") or {}, "ma3d_rafdb")
        st.caption(f"Prediction unavailable · {_variant_error_message(row.get('result') or {}, prediction)}")
    elif failed:
        st.caption("Prediction unavailable · bấm Phân tích lại để thử lại.")
    else:
        st.caption("Đang chờ dự đoán…")


def _render_face_detail_page(views, row: dict | None, face_id: int, tokens: dict, meta: dict) -> None:
    item = next((face for face in views.faces if face.face_id == face_id), None)
    if item is None:
        st.error(f"Không còn dữ liệu khuôn mặt #{face_id}. Hãy quay lại danh sách và thử lại.")
    else:
        st.markdown(f"## Face #{face_id} — Details")
        st.caption(
            f"BBox ảnh gốc: {item.detection.bbox_xyxy} · "
            f"Confidence: {item.detection.confidence:.3f} · "
            f"Crop: {item.detection.crop_size[0]}×{item.detection.crop_size[1]} px"
        )
        original_col, crop_col = st.columns(2)
        original_col.image(item.reference_image, caption="Vị trí trong ảnh gốc", width="stretch")
        processed = (row or {}).get("processed") or item.detection.cropped_face
        crop_col.image(processed, caption="YOLO crop từ ảnh gốc", width="stretch")
        if row and row.get("method"):
            st.caption(f"Nhánh xử lý: {row['method']}")
        _render_face_prediction(row, tokens, meta)

    if st.button("Back to faces", key="back_to_faces"):
        for key in ("page", "face_id"):
            if key in st.query_params:
                del st.query_params[key]
        st.rerun()


def _open_face_details(face_id: int) -> None:
    st.query_params["page"] = "face-details"
    st.query_params["face_id"] = str(face_id)
    st.rerun()


def render_multiface_analysis(image: Image.Image, image_digest: str, views, tokens: dict,
                              meta: dict, face_limit: int) -> None:
    """Render face cards and automatically run multi-object inference once per input/config."""
    run_key = (image_digest, emoca_signature(), st.session_state.get("engine_mode") or "mock", face_limit)
    cached = st.session_state.get("multiface_result")
    if cached and cached.get("key") != run_key:
        st.session_state.pop("multiface_result", None)
        st.session_state.pop("frontalize_3ddfa", None)
        cached = None
    faces = views.faces
    if len(faces) <= 1:
        return

    with st.container(border=True):
        section("image", "Multi-Object Analysis", "Tự động phân tích tất cả khuôn mặt được phát hiện; mở thẻ hoặc View Details để xem đầy đủ.")
        st.caption(f"Đã phát hiện {len(faces)} khuôn mặt. Giới hạn {face_limit}; các crop xếp theo vị trí từ trên xuống, trái sang phải.")
        attempt = st.session_state.get("multiface_attempt")
        if attempt and attempt.get("key") != run_key:
            st.session_state.pop("multiface_attempt", None)
            attempt = None
        has_result = bool(cached and cached.get("key") == run_key)
        if not has_result and not attempt:
            st.session_state["multiface_attempt"] = {"key": run_key, "status": "running"}
            status = st.status(f"Đang phân tích {len(faces)} khuôn mặt…", expanded=True)
            try:
                def update_progress(message: str) -> None:
                    status.update(label=message, state="running", expanded=True)

                state = analyze_multiface(
                    image,
                    emoca=emoca_overrides(),
                    engine_mode=st.session_state.get("engine_mode") or "mock",
                    max_faces=face_limit,
                    progress_callback=update_progress,
                    views=views,
                )
                state["key"] = run_key
                st.session_state["multiface_result"] = state
                st.session_state["multiface_attempt"] = {"key": run_key, "status": "complete"}
                cached = state
                status.update(label=f"Đã phân tích {len(state.get('rows') or [])} khuôn mặt.",
                              state="complete", expanded=False)
            except Exception as exc:
                st.session_state["multiface_attempt"] = {
                    "key": run_key, "status": "error", "message": f"{type(exc).__name__}: {exc}"
                }
                status.update(label="Multi-Object Analysis gặp lỗi.", state="error", expanded=True)
                st.error(f"{type(exc).__name__}: {exc}")

        elif not has_result and attempt and attempt.get("status") == "error":
            st.error(attempt.get("message", "Multi-Object Analysis gặp lỗi."))

        attempt = st.session_state.get("multiface_attempt")
        rows = (cached.get("rows") or []) if cached and cached.get("key") == run_key else []
        columns_per_row = min(4, len(faces))
        for start in range(0, len(faces), columns_per_row):
            cols = st.columns(columns_per_row)
            for col, item in zip(cols, faces[start:start + columns_per_row]):
                row = _face_row(rows, item.face_id)
                with col:
                    with st.container(border=True):
                        st.markdown(f"**Face #{item.face_id}**")
                        st.image(item.detection.cropped_face, caption="YOLO crop", width="stretch")
                        st.caption(f"Confidence {item.detection.confidence:.3f} · {item.detection.crop_size[0]}×{item.detection.crop_size[1]} px")
                        roll_info = (row or {}).get("roll_info") or {}
                        if roll_info.get("applied"):
                            st.caption(f"↻ {roll_info['rotation_deg']:.0f}°")
                        _render_face_prediction_summary(
                            row, meta, failed=bool(attempt and attempt.get("status") == "error")
                        )
                        with st.expander(f"Full details · Face #{item.face_id}", expanded=False):
                            st.image(item.reference_image, caption="Vị trí trong ảnh gốc", width="stretch")
                            processed = (row or {}).get("processed") or item.detection.cropped_face
                            st.image(processed, caption="YOLO crop từ ảnh gốc", width="stretch")
                            _render_face_prediction(row, tokens, meta)
                        if st.button("View Details", key=f"view_face_details_{item.face_id}"):
                            _open_face_details(item.face_id)

        if views.errors:
            with st.expander("Processing notes", expanded=False):
                for message in views.errors:
                    st.caption(message)


# ----------------------------------------------------------------------------- trang
def main():
    st.set_page_config(page_title="MA3D-Net Lab", page_icon="🎭", layout="wide", initial_sidebar_state="collapsed")

    palettes, palette_problems = _palettes(_signature(THEME_DIR))
    if not palettes:
        palette_problems = list(palette_problems) + ["dùng bảng màu mặc định tích hợp sẵn"]
        palettes = [mx.DEFAULT_PALETTE]
    meta, label_problems = _label_meta(_signature(LABELS_YAML))
    default_theme = next((p.id for p in palettes if p.id == "cold"), palettes[0].id)
    init_prefs([p.id for p in palettes], default_theme)
    init_emoca_state()

    by_id = {p.id: p for p in palettes}
    if st.session_state.get("pref_theme") not in by_id:  # bảng màu đã bị xoá/đổi tên
        st.session_state["pref_theme"] = default_theme
    palette = by_id[st.session_state["pref_theme"]]
    mode = st.session_state.get("pref_mode") or DEFAULT_PREFS["pref_mode"]
    level = st.session_state.get("pref_level") or DEFAULT_PREFS["pref_level"]
    font = st.session_state.get("pref_font") or DEFAULT_PREFS["pref_font"]
    tokens = mx.derive_tokens(palette.colors, mode, level)
    md(f"<style>{mx.build_css(tokens, font, bool(st.session_state.get('pref_still')))}\n{upload_hint_css()}</style>")

    header_slot = st.empty()  # cập nhật ngay khi có kết quả, không đợi lần chạy sau
    sync_query_params()
    if st.query_params.get("page") == "pipeline-studio":
        if ORCHESTRATOR_ERROR:
            st.error(f"Không nạp được pipeline: {ORCHESTRATOR_ERROR}")
            return
        from pipeline_studio import render_pipeline_studio

        render_pipeline_studio(
            analyze_fn=analyze,
            prepare_fn=prepare_multiface_views,
            multiface_fn=analyze_multiface,
            roll_fn=roll_correct_upload,
            emoca=emoca_overrides(),
            engine_mode=st.session_state.get("engine_mode") or "mock",
            face_limit=max_faces_limit(),
            reduce_motion=bool(st.session_state.get("pref_still")),
        )
        return

    render_settings(palettes, default_theme, list(palette_problems), list(label_problems))  # ngay dưới tiêu đề

    cached = st.session_state.get("result")
    steps_slot = st.empty()

    def set_header(mock: bool) -> None:
        # Chỉ gắn nhãn MOCK khi có bằng chứng (kết quả đã có).
        with header_slot:
            page_header(mock, palette, tokens)

    def set_step(n: int) -> None:
        with steps_slot:
            steps(n)

    set_header(bool(cached and is_mock(cached[1])))
    set_step(2 if cached else 0)
    if ORCHESTRATOR_ERROR:
        st.error(f"**[service_unavailable]** Không nạp được bộ điều phối `web/orchestrator.py`: "
                 f"`{ORCHESTRATOR_ERROR}`\n\nSửa lỗi import (xem terminal chạy Streamlit) rồi tải lại trang.")
        return
    view = emoca_status(emoca_overrides())  # KHÔNG chứa token
    with st.container(border=True):
        section("upload", "Tải ảnh khuôn mặt")
        up = st.file_uploader("Chọn ảnh khuôn mặt (JPEG/PNG/WebP)", type=ACCEPTED_TYPES, label_visibility="collapsed")
        if up is None:
            st.session_state.pop("result", None)
            st.session_state.pop("yolo26_face_shadow", None)
            st.session_state.pop("predata_views", None)
            st.session_state.pop("ma3d_predata_result", None)
            st.session_state.pop("ma3d_predata_run_key", None)
            st.session_state.pop("multiface_result", None)
            st.session_state.pop("frontalize_3ddfa", None)
            st.session_state.pop("multiface_attempt", None)
            st.session_state.pop("multiface_detection", None)
            for key in ("page", "face_id"):
                if key in st.query_params:
                    del st.query_params[key]
            set_step(0)
            empty_state()
            return

        data = up.getvalue()
        image = load_upload(data)
        if image is None:
            st.error(f"**[image_invalid]** {ERROR_HINTS['image_invalid']}")
            return
        pcol, icol = st.columns([1, 2])
        pcol.image(image, caption=f"Xem trước — {image.size[0]}×{image.size[1]} px", width=240)
        with icol:
            md(f'<div class="mx-chip">{svg("image")} {esc(up.name)}</div><div class="mx-chip">{len(data) / 1024:.0f} KB</div>')
            digest = hashlib.sha256(data).hexdigest()
            detection = st.session_state.get("multiface_detection")
            if detection and (detection.get("digest") != digest
                              or detection.get("max_faces") != max_faces_limit()):
                st.session_state.pop("multiface_detection", None)
                st.session_state.pop("multiface_result", None)
                st.session_state.pop("frontalize_3ddfa", None)
                st.session_state.pop("multiface_attempt", None)
                st.session_state.pop("result", None)
                for key in ("page", "face_id"):
                    if key in st.query_params:
                        del st.query_params[key]
                detection = None
            shadow = st.session_state.get("yolo26_face_shadow")
            if shadow and shadow.get("digest") != digest:
                st.session_state.pop("yolo26_face_shadow", None)
            predata = st.session_state.get("predata_views")
            if predata and predata.get("digest") != digest:
                st.session_state.pop("predata_views", None)
                st.session_state.pop("frontalize_3ddfa", None)
                st.session_state.pop("ma3d_predata_result", None)
                st.session_state.pop("ma3d_predata_run_key", None)
            multi = st.session_state.get("multiface_result")
            if multi and multi.get("key", (None,))[0] != digest:
                st.session_state.pop("multiface_result", None)
                st.session_state.pop("frontalize_3ddfa", None)
                st.session_state.pop("multiface_attempt", None)
            rerun = st.button("Phân tích lại", type="primary")
    if view["mode"] == "real":
        for note in view["warnings"]:
            st.caption(f"Cấu hình EMOCA: `{note}`")

    engine_mode = st.session_state.get("engine_mode") or "mock"
    face_limit = max_faces_limit()
    detection = st.session_state.get("multiface_detection")
    if rerun:
        st.session_state.pop("multiface_detection", None)
        st.session_state.pop("multiface_result", None)
        st.session_state.pop("frontalize_3ddfa", None)
        st.session_state.pop("multiface_attempt", None)
        st.session_state.pop("result", None)
        detection = None
    if not detection or detection.get("digest") != digest or detection.get("max_faces") != face_limit:
        status = st.status("Đang phát hiện khuôn mặt trước khi chọn luồng dự đoán…", expanded=True)
        try:
            def update_detection_progress(message: str) -> None:
                status.update(label=message, state="running", expanded=True)

            views = prepare_multiface_views(
                image, max_faces=face_limit, progress_callback=update_detection_progress
            )
            detection = {"digest": digest, "max_faces": face_limit, "views": views}
            st.session_state["multiface_detection"] = detection
            status.update(label=f"Đã kiểm tra ảnh: {len(views.faces)} khuôn mặt.",
                          state="complete", expanded=False)
        except Exception as exc:
            # ADR-027: a failed YOLO preflight must not block prediction; fall back to the
            # single-person baseline on the original upload (EMOCA detects the face itself).
            status.update(label="YOLO lỗi; chuyển sang luồng baseline trên ảnh gốc.", state="error", expanded=False)
            views = SimpleNamespace(source_image=None, faces=[], detection_source=None, detection_complete=False,
                                    errors=[f"{type(exc).__name__}: {exc}"], stage_times_ms={})
            detection = {"digest": digest, "max_faces": face_limit, "views": views}
            st.session_state["multiface_detection"] = detection

    views = detection["views"]
    page = st.query_params.get("page")
    if page == "face-details":
        try:
            face_id = int(st.query_params.get("face_id", "0"))
        except (TypeError, ValueError):
            face_id = 0
        cached_multi = st.session_state.get("multiface_result")
        detail_key = (digest, emoca_signature(), engine_mode, face_limit)
        detail_rows = (cached_multi or {}).get("rows", []) if (cached_multi or {}).get("key") == detail_key else []
        _render_face_detail_page(views, _face_row(detail_rows, face_id), face_id, tokens, meta)
        return

    if not getattr(views, "detection_complete", True):
        st.warning("YOLO không hoàn tất; đang dùng luồng baseline trên ảnh gốc.")
        for message in views.errors:
            st.caption(message)

    if len(views.faces) > 1:
        # Deliberately skip orchestrator.analyze(): multi-person input only enters the multi-object path.
        st.session_state.pop("result", None)
        multi_key = (digest, emoca_signature(), engine_mode, face_limit)
        multi_state = st.session_state.get("multiface_result")
        has_multi_result = bool(multi_state and multi_state.get("key") == multi_key)
        set_header(any(is_mock(row.get("result", {})) for row in (multi_state or {}).get("rows", []))
                   if has_multi_result else False)
        set_step(2 if has_multi_result else 1)
        render_multiface_analysis(image, digest, views, tokens, meta, face_limit)
        render_frontalization_experiment(image, digest, views, tokens, meta)
        multi_state = st.session_state.get("multiface_result")
        has_multi_result = bool(multi_state and multi_state.get("key") == multi_key)
        set_header(any(is_mock(row.get("result", {})) for row in (multi_state or {}).get("rows", []))
                   if has_multi_result else False)
        set_step(2 if has_multi_result else 1)
        return

    cached = st.session_state.get("result")
    corrected_image, roll_info = roll_correct_upload(image, views)
    roll_enabled, roll_threshold = roll_correction_settings()
    roll_key = (roll_enabled, roll_threshold,
                (roll_info or {}).get("roll_deg"), (roll_info or {}).get("rotation_deg"),
                (roll_info or {}).get("error"))
    if roll_info and not roll_info.get("error") and roll_info.get("rotation_deg"):
        st.caption(f"Ảnh đã được xoay {roll_info['rotation_deg']:.0f}° để dựng thẳng mặt "
                   f"(|roll| {abs(roll_info['roll_deg']):.0f}° ≥ {roll_info['threshold_deg']:.0f}°)")
    elif roll_info and roll_info.get("error"):
        st.caption(f"Không thể đo roll bằng 3DDFA-V2; tiếp tục với ảnh gốc. {roll_info['error']}")
    run_key = (digest, emoca_signature(), engine_mode, roll_key)  # ADR-029 config/estimate participates in cache identity
    if rerun or not cached or cached[0] != run_key:
        set_step(1)
        with st.spinner("Đang phân tích (EMOCA → POSTER / MA3D-Net)…" + (" Lần đầu nạp mô hình thật, có thể vài giây." if engine_mode == "real" else "")):
            try:
                prediction_image = io.BytesIO()
                corrected_image.convert("RGB").save(prediction_image, format="PNG")
                result = analyze(prediction_image.getvalue(), emoca=emoca_overrides(), engine_mode=engine_mode, max_faces=1)
            except Exception as exc:  # orchestrator không phản hồi / ném lỗi: báo, không nuốt
                st.session_state.pop("result", None)
                set_step(0)
                st.error(f"**[service_unavailable]** Bộ điều phối lỗi: `{type(exc).__name__}: {exc}`\n\n"
                         f"{ERROR_HINTS['service_unavailable']}")
                return
        if not isinstance(result, dict):
            set_step(0)
            st.error(f"**[contract_mismatch]** {ERROR_HINTS['contract_mismatch']}")
            return
        st.session_state["result"] = (run_key, result)
    set_header(is_mock(st.session_state["result"][1]))
    set_step(2)
    render_result(st.session_state["result"][1], corrected_image, tokens, meta, digest)
    render_multiface_analysis(image, digest, views, tokens, meta, face_limit)
    render_yolo_face_shadow(image, digest)
    render_frontalization_experiment(image, digest, views, tokens, meta)


def _prediction_probabilities(result: dict | None, model_id: str) -> tuple[dict | None, dict | None]:
    face, prediction = _find_prediction(result or {}, model_id)
    if not prediction:
        return face, None
    labels = prediction.get("labels") or []
    probs = prediction.get("probs") or []
    try:
        values = {str(label): float(prob) for label, prob in zip(labels, probs)}
    except (TypeError, ValueError):
        values = {}
    return face, values


def _render_frontal_prediction(col, title: str, before: dict | None, after: dict | None, meta: dict) -> None:
    with col:
        st.markdown(f"**{title}**")
        before_face, before_pred = _find_prediction(before or {}, "poster_rafdb" if "POSTER" in title else "ma3d_rafdb")
        after_face, after_pred = _find_prediction(after or {}, "poster_rafdb" if "POSTER" in title else "ma3d_rafdb")
        def top(pred):
            top1 = (pred or {}).get("top1") or {}
            return f"{label_text(str(top1.get('label', '?')), meta)} {float(top1.get('prob', 0)):.1%}" if top1 else "—"
        st.caption(f"Trước: {top(before_pred)} → Sau: {top(after_pred)}")
        _, left = _prediction_probabilities(before, "poster_rafdb" if "POSTER" in title else "ma3d_rafdb")
        _, right = _prediction_probabilities(after, "poster_rafdb" if "POSTER" in title else "ma3d_rafdb")
        left, right = left or {}, right or {}
        labels = list(dict.fromkeys([*left.keys(), *right.keys()]))
        if labels:
            st.table([{"Nhãn": label_text(label, meta), "Trước (%)": round(left.get(label, 0)*100, 2),
                       "Sau (%)": round(right.get(label, 0)*100, 2),
                       "Δ (điểm % )": round((right.get(label, 0)-left.get(label, 0))*100, 2)} for label in labels])
        for label, pred in (("Trước", before_pred), ("Sau", after_pred)):
            if pred and pred.get("status") != "ok":
                st.caption(f"{label}: {_variant_error_message(None, pred)}")


def render_frontalization_experiment(image: Image.Image | None, digest: str, views,
                                     tokens: dict, meta: dict) -> None:
    """Render the opt-in 3DDFA-V2 comparison at the end of either analysis path."""
    engine_mode = st.session_state.get("engine_mode") or "mock"
    primary = st.session_state.get("result")
    results = [primary[1]] if isinstance(primary, tuple) and len(primary) > 1 else []
    multi = st.session_state.get("multiface_result") or {}
    results.extend(row.get("result") or {} for row in multi.get("rows", []))
    real_emoca = any(str(w).startswith("emoca_real:") for result in results
                     for w in result.get("warnings") or [])
    can_run = image is not None and real_emoca and engine_mode == "real"
    run_key = (digest, emoca_signature(), engine_mode, max_faces_limit())
    cached = st.session_state.get("frontalize_3ddfa")
    if cached and cached.get("key") != run_key:
        st.session_state.pop("frontalize_3ddfa", None)
        cached = None
    with st.container(border=True):
        section("image", "Thử nghiệm: Sửa nghiêng (roll) — 3DDFA-V2",
                "YOLO crop (ảnh gốc) → 3DDFA-V2 đo roll → xoay 2D nếu |roll| ≥ ngưỡng → EMOCA → POSTER / MA3D-Net. "
                "Chỉ sửa roll; yaw và pitch không được chỉnh (ADR-028). Đây là thử nghiệm; không thay đổi kết quả phía trên.")
        if not can_run:
            st.info("Bật EMOCA thật và engine POSTER/MA3D thật để chạy thử nghiệm này.")
        run = st.button("Experiment: Sửa nghiêng (3DDFA-V2)", key="run_frontalize_3ddfa", disabled=not can_run)
        if run:
            status = st.status("Đang chạy thử nghiệm sửa roll…", expanded=True)
            progress = st.progress(0, text="Chuẩn bị theo từng khuôn mặt")
            try:
                def report(message: str) -> None:
                    status.update(label=message, state="running", expanded=True)
                    match = re.search(r"Mặt (\d+)/(\d+)", message)
                    if match:
                        progress.progress(int(match.group(1)) / max(1, int(match.group(2))), text=message)
                from orchestrator import run_frontalization_experiment
                state = run_frontalization_experiment(views, emoca=emoca_overrides(), engine_mode=engine_mode,
                                                     progress_callback=report)
                state["key"] = run_key
                st.session_state["frontalize_3ddfa"] = {"key": run_key, "state": state}
                progress.progress(1.0, text=f"Đã xử lý {len(state.get('rows') or [])} khuôn mặt")
                status.update(label="Đã hoàn tất thử nghiệm sửa roll.", state="complete", expanded=False)
            except Exception as exc:
                state = {"rows": [], "times_ms": {}, "errors": [f"{type(exc).__name__}: {exc}"], "key": run_key}
                st.session_state["frontalize_3ddfa"] = {"key": run_key, "state": state}
                status.update(label="Thử nghiệm sửa roll gặp lỗi.", state="error", expanded=True)
        cached = st.session_state.get("frontalize_3ddfa")
        if not cached or cached.get("key") != run_key:
            return
        rows = cached.get("state", {}).get("rows", [])
        for i, row in enumerate(rows or []):
            def render_row():
                st.markdown(f"**Face #{row.get('face_id', i+1)}**")
                cols = st.columns([1, 1, 2])
                with cols[0]: st.image(row["crop"], caption="YOLO crop", width="stretch")
                with cols[1]:
                    if row.get("corrected") is not None:
                        st.image(row["corrected"], caption="YOLO crop + sửa roll 2D", width="stretch")
                    else:
                        st.info("Không tạo được ảnh đã sửa roll.")
                    pose = row.get("pose") or {}
                    if pose:
                        st.caption(f"Roll {pose['roll_deg']:.1f}° · Yaw {pose['yaw_deg']:.1f}° · Pitch {pose['pitch_deg']:.1f}° · "
                                   f"Xoay {pose['rotation_deg']:.1f}° · Ngưỡng {pose['threshold_deg']:.1f}°")
                        if not pose["applied"]:
                            st.caption(f"Không xoay: |roll| = {abs(pose['roll_deg']):.1f}° < ngưỡng {pose['threshold_deg']:.1f}°")
                            st.caption("Kết quả sau giống hệt theo cách dựng nhánh dưới ngưỡng.")
                _render_frontal_prediction(cols[2], "POSTER", row.get("result_before"), row.get("result_after"), meta)
                _render_frontal_prediction(cols[2], "MA3D-Net", row.get("result_before"), row.get("result_after"), meta)
                if row.get("errors"):
                    for stage, message in row["errors"].items(): st.error(f"{stage}: {message}")
                if row.get("times_ms"):
                    st.caption("Thời gian: " + " · ".join(f"{name} {value:.0f} ms" for name, value in row["times_ms"].items()))
            if len(rows) > 1:
                with st.expander(f"Face #{row.get('face_id', i+1)}", expanded=i == 0): render_row()
            else:
                render_row()
        state = cached.get("state", {})
        for error in state.get("errors", []): st.caption(error)


if __name__ == "__main__":
    main()
