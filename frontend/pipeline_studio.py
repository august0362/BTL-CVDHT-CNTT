"""Standalone animated pipeline canvas for the MA3D research demo.

This page is intentionally isolated from the main analysis layout. It calls the
existing orchestrator and keeps uploaded pixels and intermediate images only in
the Streamlit session.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, UnidentifiedImageError


NODE_LABELS = {
    "input": "Ảnh đầu vào",
    "yolo": "YOLO · detector",
    "esrgan": "RealESRGAN · fallback",
    "roll": "3DDFA-V2 · roll",
    "emoca": "EMOCA · SFD / FAN / 3DMM",
    "alignment": "RAF-DB · căn chỉnh",
    "poster": "POSTER",
    "ma3d": "MA3D-Net",
}
NODE_ORDER = ("input", "yolo", "esrgan", "roll", "emoca", "alignment", "poster", "ma3d")
_pipeline_canvas = components.declare_component(
    "pipeline_studio_canvas", path=str(Path(__file__).with_name("pipeline_visualizer"))
)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _load_image(data: bytes) -> Image.Image | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def _decode_image(value: Any) -> Image.Image | None:
    if not isinstance(value, dict) or not value.get("b64"):
        return None
    try:
        raw = base64.b64decode(value["b64"], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _css() -> str:
    return """
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stFileUploaderDropzone"] {
      background:#000 !important; color:#e9f8ff !important;
    }
    [data-testid="stMain"] { min-height:100vh; }
    [data-testid="stMainBlockContainer"], section.main > div.block-container {
      width:100% !important; max-width:none !important; padding-left:4vw !important; padding-right:4vw !important;
    }
    .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] p {
      color:#d7e7ed !important;
    }
    .stButton > button {background:#080808 !important;border:1px solid #25645f !important;color:#bafff1 !important;
      border-radius:10px;transition:transform .2s ease,box-shadow .25s ease,border-color .25s ease}
    .stButton > button:hover {border-color:#52f7dc !important;box-shadow:0 0 18px #35f1d455;transform:translateY(-1px);color:#effffc !important}
    [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div {
      background:#030303 !important;border:1px dashed #2a5559 !important;border-radius:14px;color:#b9d1dc !important;
    }
    [data-testid="stFileUploader"] *, [data-testid="stFileUploaderDropzone"] * {
      color:#d7e7ed !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
      background:#101818 !important;color:#e4fff9 !important;border:1px solid #327e75 !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {fill:#78ffe4 !important;color:#78ffe4 !important;}
    [data-baseweb="tab-list"] {gap:4px;border-bottom:1px solid #17353d}
    [data-baseweb="tab"] {background:#0c141d;border-radius:9px 9px 0 0;color:#9db5c1}
    [aria-selected="true"][data-baseweb="tab"] {color:#aaffef !important;border-bottom-color:#52f7dc !important}
    .ps-head {display:flex;align-items:center;gap:14px;margin:4px 0 6px;color:#efffff}
    .ps-orb {width:42px;height:42px;border-radius:14px;display:grid;place-items:center;color:#041411;
      background:linear-gradient(135deg,#9bffe8,#23c9db);box-shadow:0 0 22px #36f5d777;font-weight:900}
    .ps-title {font-size:30px;font-weight:800;letter-spacing:-.04em;margin:0}
    .ps-sub {color:#c2d5dc;margin:0 0 18px 56px}
    .ps-board {padding:20px 12px;border:1px solid #1d3942;border-radius:20px;background:#020202;
      box-shadow:inset 0 1px #ffffff08,0 18px 60px #0009;overflow-x:auto}
    .ps-flow {display:flex;align-items:stretch;gap:0;min-width:950px;padding:10px 4px 16px}
    .ps-node {position:relative;flex:1;min-width:112px;max-width:190px;border-radius:15px;padding:13px 11px;
      background:linear-gradient(150deg,#0b0b0b,#030303);border:1px solid #243030;color:#d7e7ed;
      transition:transform .35s ease,border-color .35s ease,box-shadow .35s ease,opacity .35s ease;}
    .ps-node.running {border-color:#37f4dc;box-shadow:0 0 7px #37f4dc88,0 0 25px #19d5d355,inset 0 0 20px #29f4d111;
      color:#eafffb;transform:translateY(-4px);animation:ps-pulse 1.65s ease-in-out infinite}
    .ps-node.done {border-color:#2adbb9;color:#dcfff8;box-shadow:0 0 14px #22dfbd28}
    .ps-node.error {border-color:#ff6487;color:#ffdbe5;box-shadow:0 0 16px #ff52772b}
    .ps-node.skipped {border-style:dashed;color:#687c89;opacity:.7}
    .ps-node.warn {border-color:#f2c568;color:#ffebbd;box-shadow:0 0 14px #eebb4830}
    .ps-nhead {display:flex;align-items:center;gap:8px;font-size:12px;font-weight:750;white-space:nowrap}
    .ps-icon {width:24px;height:24px;border-radius:8px;display:grid;place-items:center;background:#18333b;color:#74ffe6;font-size:12px}
    .ps-node.done .ps-icon {background:#123d37;color:#5ff4d4}
    .ps-node.error .ps-icon {background:#421c2a;color:#ff809a}
    .ps-detail {font-size:10px;line-height:1.4;color:#c0d1d6;margin-top:9px;min-height:27px}
    .ps-edge {width:35px;min-width:28px;position:relative;align-self:center;height:2px;background:#223642;overflow:visible}
    .ps-edge.flowing {background:linear-gradient(90deg,#1f4550,#45ffe2,#1f4550);box-shadow:0 0 10px #37f4dc99}
    .ps-edge.flowing:after {content:"";position:absolute;top:-4px;left:-7px;width:10px;height:10px;border-radius:50%;background:#bdfff5;
      box-shadow:0 0 8px 3px #2ffff0,0 0 20px 6px #14ddf0;animation:ps-travel 1.05s linear infinite}
    .ps-branch {border:1px solid #1d3942;border-radius:14px;padding:13px 15px;background:#030303;color:#d2e2e7;margin-top:12px}
    .ps-branch b {color:#d7fff8}
    .ps-module-title {color:#8df9e8;font-weight:750;letter-spacing:.02em}
    .ps-muted {color:#bfd0d6;font-size:12px}
    @keyframes ps-travel {from{left:-7px}to{left:calc(100% - 3px)}}
    @keyframes ps-pulse {0%,100%{box-shadow:0 0 7px #37f4dc66,0 0 18px #19d5d333}50%{box-shadow:0 0 13px #37f4dccc,0 0 34px #19d5d377}}
    @media (prefers-reduced-motion: reduce) { .ps-node.running,.ps-edge.flowing:after {animation:none} * {scroll-behavior:auto !important} }
    </style>
    """


def _node_statuses(kind: str = "single") -> dict[str, dict[str, str]]:
    statuses = {key: {"state": "pending", "detail": "Chờ dữ liệu"} for key in NODE_ORDER}
    statuses["input"] = {"state": "done", "detail": "Ảnh đã sẵn sàng trong RAM"}
    if kind == "multi":
        for key in ("roll", "emoca", "alignment", "poster"):
            statuses[key] = {"state": "skipped", "detail": "Nhánh nhiều mặt không gọi module này"}
    return statuses


def _graph(statuses: dict[str, dict[str, str]], active: str | None = None) -> str:
    html_nodes = []
    visible_order = [key for key in NODE_ORDER if statuses[key]["state"] != "skipped"]
    for index, key in enumerate(visible_order):
        item = statuses[key]
        state = item["state"]
        icon = {"done": "✓", "running": "◌", "error": "!", "warn": "!", "skipped": "–"}.get(state, "·")
        html_nodes.append(
            f'<div class="ps-node {state}"><div class="ps-nhead"><span class="ps-icon">{icon}</span>'
            f'<span>{_esc(NODE_LABELS[key])}</span></div><div class="ps-detail">{_esc(item["detail"])}</div></div>'
        )
        if index < len(visible_order) - 1:
            next_key = visible_order[index + 1]
            flowing = key == active or next_key == active
            html_nodes.append(f'<div class="ps-edge{" flowing" if flowing else ""}"></div>')
    return '<div class="ps-board"><div class="ps-flow">' + "".join(html_nodes) + "</div></div>"


def _set(statuses: dict, key: str, state: str, detail: str, graph_slot=None) -> None:
    statuses[key] = {"state": state, "detail": detail}
    if graph_slot is not None:
        graph_slot.markdown(_graph(statuses, key), unsafe_allow_html=True)


def _run_image(data: bytes, name: str, analyze_fn, prepare_fn, multiface_fn, roll_fn,
               emoca: dict, engine_mode: str, face_limit: int, graph_slot) -> dict:
    image = _load_image(data)
    if image is None:
        return {"name": name, "error": "Không đọc được ảnh JPEG/PNG/WebP."}
    states = _node_statuses()
    _set(states, "yolo", "running", "Dò khuôn mặt trên ảnh gốc…", graph_slot)
    views = prepare_fn(image, max_faces=face_limit)
    count = len(views.faces)
    esrgan_error = next((message for message in views.errors if "RealESRGAN" in message), None)
    if views.stage_times_ms.get("realesrgan", 0) > 0:
        _set(states, "esrgan", "done", "Đã tăng nét rồi thử YOLO lại.", graph_slot)
    elif esrgan_error:
        _set(states, "esrgan", "error", esrgan_error, graph_slot)
    else:
        _set(states, "esrgan", "skipped", "Chỉ chạy nếu YOLO ảnh gốc trả 0 mặt.", graph_slot)
    yolo_state = "done" if views.detection_complete else "error"
    yolo_detail = f"{count} mặt · {views.detection_source or 'không có crop'}"
    if views.errors and not views.detection_complete:
        yolo_detail = views.errors[0]
    _set(states, "yolo", yolo_state, yolo_detail, graph_slot)
    route = "multi" if count > 1 else "single"
    states = _node_statuses(route)
    _set(states, "input", "done", f"{image.width}×{image.height} · RGB", graph_slot)
    _set(states, "yolo", yolo_state, yolo_detail, graph_slot)
    if views.stage_times_ms.get("realesrgan", 0) > 0:
        _set(states, "esrgan", "done", "Đã làm nét để detector thử lại.", graph_slot)
    elif esrgan_error:
        _set(states, "esrgan", "error", esrgan_error, graph_slot)
    else:
        _set(states, "esrgan", "skipped", "Không cần fallback.", graph_slot)

    if route == "multi":
        _set(states, "roll", "running", f"Đang xử lý roll và {count} crop…", graph_slot)
        _set(states, "ma3d", "running", "Đang phân tích từng crop tuần tự…", graph_slot)

        def progress(message: str) -> None:
            _set(states, "ma3d", "running", message, graph_slot)

        result = multiface_fn(image, emoca=emoca, engine_mode=engine_mode,
                              max_faces=face_limit, progress_callback=progress, views=views)
        rows = result.get("rows", [])
        good = sum(1 for row in rows if (row.get("result") or {}).get("status") == "ok")
        _set(states, "roll", "done", "Đã thử sửa roll trên từng crop.", graph_slot)
        _set(states, "ma3d", "done" if good else "error", f"{good}/{count} crop có kết quả.", graph_slot)
        return {"name": name, "image": image, "views": views, "result": result,
                "kind": route, "states": states, "roll_info": None, "corrected": image}

    corrected = image
    roll_info = None
    if count == 1:
        _set(states, "roll", "running", "Đang ước lượng roll từ crop YOLO…", graph_slot)
        corrected, roll_info = roll_fn(image, views)
        if roll_info and roll_info.get("error"):
            _set(states, "roll", "warn", "Đo roll lỗi; giữ ảnh gốc và tiếp tục.", graph_slot)
        elif roll_info and roll_info.get("rotation_deg"):
            _set(states, "roll", "done", f"Đã xoay ảnh gốc {roll_info['rotation_deg']:.1f}°.", graph_slot)
        else:
            _set(states, "roll", "done", "Roll dưới ngưỡng; không cần xoay.", graph_slot)
    else:
        _set(states, "roll", "skipped", "Không có crop mặt để đo roll.", graph_slot)

    _set(states, "emoca", "running", "Dò mặt SFD/FAN → crop224 → mã 3DMM…", graph_slot)
    payload = io.BytesIO()
    corrected.convert("RGB").save(payload, format="PNG")

    def progress(module: str, state: str, detail: str | None) -> None:
        if module == "emoca":
            _set(states, "emoca", state, detail or "EMOCA", graph_slot)
        elif module == "alignment":
            _set(states, "alignment", state, detail or "Căn chỉnh ảnh đầu vào.", graph_slot)
        elif module == "classifier":
            if state == "running":
                _set(states, "poster", "running", "Module classifier đang chạy POSTER…", graph_slot)
                _set(states, "ma3d", "running", "Module classifier đang chạy MA3D-Net…", graph_slot)
            else:
                _set(states, "poster", state, detail or "POSTER hoàn tất.", graph_slot)
                _set(states, "ma3d", state, detail or "MA3D-Net hoàn tất.", graph_slot)

    result = analyze_fn(payload.getvalue(), emoca=emoca, engine_mode=engine_mode,
                        max_faces=1, progress_callback=progress)
    if not result.get("faces") and result.get("error"):
        _set(states, "emoca", "error", result["error"].get("message", "EMOCA không trả kết quả."), graph_slot)
    return {"name": name, "image": image, "views": views, "result": result,
            "kind": route, "states": states, "roll_info": roll_info, "corrected": corrected}


def _module_images(module: str, item: dict) -> None:
    image, views = item.get("image"), item.get("views")
    result = item.get("result") or {}
    faces = result.get("faces") or []
    if module == "input" and image:
        st.image(image, caption="Ảnh sau xoay EXIF · RGB", width="stretch")
    elif module == "yolo" and views:
        st.image(views.source_image, caption="Ảnh detector", width="stretch")
        for start in range(0, len(views.faces), 3):
            cols = st.columns(min(3, len(views.faces) - start))
            for col, face in zip(cols, views.faces[start:start + 3]):
                with col:
                    col.image(face.detection.annotated_image, caption=f"BBox · mặt {face.face_id}", width="stretch")
                    st.caption(f"confidence={face.detection.confidence:.3f} · bbox={face.detection.bbox_xyxy}")
    elif module == "esrgan":
        st.info("RealESRGAN chỉ chạy khi YOLO trên ảnh gốc không thấy mặt. Ảnh nâng nét là đầu vào detector; crop model vẫn lấy từ ảnh gốc.")
        if views and views.enhanced_image is not None:
            st.image(views.enhanced_image, caption="RealESRGAN output · cùng kích thước ảnh gốc", width="stretch")
        if views and "realesrgan" in views.stage_times_ms:
            st.caption(f"Module đã chạy · {views.stage_times_ms['realesrgan']:.0f} ms.")
    elif module == "roll":
        if item.get("corrected"):
            st.image(item["corrected"], caption="Ảnh/crop sau bước sửa roll (nếu đạt ngưỡng)", width="stretch")
        st.caption(str(item.get("roll_info") or "Không có phép xoay được áp dụng."))
    elif module in ("emoca", "alignment") and faces:
        face = faces[0]
        candidates = []
        if module == "emoca":
            candidates.append(("Crop EMOCA · 224×224", _decode_image(face.get("crop_image"))))
            for key, label in (("geometry_coarse", "Geometry coarse"), ("geometry_detail", "Geometry detail"),
                               ("output_coarse", "Output coarse"), ("output_detail", "Output detail")):
                candidates.append((label, _decode_image((face.get("renders") or {}).get(key))))
        else:
            candidates.append(("model_input", _decode_image((face.get("model_input") or {}).get("image"))))
        shown = [(title, img) for title, img in candidates if img is not None]
        if shown:
            cols = st.columns(min(3, len(shown)))
            for index, (title, img) in enumerate(shown):
                with cols[index % len(cols)]:
                    st.image(img, caption=title, width="stretch")
        else:
            st.info("Không có ảnh trả về từ module ở chế độ/kết quả hiện tại.")
    elif module in ("poster", "ma3d"):
        if item.get("kind") == "multi":
            rows = (result.get("rows") or []) if isinstance(result, dict) else []
            if module == "poster":
                st.info("POSTER không chạy trong nhánh nhiều mặt.")
                return
            if not rows:
                st.info("Chưa có prediction cho các crop.")
            for row in rows:
                face_result = row.get("result") or {}
                face_result_list = face_result.get("faces") or []
                st.markdown(f"**Mặt #{row.get('face_id', '?')} · MA3D-Net**")
                face = face_result_list[0] if face_result_list else None
                if face:
                    for prediction in face.get("predictions", []):
                        if prediction.get("status") == "ok":
                            st.json(prediction.get("top1") or {})
                            model_input = ((face.get("model_input") or {}).get("image")
                                           or face.get("crop_image"))
                            preview = _decode_image(model_input)
                            if preview is not None:
                                st.image(preview, caption="Ảnh đầu vào MA3D-Net", width="stretch")
                        else:
                            error = prediction.get("error") or {}
                            st.error(f"{error.get('code', 'error')}: {error.get('message', 'Module lỗi')}")
                else:
                    st.warning(face_result.get("error", {}).get("message", "Prediction unavailable"))
            return
        if not faces:
            st.info("Không có prediction để hiển thị.")
            return
        wanted = "poster" if module == "poster" else "ma3d"
        predictions = [pred for pred in faces[0].get("predictions", [])
                       if wanted in str(pred.get("model_id", "")).lower()]
        if not predictions:
            st.info("Module này không chạy ở nhánh hiện tại (ví dụ nhánh nhiều mặt chỉ gọi MA3D-Net).")
        for prediction in predictions:
            st.markdown(f"**{_esc(prediction.get('display_name') or prediction.get('model_id'))}**")
            if prediction.get("status") == "ok":
                st.json(prediction.get("top1") or {})
                model_input = ((faces[0].get("model_input") or {}).get("image")
                               or faces[0].get("crop_image"))
                preview = _decode_image(model_input)
                if preview is not None:
                    st.image(preview, caption="Ảnh đầu vào module classifier", width="stretch")
            else:
                error = prediction.get("error") or {}
                st.error(f"{error.get('code', 'error')}: {error.get('message', 'Module lỗi')}")
    else:
        st.info("Module chưa tạo ảnh ở lượt chạy này.")


def render_pipeline_studio(*, analyze_fn, prepare_fn, multiface_fn, roll_fn,
                           emoca: dict, engine_mode: str, face_limit: int,
                           reduce_motion: bool = False) -> None:
    """Render the standalone draggable module canvas and run uploaded images on demand."""
    st.markdown(_css(), unsafe_allow_html=True)
    if reduce_motion:
        st.markdown('<style>.ps-node.running,.ps-edge.flowing:after {animation:none !important}</style>',
                    unsafe_allow_html=True)
    st.markdown('<div class="ps-head"><div class="ps-orb">⌘</div><h1 class="ps-title">Pipeline Studio</h1></div>'
                '<div class="ps-sub">Canvas độc lập · module hoá · dữ liệu chỉ lưu trong session</div>',
                unsafe_allow_html=True)
    if st.button("← Về trang phân tích", key="pipeline_studio_back"):
        if "page" in st.query_params:
            del st.query_params["page"]
        st.rerun()

    uploads = st.file_uploader(
        "Ảnh đầu vào · chọn một ảnh hoặc nhiều ảnh",
        type=("jpg", "jpeg", "png", "webp"),
        accept_multiple_files=True,
        key="ps_uploads",
        help="Có thể chọn một ảnh hoặc chọn nhiều ảnh trong cùng một lần.",
    )
    if not uploads:
        _pipeline_canvas(statuses=_node_statuses(), selected=st.session_state.get("ps_selected_module", "input"),
                         active=None, key="ps_canvas")
        st.markdown('<p class="ps-muted">Chọn một ảnh hoặc một chùm ảnh để tạo các luồng dữ liệu.</p>', unsafe_allow_html=True)
        return

    signatures = tuple(hashlib.sha256(upload.getvalue()).hexdigest() for upload in uploads)
    config_key = (engine_mode, emoca.get("mode"), emoca.get("url"), face_limit)
    cached = st.session_state.get("pipeline_studio_results")
    if st.button("▶ Chạy luồng dữ liệu", type="primary", key="ps_run_pipeline"):
        outputs = []
        for index, upload in enumerate(uploads, start=1):
            st.markdown(f"### Luồng {index} · {_esc(upload.name)}")
            graph_slot = st.empty()
            states = _node_statuses()
            graph_slot.markdown(_graph(states), unsafe_allow_html=True)
            try:
                outputs.append(_run_image(upload.getvalue(), upload.name, analyze_fn, prepare_fn,
                                          multiface_fn, roll_fn, emoca, engine_mode,
                                          face_limit, graph_slot))
            except Exception as exc:
                outputs.append({"name": upload.name, "error": f"{type(exc).__name__}: {exc}"})
                _set(states, "yolo", "error", f"{type(exc).__name__}: {exc}", graph_slot)
        st.session_state["pipeline_studio_results"] = {
            "signatures": signatures, "config": config_key, "outputs": outputs,
        }
        cached = st.session_state["pipeline_studio_results"]
    if not cached or cached.get("signatures") != signatures or cached.get("config") != config_key:
        _pipeline_canvas(statuses=_node_statuses(), selected=st.session_state.get("ps_selected_module", "input"),
                         active=None, key="ps_canvas")
        return

    output_items = [item for item in cached["outputs"] if not item.get("error")]
    shown_item = output_items[0] if output_items else None
    statuses = shown_item["states"] if shown_item else _node_statuses()
    selected = _pipeline_canvas(statuses=statuses,
                                selected=st.session_state.get("ps_selected_module", "input"),
                                active=None, key="ps_canvas")
    if isinstance(selected, dict) and selected.get("selected") in NODE_LABELS:
        st.session_state["ps_selected_module"] = selected["selected"]
    module = st.session_state.get("ps_selected_module", "input")
    st.markdown(f'<div class="ps-branch"><div class="ps-module-title">{_esc(NODE_LABELS[module])}</div>'
                '<div class="ps-muted">Bấm module để xem mô tả và dữ liệu do module tạo ra. Kéo module để sắp xếp canvas.</div></div>',
                unsafe_allow_html=True)
    if shown_item:
        _module_images(module, shown_item)
        if shown_item.get("result", {}).get("error"):
            error = shown_item["result"]["error"]
            st.warning(f"{error.get('code', 'error')}: {error.get('message', '')}")
        if shown_item["views"].errors:
            with st.expander("Nhật ký module", expanded=False):
                for note in shown_item["views"].errors:
                    st.caption(note)

    for index, item in enumerate(cached["outputs"], start=1):
        if item.get("error"):
            st.error(f"Luồng {index} · {item['name']}: {item['error']}")
