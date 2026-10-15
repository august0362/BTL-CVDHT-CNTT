"""Kiểm thử frontend/app.py bằng streamlit.testing (chạy nguyên script, không cần trình duyệt).

Chạy trong môi trường có streamlit:  <python của venv web> another/tests/test_app_ui.py
Thiếu streamlit thì báo SKIP và thoát mã 0 (để `python tests/test_*.py` bằng Python thường không bị đỏ oan).
Giao diện tối giản (ADR-032): không có Cài đặt; chỉ tiêu đề, ô tải ảnh và khối "Kết quả trực quan" (mỗi mặt một hàng).
Harness test thay `analyze` bằng lời gọi orchestrator thật với `emoca={"mode":"mock"}` và `engine_mode="mock"`.
"""
import io
import re
import sys
import time
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("SKIP test_app_ui: chưa cài streamlit (dùng python của venv web)")
    sys.exit(0)

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))
APP = str(ROOT / "frontend" / "app.py")


def png(size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.linear_gradient("L").resize(size).convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def _fake_face_views(image: Image.Image, count: int):
    from types import SimpleNamespace

    crop = image.crop((0, 0, max(1, image.width // 2), max(1, image.height // 2)))
    detection = SimpleNamespace(cropped_face=crop, bbox_xyxy=(0, 0, crop.width, crop.height),
                               confidence=0.9, crop_size=crop.size)
    return SimpleNamespace(
        source_image=image.copy(), detection_complete=True, errors=[],
        faces=[SimpleNamespace(face_id=i, detection=detection, reference_image=image.copy()) for i in range(1, count + 1)],
    )


def settle(at: AppTest, timeout: float = 60) -> AppTest:
    """Chờ job nền "main" xong rồi chạy lại script để vẽ kết quả (AppTest không tự chạy fragment run_every)."""
    deadline = time.monotonic() + timeout
    while "_background_jobs" in at.session_state:
        job = at.session_state["_background_jobs"].get("main")
        if job is None or not job.snapshot().active:
            break
        assert time.monotonic() < deadline, "job nền chưa xong"
        time.sleep(0.05)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def run(upload: tuple | None = None, face_count: int | None = None, roll_result=None, analyze_func=None,
        multiface_func=None, query: dict | None = None) -> AppTest:
    """Chạy app (giả lập EMOCA + mô hình); có `upload` thì tải ảnh lên và chờ phân tích xong."""
    import orchestrator

    at = AppTest.from_file(APP, default_timeout=90)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    names = ("prepare_multiface_views", "roll_correct_upload", "roll_correction_settings", "analyze",
             "analyze_multiface", "_EMOCA_MODE", "_ENGINE_MODE")
    original = {name: getattr(orchestrator, name) for name in names}
    orchestrator._EMOCA_MODE = orchestrator._ENGINE_MODE = "mock"
    orchestrator.analyze = analyze_func or _mock_analyze
    if upload:
        count = face_count
        if count is None:
            try:
                count = 0 if min(Image.open(io.BytesIO(upload[1])).size) < 32 else 1
            except OSError:  # cố ý tải tệp không phải ảnh: app từ chối trước khi dò mặt
                count = 0
        orchestrator.prepare_multiface_views = lambda image, **kwargs: _fake_face_views(image, count)
        orchestrator.roll_correct_upload = roll_result or (lambda image, views: (image, None))
        orchestrator.roll_correction_settings = lambda: (False, 15.)
        if analyze_func:
            orchestrator.analyze = analyze_func
        if multiface_func:
            orchestrator.analyze_multiface = multiface_func
    try:
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        if upload:
            at.file_uploader[0].upload(*upload).run()
            assert not at.exception, [e.value for e in at.exception]
            settle(at)
    finally:
        for name, value in original.items():
            setattr(orchestrator, name, value)
    return at


def md(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _vis_caps(text: str) -> list[str]:
    return re.findall(r'<div class="mx-vis-cap">([^<]*)</div>', text)


def _vis_tops(text: str) -> list[str]:
    return re.findall(r'<div class="mx-vis-top">([^<]*)</div>', text)


def _all_text(at: AppTest) -> str:
    parts = [md(at)] + [x.value for x in list(at.caption) + list(at.info) + list(at.warning) + list(at.error)]
    parts += [img.caption for item in at.image for img in item.proto.imgs]
    return "\n".join(parts)


def _mock_analyze(data, **kwargs):
    """analyze() thật của orchestrator ở chế độ giả lập (dùng khi đã thay orchestrator.analyze bằng hàm khác)."""
    import orchestrator

    return ORIGINAL_ANALYZE(data, **{**kwargs, "emoca": {"mode": "mock"}, "engine_mode": "mock"})


def _orchestrator_analyze():
    import orchestrator

    return orchestrator.analyze


ORIGINAL_ANALYZE = _orchestrator_analyze()


def test_empty_page_is_minimal():
    """Không có Cài đặt, không có bước/ghi chú: chỉ tiêu đề "MA3D-Net" và ô tải ảnh."""
    at = run()
    text = md(at)
    assert '<h1 class="mx-title">MA3D-Net</h1>' in text and "MA3D-Net Lab" not in text
    for absent in ("mx-logo", "mx-stripe", "mx-ico", "<svg"):
        assert absent not in text, f"bỏ icon/đường gradient: còn {absent}"
    assert "So sánh POSTER và MA3D-Net cho một khuôn mặt" not in text, "bỏ dòng mô tả dưới tiêu đề"
    for absent in ("mx-empty", "mx-steps", "mx-sub", "Cài đặt"):
        assert absent not in text, absent
    assert not at.toggle and not at.selectbox and not at.text_input and not at.get("button_group"), "còn control Cài đặt"
    assert not at.button and len(at.file_uploader) == 1
    assert len(at.sidebar.children) == 0
    source = Path(APP).read_text(encoding="utf-8")
    assert "st.popover" not in source and "plotly" not in source
    print("PASS empty_page_is_minimal")


def test_single_face_renders_only_visual_row():
    at = run(("face.png", png(), "image/png"))
    text = md(at)
    caps = _vis_caps(text)
    assert caps[:3] == ["Input", "Facial Landmark", "3DMM"] and len(caps) == 5, caps
    assert caps[3].startswith("POSTER") and caps[4].startswith("MA3D-Net") and caps[4].endswith("(ours)"), caps
    assert 'class="mx-vis-miss"' not in text, "mock có crop_image + attention ⇒ không ô nào thiếu"
    tops = _vis_tops(text)
    assert len(tops) == 5 and all(t.strip() and "%" not in t for t in tops[3:]), "mặc định chỉ hiện nhãn, không %"
    assert "Dữ liệu MOCK" not in text
    everything = _all_text(at)
    for absent in ("So sánh biểu cảm", "Hình học khuôn mặt", "pre-data", "Thử nghiệm", "mx-vis-note", "mx-bar",
                   "Multi-Object", "face.png", " KB", "×240 px", "Ảnh đã được xoay", "crop 224 cho EMOCA"):
        assert absent not in everything, absent
    labels = [button.label for button in at.button]
    assert labels == ["Phân tích lại", "Xóa ảnh"], labels
    assert at.session_state["result"][1]["status"] == "ok"
    print("PASS single_face_renders_only_visual_row")


def test_main_pipeline_requests_real_modes():
    calls = []

    def capture(data, **kwargs):
        calls.append(kwargs)
        return _mock_analyze(data, **kwargs)

    run(("face.png", png(), "image/png"), analyze_func=capture)
    assert len(calls) == 1
    assert calls[0]["emoca"]["mode"] == "real" and calls[0]["engine_mode"] == "real", calls[0]
    print("PASS main_pipeline_requests_real_modes")


def test_single_face_rotation_adds_tile():
    source_bytes = png((320, 240))
    source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    rotated = source.rotate(-90, resample=Image.Resampling.BICUBIC, expand=True)
    received = []

    def fake_roll(image, views):
        return rotated, {"rotation_deg": -90., "roll_deg": 90., "threshold_deg": 15.}

    def fake_analyze(data, **kwargs):
        received.append(Image.open(io.BytesIO(data)).convert("RGB"))
        return _mock_analyze(data, **kwargs)

    at = run(("face.png", source_bytes, "image/png"), roll_result=fake_roll, analyze_func=fake_analyze)
    assert received[0].size == rotated.size and received[0].tobytes() == rotated.tobytes(), "phải gửi ảnh đã xoay"
    caps = _vis_caps(md(at))
    assert caps[:3] == ["Input", "Rotated Face", "Facial Landmark"] and len(caps) == 6, caps
    assert "Ảnh đã được xoay" not in _all_text(at), "không còn dòng chú thích xoay"

    def below(image, views):
        return image, {"rotation_deg": 0., "roll_deg": 3., "threshold_deg": 15.}

    at = run(("face.png", source_bytes, "image/png"), roll_result=below, analyze_func=fake_analyze)
    caps = _vis_caps(md(at))
    assert "Rotated Face" not in caps and len(caps) == 5, caps
    print("PASS single_face_rotation_adds_tile")


def _multi(rows_spec: list[tuple[int, bool, bool]]):
    """rows_spec: (face_id, emoca_nhận_mặt, đã_xoay) → hàm thay analyze_multiface."""
    import orchestrator  # noqa: F401 — analyze giả lập lấy qua _mock_analyze

    def fake_multiface(image, **kwargs):
        buf = io.BytesIO()
        image.crop((0, 0, 80, 80)).save(buf, format="PNG")
        rows = []
        for face_id, detected, rotated in rows_spec:
            result = (_mock_analyze(buf.getvalue(), max_faces=1) if detected else
                      {"status": "error", "faces": [], "error": {"code": "no_face", "message": "EMOCA không thấy mặt"}})
            rows.append({"face_id": face_id, "reference": image.copy(), "processed": image.crop((0, 0, 80, 80)),
                         "roll_info": {"applied": rotated, "rotation_deg": 30. if rotated else 0.},
                         "result": result})
        return {"kind": "faces", "views": kwargs.get("views"), "rows": rows}

    return fake_multiface


def test_multiface_rows_drop_undetected_faces():
    """Nhiều mặt: một hàng / mặt EMOCA nhận được; mặt không nhận được bị bỏ; có mặt được xoay ⇒ mọi hàng 6 ô."""
    at = run(("group.png", png(), "image/png"), face_count=3,
             multiface_func=_multi([(1, True, False), (2, False, False), (3, True, True)]))
    assert "result" not in at.session_state, "nhiều mặt không chạy luồng một mặt"
    text = md(at)
    caps = _vis_caps(text)
    assert len(caps) == 12 and caps.count("Input") == 2 and caps.count("Rotated Face") == 2, caps
    assert caps[:3] == ["Input", "Rotated Face", "Facial Landmark"], caps
    assert text.count('class="mx-vis-miss"') == 1 and "No rotation needed" in text, "mặt 1 không xoay ⇒ ô Rotated Face ghi lý do"
    everything = _all_text(at)
    for absent in ("Face #", "View Details", "Processing notes", "EMOCA không thấy mặt", "Confidence", "mx-face-prediction"):
        assert absent not in everything, absent

    at = run(("group.png", png(), "image/png"), face_count=2, multiface_func=_multi([(1, True, False), (2, True, False)]))
    caps = _vis_caps(md(at))
    assert len(caps) == 10 and "Rotated Face" not in caps, caps

    at = run(("group.png", png(), "image/png"), face_count=2, multiface_func=_multi([(1, False, False), (2, False, False)]))
    assert not _vis_caps(md(at)) and any("[no_face]" in e.value for e in at.error), [e.value for e in at.error]
    print("PASS multiface_rows_drop_undetected_faces")


def test_error_states():
    at = run(upload=("tiny.png", png((20, 20)), "image/png"))
    assert any("[no_face]" in e.value for e in at.error), [e.value for e in at.error]
    at = run(upload=("junk.png", b"day khong phai anh", "image/png"))
    assert any("[image_invalid]" in e.value for e in at.error), [e.value for e in at.error]
    print("PASS error_states")


def test_partial_and_pipeline_errors():
    import json
    partial = json.loads((ROOT / "backend" / "contracts" / "web_analyze.example.partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "partial", partial["status"]
    at = run(("face.png", png(), "image/png"), analyze_func=lambda data, **kw: partial)
    warns = "\n".join(w.value for w in at.warning)
    assert partial["error"]["code"] in warns, (partial["error"], warns[:300])  # `partial` là cảnh báo, không phải lỗi đỏ
    assert not any(partial["error"]["code"] in e.value for e in at.error), [e.value for e in at.error]

    def boom(data, **kw):
        raise RuntimeError("giả lập orchestrator sập")

    at = run(("face.png", png(), "image/png"), analyze_func=boom)
    assert any("giả lập orchestrator sập" in e.value for e in at.error), [e.value for e in at.error]
    assert "result" not in at.session_state, "lỗi không được để lại kết quả cũ trong phiên"
    at = run(("face.png", png(), "image/png"), analyze_func=lambda data, **kw: "không phải dict")
    assert any("không trả kết quả đúng định dạng" in e.value for e in at.error), [e.value for e in at.error]
    print("PASS partial_and_pipeline_errors")


def test_orchestrator_import_error_is_reported():
    """Không nạp được orchestrator ⇒ lỗi đỏ rõ ràng, không lặng lẽ hiện fixture như dữ liệu mock."""
    saved = sys.modules.get("orchestrator")
    at = AppTest.from_file(APP, default_timeout=90)
    sys.modules["orchestrator"] = None  # `from orchestrator import ...` ⇒ ImportError
    try:
        at.run()
    finally:
        sys.modules["orchestrator"] = saved
    assert not at.exception, [x.value for x in at.exception]
    assert not at.file_uploader, "không có bộ điều phối thì không mời người dùng tải ảnh"
    assert any("Không nạp được bộ điều phối" in e.value for e in at.error), [e.value for e in at.error]
    print("PASS orchestrator_import_error_is_reported")


def test_rerun_and_remove_buttons():
    calls = []

    def counting(data, **kwargs):
        calls.append(kwargs)
        return _mock_analyze(data, **kwargs)

    import orchestrator

    original = orchestrator.analyze
    orchestrator.analyze = counting
    try:
        at = run(("face.png", png(), "image/png"), analyze_func=counting)
        assert len(calls) == 1 and calls[0]["max_faces"] == 1
        at.run()
        assert len(calls) == 1, "chạy lại trang không được phân tích lại"
        [b for b in at.button if b.label == "Phân tích lại"][0].click().run()
        settle(at)
        assert len(calls) == 2, "bấm Phân tích lại phải phân tích lại"
        [b for b in at.button if b.label == "Xóa ảnh"][0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert "_primary_upload" not in at.session_state and "result" not in at.session_state
        assert not _vis_caps(md(at)) and not at.button
    finally:
        orchestrator.analyze = original
    print("PASS rerun_and_remove_buttons")


def test_html_is_escaped():
    evil = "<img src=x onerror=alert(1)>"

    def evil_analyze(data, **kwargs):
        result = _mock_analyze(data, **kwargs)
        for pred in result["faces"][0]["predictions"]:
            pred["display_name"] = evil
            pred["top1"]["label"] = evil
        return result

    at = run(("face.png", png(), "image/png"), analyze_func=evil_analyze)
    text = md(at)
    assert "&lt;img src=x onerror=alert(1)&gt;" in text and "<img src=x" not in text, text[:300]
    print("PASS html_is_escaped")


def _b64png(img: Image.Image) -> dict:
    import base64
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return {"format": "png", "b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def test_attention_overlay_helpers():
    """Hàm dựng ảnh của khối 'Kết quả trực quan' (PIL, không HTML): phủ attention đúng kích thước, đúng hướng nóng/lạnh."""
    import app
    crop = Image.new("RGB", (224, 224), (128, 128, 128))
    grid = {"space": "crop224", "grid": [7, 7], "values": [1.0] + [0.0] * 48}   # nóng nhất ở ô trên-trái
    img, why = app.attention_overlay(crop, grid)
    assert img is not None and why == "" and img.size == (224, 224) and img.mode == "RGB", (img, why)
    tl, br = img.getpixel((4, 4)), img.getpixel((219, 219))
    assert tl != (128, 128, 128) and br != (128, 128, 128), "phải trộn màu lên crop"
    assert tl[0] > br[0] and br[2] > tl[2], f"Turbo: ô nóng ngả đỏ, ô lạnh ngả xanh ({tl} vs {br})"
    small = Image.new("RGB", (100, 100), (90, 90, 90))   # ảnh model_input 100×100 (ADR-020) ⇒ vẫn ra 224×224
    assert app.attention_overlay(small, grid)[0].size == (224, 224)
    assert app.attention_overlay(crop, None) == (None, "No attention returned by the model")
    none_img, why = app.attention_overlay(crop, {**grid, "space": "original_px"})
    assert none_img is None and "crop224" in why, why
    assert app.attention_overlay(crop, {**grid, "values": [0.5] * 10})[0] is None, "sai hình dạng ⇒ None"
    lm = app.landmark_tile(crop, [[112, 112]], ref_size=224)
    assert lm.size == (224, 224) and lm.getpixel((112, 112)) == app.VIS_LANDMARK_RGB, lm.getpixel((112, 112))
    print("PASS attention_overlay_helpers")


def test_visual_tile_missing_attention_and_model_input():
    """Thiếu attention ⇒ ô ghi lý do, không vỡ; có model_input (ADR-020) ⇒ attention phủ lên ảnh căn chỉnh."""
    def no_poster_attention(data, **kwargs):
        result = _mock_analyze(data, **kwargs)
        for pred in result["faces"][0]["predictions"]:
            if pred.get("model_id") == "poster_rafdb":
                pred.pop("attention", None)
        return result

    at = run(("face.png", png(), "image/png"), analyze_func=no_poster_attention)
    text = md(at)
    assert len(_vis_caps(text)) == 5 and text.count('class="mx-vis-miss"') == 1, text.count('class="mx-vis-miss"')
    assert "No attention returned by the model" in text

    import app
    face = {"model_input": {"image": _b64png(Image.new("RGB", (100, 100), (60, 120, 180)))},
            "predictions": [{"model_id": "ma3d_rafdb", "status": "ok", "top1": {"label": "happiness", "prob": .9},
                             "attention": {"space": "crop224", "grid": [7, 7], "values": [0.0] * 49}}]}
    assert app.analysis_tiles(face, None, {"image": {"width": 1, "height": 1}}, {})[2]["top"] == "happiness"
    tiles = app.analysis_tiles(face, None, {"image": {"width": 1, "height": 1}}, {}, show_prob=True)
    assert [t["title"] for t in tiles][:2] == ["Facial Landmark", "3DMM"] and tiles[0]["img"] is None
    ma3d = tiles[2]
    assert ma3d["img"] is not None and ma3d["img"].getpixel((112, 112))[2] > 100, "phủ lên ảnh model_input (xanh)"
    assert ma3d["top"].endswith("90%"), ma3d["top"]
    print("PASS visual_tile_missing_attention_and_model_input")


def test_upload_dropzone():
    """Ô tải ảnh: dropzone lớn viền đứt, chữ hướng dẫn tiếng Việt, giới hạn dung lượng lấy từ configs/services.yaml,
    sáng viền khi có focus bàn phím; ẩn chip tên/dung lượng tệp (ADR-032)."""
    import yaml
    limit = yaml.safe_load((ROOT / "backend" / "configs" / "services.yaml").read_text(encoding="utf-8"))["limits"]["max_image_bytes"]
    mb = f"{limit / (1024 * 1024):g}"
    at = run()
    css = md(at)
    zone = re.search(r'\[data-testid="stFileUploaderDropzone"\] \{([^}]*)\}', css)
    assert zone and "min-height:168px" in zone.group(1) and "dashed" in zone.group(1), "dropzone phải là ô lớn viền đứt"
    assert 'content:"Kéo thả ảnh vào đây hoặc bấm để chọn tệp"' in css
    assert f'--mx-upload-hint: "JPEG · PNG · WebP · tối đa {mb} MB · ảnh chỉ nằm trong bộ nhớ, không lưu lại"' in css, \
        "giới hạn dung lượng phải lấy từ configs/services.yaml"
    assert 'content:var(--mx-upload-hint' in css and ':has(:focus-visible)' in css, "thiếu dòng giới hạn / viền focus bàn phím"
    assert '[data-testid="stFileChips"] { display:none !important; }' in css, "phải ẩn tên/dung lượng tệp"
    assert "Kéo thả ảnh khác vào đây hoặc bấm vào ô để thay ảnh" in css, "đã có ảnh: phải hướng dẫn thay ảnh"
    print("PASS upload_dropzone")


def main():
    test_empty_page_is_minimal()
    test_upload_dropzone()
    test_single_face_renders_only_visual_row()
    test_main_pipeline_requests_real_modes()
    test_single_face_rotation_adds_tile()
    test_multiface_rows_drop_undetected_faces()
    test_error_states()
    test_partial_and_pipeline_errors()
    test_orchestrator_import_error_is_reported()
    test_rerun_and_remove_buttons()
    test_html_is_escaped()
    test_attention_overlay_helpers()
    test_visual_tile_missing_attention_and_model_input()
    print("ALL PASS")


if __name__ == "__main__":
    main()
