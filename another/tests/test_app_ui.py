"""Kiểm thử frontend/app.py bằng streamlit.testing (chạy nguyên script, không cần trình duyệt).

Chạy trong môi trường có streamlit:  <python của venv web> tests/test_app_ui.py
Thiếu streamlit thì báo SKIP và thoát mã 0 (để `python tests/test_*.py` bằng Python thường không bị đỏ oan).
"""
import io
import re
import sys
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


def run(query: dict | None = None, upload: tuple | None = None, face_count: int | None = None,
        roll_result=None, analyze_func=None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    for k, v in (query or {}).items():
        at.query_params[k] = v
    # Only uploads need the detector stub; importing here keeps
    # test_orchestrator_import_error_is_reported able to block the module.
    orchestrator = None
    if upload:
        import orchestrator
    original_prepare = orchestrator.prepare_multiface_views if orchestrator else None
    original_roll = orchestrator.roll_correct_upload if orchestrator else None
    original_settings = orchestrator.roll_correction_settings if orchestrator else None
    original_analyze = orchestrator.analyze if orchestrator else None
    if upload:
        count = face_count
        if count is None:
            try:
                count = 0 if min(Image.open(io.BytesIO(upload[1])).size) < 32 else 1
            except OSError:  # deliberately invalid upload (test_error_states): app rejects it before detection
                count = 0
        orchestrator.prepare_multiface_views = lambda image, **kwargs: _fake_face_views(image, count)
        orchestrator.roll_correct_upload = roll_result or (lambda image, views: (image, None))
        orchestrator.roll_correction_settings = lambda: (False, 15.)
        if analyze_func:
            orchestrator.analyze = analyze_func
    try:
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        if upload:
            at.file_uploader[0].upload(*upload).run()
            assert not at.exception, [e.value for e in at.exception]
    finally:
        if orchestrator:
            orchestrator.prepare_multiface_views = original_prepare
            orchestrator.roll_correct_upload = original_roll
            orchestrator.roll_correction_settings = original_settings
            orchestrator.analyze = original_analyze
    return at


def md(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_empty_state_all_modes():
    for mode in ("light", "dark"):
        for level in ("normal", "high"):
            at = run({"m": mode, "c": level})
            text = md(at)
            assert "--brand:" in text and "mx-empty" in text, (mode, level)
            assert ("color-scheme: only light" in text) == (mode == "light")
    print("PASS empty_state_all_modes")


def test_upload_flow_renders_components():
    at = run({"m": "light"}, ("face.png", png(), "image/png"))
    text = md(at)
    for cls in ("mx-bar", "mx-top1", "mx-insight"):
        assert cls in text, f"thiếu {cls}"
    assert "Radar Chart" not in text and "mx-radar" not in text, "radar chart section must be removed"
    assert not any("Multi-Object Analysis" in button.label for button in at.button), \
        "Multi-Object Analysis is automatic and has no action button"
    result = at.session_state["result"][1]
    faces = result.get("faces") or []
    if faces and any(p.get("feature_maps") or p.get("attention") for p in faces[0].get("predictions", [])):
        assert "mx-heat" in text, "có feature_maps/attention nhưng không vẽ bản đồ nhiệt"
        assert "độ lớn kích hoạt" in "\n".join(c.value for c in at.caption), "thiếu chú thích ADR-013"
    assert "Dữ liệu MOCK" in text
    assert "plotly" not in Path(APP).read_text(encoding="utf-8"), "đã bỏ Plotly (biểu đồ vẽ bằng HTML/SVG), không được dùng lại"
    print("PASS upload_flow_renders_components")


def test_frontalization_experiment_section_is_disabled_in_mock_mode():
    at = run({"m": "light"}, ("face.png", png(), "image/png"))
    text = md(at)
    assert "Thử nghiệm: Sửa nghiêng (roll) — 3DDFA-V2" in text
    assert "3DDFA-V2 đo roll → xoay 2D" in text and "yaw và pitch không được chỉnh (ADR-028)" in text
    buttons = [button for button in at.button if button.key == "run_frontalize_3ddfa"]
    assert len(buttons) == 1 and buttons[0].disabled and buttons[0].label == "Experiment: Sửa nghiêng (3DDFA-V2)"
    assert any("Bật EMOCA thật" in item.value for item in at.info)
    print("PASS frontalization_experiment_section_is_disabled_in_mock_mode")


def test_single_face_roll_upload_bytes_and_caption():
    from types import SimpleNamespace

    source_bytes = png((320, 240))
    source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    received = []
    rotated = source.rotate(-90, resample=Image.Resampling.BICUBIC, expand=True)
    def fake_roll(image, views):
        return rotated, {"applied": True, "rotation_deg": -90., "roll_deg": 90., "threshold_deg": 15.}
    def fake_analyze(data, **kwargs):
        sent = Image.open(io.BytesIO(data)).convert("RGB")
        received.append(sent)
        return {"status": "error", "faces": [], "error": {"code": "internal_error", "message": "fake"}}
    at = run(upload=("face.png", source_bytes, "image/png"), roll_result=fake_roll, analyze_func=fake_analyze)
    assert received[0].size == rotated.size and received[0].tobytes() == rotated.tobytes()
    assert any("Ảnh đã được xoay -90°" in item.value and "|roll| 90° ≥ 15°" in item.value
               for item in at.caption)
    def below(image, views):
        return image, {"applied": False, "rotation_deg": 0., "roll_deg": 3., "threshold_deg": 15.}
    at = run(upload=("face.png", source_bytes, "image/png"), roll_result=below, analyze_func=fake_analyze)
    assert not any("Ảnh đã được xoay" in item.value for item in at.caption)
    print("PASS single_face_roll_upload_bytes_and_caption")


def test_yolo_shadow_is_opt_in_and_preserves_primary_result():
    from hashlib import sha256
    from types import SimpleNamespace

    import predata_experiment
    import orchestrator

    calls = []
    def fake_build(image):
        calls.append(image.size)
        crop = image.crop((0, 0, max(1, image.width // 2), max(1, image.height // 2)))
        detection = SimpleNamespace(cropped_face=crop, bbox_xyxy=(0, 0, crop.width, crop.height),
                                    confidence=0.9, crop_size=crop.size)
        return SimpleNamespace(
            source_image=image.copy(), enhanced_image=image.copy(), enhanced_detection=detection,
            branch_errors={"enhanced_detection": None},
            stage_times_ms={"general_restoration": 1.0, "enhanced_detection": 2.0},
        )

    original = predata_experiment.build_predata_views
    original_roll, original_settings = orchestrator.roll_correct_upload, orchestrator.roll_correction_settings
    predata_experiment.build_predata_views = fake_build
    orchestrator.roll_correct_upload = lambda image, views: (image, None)
    orchestrator.roll_correction_settings = lambda: (False, 15.)
    try:
        data = png()
        at = run(upload=("face.png", data, "image/png"))
        assert calls == [], "YOLO shadow must stay opt-in"
        primary_before = at.session_state["result"][1]
        [button for button in at.button if button.key == "run_yolo26_shadow"][0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert calls == [(320, 240)]
        shadow = at.session_state["predata_views"]
        assert shadow["digest"] == sha256(data).hexdigest()
        assert shadow["views"].enhanced_detection.cropped_face.size == (160, 120)
        assert "RealESRGAN" in "\n".join(c.value for c in at.caption)
        assert at.session_state["result"][1] == primary_before, "shadow crop must not alter primary analysis"
    finally:
        predata_experiment.build_predata_views = original
        orchestrator.roll_correct_upload, orchestrator.roll_correction_settings = original_roll, original_settings
    print("PASS yolo_shadow_is_opt_in_and_preserves_primary_result")


def test_multi_object_grid_skips_single_prediction_and_renders_face_cards():
    import copy
    from types import SimpleNamespace

    import orchestrator

    original = orchestrator.analyze_multiface
    calls = []

    def fake_multiface(image, **kwargs):
        from io import BytesIO

        calls.append(kwargs.get("max_faces"))
        encoded = BytesIO()
        image.save(encoded, format="PNG")
        prediction = orchestrator.analyze(
            encoded.getvalue(), emoca={"mode": "mock"}, engine_mode="mock",
            model_ids=["ma3d_rafdb"], max_faces=1,
        )
        return {
            "kind": "faces",
            "views": SimpleNamespace(source_image=image.copy(), errors=[]),
            "rows": [
                {"face_id": face_id, "reference": image.copy(), "processed": image.crop((0, 0, 80, 80)),
                 "result": prediction}
                for face_id in (1, 2)
            ],
        }

    orchestrator.analyze_multiface = fake_multiface
    try:
        at = run({"m": "light"}, ("group.png", png(), "image/png"), face_count=2)
        assert "result" not in at.session_state, "multi-person input must skip single-person prediction"

        assert not at.exception, [error.value for error in at.exception]
        assert calls == [30]
        assert "result" not in at.session_state
        text = md(at)
        assert "Face #1" in text and "Face #2" in text
        labels = [button.label for button in at.button]
        assert "View Details" in labels
        assert not any("Multi-Object Analysis" in label for label in labels)
        assert "So sánh biểu cảm" not in text and "Radar Chart" not in text
        assert "mx-face-prediction" in text, "each face card must show a compact prediction"
    finally:
        orchestrator.analyze_multiface = original
    print("PASS multi_object_grid_skips_single_prediction_and_renders_face_cards")


def test_predata_branch_errors_are_visible():
    import app

    message = app._variant_error_message(
        {"status": "error", "error": {"code": "no_face", "message": "EMOCA found no face"}}, None
    )
    assert message == "[no_face] EMOCA found no face", message
    fallback = app._variant_error_message({"status": "error", "warnings": ["emoca_svc_unavailable"]}, None)
    assert fallback == "emoca_svc_unavailable", fallback
    print("PASS predata_branch_errors_are_visible")


def test_error_states():
    at = run(upload=("tiny.png", png((20, 20)), "image/png"))
    assert any("[no_face]" in e.value for e in at.error), [e.value for e in at.error]
    at = run(upload=("junk.png", b"day khong phai anh", "image/png"))
    assert any("[image_invalid]" in e.value for e in at.error), [e.value for e in at.error]
    print("PASS error_states")


def test_settings_from_query_and_validation():
    at = run({"m": "dark", "c": "high", "f": "L", "rm": "1", "lm": "0"})
    ss = at.session_state
    assert (ss["pref_mode"], ss["pref_level"], ss["pref_font"], ss["pref_still"], ss["pref_lm"]) == ("dark", "high", "L", True, False)
    at = run({"t": "khong-co-chu-de-nay", "m": "tim", "c": "x", "f": "?", "rm": "maybe"})
    ss = at.session_state
    assert ss["pref_theme"] == "cold" and ss["pref_mode"] == "light" and ss["pref_level"] == "normal", dict(ss)
    assert ss["pref_font"] == "M" and ss["pref_still"] is False, dict(ss)  # giá trị lạ → mặc định, không lỗi
    print("PASS settings_from_query_and_validation")


def rerun_btn(at: AppTest):
    return [b for b in at.button if b.label == "Phân tích lại"][0]


def settings_panel(at: AppTest):
    # expander có icon được AppTest xếp vào at.status
    found = [e for e in list(at.expander) + list(at.status) if e.label == "Cài đặt"]
    assert len(found) == 1, [e.label for e in list(at.expander) + list(at.status)]
    return found[0]


def test_settings_panel_at_top():
    """Cài đặt nằm trong nút mở/đóng "Cài đặt" ở đầu trang, không phụ thuộc thanh bên (thu gọn thì không mở lại được)."""
    import json
    at = run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.sidebar.children) == 0, "không còn gì trong thanh bên"
    panel = settings_panel(at)
    assert panel.proto.expanded is False, "mặc định đóng"
    first_block = next(c for c in at.main.children.values() if c.type != "markdown")
    assert first_block is panel, "bảng Cài đặt là khối đầu tiên, ngay dưới tiêu đề"
    keys = {w.key for w in list(panel.selectbox) + list(panel.toggle) + list(panel.button)
            + list(panel.get("button_group"))}
    need = {"pref_theme", "pref_mode", "pref_level", "pref_font", "pref_still", "pref_lm", "pref_bb", "pref_att",
            "engine_mode", "emoca_mode", "btn_reset"}
    assert need <= keys, f"thiếu control trong bảng Cài đặt: {need - keys}"
    # đổi cài đặt bên trong bảng vẫn có tác dụng như trước
    panel.toggle(key="pref_still").set_value(True).run()
    assert at.session_state["pref_still"] is True and "--motion:0s" in md(at)
    # EMOCA thật: ô địa chỉ/token cũng nằm trong bảng, token không lên địa chỉ trang
    at.session_state["emoca_mode"] = "real"
    at.run()
    panel = settings_panel(at)
    assert {"emoca_url", "emoca_token"} <= {t.key for t in panel.text_input}, "ô EMOCA phải nằm trong bảng Cài đặt"
    panel.text_input(key="emoca_token").set_value("tok-khong-len-url").run()
    assert "tok-khong-len-url" not in json.dumps(dict(at.query_params)), "token lọt lên query params"
    print("PASS settings_panel_at_top")


def test_settings_widgets_and_persistence():
    at = run()
    assert len(at.selectbox) == 1 and len(at.toggle) == 4, (len(at.selectbox), len(at.toggle))
    at.toggle[0].set_value(True).run()  # "Giảm chuyển động"
    assert at.session_state["pref_still"] is True and "--motion:0s" in md(at)
    assert at.query_params["rm"] in ("1", ["1"]), dict(at.query_params)  # cài đặt được ghi lên địa chỉ trang
    # đặt lại
    [b for b in at.button if b.key == "btn_reset"][0].click().run()
    assert at.session_state["pref_still"] is False
    print("PASS settings_widgets_and_persistence")


def test_theme_switch_changes_css():
    at = run({"t": "cold"})
    css_a = md(at)
    before = at.session_state["pref_theme"]
    at.selectbox[0].select_index(3).run()  # chọn theo vị trí (nhãn hiển thị khác id)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["pref_theme"] != before, "chọn chủ đề khác phải đổi cài đặt"
    assert md(at) != css_a, "đổi chủ đề phải đổi CSS"
    print("PASS theme_switch_changes_css")


def test_html_is_escaped():
    import app  # nhập được vì main() chỉ chạy khi __name__ == "__main__"
    from theme import derive_tokens, DEFAULT_PALETTE
    tokens = derive_tokens(DEFAULT_PALETTE.colors, "light")
    evil = "<img src=x onerror=alert(1)>"
    pred = {"status": "ok", "model_id": evil, "display_name": evil, "dataset": evil, "labels": [evil, "b", "c"],
            "probs": [0.5, 0.3, 0.2], "top1": {"index": 0, "label": evil, "prob": 0.5},
            "feature_maps": {"image_backbone": {"space": "crop224", "grid": [7, 7], "values": [0.1] * 49}}}
    for html_out in (app.bars_html(pred, tokens, {}), app.feature_maps_html(pred, tokens) or ""):
        assert "<img" not in html_out and "onerror=alert" not in html_out.replace("&lt;img src=x onerror=alert(1)&gt;", ""), html_out[:200]
    assert app.heat_html({"grid": [7, 7], "values": [0.1] * 48}, tokens, "x") is None  # sai hình dạng → không vẽ
    print("PASS html_is_escaped")


def _with_analyze(fn, body):
    """Chạy `body` khi orchestrator.analyze bị thay bằng `fn` (AppTest chạy cùng tiến trình nên thay được)."""
    import orchestrator
    real = orchestrator.analyze
    orchestrator.analyze = fn
    try:
        return body()
    finally:
        orchestrator.analyze = real


def test_partial_and_service_errors():
    import json
    partial = json.loads((ROOT / "backend" / "contracts" / "web_analyze.example.partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "partial", partial["status"]

    def show_partial():
        at = run(upload=("face.png", png(), "image/png"))
        warns = "\n".join(w.value for w in at.warning)
        assert partial["error"]["code"] in warns, (partial["error"], warns[:300])  # `partial` hiện dạng cảnh báo, không phải lỗi đỏ
        assert not any(partial["error"]["code"] in e.value for e in at.error), [e.value for e in at.error]
    _with_analyze(lambda data, **kw: partial, show_partial)

    def boom(data, **kw):
        raise RuntimeError("giả lập orchestrator sập")

    def show_crash():
        at = run(upload=("face.png", png(), "image/png"))
        assert any("[service_unavailable]" in e.value and "RuntimeError" in e.value for e in at.error), [e.value for e in at.error]
        assert "result" not in at.session_state, "lỗi không được để lại kết quả cũ trong phiên"
    _with_analyze(boom, show_crash)

    def show_bad_type():
        at = run(upload=("face.png", png(), "image/png"))
        assert any("[contract_mismatch]" in e.value for e in at.error), [e.value for e in at.error]
    _with_analyze(lambda data, **kw: "không phải dict", show_bad_type)
    print("PASS partial_and_service_errors")


def test_orchestrator_import_error_is_reported():
    """Không nạp được orchestrator ⇒ lỗi đỏ rõ ràng, không lặng lẽ hiện fixture như dữ liệu mock."""
    import sys
    saved = sys.modules.get("orchestrator")
    sys.modules["orchestrator"] = None  # `from orchestrator import ...` ⇒ ImportError
    try:
        at = run()
    finally:
        sys.modules["orchestrator"] = saved
    assert not at.file_uploader, "không có bộ điều phối thì không mời người dùng tải ảnh"
    errors = [e.value for e in at.error]
    assert any("[service_unavailable]" in e and "orchestrator" in e for e in errors), errors
    assert not at.exception, [x.value for x in at.exception]
    assert "result" not in at.session_state
    print("PASS orchestrator_import_error_is_reported")


def test_emoca_panel_and_consent():
    """ADR-015: chọn EMOCA từ xa thì ảnh KHÔNG được gửi cho tới khi người dùng xác nhận; đổi địa chỉ thu hồi xác nhận;
    token không lọt vào trang/kết quả; địa chỉ loopback không cần xác nhận."""
    import json
    import os

    import emoca_client

    saved_env = {k: os.environ.pop(k, None) for k in ("EMOCA_MODE", "EMOCA_URL", "EMOCA_TOKEN")}
    real_get = emoca_client.get_analysis
    calls: list[dict] = []

    def spy(image, **kw):  # thay đường mạng: đếm số lần được gọi, trả dữ liệu mock
        calls.append(kw)
        return real_get(image, **{**kw, "mode": "mock", "endpoint": None})

    emoca_client.get_analysis = spy
    try:
        at = run()
        assert not at.text_input, "chế độ Giả lập không có ô địa chỉ/token"
        assert at.session_state["emoca_mode"] == "mock" and at.session_state["emoca_consent"] is False

        at.session_state["emoca_mode"] = "real"
        at.session_state["emoca_url"] = "https://abc.trycloudflare.com"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(at.text_input) == 2 and at.toggle(key="emoca_consent").value is False
        assert "abc.trycloudflare.com" in "\n".join(c.value for c in at.caption), "phải nói rõ ảnh sẽ tới máy nào"

        at.file_uploader[0].upload("face.png", png(), "image/png").run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("[consent_required]" in e.value for e in at.error), [e.value for e in at.error]
        assert calls == [], "chưa xác nhận mà vẫn gọi ra ngoài"

        at.toggle(key="emoca_consent").set_value(True).run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(calls) == 1 and calls[0]["mode"] == "real" and calls[0]["endpoint"]["url"] == "https://abc.trycloudflare.com"
        assert not any("[consent_required]" in e.value for e in at.error), [e.value for e in at.error]

        at.text_input(key="emoca_token").set_value("tok-bi-mat-xyz").run()
        assert len(calls) == 1, "đổi token không tự phân tích lại"
        blob = md(at) + "\n".join(x.value for x in list(at.caption) + list(at.error) + list(at.warning))
        assert "tok-bi-mat-xyz" not in blob and "tok-bi-mat-xyz" not in json.dumps(at.session_state["result"][1]), "token lọt vào trang/kết quả"
        rerun_btn(at).click().run()  # "Phân tích lại" dùng token mới
        assert len(calls) == 2 and calls[-1]["endpoint"]["token"] == "tok-bi-mat-xyz"

        at.text_input(key="emoca_url").set_value("https://khac.example.com").run()
        assert at.session_state["emoca_consent"] is False, "đổi địa chỉ phải thu hồi xác nhận"
        assert any("[consent_required]" in e.value for e in at.error) and len(calls) == 2, "địa chỉ mới, chưa xác nhận ⇒ không gửi"

        at.text_input(key="emoca_url").set_value("http://127.0.0.1:8765").run()
        assert not at.exception, [e.value for e in at.exception]
        assert not [t for t in at.toggle if t.key == "emoca_consent"], "địa chỉ loopback không cần ô xác nhận"
        assert len(calls) == 3 and not any("[consent_required]" in e.value for e in at.error), "loopback phải chạy thẳng"
        assert "không rời khỏi máy" in "\n".join(c.value for c in at.caption)
    finally:
        emoca_client.get_analysis = real_get
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v
    print("PASS emoca_panel_and_consent")


def test_engine_mode_control():
    """Ô "Mô hình POSTER / MA3D-Net": chọn Thật (CPU) ⇒ analyze nhận engine_mode="real", cảnh báo nói đúng phần nào là mock;
    đổi chế độ ⇒ phân tích lại. Theo dõi engine.predict (không nạp torch thật trong test giao diện)."""
    import os

    import engine

    saved_env = os.environ.pop("ENGINE_MODE", None)
    real_predict = engine.predict
    modes: list[str] = []

    def spy(image, codes, models=None, *, attention=False, feature_maps=True, mode="mock", **kw):
        modes.append(mode)
        return real_predict(image, codes, models, attention=attention, feature_maps=feature_maps, mode="mock", **kw)

    engine.predict = spy
    try:
        at = run()
        assert at.session_state["engine_mode"] == "mock", "mặc định là Giả lập (config engine.mode = mock)"
        at.file_uploader[0].upload("face.png", png(), "image/png").run()
        assert not at.exception, [e.value for e in at.exception]
        assert modes == ["mock"]
        at.session_state["engine_mode"] = "real"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert modes == ["mock", "real"], f"đổi chế độ phải phân tích lại với mode=real (được {modes})"
        assert any("mã 3DMM do EMOCA thật" in c.value and "EMOCA giả lập" in c.value for c in at.caption), \
            "phải giải thích vì sao MA3D báo lỗi (cần EMOCA thật, ADR-017)"
        assert not any("thiếu file thống kê" in c.value for c in at.caption), "caption cũ: stats RAF-DB đã có (ADR-017)"

        rerun_btn(at).click().run()  # "Phân tích lại"
        assert modes == ["mock", "real", "real"] and not at.exception
    finally:
        engine.predict = real_predict
        if saved_env is not None:
            os.environ["ENGINE_MODE"] = saved_env
    print("PASS engine_mode_control")


def test_real_engine_with_mock_emoca_blocks_ma3d():
    """Engine "Thật" + EMOCA giả lập ⇒ orchestrator truyền codes_real=False; MA3D-Net hiện lỗi codes_missing nói rõ
    "giả lập" (engine thật từ chối trước khi nạp mô hình). POSTER lấy từ mock để khỏi nạp torch trong test giao diện."""
    import os

    import engine

    saved_env = os.environ.pop("ENGINE_MODE", None)
    real_predict = engine.predict
    seen: list = []

    def spy(image, codes, models=None, *, attention=False, feature_maps=True, mode="mock", **kw):
        seen.append((mode, kw.get("codes_real")))
        out = real_predict(image, codes, models, attention=attention, feature_maps=feature_maps, mode="mock")
        if mode == "real" and kw.get("codes_real") is False:  # đường thật của engine: chặn trước khi nạp mô hình
            blocked = real_predict(image, codes, ["ma3d_rafdb"], mode="real", codes_real=False)["predictions"][0]
            out["predictions"] = [blocked if p["model_id"] == "ma3d_rafdb" else p for p in out["predictions"]]
        return out

    engine.predict = spy
    try:
        at = run()
        at.session_state["engine_mode"] = "real"
        at.file_uploader[0].upload("face.png", png(), "image/png").run()
        assert not at.exception, [e.value for e in at.exception]
        assert seen and seen[-1] == ("real", False), f"EMOCA giả lập ⇒ engine phải nhận codes_real=False ({seen})"
        preds = [p for f in at.session_state["result"][1]["faces"] for p in f["predictions"] if p["model_id"] == "ma3d_rafdb"]
        assert preds and preds[0]["status"] == "error" and preds[0]["error"]["code"] == "codes_missing", preds
        assert "giả lập" in preds[0]["error"]["message"], preds[0]["error"]
        text = md(at)
        errs = "\n".join(e.value for e in at.error) + "\n".join(w.value for w in at.warning)
        assert "[codes_missing]" in errs, "mô hình MA3D-Net phải báo lỗi codes_missing"
        assert "EMOCA giả lập" in errs + text, "giao diện phải nói MA3D-Net bị chặn vì EMOCA giả lập"
    finally:
        engine.predict = real_predict
        if saved_env is not None:
            os.environ["ENGINE_MODE"] = saved_env
    print("PASS real_engine_with_mock_emoca_blocks_ma3d")


def test_failed_model_card_and_table():
    """Mô hình lỗi vẫn hiện tên từ registry (thẻ mô hình), kèm lý do lỗi; gợi ý codes_missing không đổ lỗi sai cho
    mã 3DMM khi thực ra thiếu file thống kê."""
    import engine

    real_predict = engine.predict

    def spy(image, codes, models=None, *, attention=False, feature_maps=True, mode="mock", **kw):
        out = real_predict(image, codes, models, attention=attention, feature_maps=feature_maps, mode="mock", **kw)
        for i, p in enumerate(out["predictions"]):
            if p["model_id"] == "ma3d_rafdb":  # giống engine thật khi thiếu *_emoca_stats.npz
                out["predictions"][i] = {"model_id": "ma3d_rafdb", "status": "error", "error": {
                    "code": "codes_missing", "retryable": False,
                    "message": "Chưa có file thống kê chuẩn hoá 3DMM (*_emoca_stats.npz); không điền 0."}}
        return out

    engine.predict = spy
    try:
        at = run(upload=("face.png", png(), "image/png"))
    finally:
        engine.predict = real_predict
    assert not at.exception, [e.value for e in at.exception]
    text = md(at)
    assert "MA3D-Net (RAF-DB)" in text, "thẻ mô hình lỗi phải lấy tên hiển thị từ configs/models.yaml"
    errs = "\n".join(e.value for e in at.error) + "\n".join(w.value for w in at.warning)
    assert "[codes_missing]" in errs, "thẻ mô hình lỗi phải hiện mã lỗi"
    assert "Thiếu mã 3DMM từ EMOCA nên MA3D-Net không chạy được" not in errs, "gợi ý cũ đổ lỗi sai nguyên nhân"
    assert "file thống kê chuẩn hoá" in errs, errs[:300]
    print("PASS failed_model_card_and_table")


def _peak(pred: dict, k: int) -> dict:
    """Xác suất dồn vào nhãn k (0.9, còn lại chia đều) và top1 khớp — mô phỏng một mô hình dự đoán nhãn k."""
    n = len(pred["labels"])
    probs = [0.1 / (n - 1)] * n
    probs[k] = 0.9
    return {**pred, "probs": probs, "top1": {"index": k, "label": pred["labels"][k], "prob": 0.9}}


def test_comparison_agree_disagree_and_checkpoint():
    """M3-05: top-1 hai mô hình KHÁC nhau ⇒ dải "bất đồng" + cờ trên cả hai thẻ; GIỐNG nhau ⇒ "cùng dự đoán", không cờ."""
    import engine

    real_predict = engine.predict

    def run_with(tops: dict[str, int]) -> AppTest:
        def spy(image, codes, models=None, *, mode="mock", **kw):
            out = real_predict(image, codes, models, mode="mock", **kw)
            out["predictions"] = [_peak(p, tops[p["model_id"]]) if p.get("status") == "ok" else p
                                  for p in out["predictions"]]
            return out

        engine.predict = spy
        try:
            return run(upload=("face.png", png(), "image/png"))
        finally:
            engine.predict = real_predict

    at = run_with({"poster_rafdb": 3, "ma3d_rafdb": 5})  # happiness vs anger
    preds = at.session_state["result"][1]["faces"][0]["predictions"]
    assert {p["model_id"]: p["top1"]["index"] for p in preds} == {"poster_rafdb": 3, "ma3d_rafdb": 5}, preds
    text = md(at)
    assert 'class="mx-insight diff"' in text and "<b>bất đồng</b>" in text, "thiếu dải bất đồng"
    assert text.count("Bất đồng với mô hình còn lại") == 2, "cả hai thẻ phải có cờ bất đồng"
    assert "cùng dự đoán" not in text

    at = run_with({"poster_rafdb": 6, "ma3d_rafdb": 6})  # cùng neutral
    text = md(at)
    assert "<b>cùng dự đoán</b>" in text, "thiếu dải cùng dự đoán"
    assert 'class="mx-insight diff"' not in text and "Bất đồng với mô hình còn lại" not in text \
        and 'class="mx-flag"' not in text, \
        "top-1 giống nhau thì không được hiện đánh dấu bất đồng"
    print("PASS comparison_agree_disagree_and_checkpoint")


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
    assert app.attention_overlay(crop, None) == (None, "mô hình không trả attention")
    none_img, why = app.attention_overlay(crop, {**grid, "space": "original_px"})
    assert none_img is None and "crop224" in why, why
    assert app.attention_overlay(crop, {**grid, "values": [0.5] * 10})[0] is None, "sai hình dạng ⇒ None"
    lm = app.landmark_tile(crop, [[112, 112]], ref_size=224)
    assert lm.size == (224, 224) and lm.getpixel((112, 112)) == app.VIS_LANDMARK_RGB, lm.getpixel((112, 112))
    print("PASS attention_overlay_helpers")


def _vis_caps(text: str) -> list[str]:
    import re
    return re.findall(r'<div class="mx-vis-cap">([^<]*)</div>', text)


def test_visual_block_mock():
    """Khối 5 ảnh ngay sau phân tích, trước phần so sánh; mock ⇒ attention phủ lên ảnh căn chỉnh RAF-DB, ghi 'giả lập'."""
    at = run({"m": "light"}, ("face.png", png(), "image/png"))
    text = md(at)
    caps = _vis_caps(text)
    assert caps[:3] == ["Input", "Facial Landmark", "3DMM"] and len(caps) == 5, caps
    assert caps[3].startswith("POSTER") and caps[4].startswith("MA3D-Net") and caps[4].endswith("(ours)"), caps
    assert text.find("Kết quả trực quan") < text.find("So sánh biểu cảm"), "khối trực quan phải đứng trên phần so sánh"
    assert 'class="mx-vis-miss"' not in text, "mock có crop_image + attention ⇒ không ô nào thiếu"
    tops = re.findall(r'<div class="mx-vis-top">([^<]*)</div>', text)
    assert len(tops) == 5 and all("%" in t for t in tops[3:]), tops
    assert text.count("attention giả lập, trên ảnh căn chỉnh RAF-DB mà mô hình nhận") == 2, "mock: ghi rõ giả lập + nguồn ảnh nền"
    assert "3DMM" in caps and "EMOCA giả lập" in text
    caps_all = "\n".join(c.value for c in at.caption)
    assert "không phải mức đóng góp" in caps_all
    print("PASS visual_block_mock")


def test_visual_block_missing_attention_and_model_input():
    """Thiếu attention ⇒ ô ghi lý do, không vỡ; có model_input (ADR-020) ⇒ phủ lên ảnh căn chỉnh, ghi rõ."""
    import engine
    import orchestrator
    real_predict, real_analyze = engine.predict, orchestrator.analyze

    def spy(image, codes, models=None, **kw):
        out = real_predict(image, codes, models, **kw)
        for p in out["predictions"]:
            if p.get("model_id") == "poster_rafdb":
                p.pop("attention", None)
        return out

    engine.predict = spy
    try:
        at = run(upload=("face.png", png(), "image/png"))
    finally:
        engine.predict = real_predict
    text = md(at)
    assert len(_vis_caps(text)) == 5 and text.count('class="mx-vis-miss"') == 1, text.count('class="mx-vis-miss"')
    assert "mô hình không trả attention" in text

    def with_model_input(*a, **kw):
        r = real_analyze(*a, **kw)
        for f in r.get("faces") or []:
            f["model_input"] = {"image": _b64png(Image.new("RGB", (100, 100), (60, 120, 180))),
                                "transform": {"from": "original_px", "to": "model_input",
                                              "matrix_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, "size": [100, 100]}
        return r

    orchestrator.analyze = with_model_input
    try:
        at = run(upload=("face.png", png(), "image/png"))
    finally:
        orchestrator.analyze = real_analyze
    text = md(at)
    assert text.count("trên ảnh căn chỉnh RAF-DB mà mô hình nhận") == 2 and "trên crop EMOCA" not in text
    print("PASS visual_block_missing_attention_and_model_input")


def test_upload_dropzone_and_notes_title():
    """Ô tải ảnh: dropzone LỚN, luôn hiện (không thu lại khi đã có ảnh), chữ hướng dẫn tiếng Việt, giới hạn dung lượng lấy
    từ configs/services.yaml, sáng viền khi có focus bàn phím."""
    import yaml
    limit = yaml.safe_load((ROOT / "backend" / "configs" / "services.yaml").read_text(encoding="utf-8"))["limits"]["max_image_bytes"]
    mb = f"{limit / (1024 * 1024):g}"
    at = run({"m": "light"})
    css = md(at)
    zone = re.search(r'\[data-testid="stFileUploaderDropzone"\] \{([^}]*)\}', css)
    assert zone and "min-height:230px" in zone.group(1) and "dashed" in zone.group(1), "dropzone phải là ô lớn viền đứt"
    assert 'content:"Kéo thả ảnh vào đây hoặc bấm để chọn tệp"' in css
    assert f'--mx-upload-hint: "JPEG · PNG · WebP · tối đa {mb} MB · ảnh chỉ nằm trong bộ nhớ, không lưu lại"' in css, \
        "giới hạn dung lượng phải lấy từ configs/services.yaml"
    assert 'content:var(--mx-upload-hint' in css and ':has(:focus-visible)' in css, "thiếu dòng giới hạn / viền focus bàn phím"
    assert "padding:10px 14px" not in css and "thu dropzone" not in css, "không được thu dropzone lại khi đã có ảnh"
    assert "Kéo thả ảnh khác vào đây hoặc bấm vào ô để thay ảnh" in css, "đã có ảnh: phải hướng dẫn thay ảnh"
    at.file_uploader[0].upload("face.png", png(), "image/png").run()
    assert not at.exception, [e.value for e in at.exception]
    text = md(at)
    assert at.file_uploader, "đã có ảnh vẫn phải còn ô tải ảnh để thay ảnh"
    print("PASS upload_dropzone_and_notes_title")


def main():
    test_upload_dropzone_and_notes_title()
    test_yolo_shadow_is_opt_in_and_preserves_primary_result()
    test_multi_object_grid_skips_single_prediction_and_renders_face_cards()
    test_predata_branch_errors_are_visible()
    test_attention_overlay_helpers()
    test_visual_block_mock()
    test_visual_block_missing_attention_and_model_input()
    test_empty_state_all_modes()
    test_upload_flow_renders_components()
    test_frontalization_experiment_section_is_disabled_in_mock_mode()
    test_single_face_roll_upload_bytes_and_caption()
    test_error_states()
    test_partial_and_service_errors()
    test_emoca_panel_and_consent()
    test_engine_mode_control()
    test_real_engine_with_mock_emoca_blocks_ma3d()
    test_failed_model_card_and_table()
    test_comparison_agree_disagree_and_checkpoint()
    test_orchestrator_import_error_is_reported()
    test_settings_from_query_and_validation()
    test_settings_panel_at_top()
    test_settings_widgets_and_persistence()
    test_theme_switch_changes_css()
    test_html_is_escaped()
    print("ALL PASS")


if __name__ == "__main__":
    main()
