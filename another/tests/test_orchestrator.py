"""Tự kiểm web/orchestrator.py (giai đoạn fixture). Chạy: python tests/test_orchestrator.py

Ảnh giả sinh trong bộ nhớ; không ghi file ảnh ra đĩa.
"""

import io
import json
import sys
import tempfile
from pathlib import Path

import jsonschema
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from web import orchestrator  # noqa: E402
from web.orchestrator import analyze  # noqa: E402

SCHEMA = json.loads((ROOT / "backend" / "contracts" / "web_analyze.schema.json").read_text(encoding="utf-8"))


def make_image(fmt: str, size=(64, 48), exif_orientation: int | None = None) -> bytes:
    img = Image.new("RGB", size, (200, 150, 120))
    buf = io.BytesIO()
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif.tobytes()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


class FakeUpload:
    """Giống Streamlit UploadedFile ở mức cần dùng."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def check(name, result, status, code=None, width=None, height=None):
    jsonschema.validate(result, SCHEMA)
    assert result["status"] == status, (name, result["status"], result["error"])
    if code is None:
        assert result["error"] is None or status == "partial", (name, result["error"])
    else:
        assert result["error"]["code"] == code, (name, result["error"])
    if width is not None:
        assert result["image"] == {"width": width, "height": height}, (name, result["image"])
    print(f"PASS {name}: status={result['status']} error={result['error'] and result['error']['code']}"
          f" image={result['image']} faces={len(result['faces'])}")


def main():
    check("jpeg_ok", analyze(make_image("JPEG")), "ok", width=64, height=48)
    check("png_upload_ok", analyze(FakeUpload(make_image("PNG"))), "ok", width=64, height=48)
    check("webp_ok", analyze(make_image("WEBP")), "ok", width=64, height=48)
    check("partial", analyze(make_image("JPEG"), mock_case="partial"), "partial",
          code="service_unavailable", width=64, height=48)
    # EXIF orientation 6 = xoay 90°, nên width/height phải đổi chỗ.
    check("exif_rotate", analyze(make_image("JPEG", exif_orientation=6)), "ok", width=48, height=64)
    check("gif_rejected", analyze(make_image("GIF")), "error", code="image_invalid")
    check("garbage", analyze(b"not an image at all"), "error", code="image_invalid")
    check("empty", analyze(b""), "error", code="image_invalid")
    check("truncated_jpeg", analyze(make_image("JPEG", size=(256, 256))[:200]), "error", code="image_invalid")
    check("bad_type", analyze(12345), "error", code="image_invalid")
    check("bad_mock_case", analyze(make_image("PNG"), mock_case="nope"), "error", code="contract_mismatch")

    # Giới hạn đọc từ configs/services.yaml khi file tồn tại (_load_limits), chỉ rơi về
    # _FALLBACK_LIMITS khi thiếu file ⇒ trỏ tạm _SERVICES_CONFIG sang một config nhỏ.
    orig = orchestrator._SERVICES_CONFIG
    with tempfile.TemporaryDirectory() as tmp:
        small = Path(tmp) / "services.yaml"
        small.write_text("limits:\n  max_image_bytes: 100\n  max_faces: 1\n  max_concurrent: 1\n", encoding="utf-8")
        orchestrator._SERVICES_CONFIG = small
        try:
            check("too_large", analyze(make_image("PNG")), "error", code="image_too_large")
        finally:
            orchestrator._SERVICES_CONFIG = orig

    # analyze() không được sửa fixture dùng chung giữa các lần gọi.
    a = analyze(make_image("JPEG"))
    b = analyze(make_image("JPEG"))
    assert a["warnings"] == b["warnings"], "warnings bị cộng dồn giữa các lần gọi"
    print("PASS no_state_leak")

    # --- Sau khi nối AI (web/emoca_client.py + web/engine.py, chế độ mock): không còn đọc fixture ---
    ok = analyze(make_image("PNG", size=(320, 240)))
    check("ai_wired_ok", ok, "ok", width=320, height=240)
    face = ok["faces"][0]
    assert len(ok["faces"]) == 1 and face["landmarks"]["space"] == "original_px"
    assert len(face["landmarks"]["points"]) == 68 == face["landmarks"]["n_points"]
    assert all(0 <= x <= 320 and 0 <= y <= 240 for x, y in face["landmarks"]["points"]), "landmark ngoài ảnh"
    assert {p["model_id"] for p in face["predictions"]} == {"poster_rafdb", "ma3d_rafdb"}
    assert all(p["status"] == "ok" for p in face["predictions"])
    assert [m["id"] for m in ok["models_info"]] == [p["model_id"] for p in face["predictions"]]
    assert {"emoca", "ma3d", "total"} <= set(ok["timing_ms"]), ok["timing_ms"]
    assert any(w.startswith("mock_ai") for w in ok["warnings"]), ok["warnings"]
    assert not any("mock_fixture" in w for w in ok["warnings"]), "còn đọc fixture"
    print("PASS ai_wired_details")

    # Model selection reaches engine.predict; omitted selection preserves engine defaults.
    selected_models = []
    original_predict = orchestrator.engine.predict

    def capture_model_selection(image, codes, models=None, **kwargs):
        selected_models.append(models)
        return original_predict(image, codes, models=models, **kwargs)

    orchestrator.engine.predict = capture_model_selection
    try:
        default_models = analyze(make_image("PNG"))
        assert selected_models == [None], selected_models
        assert {p["model_id"] for p in default_models["faces"][0]["predictions"]} == {
            "poster_rafdb", "ma3d_rafdb"
        }

        selected_models.clear()
        ma3d_only = analyze(make_image("PNG"), model_ids=["ma3d_rafdb"])
        assert selected_models == [["ma3d_rafdb"]], selected_models
        jsonschema.validate(ma3d_only, SCHEMA)
        assert ma3d_only["status"] == "ok", ma3d_only
        assert [p["model_id"] for p in ma3d_only["faces"][0]["predictions"]] == ["ma3d_rafdb"]
        assert [m["id"] for m in ma3d_only["models_info"]] == ["ma3d_rafdb"]
    finally:
        orchestrator.engine.predict = original_predict
    print("PASS engine_model_selection_default_and_ma3d_only")

    # ADR-019 (0.3.0): face mang crop EMOCA để giao diện phủ attention đúng khung — chép nguyên từ emoca_analyze.
    assert ok["contract_version"] == "0.3.0"
    assert face["crop_image"]["format"] == "png" and face["crop_image"]["b64"], "thiếu crop_image"
    assert face["crop_transform"]["from"] == "original_px" and face["crop_transform"]["to"] == "crop224"
    assert len(face["landmarks_crop224"]) == 68 and all(0 <= x <= 224 and 0 <= y <= 224 for x, y in face["landmarks_crop224"])
    # ADR-020 BẬT (configs/align.yaml enabled: true) ⇒ engine nhận ảnh căn chỉnh RAF-DB (model_input), không cảnh báo.
    assert "model_input" in face and not any("align_failed" in w for w in face["warnings"]), face["warnings"]
    print("PASS adr019_crop_fields + adr020_default_on")

    fa = orchestrator.face_align
    if fa is not None:  # bật tạm (không sửa config): model_input có mặt, đúng schema, ảnh đúng kích thước mẫu
        real_load = fa.load_template

        def enabled_template(*a, **kw):
            tpl, err = real_load(*a, **kw)
            return ({**tpl, "enabled": True} if tpl else tpl), err

        fa.load_template = enabled_template
        try:
            on = analyze(make_image("PNG", size=(320, 240)))
        finally:
            fa.load_template = real_load
        check("adr020_enabled", on, "ok", width=320, height=240)
        mi = on["faces"][0].get("model_input")
        assert mi and mi["transform"]["from"] == "original_px" and mi["transform"]["to"] == "model_input", mi and mi.keys()
        size = tuple(mi["size"])
        from PIL import Image as _I
        import base64 as _b64
        got = _I.open(io.BytesIO(_b64.b64decode(mi["image"]["b64"]))).size
        assert got == size, (got, size)
        assert not any("align_failed" in w for w in on["faces"][0]["warnings"]), on["faces"][0]["warnings"]
        print(f"PASS adr020_enabled_model_input size={size}")

    # Ảnh quá nhỏ → mock EMOCA báo no_face → lỗi đầu vào, status "error" (không phải partial).
    check("no_face_tiny", analyze(make_image("PNG", size=(20, 20))), "error", code="no_face", width=20, height=20)

    # partial giả lập: EMOCA lỗi dịch vụ → không có mặt, không có mô hình.
    part = analyze(make_image("PNG"), mock_case="partial")
    assert part["faces"] == [] and part["models_info"] == [], part
    assert "emoca_svc_unavailable" in part["warnings"]
    print("PASS partial_no_faces")

    # partial thật: chế độ "real" của AI chưa làm → model_unavailable, không âm thầm rơi về mock.
    orig_modes = (orchestrator._EMOCA_MODE, orchestrator._ENGINE_MODE)
    try:
        orchestrator._EMOCA_MODE = "real"
        # Tất định, không phụ thuộc configs/services.yaml (từ 2026-09-26 url = http://127.0.0.1:8765, ADR-018):
        # (a) url rỗng ⇒ model_unavailable; (b) url trỏ tới cổng loopback ĐÓNG ⇒ service_unavailable.
        orig_load = orchestrator._load_emoca_config
        orchestrator._load_emoca_config = lambda: ({**orig_load()[0], "url": ""}, [])
        try:
            r = analyze(make_image("PNG"))
        finally:
            orchestrator._load_emoca_config = orig_load
        check("emoca_real_partial", r, "partial", code="model_unavailable", width=64, height=48)
        assert r["faces"] == []
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            closed_port = s.getsockname()[1]
        r = analyze(make_image("PNG"), emoca={"mode": "real", "url": f"http://127.0.0.1:{closed_port}", "timeout_s": 5})
        check("emoca_real_closed_port", r, "partial", code="service_unavailable", width=64, height=48)
        assert r["faces"] == []

        # Engine `real` KHÔNG khả dụng một cách tất định (tiêm handle lỗi, kết quả không phụ thuộc máy có torch hay không —
        # trong venv web torch có sẵn nên POSTER thật chạy được; phần đó do tests/test_engine_poster_real.py đo).
        engine = orchestrator.engine  # cùng module mà orchestrator gọi (test nạp qua gói `web`)
        orchestrator._EMOCA_MODE, orchestrator._ENGINE_MODE = "mock", "real"
        saved_models = engine._REAL_MODELS
        engine._REAL_MODELS = {mid: {"error": "giả lập thiếu checkpoint", "spec": {"id": mid}} for mid in ("poster_rafdb", "ma3d_rafdb")}
        try:
            r = analyze(make_image("PNG"))
        finally:
            engine._REAL_MODELS = saved_models
        check("engine_real_partial", r, "partial", code="model_unavailable", width=64, height=48)
        assert len(r["faces"]) == 1 and r["models_info"] == []
        assert all(p["status"] == "error" for p in r["faces"][0]["predictions"])
        assert not any(w.startswith("mock_ai: engine") for w in r["warnings"]), "engine real KHÔNG được gắn nhãn mock"
        assert any(w.startswith("mock_ai: emoca_client") for w in r["warnings"]), "EMOCA vẫn mock ⇒ phải ghi rõ"
        assert any(w.startswith("engine_real:") and "không mô hình nào chạy được" in w for w in r["warnings"]), r["warnings"]
        assert "engine_total" in r["timing_ms"] and r["timing_ms"]["engine_total"] >= 0, r["timing_ms"]  # gồm cả thời gian nạp (E2E U4)
        print("PASS engine_real_warnings")
    finally:
        orchestrator._EMOCA_MODE, orchestrator._ENGINE_MODE = orig_modes

    # Thứ tự ưu tiên chọn chế độ engine: tham số > hằng > biến môi trường > config > mock; sai chính tả ⇒ cảnh báo + mock.
    import os
    saved_env = os.environ.pop("ENGINE_MODE", None)
    try:
        assert orchestrator.resolve_engine_mode() == ("mock", []), orchestrator.resolve_engine_mode()  # config mặc định
        os.environ["ENGINE_MODE"] = "real"
        assert orchestrator.resolve_engine_mode()[0] == "real", "biến môi trường thắng config"
        orchestrator._ENGINE_MODE = "mock"
        assert orchestrator.resolve_engine_mode()[0] == "mock", "hằng _ENGINE_MODE thắng biến môi trường"
        assert orchestrator.resolve_engine_mode("real")[0] == "real", "tham số thắng tất cả"
        orchestrator._ENGINE_MODE = None
        os.environ["ENGINE_MODE"] = "rael"
        mode, warns = orchestrator.resolve_engine_mode()
        assert mode == "mock" and any("config_invalid" in w for w in warns), (mode, warns)
        with tempfile.TemporaryDirectory() as tmp:
            os.environ.pop("ENGINE_MODE", None)
            cfg = Path(tmp) / "services.yaml"
            cfg.write_text("engine:\n  mode: real\n", encoding="utf-8")
            orig_cfg = orchestrator._SERVICES_CONFIG
            orchestrator._SERVICES_CONFIG = cfg
            try:
                assert orchestrator.resolve_engine_mode()[0] == "real", "đọc engine.mode từ configs/services.yaml"
            finally:
                orchestrator._SERVICES_CONFIG = orig_cfg
    finally:
        orchestrator._ENGINE_MODE = orig_modes[1]
        os.environ.pop("ENGINE_MODE", None)
        if saved_env is not None:
            os.environ["ENGINE_MODE"] = saved_env
    print("PASS engine_mode_resolution")

    # Giới hạn cạnh ảnh được truyền cho client EMOCA (D3 của docs/qa/REVIEW_ADR015_EMOCA_COLAB.md): mặc định 4096 từ config.
    s, _ = orchestrator.resolve_emoca_settings(None, 1000, 321)
    assert (s["max_image_bytes"], s["max_image_side_px"]) == (1000, 321), s
    s, _ = orchestrator.resolve_emoca_settings(None)
    assert s["max_image_side_px"] == 4096, s
    lim, _ = orchestrator._load_limits()
    assert lim["max_image_side_px"] == 4096, lim
    print("PASS emoca_side_limit_plumbing")

    # Chốt chặn mã 3DMM giả: MA3D-Net (arch MA3D) chỉ được suy luận thật khi mã đến từ EMOCA THẬT.
    assert orchestrator._codes_are_real({"name": "MOCK-EMOCA_v2_lr_mse_20"}) is False
    assert orchestrator._codes_are_real({"name": "MOCK-emoca_svc"}) is False
    assert orchestrator._codes_are_real({"name": "EMOCA_v2_lr_mse_20"}) is True
    assert orchestrator._codes_are_real(None) is True, "không có thông tin ⇒ không tự kết luận là giả (engine vẫn kiểm mã)"
    seen = []
    orig_predict, orig_call = orchestrator.engine.predict, orchestrator._emoca_call

    def spy(image, codes, models=None, **kw):
        seen.append(kw.get("codes_real"))
        return orig_predict(image, codes, models=models, **kw)

    def emoca_named_real(*a, **k):  # mô phỏng EMOCA thật: cùng dữ liệu mock nhưng tên mô hình không phải MOCK
        out = orig_call(*a, **k)
        out["model"] = {"name": "EMOCA_v2_lr_mse_20", "mode": "detail", "upstream_commit": "e0be0db"}
        return out

    orchestrator.engine.predict = spy
    try:
        r = analyze(make_image("PNG"), emoca={"mode": "mock"}, engine_mode="real")
        assert seen == [False], f"EMOCA giả ⇒ codes_real=False ({seen})"
        ma3d = [p for f in r["faces"] for p in f["predictions"] if p["model_id"] == "ma3d_rafdb"]
        assert ma3d and ma3d[0]["status"] == "error" and ma3d[0]["error"]["code"] == "codes_missing"             and "giả lập" in ma3d[0]["error"]["message"], r
        seen.clear()
        orchestrator._emoca_call = emoca_named_real
        analyze(make_image("PNG"), emoca={"mode": "mock"}, engine_mode="real")
        assert seen == [True], f"EMOCA báo tên thật ⇒ codes_real=True ({seen})"
    finally:
        orchestrator.engine.predict, orchestrator._emoca_call = orig_predict, orig_call
    print("PASS ma3d_gate_on_mock_codes")
    print("ALL PASS")


if __name__ == "__main__":
    main()
