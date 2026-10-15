"""Light M7-04 UI acceptance checks (requires Streamlit in the venv web)."""

from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))
sys.path.insert(0, str(ROOT / "another" / "tests"))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("SKIP test_m7_expanded_acceptance: Streamlit is unavailable (use venv web)")
    raise SystemExit(0)

from test_app_ui import md, png, settle
from web import predata_experiment

APP = str(ROOT / "frontend" / "app.py")


def _views(image: Image.Image, count: int, *, complete: bool = True, error: str | None = None):
    """Construct real M7 view dataclasses using only in-memory image data."""
    source = image.convert("RGB").copy()
    faces = []
    for index in range(count):
        x1 = min(max(0, 4 + (index % 5) * max(1, source.width // 6)), max(0, source.width - 2))
        y1 = min(max(0, 4 + (index // 5) * max(1, source.height // 7)), max(0, source.height - 2))
        x2, y2 = min(source.width, x1 + 20), min(source.height, y1 + 20)
        crop_bbox = (x1, y1, x2, y2)
        crop = source.crop(crop_bbox)
        detection = predata_experiment.FaceDetectionResult(
            annotated_image=source.copy(), cropped_face=crop, bbox_xyxy=crop_bbox,
            crop_bbox_xyxy=crop_bbox, confidence=0.91 - index / 1000,
            image_size=source.size, crop_size=crop.size, inference_ms=1.0,
        )
        faces.append(predata_experiment.MultiFaceItem(index + 1, detection, source.copy()))
    return predata_experiment.MultiFaceViews(
        source_image=source, faces=faces, detection_source="fake YOLO",
        detection_complete=complete, errors=[error] if error else [],
    )


def _result(marker: int = 0) -> dict:
    def prediction(model_id: str, label: str, probability: float) -> dict:
        return {
            "model_id": model_id, "status": "ok", "labels": ["Happy", "Sad", "Neutral"],
            "probs": [probability, (1 - probability) / 2, (1 - probability) / 2],
            "top1": {"index": 0, "label": f"{label}{marker}", "prob": probability},
        }

    return {
        "contract_version": "1.0", "status": "ok", "image": {"width": 320, "height": 240},
        "models_info": [], "timing_ms": {}, "warnings": [], "error": None,
        "faces": [{"predictions": [prediction("poster_rafdb", "POSTER Happy ", .82),
                                    prediction("ma3d_rafdb", "MA3D Happy ", .91)]}],
    }


def _multi_result(image: Image.Image, views, *, fail: bool = False) -> dict:
    if fail:
        raise RuntimeError("fake multi analysis failure")
    return {
        "kind": "faces", "views": views,
        "rows": [{"face_id": item.face_id, "reference": item.reference_image,
                  "processed": item.detection.cropped_face, "result": _result(item.face_id)}
                 for item in views.faces],
    }


class _Harness:
    """Keep orchestrator fakes installed until every AppTest rerun has finished."""

    def __init__(self, count: int, *, detector_error: str | None = None, detector_exception: bool = False,
                 fail_multi: bool = False):
        import orchestrator

        self.orchestrator = orchestrator
        self.original = {name: getattr(orchestrator, name) for name in
                         ("prepare_multiface_views", "analyze", "analyze_multiface")}
        self.count = count
        self.detector_error = detector_error
        self.detector_exception = detector_exception
        self.fail_multi = fail_multi
        self.calls = {"prepare": 0, "analyze": [], "multi": []}

        def prepare(image, **kwargs):
            self.calls["prepare"] += 1
            if self.detector_exception:
                raise RuntimeError("fake detector exception")
            return _views(image, count, complete=detector_error is None, error=detector_error)

        def analyze(data, **kwargs):
            self.calls["analyze"].append(kwargs)
            return _result()

        def analyze_multiface(image, **kwargs):
            self.calls["multi"].append(kwargs)
            return _multi_result(image, kwargs["views"], fail=fail_multi)

        orchestrator.prepare_multiface_views = prepare
        orchestrator.analyze = analyze
        orchestrator.analyze_multiface = analyze_multiface

    def close(self):
        for name, original in self.original.items():
            setattr(self.orchestrator, name, original)


def _upload(at: AppTest, name: str = "group.png") -> AppTest:
    at.run()
    assert not at.exception, [error.value for error in at.exception]
    at.file_uploader[0].upload(name, png(), "image/png").run()
    assert not at.exception, [error.value for error in at.exception]
    return settle(at)


def _run_count(count: int) -> tuple[AppTest, _Harness]:
    harness = _Harness(count)
    try:
        at = _upload(AppTest.from_file(APP, default_timeout=90))
        return at, harness
    except BaseException:
        harness.close()
        raise


def _assert_single_route(count: int) -> None:
    at, harness = _run_count(count)
    try:
        assert harness.calls["prepare"] == 1
        assert len(harness.calls["analyze"]) == 1
        assert harness.calls["analyze"][0]["max_faces"] == 1
        assert harness.calls["analyze"][0]["emoca"]["mode"] == "real"
        assert harness.calls["analyze"][0]["engine_mode"] == "real"
        assert harness.calls["multi"] == []
        text = md(at)
        assert "Multi-Object Analysis" not in text
        assert not any("Multi-Object Analysis" in button.label for button in at.button)
        assert "Face #1" not in text
    finally:
        harness.close()


def test_zero_faces_route_to_single_baseline():
    _assert_single_route(0)
    print("PASS zero_faces_route_to_single_baseline")


def test_one_face_route_to_single_baseline():
    _assert_single_route(1)
    print("PASS one_face_route_to_single_baseline")


def test_two_faces_render_only_compact_multi_cards():
    at, harness = _run_count(2)
    try:
        text = md(at)
        assert harness.calls["analyze"] == [] and len(harness.calls["multi"]) == 1
        assert harness.calls["multi"][0]["emoca"]["mode"] == "real"
        assert harness.calls["multi"][0]["engine_mode"] == "real"
        assert text.count('class="mx-vis-cap"') == 10, text
        assert "Face #" not in text and "Happy" in text
        for absent in ("Kết quả trực quan — MA3D + pre-data",
                       "So sánh biểu cảm", "Hình học khuôn mặt", "Radar Chart", "mx-radar"):
            assert absent not in text, absent
        assert not any("Multi-Object Analysis" in button.label for button in at.button)
    finally:
        harness.close()
    print("PASS two_faces_render_only_compact_multi_cards")


def test_thirty_faces_render_in_stable_order():
    at, harness = _run_count(30)
    try:
        text = md(at)
        assert text.count('class="mx-vis-cap"') == 150
        assert len(harness.calls["multi"]) == 1 and harness.calls["analyze"] == []
    finally:
        harness.close()
    print("PASS thirty_faces_render_in_stable_order")


def test_automatic_analysis_runs_once_across_unrelated_reruns():
    at, harness = _run_count(2)
    try:
        assert len(harness.calls["multi"]) == 1
        at.run()
        assert not at.exception, [error.value for error in at.exception]
        assert len(harness.calls["multi"]) == 1 and harness.calls["prepare"] == 1
    finally:
        harness.close()
    print("PASS automatic_analysis_runs_once_across_unrelated_reruns")


def test_failed_multi_analysis_is_cached_until_explicit_retry():
    harness = _Harness(2, fail_multi=True)
    try:
        at = _upload(AppTest.from_file(APP, default_timeout=90))
        assert len(harness.calls["multi"]) == 1
        assert any("fake multi analysis failure" in error.value for error in at.error)
        at.run()
        assert len(harness.calls["multi"]) == 1 and harness.calls["prepare"] == 1
        [button for button in at.button if button.label == "Phân tích lại"][0].click().run()
        assert not at.exception, [error.value for error in at.exception]
        assert len(harness.calls["multi"]) == 2 and harness.calls["prepare"] == 2
    finally:
        harness.close()
    print("PASS failed_multi_analysis_is_cached_until_explicit_retry")


def _assert_detector_fallback(*, detector_error: str | None = None, detector_exception: bool = False) -> None:
    harness = _Harness(0, detector_error=detector_error, detector_exception=detector_exception)
    try:
        at = _upload(AppTest.from_file(APP, default_timeout=90))
        assert not at.exception, [error.value for error in at.exception]
        assert len(harness.calls["analyze"]) == 1 and harness.calls["analyze"][0]["max_faces"] == 1
        assert harness.calls["multi"] == []
        assert any("YOLO không hoàn tất" in warning.value for warning in at.warning)
        assert "đã phân tích trên ảnh gốc." in "\n".join(warning.value for warning in at.warning)
    finally:
        harness.close()


def test_detector_error_view_warns_and_falls_back_to_baseline():
    _assert_detector_fallback(detector_error="YOLO test error")
    print("PASS detector_error_view_warns_and_falls_back_to_baseline")


def test_detector_exception_warns_and_falls_back_to_baseline():
    _assert_detector_fallback(detector_exception=True)
    print("PASS detector_exception_warns_and_falls_back_to_baseline")


def test_unknown_query_param_is_ignored():
    at = AppTest.from_file(APP, default_timeout=90)
    at.query_params["page"] = "face-details"
    at.run()
    assert not at.exception, [error.value for error in at.exception]
    assert "Face #2 — Details" not in md(at)
    assert "page=face-details" not in md(at)
    print("PASS unknown_query_param_is_ignored")


def test_multiface_flow_never_saves_images_to_disk():
    import PIL.Image

    harness = _Harness(2)
    original_save = PIL.Image.Image.save

    def guarded_save(self, fp, *args, **kwargs):
        assert isinstance(fp, BytesIO), f"PIL image attempted non-memory save to {fp!r}"
        return original_save(self, fp, *args, **kwargs)

    PIL.Image.Image.save = guarded_save
    try:
        at = _upload(AppTest.from_file(APP, default_timeout=90))
        assert not at.exception, [error.value for error in at.exception]
        assert len(harness.calls["multi"]) == 1
    finally:
        PIL.Image.Image.save = original_save
        harness.close()
    print("PASS multiface_flow_never_saves_images_to_disk")


def main():
    test_zero_faces_route_to_single_baseline()
    test_one_face_route_to_single_baseline()
    test_two_faces_render_only_compact_multi_cards()
    test_thirty_faces_render_in_stable_order()
    test_automatic_analysis_runs_once_across_unrelated_reruns()
    test_failed_multi_analysis_is_cached_until_explicit_retry()
    test_detector_error_view_warns_and_falls_back_to_baseline()
    test_detector_exception_warns_and_falls_back_to_baseline()
    test_unknown_query_param_is_ignored()
    test_multiface_flow_never_saves_images_to_disk()
    print("ALL PASS")


if __name__ == "__main__":
    main()
