from PIL import Image

from web import predata_experiment as predata
from web.face_preprocessor import FaceDetectionResult, NoFaceDetectedError


def _detection(image, marker):
    return FaceDetectionResult(
        annotated_image=image.copy(), cropped_face=image.crop((2, 1, 6, 5)),
        bbox_xyxy=(3, 2, 5, 4), crop_bbox_xyxy=(2, 1, 6, 5), confidence=0.9,
        image_size=image.size, crop_size=(4, 4), inference_ms=1.0,
    )


class _General:
    def restore(self, image):
        return Image.eval(image, lambda value: 255 - value)


def _source_image():
    source = Image.new("RGB", (8, 6))
    for y in range(6):
        for x in range(8):
            source.putpixel((x, y), (x * 10, y * 10, x + y))
    return source


def _prepare_multiface(monkeypatch, stack):
    monkeypatch.setattr(predata, "_get_multiface_model_stack", lambda: stack)
    monkeypatch.setattr(predata, "_limit_windows_cpu_cores", lambda *_a, **_k: _NullContext())


def test_multiface_faces_on_original_skip_restorer(monkeypatch):
    class Restorer:
        calls = 0
        def restore(self, image):
            self.calls += 1
            return _General().restore(image)
    class Detector:
        def process_many(self, image, *, max_faces):
            assert image.getpixel((0, 0)) == _source_image().getpixel((0, 0))
            return [_detection(image, 0)]
    restorer = Restorer()
    _prepare_multiface(monkeypatch, predata._ModelStack(general=restorer, detector=Detector()))
    source = _source_image()
    result = predata._build_multiface_views(source, 30)
    assert len(result.faces) == 1 and result.detection_source == "YOLO trên ảnh gốc"
    assert restorer.calls == 0 and "realesrgan" not in result.stage_times_ms
    assert result.faces[0].detection.cropped_face.tobytes() == source.crop((2, 1, 6, 5)).tobytes()


def test_multiface_original_zero_then_sharpened_faces_crop_original(monkeypatch):
    class Restorer:
        calls = 0
        def restore(self, image):
            self.calls += 1
            return _General().restore(image)
    class Detector:
        calls = 0
        def process_many(self, image, *, max_faces):
            self.calls += 1
            if self.calls == 1:
                raise NoFaceDetectedError("no original faces")
            assert image.getpixel((0, 0)) != _source_image().getpixel((0, 0))
            return [_detection(image, 0)]
    restorer, detector = Restorer(), Detector()
    _prepare_multiface(monkeypatch, predata._ModelStack(general=restorer, detector=detector))
    source = _source_image()
    result = predata._build_multiface_views(source, 30)
    assert len(result.faces) == 1 and result.detection_source == "RealESRGAN + YOLO"
    assert detector.calls == 2 and restorer.calls == 1 and "realesrgan" in result.stage_times_ms
    assert result.faces[0].detection.cropped_face.tobytes() == source.crop((2, 1, 6, 5)).tobytes()


def test_multiface_both_detectors_find_zero_is_complete(monkeypatch):
    class Detector:
        calls = 0
        def process_many(self, image, *, max_faces):
            self.calls += 1
            raise NoFaceDetectedError("no faces")
    detector = Detector()
    _prepare_multiface(monkeypatch, predata._ModelStack(general=_General(), detector=detector))
    result = predata._build_multiface_views(_source_image(), 30)
    assert result.faces == [] and detector.calls == 2 and result.detection_complete is True
    assert result.detection_source is None and "realesrgan" in result.stage_times_ms


def test_multiface_original_detector_error_skips_restorer(monkeypatch):
    class Restorer:
        calls = 0
        def restore(self, image):
            self.calls += 1
            return image
    class Detector:
        def process_many(self, image, *, max_faces):
            raise RuntimeError("broken YOLO")
    restorer = Restorer()
    _prepare_multiface(monkeypatch, predata._ModelStack(general=restorer, detector=Detector()))
    result = predata._build_multiface_views(_source_image(), 30)
    assert result.faces == [] and result.detection_complete is False
    assert restorer.calls == 0 and "realesrgan" not in result.stage_times_ms
    assert any("RuntimeError: broken YOLO" in error for error in result.errors)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_removed_adapter_and_stage_symbols_are_not_importable():
    from services.image_preprocessing import adapters, cli

    assert not hasattr(adapters, "Code" + "FormerRestorer")
    assert not hasattr(predata, "restore_multiface_crop")
    assert not hasattr(cli, "Code" + "FormerRestorer")
