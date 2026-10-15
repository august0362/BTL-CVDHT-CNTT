"""Independent, offline tests for the optional face preprocessor."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from web.face_preprocessor import (
    FaceDetectionPreprocessor,
    NoFaceDetectedError,
)


class TensorLike:
    """Small stand-in for the tensor methods used on Ultralytics results."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Boxes:
    def __init__(self, xyxy=(), conf=()):
        self.xyxy = TensorLike(xyxy)
        self.conf = TensorLike(conf)

    def __len__(self):
        return len(self.xyxy.value)


class Prediction:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeDetector:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, source, *, conf, imgsz, verbose=False):
        self.calls.append((source, conf, imgsz, verbose))
        return self.predictions


def detector(boxes=(), confidences=()):
    return FakeDetector([Prediction(Boxes(boxes, confidences))])


def processor(fake, **kwargs):
    return FaceDetectionPreprocessor(detector=fake, **kwargs)


def test_no_face_raises_no_face_detected_error():
    with pytest.raises(NoFaceDetectedError):
        processor(detector()).process(Image.new("RGB", (12, 10)))


@pytest.mark.parametrize(
    ("boxes", "scores", "expected"),
    [
        ([[1, 1, 5, 5]], [0.8], (1, 1, 5, 5)),
        ([[1, 1, 5, 5], [6, 2, 11, 9]], [0.7, 0.95], (6, 2, 11, 9)),
        ([[1, 1, 5, 5], [6, 2, 11, 9]], [0.9, 0.9], (1, 1, 5, 5)),
    ],
)
def test_selects_highest_confidence_and_stable_first_tie(boxes, scores, expected):
    result = processor(detector(boxes, scores)).process(Image.new("RGB", (12, 10)))
    assert result.bbox_xyxy == expected


@pytest.mark.parametrize(
    ("raw_box", "expected"),
    [
        ([-2.2, -1.1, 4.1, 3.2], (0, 0, 5, 4)),
        ([7.2, 6.8, 20.1, 20.1], (7, 6, 12, 10)),
    ],
)
def test_bbox_rounds_outward_clamps_and_returns_nonempty_crop(raw_box, expected):
    result = processor(detector([raw_box], [0.9])).process(Image.new("RGB", (12, 10)))
    assert result.bbox_xyxy == expected
    assert result.cropped_face.size == result.crop_size
    assert result.crop_size[0] > 0 and result.crop_size[1] > 0


def test_accepts_path_and_pil_rgb_and_returns_rgb_products(tmp_path):
    source = Image.new("RGB", (80, 70), (231, 19, 47))
    path = tmp_path / "input.png"
    source.save(path)
    for input_value in (path, source):
        result = processor(detector([[10, 20, 60, 55]], [0.87])).process(input_value)
        assert result.cropped_face.mode == result.annotated_image.mode == "RGB"
        assert result.cropped_face.getpixel((0, 0)) == (231, 19, 47)
        assert result.annotated_image.getpixel((79, 69)) == (231, 19, 47)


def test_numpy_bgr_and_rgb_color_order_are_explicit_and_rgb_on_output():
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[:] = (240, 30, 10)
    bgr = rgb[:, :, ::-1].copy()
    bgr_result = processor(detector([[1, 1, 8, 7]], [0.9])).process(bgr)
    rgb_result = processor(detector([[1, 1, 8, 7]], [0.9])).process(
        rgb, array_color_order="RGB"
    )
    for result in (bgr_result, rgb_result):
        assert result.annotated_image.mode == result.cropped_face.mode == "RGB"
        assert result.cropped_face.getpixel((0, 0)) == (240, 30, 10)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((5, 5), dtype=np.uint8),
        np.zeros((5, 5, 4), dtype=np.uint8),
        np.zeros((5, 5, 3), dtype=np.float32),
    ],
)
def test_rejects_invalid_numpy_shape_or_dtype(array):
    with pytest.raises(ValueError):
        processor(detector()).process(array)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_rejects_confidence_threshold_outside_unit_interval(threshold):
    with pytest.raises(ValueError):
        FaceDetectionPreprocessor(detector=detector(), conf_threshold=threshold)


@pytest.mark.parametrize("imgsz", [0, -1])
def test_rejects_nonpositive_imgsz(imgsz):
    with pytest.raises(ValueError):
        FaceDetectionPreprocessor(detector=detector(), imgsz=imgsz)


def test_passes_configured_confidence_and_imgsz_to_detector():
    fake = detector([[1, 1, 5, 5]], [0.75])
    processor(fake, conf_threshold=0.65, imgsz=320).process(Image.new("RGB", (8, 8)))
    _, conf, imgsz, verbose = fake.calls[0]
    assert (conf, imgsz, verbose) == (0.65, 320, False)


def test_annotated_image_contains_selected_box_and_confidence_label():
    original = Image.new("RGB", (40, 30), (10, 20, 30))
    result = processor(detector([[5, 6, 25, 22]], [0.93])).process(original)
    assert result.annotated_image.getpixel((5, 10)) == (0, 200, 255)
    assert result.annotated_image.getpixel((15, 15)) == original.getpixel((15, 15))
    assert result.annotated_image.getpixel((40 - 1, 30 - 1)) == original.getpixel((39, 29))
    # Label is drawn over the image near the selected box, with the detector's color.
    assert (0, 200, 255) in set(result.annotated_image.crop((5, 0, 30, 6)).getdata())


def test_metadata_has_selected_box_confidence_sizes_and_finite_time():
    result = processor(detector([[2, 3, 11, 12]], [0.876])).process(Image.new("RGB", (16, 14)))
    assert result.bbox_xyxy == (2, 3, 11, 12)
    assert result.confidence == pytest.approx(0.876)
    assert result.image_size == (16, 14)
    assert result.crop_bbox_xyxy == (0, 1, 13, 14)
    assert result.crop_size == (13, 13)
    assert result.cropped_face.size == result.crop_size
    assert np.isfinite(result.inference_ms) and result.inference_ms >= 0


def test_empty_detector_result_list_means_no_face():
    with pytest.raises(NoFaceDetectedError):
        processor(FakeDetector([])).process(Image.new("RGB", (8, 8)))


def test_malformed_detector_prediction_raises_runtime_error():
    with pytest.raises(RuntimeError, match="output|trả|prediction"):
        processor(FakeDetector([object()])).process(Image.new("RGB", (8, 8)))


def test_none_detector_output_is_malformed_not_no_face():
    with pytest.raises(RuntimeError, match="None"):
        processor(FakeDetector(None)).process(Image.new("RGB", (8, 8)))


def test_prediction_with_missing_boxes_is_malformed():
    with pytest.raises(RuntimeError, match="boxes"):
        processor(FakeDetector([object()])).process(Image.new("RGB", (8, 8)))


def test_valid_prediction_with_empty_boxes_means_no_face():
    with pytest.raises(NoFaceDetectedError):
        processor(FakeDetector([Prediction(Boxes())])).process(Image.new("RGB", (8, 8)))


def test_mismatched_box_and_confidence_counts_raise_runtime_error():
    with pytest.raises(RuntimeError, match="bounding box/confidence"):
        processor(detector([[1, 1, 4, 4]], [])).process(Image.new("RGB", (8, 8)))


@pytest.mark.parametrize(
    "box",
    [
        [np.nan, 1, 4, 4],
        [1, 1, np.inf, 4],
        [1, 1, 2],
    ],
)
def test_nonfinite_or_malformed_bbox_raises_runtime_error(box):
    with pytest.raises(RuntimeError, match="bounding box"):
        processor(detector([box], [0.9])).process(Image.new("RGB", (8, 8)))


@pytest.mark.parametrize("confidences", [[np.nan], [np.inf], [-0.1], [1.1]])
def test_malformed_confidence_raises_runtime_error(confidences):
    with pytest.raises(RuntimeError, match="confidence"):
        processor(detector([[1, 1, 4, 4]], confidences)).process(Image.new("RGB", (8, 8)))


def test_box_count_mismatch_raises_runtime_error():
    boxes = Boxes([[1, 1, 4, 4], [2, 2, 5, 5]], [0.8])
    with pytest.raises(RuntimeError, match="bounding box/confidence"):
        processor(FakeDetector([Prediction(boxes)])).process(Image.new("RGB", (8, 8)))


def test_invalid_image_path_and_input_type_raise_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="Không đọc được"):
        processor(detector()).process(tmp_path / "missing.png")
    with pytest.raises(TypeError, match="image_input"):
        processor(detector()).process(object())


def test_invalid_numpy_color_order_raises_value_error():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="array_color_order"):
        processor(detector()).process(image, array_color_order="XYZ")


def test_box_clamped_to_empty_region_raises_runtime_error():
    with pytest.raises(RuntimeError):
        processor(detector([[20, 20, 30, 30]], [0.9])).process(Image.new("RGB", (8, 8)))


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), float("-inf"), 0.99])
def test_rejects_invalid_crop_scale(scale):
    with pytest.raises(ValueError, match="crop_scale"):
        FaceDetectionPreprocessor(detector=detector(), crop_scale=scale)


def test_crop_scale_preserves_center_and_rounds_edges_outward():
    # bbox 10x8 centered at (20, 20); 1.3x gives 13x10.4 bounds.
    result = processor(detector([[15, 16, 25, 24]], [0.9])).process(
        Image.new("RGB", (50, 50))
    )
    assert result.bbox_xyxy == (15, 16, 25, 24)
    assert result.crop_bbox_xyxy == (13, 14, 27, 26)
    assert result.cropped_face.size == (14, 12)
    assert result.crop_size == result.cropped_face.size


@pytest.mark.parametrize(
    ("box", "expected_crop"),
    [
        ([0, 10, 10, 20], (0, 8, 12, 22)),
        ([90, 10, 100, 20], (88, 8, 100, 22)),
        ([10, 0, 20, 10], (8, 0, 22, 12)),
        ([10, 90, 20, 100], (8, 88, 22, 100)),
    ],
)
def test_expanded_crop_clamps_at_each_image_edge(box, expected_crop):
    result = processor(detector([box], [0.9])).process(Image.new("RGB", (100, 100)))
    assert result.crop_bbox_xyxy == expected_crop
    assert result.cropped_face.size == result.crop_size == (
        expected_crop[2] - expected_crop[0], expected_crop[3] - expected_crop[1]
    )
    assert result.crop_size[0] > 0 and result.crop_size[1] > 0
