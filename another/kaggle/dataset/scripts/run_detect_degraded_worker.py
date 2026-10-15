from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CLASSES = ("surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral")
CANVAS = (640, 480)
BRANCHES = ("yolo", "esrgan_yolo")
DEFAULT_SIZES = (24, 48, 96)
DEFAULT_DEGRADATIONS = ("clean", "blur", "down4", "jpeg10", "noise20", "combo")
VALID_DEGRADATIONS = set(DEFAULT_DEGRADATIONS)
DETECTION_THRESHOLD = 0.25


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_scene(face: Image.Image, side: int, seed: int, canvas: tuple[int, int] = CANVAS,
               field_size: tuple[int, int] = (10, 8), rectangle_count: int = 5,
               field_range: tuple[int, int] = (24, 232), gradient_x: float = 22.0,
               gradient_y: float = 18.0, tint_range: int = 12) -> tuple[Image.Image, list[int]]:
    """Create a deterministic synthetic scene and return its pasted-face xyxy box."""
    width, height = canvas
    if side < 1 or side > min(canvas):
        raise ValueError("face side must fit inside the canvas")
    rng = np.random.default_rng(seed)
    # Smoothly upsample a small random colour field, then add a broad colour gradient.
    field = rng.integers(field_range[0], field_range[1], size=(field_size[1], field_size[0], 3), dtype=np.uint8)
    background = Image.fromarray(field, "RGB").resize(canvas, Image.Resampling.BICUBIC)
    pixels = np.asarray(background, dtype=np.int16).copy()
    gx = np.linspace(-gradient_x, gradient_x, width, dtype=np.float32)[None, :, None]
    gy = np.linspace(gradient_y, -gradient_y, height, dtype=np.float32)[:, None, None]
    tint = rng.integers(-tint_range, tint_range + 1, size=(1, 1, 3), dtype=np.int16)
    pixels = np.clip(pixels + gx + gy + tint, 0, 255).astype(np.uint8)
    scene = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(scene)
    for _ in range(rectangle_count):
        x1, x2 = sorted((int(rng.integers(0, width)), int(rng.integers(0, width))))
        y1, y2 = sorted((int(rng.integers(0, height)), int(rng.integers(0, height))))
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        draw.rectangle((x1, y1, x2, y2), fill=color)
    py_rng = random.Random(seed ^ 0x9E3779B9)
    x = py_rng.randint(0, width - side)
    y = py_rng.randint(0, height - side)
    resized = face.convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
    scene.paste(resized, (x, y))
    return scene, [x, y, x + side, y + side]


def degrade_scene(image: Image.Image, degradation: str, seed: int, *, blur_sigma: float = 2.0,
                  down4_factor: int = 4, jpeg10_quality: int = 10, noise_sigma: float = 20.0,
                  combo_downscale: int = 2, combo_blur_sigma: float = 1.0,
                  combo_jpeg_quality: int = 20) -> Image.Image:
    if degradation not in VALID_DEGRADATIONS:
        raise ValueError(f"unknown degradation: {degradation}")
    result = image.convert("RGB")
    if degradation == "blur":
        return result.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    if degradation == "down4":
        w, h = result.size
        return result.resize((max(1, w // down4_factor), max(1, h // down4_factor)), Image.Resampling.BICUBIC).resize((w, h), Image.Resampling.BICUBIC)
    if degradation == "jpeg10":
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=jpeg10_quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if degradation == "noise20":
        rng = np.random.default_rng(seed)
        pixels = np.asarray(result, dtype=np.float32) + rng.normal(0.0, noise_sigma, size=(result.height, result.width, 3))
        return Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), "RGB")
    if degradation == "combo":
        w, h = result.size
        result = result.resize((max(1, w // combo_downscale), max(1, h // combo_downscale)), Image.Resampling.BICUBIC).resize((w, h), Image.Resampling.BICUBIC)
        result = result.filter(ImageFilter.GaussianBlur(radius=combo_blur_sigma))
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=combo_jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    return result


def parse_csv_values(value: str, cast, label: str) -> list:
    try:
        values = [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated {cast.__name__} values") from exc
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{label} must contain unique values")
    return values


def stratified_entries(bundle: Path, count: int, seed: int) -> list[tuple[str, int]]:
    with (bundle / "dataset" / "test_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        labels = [(f"{int(row['label'])}/{Path(row['image']).name}", int(row["label"])) for row in csv.DictReader(stream)]
    by_class = {index: sorted((item for item in labels if item[1] == index), key=lambda item: item[0]) for index in range(1, 8)}
    if any(not values for values in by_class.values()):
        raise ValueError("RAF-DB test labels must contain all seven classes")
    if count < 1 or count > len(labels):
        raise ValueError(f"DET_IMAGES must be between 1 and {len(labels)}")
    # Equal share per class, capped by availability (RAF-DB fear has only 74 test images);
    # the remainder is spread over classes that still have images (water-filling).
    quotas = {label: 0 for label in range(1, 8)}
    remaining = count
    while remaining:
        open_labels = [label for label in range(1, 8) if quotas[label] < len(by_class[label])]
        share, extra = divmod(remaining, len(open_labels))
        for position, label in enumerate(open_labels):
            take = min(share + int(position < extra), len(by_class[label]) - quotas[label])
            quotas[label] += take
            remaining -= take
    rng = random.Random(seed)
    selected = []
    for label in range(1, 8):
        selected.extend(rng.sample(by_class[label], quotas[label]))
    return sorted(selected, key=lambda item: (item[1], item[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect faces in synthetic degraded RAF-DB scenes, with and without RealESRGAN.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0, help="Override sampled image count; 0 uses --images.")
    parser.add_argument("--sizes", default="24,48,96")
    parser.add_argument("--degradations", default=",".join(DEFAULT_DEGRADATIONS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--canvas", default="640,480")
    parser.add_argument("--background-field", default="10,8")
    parser.add_argument("--background-rectangles", type=int, default=5)
    parser.add_argument("--background-color-range", default="24,232")
    parser.add_argument("--background-gradient-x", type=float, default=22.0)
    parser.add_argument("--background-gradient-y", type=float, default=18.0)
    parser.add_argument("--background-tint-range", type=int, default=12)
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--down4-factor", type=int, default=4)
    parser.add_argument("--jpeg10-quality", type=int, default=10)
    parser.add_argument("--noise-sigma", type=float, default=20.0)
    parser.add_argument("--combo-downscale", type=int, default=2)
    parser.add_argument("--combo-blur-sigma", type=float, default=1.0)
    parser.add_argument("--combo-jpeg-quality", type=int, default=20)
    parser.add_argument("--confidence-min", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--tile-pad", type=int, default=32)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("require --shards >= 1 and 0 <= --shard-index < --shards")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    try:
        sizes = parse_csv_values(args.sizes, int, "sizes")
        degradations = parse_csv_values(args.degradations, str, "degradations")
        canvas = tuple(parse_csv_values(args.canvas, int, "canvas"))
        field_size = tuple(parse_csv_values(args.background_field, int, "background-field"))
        field_range = tuple(parse_csv_values(args.background_color_range, int, "background-color-range"))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if any(size < 1 for size in sizes):
        parser.error("sizes must be positive and fit inside the canvas")
    if any(name not in VALID_DEGRADATIONS for name in degradations):
        parser.error(f"degradations must be selected from {sorted(VALID_DEGRADATIONS)}")
    if len(canvas) != 2 or len(field_size) != 2 or len(field_range) != 2 or min(canvas) < 1 or min(field_size) < 1:
        parser.error("canvas, background-field, and background-color-range must each contain two valid integers")
    if not 0 <= field_range[0] < field_range[1] <= 256:
        parser.error("background color range must satisfy 0 <= low < high <= 256")
    if any(size > min(canvas) for size in sizes):
        parser.error("sizes must fit inside the configured canvas")
    if args.background_rectangles < 0 or args.down4_factor < 1 or args.combo_downscale < 1:
        parser.error("rectangle count cannot be negative and downscale factors must be positive")
    if args.background_tint_range < 0 or min(args.background_gradient_x, args.background_gradient_y) < 0:
        parser.error("background tint range and gradient amplitudes must be nonnegative")
    if not 1 <= args.jpeg10_quality <= 100 or not 1 <= args.combo_jpeg_quality <= 100:
        parser.error("JPEG qualities must be in [1, 100]")
    if args.confidence_min < 0 or args.tile_size < 64 or args.tile_pad < 0:
        parser.error("confidence must be nonnegative, tile size >=64 and tile pad >=0")
    if any(value < 0 for value in (args.blur_sigma, args.noise_sigma, args.combo_blur_sigma)):
        parser.error("noise and blur sigmas must be nonnegative")
    count = args.limit or args.images
    entries = stratified_entries(args.bundle_root.resolve(), count, args.seed)
    entries = [entry for index, entry in enumerate(entries) if index % args.shards == args.shard_index]

    import sys
    project = args.project_root.resolve()
    sys.path.insert(0, str(project))
    from services.image_preprocessing.adapters import SpandrelRestorer, YoloDetector

    model_root = project / "model"
    restorer = SpandrelRestorer(model_root / "realesr-general-x4v3" / "realesr-general-x4v3.pth",
                                "realesr-general-x4v3 via Spandrel (same-size output)",
                                tile_size=args.tile_size, tile_pad=args.tile_pad)
    detector = YoloDetector(model_root / "yolo26_face" / "yolo26_face.pt")

    out_path = args.out_jsonl.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int, str]] = set()
    if out_path.is_file():
        with out_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    done.add((str(row["image"]), int(row["size"]), str(row["degradation"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    todo = [(image, label, size, degradation) for image, label in entries for size in sizes
            for degradation in degradations if (image, size, degradation) not in done]
    print(f"Shard {args.shard_index}/{args.shards}: images={len(entries)} scenes_pending={len(todo)}", flush=True)
    started_all = time.perf_counter()
    with out_path.open("a", encoding="utf-8", buffering=1) as stream:
        for index, (relative, label_index, size, degradation) in enumerate(todo, 1):
            key_seed = stable_seed(args.seed, relative, size, degradation)
            path = args.bundle_root / "dataset" / "DATASET" / "test" / relative
            row = {"image": relative, "class": CLASSES[label_index - 1], "class_id": label_index,
                   "size": size, "degradation": degradation, "gt_box": None, "branches": {}, "seed": args.seed}
            try:
                with Image.open(path) as source:
                    face = source.convert("RGB")
                scene, gt = make_scene(face, size, stable_seed(args.seed, relative, size, "scene"), canvas,
                                       field_size, args.background_rectangles, field_range,
                                       args.background_gradient_x, args.background_gradient_y,
                                       args.background_tint_range)
                degraded = degrade_scene(scene, degradation, stable_seed(args.seed, relative, size, degradation, "degrade"),
                                         blur_sigma=args.blur_sigma, down4_factor=args.down4_factor,
                                         jpeg10_quality=args.jpeg10_quality, noise_sigma=args.noise_sigma,
                                         combo_downscale=args.combo_downscale, combo_blur_sigma=args.combo_blur_sigma,
                                         combo_jpeg_quality=args.combo_jpeg_quality)
                row["gt_box"] = gt
                for branch in BRANCHES:
                    esrgan_ms = 0.0
                    detector_image = degraded
                    if branch == "esrgan_yolo":
                        t0 = time.perf_counter()
                        detector_image = restorer.restore(degraded)
                        if detector_image.size != degraded.size:
                            raise ValueError("RealESRGAN output changed scene dimensions")
                        esrgan_ms = (time.perf_counter() - t0) * 1000.0
                    t0 = time.perf_counter()
                    boxes = detector.detect(detector_image, confidence=args.confidence_min)
                    yolo_ms = (time.perf_counter() - t0) * 1000.0
                    row["branches"][branch] = {
                        "boxes": [{"xyxy": [round(float(v), 3) for v in box["bbox_xyxy"]],
                                   "conf": round(float(box["confidence"]), 6)} for box in boxes],
                        "timings_ms": {"esrgan": round(esrgan_ms, 3), "yolo": round(yolo_ms, 3)},
                    }
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if index % 25 == 0 or index == len(todo):
                rate = (time.perf_counter() - started_all) / index
                print(f"shard={args.shard_index} progress={index}/{len(todo)} image={relative} rate={rate:.2f}s/scene", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
