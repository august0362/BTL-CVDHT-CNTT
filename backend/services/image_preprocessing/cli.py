from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from PIL import Image

from .adapters import SpandrelRestorer, YoloDetector
from .pipeline import ImagePipeline, PipelineConfig

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_GENERAL_WEIGHTS = Path("model/realesr-general-x4v3/realesr-general-x4v3.pth")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone M6 image restoration batch CLI")
    parser.add_argument("input", type=Path, help="Image file or directory (searched recursively)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--general-weights", type=Path, default=DEFAULT_GENERAL_WEIGHTS,
                        help=f"Real-ESRGAN general compact x4v3 checkpoint (default: {DEFAULT_GENERAL_WEIGHTS})")
    parser.add_argument("--yolo-weights", type=Path, default=Path("model/yolo26_face/yolo26_face.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--no-face-policy", choices=("keep", "error"), default="keep")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--max-side", type=int, default=4096)
    return parser


def _atomic_save_png(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png.tmp", dir=path.parent)
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _atomic_write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json.tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _inputs(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in EXTS: raise ValueError(f"unsupported image extension: {source.suffix}")
        return [source]
    if not source.is_dir(): raise FileNotFoundError(f"input path does not exist: {source}")
    return sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTS), key=lambda p: str(p).casefold())


def _artifact_record(path: Path, role: str) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"role": role, "filename": path.name, "size_bytes": size, "sha256": digest.hexdigest()}


def _relative_name(path: Path, source: Path) -> Path:
    return Path(path.name) if source.is_file() else path.relative_to(source)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = _inputs(args.input)
        if not inputs: raise ValueError("input directory contains no supported images")
        config = PipelineConfig(confidence=args.confidence, padding=args.padding,
                                max_side=args.max_side, no_face_policy=args.no_face_policy)
        # Validate numerical options before loading GPU models.
        if not math.isfinite(config.confidence) or not 0 <= config.confidence <= 1: raise ValueError("confidence must be in [0, 1]")
        if not math.isfinite(config.padding) or config.padding < 0: raise ValueError("padding must be finite and non-negative")
        if config.max_side < 1: raise ValueError("max-side must be positive")
        for checkpoint in (args.general_weights, args.yolo_weights):
            if not checkpoint.is_file(): raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
        artifact_records = [_artifact_record(args.general_weights, "general_restoration"),
                            _artifact_record(args.yolo_weights, "face_detection")]
        for name in ("enhanced_images", "faces", "metadata"):
            (args.output / name).mkdir(parents=True, exist_ok=True)
        general = SpandrelRestorer(args.general_weights, "realesr-general-x4v3 via Spandrel; resized to source dimensions", args.device)
        detector = YoloDetector(args.yolo_weights, args.device)
        resolved_device = getattr(general, "device", args.device)
        pipeline = ImagePipeline(general, detector, config)
    except Exception as exc:
        # If inputs were enumerated but model initialization failed, still emit
        # one explicit error record per file so batch status is auditable.
        if "inputs" in locals():
            try:
                source_root = args.input if args.input.is_dir() else args.input.parent
                (args.output / "metadata").mkdir(parents=True, exist_ok=True)
                for path in inputs:
                    relative = _relative_name(path, source_root)
                    dimensions = None
                    try:
                        with Image.open(path) as image: dimensions = list(image.size)
                    except Exception:
                        pass
                    stem = f"{relative.stem}__{relative.suffix.lstrip('.').lower()}"
                    _atomic_write_json({"input_name": str(relative), "input_dimensions": dimensions,
                        "output_dimensions": None, "model_ids": {"general": args.general_weights.name,
                        "detector": args.yolo_weights.name},
                        "model_artifacts": locals().get("artifact_records", []), "face_count": 0, "faces": [],
                        "stage_times_ms": {}, "status": "error", "error": f"{type(exc).__name__}: {exc}"},
                        args.output / "metadata" / relative.parent / f"{stem}.json")
            except Exception:
                pass
        print(json.dumps({"status": "initialization_error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2

    failures = 0
    source_root = args.input if args.input.is_dir() else args.input.parent
    for path in inputs:
        started = time.perf_counter()
        result = pipeline.process(path)
        relative = _relative_name(path, source_root)
        # Include original extension in the stem to avoid collisions such as a.jpg/a.png.
        output_stem = f"{relative.stem}__{relative.suffix.lstrip('.').lower()}"
        relative_parent = relative.parent
        if result.enhanced is not None:
            _atomic_save_png(result.enhanced, args.output / "enhanced_images" / relative_parent / f"{output_stem}.png")
        if result.status in ("ok", "no_face"):
            for index, face_result in enumerate(result.faces, start=1):
                _atomic_save_png(face_result.crop, args.output / "faces" / relative_parent / f"{output_stem}_face_{index:02d}.png")
        metadata = {
            "input_name": str(relative), "input_dimensions": result.input_dimensions,
            "output_dimensions": result.output_dimensions, "model_ids": result.model_ids,
            "config": {"confidence": config.confidence, "padding": config.padding,
                       "max_side": config.max_side, "no_face_policy": config.no_face_policy,
                       "device_requested": args.device, "device": resolved_device},
            "model_artifacts": artifact_records,
            "face_count": len(result.faces),
            "faces": [{"bbox_xyxy": face_result.bbox_xyxy, "crop_bbox_xyxy": face_result.crop_bbox_xyxy,
                       "confidence": face_result.confidence, "crop_dimensions": list(face_result.crop.size)}
                      for face_result in result.faces],
            "stage_times_ms": result.stage_times_ms, "total_ms": round((time.perf_counter() - started) * 1000, 3),
            "status": result.status, "error": result.error,
        }
        _atomic_write_json(metadata, args.output / "metadata" / relative_parent / f"{output_stem}.json")
        print(f"{relative}: {result.status}" + (f" ({result.error})" if result.error else ""))
        failures += result.status == "error"
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
