from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


KAGGLE_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = KAGGLE_ROOT / "bundle"
MODEL_FILES = {
    "poster_rafdb": "models/ma3d/checkpoints/rafdb_POSTER.pth",
    "ma3d_rafdb": "models/ma3d/checkpoints/raf_MA3D.pth",
    "ir50": "models/ma3d/checkpoints/ir50.pth",
    "mobilefacenet": "models/ma3d/checkpoints/mobilefacenet_model_best.pth.tar",
    "rafdb_emoca_stats": "models/ma3d/checkpoints/rafdb_emoca_stats.npz",
    "yolo26_face": "models/yolo26_face/yolo26_face.pt",
    "realesr_general_x4v3": "models/realesr-general-x4v3/realesr-general-x4v3.pth",
    "codeformer": "models/codeformer/codeformer.pth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative: str) -> dict:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    labels_path = BUNDLE_ROOT / "dataset" / "test_labels.csv"
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != 3068:
        raise SystemExit(f"Expected 3068 RAF-DB test labels; found {len(labels)}")

    image_records = []
    class_counts = {str(label): 0 for label in range(1, 8)}
    for row in labels:
        label = int(row["label"])
        if label not in range(1, 8):
            raise SystemExit(f"Invalid RAF-DB label: {label}")
        rel = f"dataset/DATASET/test/{label}/{row['image']}"
        path = BUNDLE_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"Missing test image: {rel}")
        image_records.append(file_record(path, rel))
        class_counts[str(label)] += 1

    model_records = {
        name: file_record(BUNDLE_ROOT / rel, rel)
        for name, rel in MODEL_FILES.items()
    }
    manifest = {
        "bundle_version": 1,
        "created_from": "database/image/archive/DATASET/test + test_labels.csv",
        "dataset": {
            "name": "RAF-DB local test archive",
            "label_file": file_record(labels_path, "dataset/test_labels.csv"),
            "test_images": len(image_records),
            "class_counts": class_counts,
            "images": image_records,
        },
        "models": model_records,
        "flows": {
            "POSTER": "original input -> EMOCA landmarks/alignment -> POSTER",
            "MA3DNRe": "original input -> YOLO crop 1.3x -> CodeFormer -> EMOCA -> MA3D",
            "MA3DNCo": "original input -> RealESRGAN Compact -> YOLO crop 1.3x -> EMOCA -> MA3D",
            "MA3D": "original input -> EMOCA -> MA3D",
            "MA3DReCo": "original input -> RealESRGAN Compact -> YOLO crop 1.3x -> CodeFormer -> EMOCA -> MA3D",
        },
        "emoca": {
            "model_name": "EMOCA_v2_lr_mse_20",
            "upstream_revision": "e0be0db",
            "assets_in_bundle": False,
            "reason": "FLAME/EMOCA assets require personal registration and license acceptance; download step is gated in the Kaggle notebook.",
        },
        "compute": {"recommended_kaggle_accelerator": "NvidiaTeslaT4", "gpus": 2},
        "results": "Write numeric rows and metrics under /kaggle/working; no images/crops are persisted.",
    }
    output = BUNDLE_ROOT / "BUNDLE_MANIFEST.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_bytes = sum(r["bytes"] for r in model_records.values())
    image_bytes = sum(r["bytes"] for r in image_records)
    print(f"Wrote {output}")
    print(f"RAF-DB test images: {len(image_records)} ({image_bytes:,} bytes), classes={class_counts}")
    print(f"Model artifacts: {len(model_records)} ({model_bytes:,} bytes)")


if __name__ == "__main__":
    main()
