from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    bundle = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "bundle"
    manifest_path = bundle / "BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [manifest["dataset"]["label_file"], *manifest["dataset"]["images"], *manifest["models"].values()]
    errors = []
    for item in records:
        path = bundle / item["path"]
        if not path.is_file():
            errors.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != item["bytes"]:
            errors.append(f"size mismatch: {item['path']}")
            continue
        actual = sha256(path)
        if actual != item["sha256"]:
            errors.append(f"sha256 mismatch: {item['path']}")
    if errors:
        print("Bundle verification FAILED:")
        print("\n".join(errors))
        return 2
    test_root = bundle / "dataset" / "DATASET" / "test"
    actual_count = sum(1 for p in test_root.rglob("*") if p.is_file())
    if actual_count != manifest["dataset"]["test_images"]:
        print(f"Image count mismatch: {actual_count} != {manifest['dataset']['test_images']}")
        return 2
    print(f"PASS: {len(records)} files verified; {actual_count} RAF-DB test images; {len(manifest['models'])} model artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
