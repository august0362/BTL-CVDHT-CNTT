from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def ensure_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        raise RuntimeError(f"Refusing to replace an existing runtime link: {link}")
    if link.exists():
        raise RuntimeError(f"Runtime target already exists; remove or choose a fresh workdir: {link}")
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage the Kaggle bundle into writable working storage.")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=Path("/kaggle/working/rafdb5"))
    args = parser.parse_args()

    bundle = args.bundle_root.resolve()
    if not (bundle / "BUNDLE_MANIFEST.json").is_file():
        raise SystemExit(f"Not a Kaggle bundle root: {bundle}")
    source = bundle / "source" / "project"
    if not source.is_dir():
        raise SystemExit(f"Missing project source tree: {source}")

    workdir = args.workdir.resolve()
    project = workdir / "project"
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, project, dirs_exist_ok=True)

    ensure_link(project / "model" / "ma3d" / "checkpoints", bundle / "models" / "ma3d" / "checkpoints")
    ensure_link(project / "model" / "yolo26_face", bundle / "models" / "yolo26_face")
    ensure_link(project / "model" / "realesr-general-x4v3", bundle / "models" / "realesr-general-x4v3")
    ensure_link(project / "model" / "codeformer", bundle / "models" / "codeformer")

    emoca_source = project / "model" / "emoca"
    emoca_source.mkdir(parents=True, exist_ok=True)
    results = workdir / "results"
    results.mkdir(parents=True, exist_ok=True)
    paths = {
        "bundle_root": str(bundle),
        "project_root": str(project),
        "emoca_source_root": str(emoca_source),
        "dataset_root": str(bundle / "dataset" / "DATASET" / "test"),
        "labels_csv": str(bundle / "dataset" / "test_labels.csv"),
        "results_root": str(results),
    }
    path_file = workdir / "runtime_paths.json"
    path_file.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
