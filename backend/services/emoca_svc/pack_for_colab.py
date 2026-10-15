"""Đóng gói MÃ của dự án cần cho notebook Colab thành một file zip (ADR-015).

Chạy (từ gốc dự án):
    python backend/services/emoca_svc/pack_for_colab.py [--out another/notebooks/emoca_svc_bundle.zip]
rồi tải file zip lên Colab bằng ô "Upload" của another/notebooks/emoca_colab.ipynb.

Trong zip (đường dẫn gốc phẳng, giải nén vào /content/svc):
    server.py  emoca_runner.py  mock_server.py       ← dịch vụ EMOCA thật (+ tầng HTTP dùng chung)
    patches/*.patch  patches/optional/*.patch  patches/apply.sh  patches/revert.sh
    contracts/emoca_analyze.schema.json               ← để notebook tự kiểm phản hồi
    configs/services.yaml                             ← giới hạn (kích thước ảnh…) cho server.py
    MANIFEST.json                                     ← sha256 từng file để đối chiếu

Quy tắc:
* CHỈ mã của dự án. Từ chối mọi tệp có đuôi của tài sản có giấy phép/nặng (.pth, .npz, .pkl, .npy, .obj, .zip…)
  hoặc lớn hơn 2 MiB — FLAME, texture, checkpoint EMOCA KHÔNG BAO GIỜ đi kèm (CLAUDE.md §1.1).
* Chuẩn hoá xuống dòng CRLF → LF cho mọi tệp văn bản: script `.sh` bị CRLF sẽ hỏng trên Linux/Colab.
* Tất định: cùng nội dung ⇒ cùng byte (dấu thời gian cố định, thứ tự sắp xếp), nên sha256 của zip so được giữa các lần.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
SVC = ROOT / "services" / "emoca_svc"

# (nguồn tương đối so với gốc dự án, đường dẫn trong zip)
FILES: List[Tuple[str, str]] = [
    ("services/emoca_svc/server.py", "server.py"),
    ("services/emoca_svc/emoca_runner.py", "emoca_runner.py"),
    ("services/emoca_svc/mock_server.py", "mock_server.py"),
    ("services/emoca_svc/patches/apply.sh", "patches/apply.sh"),
    ("services/emoca_svc/patches/revert.sh", "patches/revert.sh"),
    ("contracts/emoca_analyze.schema.json", "contracts/emoca_analyze.schema.json"),
    ("configs/services.yaml", "configs/services.yaml"),
]
GLOBS = [("services/emoca_svc/patches", "0*.patch", "patches"), ("services/emoca_svc/patches/optional", "0*.patch", "patches/optional")]
DENY_SUFFIXES = {".pth", ".pt", ".ckpt", ".tar", ".npz", ".npy", ".pkl", ".pickle", ".obj", ".zip", ".onnx", ".bin", ".safetensors"}
MAX_BYTES = 2 * 1024 * 1024
FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # dấu thời gian cố định ⇒ zip tất định


def _normalize(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _collect(root: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = [(arc, root / src) for src, arc in FILES]
    for folder, pattern, arc_dir in GLOBS:
        for p in sorted((root / folder).glob(pattern)):
            items.append((f"{arc_dir}/{p.name}", p))
    return sorted(items, key=lambda t: t[0])


def _regex_value(path: Path, pattern: str) -> Optional[str]:
    m = re.search(pattern, path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def build_bundle(root: Path = ROOT) -> Tuple[bytes, Dict]:
    """Trả (nội dung zip, manifest). Ném ValueError nếu có tệp bị cấm hoặc thiếu."""
    entries = []
    blobs: Dict[str, bytes] = {}
    for arc, path in _collect(root):
        if not path.is_file():
            raise ValueError(f"thiếu tệp cần đóng gói: {path}")
        if path.suffix.lower() in DENY_SUFFIXES:
            raise ValueError(f"tệp bị cấm (tài sản có giấy phép/nặng): {path.name}")
        data = _normalize(path.read_bytes())
        if len(data) > MAX_BYTES:
            raise ValueError(f"{path.name} lớn {len(data)} byte > {MAX_BYTES} — không phải mã của dự án?")
        blobs[arc] = data
        entries.append({"path": arc, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "bundle_version": 1,
        "contract_version": _regex_value(root / "services/emoca_svc/mock_server.py", r'CONTRACT_VERSION\s*=\s*"([^"]+)"'),
        "upstream_commit_expected": _regex_value(root / "services/emoca_svc/patches/apply.sh", r'EXPECTED_COMMIT="([^"]+)"'),
        "files": entries,
    }
    blobs["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for arc in sorted(blobs):
            info = zipfile.ZipInfo(arc, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if arc.endswith(".sh") else 0o644) << 16
            zf.writestr(info, blobs[arc])
    return buf.getvalue(), manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Đóng gói mã emoca_svc cho Colab")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "another" / "notebooks" / "emoca_svc_bundle.zip")
    args = ap.parse_args(argv)
    try:
        data, manifest = build_bundle()
    except ValueError as e:
        print(f"[LOI] {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"Đã ghi {args.out} ({len(data)} byte, sha256 {hashlib.sha256(data).hexdigest()[:16]}…)")
    for f in manifest["files"]:
        print(f"  {f['path']:<44} {f['bytes']:>7} byte")
    print(f"contract_version {manifest['contract_version']} · upstream EMOCA {manifest['upstream_commit_expected']}")
    print("Tải file này lên Colab ở ô Upload của another/notebooks/emoca_colab.ipynb. KHÔNG kèm FLAME/checkpoint (giấy phép).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
