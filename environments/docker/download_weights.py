"""Tải trọng số cho tiến trình web (POSTER/MA3D-Net, YOLO, RealESRGAN, 3DDFA-V2) và kiểm SHA-256.

Chạy từ gốc dự án, chỉ cần Python >= 3.8 (thư viện chuẩn):
    python environments/docker/download_weights.py            # tải file còn thiếu
    python environments/docker/download_weights.py --check    # chỉ kiểm, không tải

KHÔNG tải FLAME/EMOCA: hai bộ này cần người dùng tự đăng ký và đồng ý giấy phép; container `emoca` tự tải
khi đặt ACCEPT_FLAME_EMOCA_LICENSES=1 (xem README.md).
Checkpoint POSTER/MA3D-Net lấy từ GitHub Release của repo dự án (tag WEIGHTS_TAG); SHA-256 trùng
backend/configs/models.yaml. Đổi nguồn bằng biến môi trường MA3D_WEIGHTS_BASE_URL.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

WEIGHTS_TAG = "weights-v1"
MA3D_BASE = os.environ.get(
    "MA3D_WEIGHTS_BASE_URL",
    f"https://github.com/august0362/BTL-CVDHT-CNTT/releases/download/{WEIGHTS_TAG}",
)
TDDFA_BASE = "https://raw.githubusercontent.com/cleardusk/3DDFA_V2/1b6c67601abffc1e9f248b291708aef0e43b55ae"

# (đường dẫn tương đối từ backend/, URL, sha256 viết thường)
FILES = [
    ("model/ma3d/checkpoints/ir50.pth", f"{MA3D_BASE}/ir50.pth",
     "f696a1d2216439c05dec042b42249a1c835f4a2106b91e14fbf09ed0fbd8acfa"),
    ("model/ma3d/checkpoints/mobilefacenet_model_best.pth.tar", f"{MA3D_BASE}/mobilefacenet_model_best.pth.tar",
     "b994af026bfddbafc507a6f1c8737a9896bab20ed2b0cfb6ae90b81736970313"),
    ("model/ma3d/checkpoints/raf_MA3D.pth", f"{MA3D_BASE}/raf_MA3D.pth",
     "8721625d74105cd6d61f0d427cbd9f63bb056d416755c8f2ba06ba84203eae95"),
    ("model/ma3d/checkpoints/rafdb_POSTER.pth", f"{MA3D_BASE}/rafdb_POSTER.pth",
     "cd548425d5fd50495f13c24b14a69971f13112f58024e6e3b80ac89e7c13f0ad"),
    ("model/ma3d/checkpoints/rafdb_emoca_stats.npz", f"{MA3D_BASE}/rafdb_emoca_stats.npz",
     "7aa56e9fe090e33a200342d1a2d81e4d17a741539c847a0c0d94600b77592c29"),
    ("model/yolo26_face/yolo26_face.pt",
     "https://github.com/marceloeatworld/yolo26-training/raw/main/checkpoints/yolo26_face.pt",
     "9fb5c9e36d3257c2979842e7573082bc05e4e78309d30a7333971c47b4b725b4"),
    ("model/realesr-general-x4v3/realesr-general-x4v3.pth",
     "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
     "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"),
    ("model/3ddfa_v2/mb1_120x120.pth", f"{TDDFA_BASE}/weights/mb1_120x120.pth",
     "a45a946c6e9b16f8d3cf2e69376da9560a7cf9afae671bebceb7e437a405ea79"),
    ("model/3ddfa_v2/bfm_noneck_v3.pkl", f"{TDDFA_BASE}/configs/bfm_noneck_v3.pkl",
     "89ac96480eddc331120f2c8401737c55a5fcc97b10f8a64574e63976c3d246ca"),
    ("model/3ddfa_v2/param_mean_std_62d_120x120.pkl", f"{TDDFA_BASE}/configs/param_mean_std_62d_120x120.pkl",
     "090d7150f77cb66c29ddae21e4508fbde59123dcd1dea7facc24a7ed06d1c795"),
    ("model/3ddfa_v2/tri.pkl", f"{TDDFA_BASE}/configs/tri.pkl",
     "0562a594d8369f7d1c86306522ff76fd3a2c81cf90ba9443893c1191ad0abbb8"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ma3d-demo-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()
    part.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="chỉ kiểm, không tải")
    args = ap.parse_args()

    bad = []
    for rel, url, digest in FILES:
        dest = BACKEND / rel
        if dest.is_file() and sha256(dest) == digest:
            print(f"[ok]    {rel}")
            continue
        if args.check:
            print(f"[thiếu] {rel}" if not dest.is_file() else f"[lệch]  {rel} (SHA-256 không khớp)")
            bad.append(rel)
            continue
        print(f"[tải]   {rel}\n    <- {url}")
        try:
            download(url, dest)
        except Exception as e:  # noqa: BLE001 — báo rồi đi tiếp file khác
            print(f"[LỖI]   {rel}: {e}")
            bad.append(rel)
            continue
        if sha256(dest) != digest:
            print(f"[LỖI]   {rel}: SHA-256 không khớp sau khi tải — giữ file để kiểm tay")
            bad.append(rel)
        else:
            print(f"[ok]    {rel}")

    if bad:
        print(f"\n{len(bad)} file chưa dùng được. Checkpoint POSTER/MA3D-Net lấy từ {MA3D_BASE}")
        return 1
    print("\nĐủ trọng số cho tiến trình web.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
