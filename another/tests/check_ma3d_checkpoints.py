"""Bằng chứng cho ADR-011 (POSTER = MA3D bỏ qua mm_fusion) và cho danh sách `quarantined`
trong configs/models.yaml — không cần torch.

Chạy:  python tests/check_ma3d_checkpoints.py      (đọc ~2.5 GB, khoảng 1–3 phút)

Phép thử:
  1. Bộ khoá + shape: POSTER và MA3D cùng bộ dữ liệu có giống nhau không.
  2. Trùng lặp: hai file có cùng toàn bộ bytes tensor không (bất kể vỏ zip).
  3. Biên khởi tạo: nn.Linear/nn.Conv2d mặc định khởi tạo đều trong ±1/sqrt(fan_in) (cả weight và
     bias). Một lớp CHƯA từng nhận gradient phải còn nằm trọn trong biên; lớp đã huấn luyện thì có
     phần tử vượt biên. So mm_fusion.* (nhánh 3DMM) với ir_layer/se_block/head (luôn được huấn luyện).
  4. Backbone: face_landback.* trong checkpoint so với mobilefacenet_model_best.pth.tar.

In PASS/FAIL cho những kỳ vọng đã ghi trong configs/models.yaml.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torchfree_ckpt import Checkpoint  # noqa: E402

CK = Path(__file__).resolve().parents[2] / "backend/model/ma3d/checkpoints"
PAIRS = {  # dataset: (MA3D, POSTER)
    "rafdb": ("raf_MA3D.pth", "rafdb_POSTER.pth"),
    "caers": ("caers_MA3D.pth", "caers_POSTER.pth"),
    "ferplus": ("ferplus_MA3D.pth", "ferplus_POSTER.pth"),
    "cheofamo": ("cheofamo_MA3D.pth", "CheoFaMo_POSTER.pth"),
}
# (khoá, fan_in) — mm_fusion theo models/ThreeDMM_Adaptive.py; đối chứng là các lớp luôn huấn luyện.
FUSION_PROBES = [
    ("mm_fusion.encoder_3dmm.net.0.weight", 334),
    ("mm_fusion.encoder_3dmm.net.0.bias", 334),
    ("mm_fusion.encoder_3dmm.net.3.weight", 512),
    ("mm_fusion.film_generator.net.2.weight", 512),
    ("mm_fusion.film_generator.net.2.bias", 512),
    ("mm_fusion.blocks.0.conv1.weight", 512),
    ("mm_fusion.blocks.2.conv2.weight", 128 * 9),
    ("mm_fusion.blocks.4.conv3.weight", 128),
]
CONTROL_PROBES = [("ir_layer.weight", 1024), ("se_block.linear1.weight", 512)]

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def out_of_bound(ck: Checkpoint, sd, key: str, fan_in: int) -> int:
    bound = 1 / math.sqrt(fan_in) * (1 + 1e-6)
    return sum(1 for v in ck.tensor_values(sd[key]) if abs(v) > bound)


def tensor_digest(ck: Checkpoint, sd) -> list[str]:
    return sorted(hashlib.sha1(ck.tensor_bytes(t)).hexdigest() for t in sd.values())


def main() -> int:
    loaded = {}
    for ds, (ma, po) in PAIRS.items():
        for f in (ma, po):
            ck = Checkpoint(CK / f)
            loaded[f] = (ck, ck.state_dict("model"))

    print("== 1. Bộ khoá và shape")
    for ds, (ma, po) in PAIRS.items():
        a, b = loaded[ma][1], loaded[po][1]
        same = set(a) == set(b) and all(a[k].shape == b[k].shape for k in a)
        check(same, f"{ds}: {po} cùng bộ khoá + shape với {ma} ({len(a)} khoá, có mm_fusion.*)")

    print("== 2. Trùng lặp toàn bộ tensor")
    for ds, (ma, po) in PAIRS.items():
        dup = tensor_digest(*loaded[ma]) == tensor_digest(*loaded[po])
        expect_dup = ds == "cheofamo"
        check(dup == expect_dup,
              f"{ds}: {po} {'TRÙNG' if dup else 'khác'} {ma} (archive_name={loaded[po][0].archive_name})"
              + (" — kỳ vọng trùng (quarantined)" if expect_dup else ""))

    print("== 3. Biên khởi tạo (số phần tử vượt ±1/sqrt(fan_in))")
    for ds, (ma, po) in PAIRS.items():
        if ds == "cheofamo":
            continue  # đã trùng MA3D ở phép thử 2
        for f in (po, ma):
            ck, sd = loaded[f]
            fusion = {k: out_of_bound(ck, sd, k, n) for k, n in FUSION_PROBES}
            control = {k: out_of_bound(ck, sd, k, n) for k, n in CONTROL_PROBES}
            untrained = all(v == 0 for v in fusion.values())
            print(f"     {f:22s} mm_fusion vượt biên: {sum(fusion.values()):7d} | đối chứng: {sum(control.values()):7d}")
            check(all(v > 0 for v in control.values()), f"{f}: lớp đối chứng đã huấn luyện (có vượt biên)")
            if f == po and ds in ("rafdb", "caers"):
                check(untrained, f"{f}: mm_fusion CHƯA huấn luyện ⇒ POSTER = MA3D bỏ qua mm_fusion")
            else:
                check(not untrained, f"{f}: mm_fusion ĐÃ huấn luyện"
                      + (" ⇒ không phải POSTER (quarantined)" if f == po else ""))

    print("== 4. Backbone MobileFaceNet")
    mb = Checkpoint(CK / "mobilefacenet_model_best.pth.tar")
    msd = mb.state_dict("state_dict")
    ck, sd = loaded["raf_MA3D.pth"]
    fl = {k[len("face_landback."):]: v for k, v in sd.items() if k.startswith("face_landback.")}
    check(set(fl) == set(msd), f"face_landback.* ({len(fl)}) cùng bộ khoá với mobilefacenet tar ({len(msd)})")
    diff = [k for k in msd if mb.tensor_bytes(msd[k]) != ck.tensor_bytes(fl[k])]
    bn_buffers = ("running_mean", "running_var", "num_batches_tracked")
    check(all(k.endswith(bn_buffers) for k in diff),
          f"khác nhau chỉ ở buffer BatchNorm ({len(diff)} tensor); trọng số học được trùng từng byte")

    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
