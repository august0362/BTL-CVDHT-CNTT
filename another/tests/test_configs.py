"""Kiểm configs/*.yaml (CONTRACTS_ADR Phần C) — không cần torch.

Chạy:  python tests/test_configs.py [--sha256] [--require-artifacts] [--config-dir DIR]
  --sha256             băm lại mọi artifact trong models.yaml (~2.5 GB, vài chục giây).
  --require-artifacts  file artifact vắng mặt ⇒ FAIL. Mặc định: SKIP có lý do (máy chưa tải checkpoint
                       ~2.47 GB vẫn kiểm được config); file CÓ mặt mà sai byte/sha256/shape vẫn luôn FAIL.
  --config-dir DIR     đọc models/labels/services.yaml từ DIR thay vì configs/ (để thử; đường dẫn
                       artifact vẫn tính từ gốc dự án).

Kiểm: schema_version; id duy nhất; mọi tham chiếu artifact tồn tại; file tồn tại (hoặc SKIP) và đúng số byte;
len(labels) == num_classes; labels null ⇒ enabled false; nhãn nằm trong labels.yaml; shape
`head.linear.weight` của checkpoint == (num_classes, 512); file quarantined không được model nào dùng;
services.yaml có đủ khoá, mode hợp lệ.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torchfree_ckpt import Checkpoint  # noqa: E402

ARCHS = {"MA3D", "MA3D_no_fusion"}
failures: list[str] = []
skipped: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def skip(msg: str) -> None:
    print("  SKIP " + msg)
    skipped.append(msg)


def main(verify_sha: bool, require_artifacts: bool, cfg_dir: Path) -> int:
    models = yaml.safe_load((cfg_dir / "models.yaml").read_text(encoding="utf-8"))
    labels_cfg = yaml.safe_load((cfg_dir / "labels.yaml").read_text(encoding="utf-8"))
    services = yaml.safe_load((cfg_dir / "services.yaml").read_text(encoding="utf-8"))
    frontalization_path = cfg_dir / "frontalization.yaml"
    print("== frontalization.yaml (ADR-029)")
    check(frontalization_path.is_file(), "có configs/frontalization.yaml")
    if frontalization_path.is_file():
        frontalization = yaml.safe_load(frontalization_path.read_text(encoding="utf-8")) or {}
        threshold = frontalization.get("roll_threshold_deg")
        check(frontalization.get("schema_version") == 1, "schema_version == 1")
        check(isinstance(frontalization.get("enabled_in_baseline"), bool),
              "enabled_in_baseline là bool")
        check(isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
              and 0 < threshold <= 90, "roll_threshold_deg là số trong (0, 90]")

    print("== labels.yaml")
    check(labels_cfg.get("schema_version") == 1, "schema_version == 1")
    known = set(labels_cfg["labels"])
    for name, entry in labels_cfg["labels"].items():
        d = entry.get("display", {})
        check(bool(d.get("vi")) and bool(d.get("en")), f"nhãn {name}: có display vi/en")

    print("== models.yaml")
    check(models.get("schema_version") == 1, "schema_version == 1")
    arts = models["artifacts"]
    for aid, a in arts.items():
        p = ROOT / a["path"]
        if require_artifacts or p.is_file():
            check(p.is_file(), f"artifact {aid}: tồn tại {a['path']}")
        else:
            skip(f"artifact {aid}: chưa có {a['path']} — tải checkpoint theo docs/RUNBOOK_DEMO.md "
                 f"(dùng --require-artifacts để bắt buộc)")
        if p.is_file():
            check(p.stat().st_size == a["bytes"], f"artifact {aid}: đúng {a['bytes']} byte")
            check(len(a["sha256"]) == 64, f"artifact {aid}: sha256 dạng 64 hex")
            if verify_sha:
                check(sha256_of(p) == a["sha256"], f"artifact {aid}: sha256 khớp")

    quarantined = {q["path"] for q in models.get("quarantined", [])}
    ids = [m["id"] for m in models["models"]]
    check(len(ids) == len(set(ids)), "id mô hình duy nhất")

    for m in models["models"]:
        mid = m["id"]
        check(m["arch"] in ARCHS, f"{mid}: arch thuộc {sorted(ARCHS)}")
        check(m["ckpt"] in arts, f"{mid}: ckpt trỏ tới artifact có thật")
        for role in ("ir50", "mobilefacenet"):
            check(m["backbones"].get(role) in arts, f"{mid}: backbone {role} trỏ tới artifact có thật")
        if m["stats"] is not None:
            check(m["stats"] in arts, f"{mid}: stats trỏ tới artifact có thật")
        if m["labels"] is None:
            check(m["enabled"] is False, f"{mid}: labels chưa xác minh ⇒ enabled false")
        else:
            check(len(m["labels"]) == m["num_classes"], f"{mid}: len(labels) == num_classes ({m['num_classes']})")
            check(len(set(m["labels"])) == len(m["labels"]), f"{mid}: nhãn không trùng")
            check(set(m["labels"]) <= known, f"{mid}: mọi nhãn có trong labels.yaml")
        if m["ckpt"] in arts:
            a = arts[m["ckpt"]]
            check(a["path"] not in quarantined, f"{mid}: không dùng file bị cách ly")
            p = ROOT / a["path"]
            if not p.is_file():
                skip(f"{mid}: không kiểm shape head.linear.weight — thiếu {a['path']}")
            else:
                ck = Checkpoint(p)
                sd = ck.state_dict(a.get("state_key"))
                head = sd.get("head.linear.weight")
                check(head is not None and head.shape == (m["num_classes"], 512),
                      f"{mid}: head.linear.weight = ({m['num_classes']}, 512) — đo được {head and head.shape}")
                if "archive_name" in a:
                    check(ck.archive_name == a["archive_name"], f"{mid}: archive_name = {a['archive_name']}")

    for mid in models.get("default_comparison", []):
        m = next((x for x in models["models"] if x["id"] == mid), None)
        check(m is not None and m["enabled"], f"default_comparison {mid}: tồn tại và enabled")

    print("== services.yaml")
    check(services.get("schema_version") == 1, "schema_version == 1")
    e = services.get("emoca_svc", {})
    check({"host", "port", "timeout_s", "mode"} <= set(e), "emoca_svc có host/port/timeout_s/mode")
    check(e.get("mode") in ("mock", "real"), "emoca_svc.mode ∈ {mock, real}")
    check("ma3d_svc" not in services, "không còn ma3d_svc (ADR-012)")
    lim = services.get("limits", {})
    check(all(isinstance(lim.get(k), int) and lim[k] > 0
              for k in ("max_image_bytes", "max_faces", "max_concurrent", "min_face_side_px")),
          "limits: số nguyên dương")

    print("== align.yaml (ADR-020)")
    align_path = cfg_dir / "align.yaml"
    check(align_path.is_file(), "có configs/align.yaml")
    if align_path.is_file():
        align = yaml.safe_load(align_path.read_text(encoding="utf-8"))
        check(align.get("schema_version") == 1, "schema_version == 1")
        check(isinstance(align.get("enabled"), bool), "enabled là bool (căn chỉnh là tuỳ chọn)")
        for name, t in (align.get("templates") or {}).items():
            size, pts = t.get("size"), t.get("points")
            check(isinstance(size, list) and len(size) == 2 and all(isinstance(v, int) and v > 0 for v in size),
                  f"{name}: size [w, h] nguyên dương")
            ok_pts = (isinstance(pts, list) and len(pts) == 5
                      and all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in pts))
            check(ok_pts, f"{name}: đúng 5 điểm [x, y]")
            if ok_pts and isinstance(size, list) and len(size) == 2:
                check(all(0 <= x <= size[0] - 1 and 0 <= y <= size[1] - 1 for x, y in pts), f"{name}: mọi điểm nằm trong ảnh")
                check(pts[0][0] < pts[2][0] < pts[1][0] and pts[3][0] < pts[4][0] and pts[0][1] < pts[2][1] < pts[3][1],
                      f"{name}: thứ tự mắt trái / mắt phải / mũi / miệng trái / miệng phải hợp lý")
            check(t.get("status") in ("provisional", "measured") and bool(t.get("source")),
                  f"{name}: có status (provisional|measured) và source")

    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    if skipped:
        print(f"ALL PASS — {len(skipped)} mục SKIP vì thiếu file artifact (xem dòng SKIP ở trên)")
    else:
        print("ALL PASS")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    cfg_dir = ROOT / "backend" / "configs"
    if "--config-dir" in args:
        i = args.index("--config-dir")
        cfg_dir = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    unknown = set(args) - {"--sha256", "--require-artifacts"}
    if unknown:
        print(f"Tham số không hợp lệ: {sorted(unknown)}")
        sys.exit(2)
    sys.exit(main("--sha256" in args, "--require-artifacts" in args, cfg_dir))
