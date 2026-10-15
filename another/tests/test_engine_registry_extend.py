"""Đo AC M2-06: POSTER và MA3D-Net phục vụ qua MỘT hàm `engine.predict` bằng config — thêm mục registry mới KHÔNG đổi code.

Chạy bằng python CÓ torch (venv web):  <venv>/Scripts/python.exe tests/test_engine_registry_extend.py
Thiếu torch hoặc checkpoint ⇒ in SKIP và thoát mã 0.

Cách đo: chép configs/models.yaml sang tệp TẠM, thêm hai mục MỚI (id chưa từng có trong code/config) dùng lại checkpoint đã có:
  - `registry_probe_poster`  arch MA3D_no_fusion, ckpt raf_poster_ckpt  (như poster_rafdb)
  - `registry_probe_ma3d`    arch MA3D,           ckpt raf_ma3d_ckpt, stats raf_emoca_stats (như ma3d_rafdb)
rồi gọi `engine.predict(mode="real")` với id mới ⇒ phải ra status ok; xác suất phải TRÙNG mục gốc cùng checkpoint (chứng tỏ
mục mới đi đúng đường nạp/suy luận của mục gốc, không phải nhánh riêng). Kiểm thêm: web/engine.py và configs/models.yaml
không bị sửa (sha256 trước/sau), id mới KHÔNG có trong registry thật (⇒ model_unavailable), mục mới `enabled: false` ⇒
model_unavailable không nạp.

GIỚI HẠN (ghi rõ theo yêu cầu điều phối): engine không nhận đường dẫn registry qua tham số — đường dẫn cứng `_MODELS_CONFIG`
(web/engine.py:38). Test này MONKEYPATCH `engine._MODELS_CONFIG` sang tệp tạm (và khôi phục). Ở vận hành thật, "thêm mục
mới" = sửa chính configs/models.yaml — cũng chỉ là sửa config, không sửa code. Mã 3DMM ở ca MA3D là ngẫu nhiên ⇒ chỉ kiểm
đường chạy + trùng khớp với mục gốc, KHÔNG phải kết quả MA3D-Net có nghĩa.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))

try:
    import torch  # noqa: F401 — chỉ để biết môi trường có torch
except ImportError:
    print("SKIP test_engine_registry_extend: chưa cài torch (dùng python của venv web)")
    sys.exit(0)

import numpy as np
import yaml
from PIL import Image

import engine

MODELS_YAML = ROOT / "backend" / "configs" / "models.yaml"
ENGINE_PY = ROOT / "backend" / "web" / "engine.py"
CONFIG = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
SPECS = {m["id"]: m for m in CONFIG["models"]}
ARTIFACTS = CONFIG["artifacts"]
needed = [ROOT / ARTIFACTS[k]["path"] for k in ("raf_poster_ckpt", "raf_ma3d_ckpt", "ir50", "mobilefacenet", "raf_emoca_stats")]
missing = [p.name for p in needed if not p.is_file()]
if missing:
    print(f"SKIP test_engine_registry_extend: thiếu artifact {missing} (xem configs/models.yaml)")
    sys.exit(0)

NEW_POSTER, NEW_MA3D, NEW_OFF = "registry_probe_poster", "registry_probe_ma3d", "registry_probe_disabled"
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def face_image(seed: int = 3) -> Image.Image:
    """Ảnh có cấu trúc (không phải mặt thật) để xác suất không suy biến."""
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (26, 32, 3)).astype("uint8")).resize((320, 260), Image.BICUBIC)


def codes(seed: int = 5) -> dict:
    rng = np.random.RandomState(seed)
    return {k: [float(v) for v in rng.randn(n)] for k, n in {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}.items()}


def by_id(out: dict, mid: str) -> dict:
    return next(p for p in out["predictions"] if p["model_id"] == mid)


def extended_config() -> dict:
    cfg = copy.deepcopy(CONFIG)
    for new_id, src_id, name in ((NEW_POSTER, "poster_rafdb", "Probe POSTER (registry)"),
                                 (NEW_MA3D, "ma3d_rafdb", "Probe MA3D-Net (registry)"),
                                 (NEW_OFF, "poster_rafdb", "Probe tắt")):
        spec = copy.deepcopy(SPECS[src_id])
        spec.update(id=new_id, display_name=name, notes="Mục TẠM của tests/test_engine_registry_extend.py")
        if new_id == NEW_OFF:
            spec["enabled"] = False
        cfg["models"].append(spec)
    return cfg


class registry_file:
    """GIỚI HẠN: engine không nhận registry qua tham số ⇒ monkeypatch `engine._MODELS_CONFIG` (web/engine.py:38) sang tệp
    tạm; cache mô hình `_REAL_MODELS` cũng tách riêng. Thoát ⇒ khôi phục cả hai."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.saved = (engine._MODELS_CONFIG, engine._REAL_MODELS)
        engine._MODELS_CONFIG, engine._REAL_MODELS = self.path, None
        return self

    def __exit__(self, *exc):
        engine._MODELS_CONFIG, engine._REAL_MODELS = self.saved
        gc.collect()


def drop_models() -> None:
    engine._REAL_MODELS = None
    gc.collect()


def run(img, mid: str, **kw) -> tuple[dict, float]:
    t0 = time.perf_counter()
    out = engine.predict(img, codes(), models=[mid], mode="real", **kw)
    return by_id(out, mid), time.perf_counter() - t0


def main() -> int:
    t_all = time.perf_counter()
    print(f"torch {torch.__version__} | suy luận trên CPU")
    before = {p.name: sha(p) for p in (ENGINE_PY, MODELS_YAML)}
    img = face_image()

    print("== registry thật KHÔNG biết id mới (đối chứng)")
    p = by_id(engine.predict(img, codes(), models=[NEW_POSTER], mode="real"), NEW_POSTER)
    check(p["status"] == "error" and p["error"]["code"] == "model_unavailable",
          f"configs/models.yaml gốc: '{NEW_POSTER}' ⇒ model_unavailable ({p.get('error', {}).get('code')})")

    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "models.yaml"
        cfg_path.write_text(yaml.safe_dump(extended_config(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        with registry_file(cfg_path):
            check(engine._read_models_config()["models"][-1]["id"] == NEW_OFF,
                  "engine đọc registry tạm (monkeypatch _MODELS_CONFIG — giới hạn, xem docstring)")

            print("== mục MỚI arch MA3D_no_fusion (POSTER) qua cùng hàm predict")
            new, secs_new = run(img, NEW_POSTER, attention=True)
            check(new["status"] == "ok", f"{NEW_POSTER}: status ok ({new.get('error')}) — {secs_new:.1f} s gồm nạp")
            if new["status"] == "ok":
                check(new["display_name"] == "Probe POSTER (registry)" and new["labels"] == SPECS["poster_rafdb"]["labels"],
                      "tên hiển thị + thứ tự nhãn lấy từ mục mới trong config")
                check(len(new["probs"]) == 7 and abs(sum(new["probs"]) - 1) < 1e-5 and new["attention"] is not None,
                      f"7 xác suất, tổng {sum(new['probs']):.6f}, có attention")
            drop_models()
            ref, secs_ref = run(img, "poster_rafdb", attention=True)
            check(ref["status"] == "ok" and new.get("probs") == ref.get("probs"),
                  f"xác suất TRÙNG poster_rafdb (cùng checkpoint ⇒ cùng đường suy luận) — {secs_ref:.1f} s")
            drop_models()

            print("== mục MỚI arch MA3D (MA3D-Net) qua cùng hàm predict (mã 3DMM ngẫu nhiên: chỉ kiểm đường chạy)")
            new_m, secs_m = run(img, NEW_MA3D, codes_real=True)
            check(new_m["status"] == "ok", f"{NEW_MA3D}: status ok ({new_m.get('error')}) — {secs_m:.1f} s gồm nạp")
            drop_models()
            ref_m, secs_rm = run(img, "ma3d_rafdb", codes_real=True)
            check(ref_m["status"] == "ok" and new_m.get("probs") == ref_m.get("probs"),
                  f"xác suất TRÙNG ma3d_rafdb — {secs_rm:.1f} s")
            fake, _ = run(img, NEW_MA3D, codes_real=False)
            check(fake["status"] == "error" and fake["error"]["code"] == "codes_missing",
                  "mục MA3D mới vẫn theo luật chung: mã từ EMOCA giả ⇒ codes_missing")
            drop_models()

            print("== mục MỚI enabled: false")
            loads = []
            orig_load = engine._load_model
            engine._load_model = lambda mid: loads.append(mid) or orig_load(mid)
            try:
                off, _ = run(img, NEW_OFF)
            finally:
                engine._load_model = orig_load
            check(off["status"] == "error" and off["error"]["code"] == "model_unavailable" and not loads,
                  "enabled: false ⇒ model_unavailable, không nạp mô hình")

    after = {p.name: sha(p) for p in (ENGINE_PY, MODELS_YAML)}
    check(before == after, "web/engine.py và configs/models.yaml không đổi (sha256 trước = sau)")
    check(engine._MODELS_CONFIG == MODELS_YAML, "đã khôi phục engine._MODELS_CONFIG")
    print(f"\nTổng thời gian: {time.perf_counter() - t_all:.1f} s")
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
