"""Kiểm web/engine.py chế độ `real` (M2-05/M2-06): POSTER (và đường chạy của MA3D-Net) trên CPU.

Chạy bằng python CÓ torch (venv web):  <venv>/Scripts/python.exe tests/test_engine_poster_real.py
Thiếu torch hoặc checkpoint ⇒ in SKIP và thoát mã 0 (để `python tests/test_*.py` bằng Python thường không đỏ oan).

Đo được ở đây: nạp strict, mô hình chỉ nạp MỘT lần, xác suất hợp lệ, đúng thứ tự nhãn của config, tiền xử lý KHỚP val_transform
của repo gốc (so với torchvision), thứ tự ghép + chuẩn hoá x_3d (kể cả với file rafdb_emoca_stats.npz THẬT), đường chạy MA3D với
stats thật (chỉ kiểm hình dạng/luồng vì mã 3DMM ở đây là ngẫu nhiên), MA3D thiếu thống kê ⇒ codes_missing (không điền 0),
MA3D từ chối chạy khi mã 3DMM đến từ EMOCA giả (`codes_real=False`), không rơi âm thầm về mock.
KHÔNG đo được ở đây: độ chính xác / tương đương với `eval.py` gốc trên RAF-DB (M2-07 — cần dữ liệu), kết quả MA3D-Net CÓ NGHĨA
(cần mã 3DMM của một ảnh thật do EMOCA thật sinh ra, tức là M0 trên Colab).
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))

try:
    import torch
except ImportError:
    print("SKIP test_engine_poster_real: chưa cài torch (dùng python của venv web)")
    sys.exit(0)

import numpy as np
import yaml
from PIL import Image

import engine

CONFIG = yaml.safe_load((ROOT / "backend" / "configs" / "models.yaml").read_text(encoding="utf-8"))
SPECS = {m["id"]: m for m in CONFIG["models"]}
ARTIFACTS = CONFIG["artifacts"]
needed = [ROOT / ARTIFACTS[k]["path"] for k in ("raf_poster_ckpt", "raf_ma3d_ckpt", "ir50", "mobilefacenet")]
missing = [p.name for p in needed if not p.is_file()]
if missing:
    print(f"SKIP test_engine_poster_real: thiếu checkpoint {missing} (xem configs/models.yaml)")
    sys.exit(0)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def face_image(seed: int = 3) -> Image.Image:
    """Ảnh 320×260 có cấu trúc (không phải một màu) để xác suất không suy biến; KHÔNG phải ảnh mặt thật."""
    rng = np.random.RandomState(seed)
    base = rng.randint(0, 255, (26, 32, 3)).astype("uint8")
    return Image.fromarray(base).resize((320, 260), Image.BICUBIC)


def codes(seed: int = 5) -> dict:
    rng = np.random.RandomState(seed)
    return {k: [float(v) for v in rng.randn(n)] for k, n in {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}.items()}


def by_id(out: dict, mid: str) -> dict:
    return next(p for p in out["predictions"] if p["model_id"] == mid)


def out_of_memory(reason: str) -> bool:
    """Lỗi nạp do máy thiếu RAM/commit (môi trường), không phải lỗi mã — test báo SKIP có lý do thay vì FAIL."""
    r = (reason or "").lower()
    return any(s in r for s in ("alloc_cpu", "defaultcpuallocator", "not enough memory", "memoryerror", "openblas"))


def rss_mb() -> float | None:
    """RSS (working set) của tiến trình, MB — chỉ để IN, không assert. None nếu không đo được."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        pass
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    fn = getattr(k32, "K32GetProcessMemoryInfo", None)
    if fn is None:
        return None
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    if not fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return pmc.WorkingSetSize / 2**20


def fmt_mb(v: float | None) -> str:
    return "?" if v is None else f"{v:.0f} MB"


class count_calls:
    """Đếm lời gọi engine.<name> (monkeypatch có khôi phục)."""

    def __init__(self, name: str, key=lambda *a, **k: None):
        self.name, self.key, self.calls = name, key, []

    def __enter__(self):
        self.orig = getattr(engine, self.name)

        def wrapper(*a, **k):
            self.calls.append(self.key(*a, **k))
            return self.orig(*a, **k)
        setattr(engine, self.name, wrapper)
        return self

    def __exit__(self, *exc):
        setattr(engine, self.name, self.orig)


class without_ma3d_stats:
    """Registry giả: ma3d_rafdb KHÔNG có stats — để vẫn kiểm nhánh "thiếu stats ⇒ codes_missing, không nạp" dù
    configs/models.yaml đã khai báo file thật. Trả lại cấu hình và cache mô hình như cũ khi thoát."""

    def __enter__(self):
        self.orig_cfg, self.saved_models = engine._read_models_config, engine._REAL_MODELS

        def cfg():
            c = copy.deepcopy(self.orig_cfg())
            for m in c["models"]:
                if m["id"] == "ma3d_rafdb":
                    m["stats"] = None
            return c
        engine._read_models_config, engine._REAL_MODELS = cfg, None
        return self

    def __exit__(self, *exc):
        engine._read_models_config, engine._REAL_MODELS = self.orig_cfg, self.saved_models


def test_lazy_loading():
    with without_ma3d_stats():
        _lazy_loading_body()


def _lazy_loading_body():
    print("== nạp LƯỜI: cả hai mô hình, (registry giả) MA3D stats null ⇒ codes_missing KHÔNG nạp MA3D; nạp tường minh ⇒ POSTER không nạp lại")
    check(all(not m.get("stats") for m in engine._read_models_config()["models"] if m["id"] == "ma3d_rafdb"),
          "tiền đề: trong registry giả, ma3d_rafdb không có stats")
    engine._REAL_MODELS = None
    img = face_image()
    rss0 = rss_mb()
    with count_calls("_load_one", key=lambda torch, MA3D, spec, *r: spec["id"]) as loads, \
            count_calls("_sha256", key=lambda p: Path(p).name) as hashes:
        t0 = time.perf_counter()
        out = engine.predict(img, codes(), models=["poster_rafdb", "ma3d_rafdb"], mode="real")
        t_poster = time.perf_counter() - t0
        rss1 = rss_mb()
        p = by_id(out, "poster_rafdb")
        if p["status"] != "ok" and out_of_memory(p["error"]["message"]):
            print(f"  SKIP nạp lười: máy thiếu RAM khi nạp POSTER ({p['error']['message'][:80]}…)")
            check(p["error"]["retryable"] is True, "lỗi hết RAM ⇒ retryable=True (được thử nạp lại)")
            return
        m = by_id(out, "ma3d_rafdb")
        check(p["status"] == "ok", f"poster_rafdb ok ({p.get('error')})")
        check(m["status"] == "error" and m["error"]["code"] == "codes_missing" and "emoca_stats" in m["error"]["message"],
              f"ma3d_rafdb ⇒ codes_missing (thiếu *_emoca_stats.npz) ({m.get('error', {}).get('message', '')[:60]}…)")
        check("ma3d_rafdb" not in engine._REAL_MODELS, f"MA3D chắc chắn codes_missing ⇒ KHÔNG nạp ({sorted(engine._REAL_MODELS)})")
        check(loads.calls == ["poster_rafdb"], f"_load_one chỉ gọi cho POSTER ({loads.calls})")
        bad = by_id(engine.predict(img, {"exp": [0.0]}, models=["ma3d_rafdb"], mode="real"), "ma3d_rafdb")
        check(bad["error"]["code"] == "codes_missing" and "không hợp lệ" in bad["error"]["message"] and loads.calls == ["poster_rafdb"],
              "codes sai + chưa nạp MA3D ⇒ vẫn báo lỗi mã trước (không điền 0, không nạp)")
        hashed_first = list(hashes.calls)

        t0 = time.perf_counter()
        engine._get_real_handle("ma3d_rafdb")  # nạp tường minh (như _get_real_models / khi có stats thật)
        t_ma3d = time.perf_counter() - t0
        rss2 = rss_mb()
        check("model" in engine._REAL_MODELS.get("ma3d_rafdb", {}), "nạp tường minh ⇒ có ma3d_rafdb")
        check(loads.calls == ["poster_rafdb", "ma3d_rafdb"], f"POSTER không nạp lại; MA3D nạp một lần ({loads.calls})")
        hashed_second = hashes.calls[len(hashed_first):]
        shared = {Path(ARTIFACTS[k]["path"]).name for k in ("ir50", "mobilefacenet")}
        check(shared <= set(hashed_first) and not (shared & set(hashed_second)),
              f"backbone dùng chung chỉ băm sha256 một lần (lần 1: {hashed_first}; lần 2: {hashed_second})")
        out = engine.predict(img, codes(), models=["ma3d_rafdb", "poster_rafdb"], mode="real")
        m2 = by_id(out, "ma3d_rafdb")
        check(m2["error"]["message"] == m["error"]["message"] and by_id(out, "poster_rafdb")["status"] == "ok",
              "đã nạp nhưng stats None ⇒ cùng thông điệp codes_missing; POSTER vẫn ok")
    print(f"  đo: RSS trước {fmt_mb(rss0)} → POSTER+MA3D(không nạp) {fmt_mb(rss1)} ({t_poster:.1f} s, gồm băm sha256 + nạp POSTER + 1 lần suy luận)"
          f" → sau khi nạp tường minh MA3D {fmt_mb(rss2)} ({t_ma3d:.1f} s)")


def test_transient_load_error_retry():
    print("== lỗi nạp do hết RAM ⇒ thử lại lần sau; lỗi thường ⇒ cache (không nạp mô hình thật)")
    saved = engine._REAL_MODELS
    orig = engine._load_one
    attempts = []
    errors = {"poster_rafdb": [RuntimeError("[enforce fail at alloc_cpu.cpp:114] DefaultCPUAllocator: not enough memory: you tried to allocate 1024 bytes."),
                               None],
              "ma3d_rafdb": [RuntimeError("thiếu file checkpoints/raf_MA.pth"), None]}

    def fake_load_one(torch, MA3D, spec, *rest):
        attempts.append(spec["id"])
        exc = errors[spec["id"]].pop(0)
        if exc is not None:
            raise exc
        return {"model": object(), "spec": spec, "stats": None}
    engine._REAL_MODELS = {}
    engine._load_one = fake_load_one
    try:
        img = face_image()
        first = by_id(engine.predict(img, {}, models=["poster_rafdb"], mode="real"), "poster_rafdb")
        check(first["status"] == "error" and first["error"]["code"] == "model_unavailable" and first["error"]["retryable"] is True,
              f"hết RAM ⇒ model_unavailable, retryable=True ({first.get('error', {}).get('message', '')[:60]}…)")
        check(engine._REAL_MODELS["poster_rafdb"].get("transient") is True, "handle lỗi được đánh dấu transient")
        engine._get_real_handle("poster_rafdb")
        check(attempts == ["poster_rafdb", "poster_rafdb"] and "model" in engine._REAL_MODELS["poster_rafdb"],
              f"lần gọi sau THỬ NẠP LẠI và thành công ({attempts})")
        # MA3D stats null ⇒ predict không tự nạp; nạp tường minh (như _get_real_models) để kiểm cache lỗi thường.
        engine._get_real_handle("ma3d_rafdb")
        engine._get_real_handle("ma3d_rafdb")
        second = by_id(engine.predict(img, codes(), models=["ma3d_rafdb"], mode="real"), "ma3d_rafdb")
        check(second["error"]["code"] == "model_unavailable" and second["error"]["retryable"] is False
              and "thiếu file" in second["error"]["message"] and attempts.count("ma3d_rafdb") == 1,
              "lỗi thường (thiếu file) ⇒ cache, KHÔNG nạp lại, retryable=False; predict dùng handle lỗi đã có")
    finally:
        engine._load_one = orig
        engine._REAL_MODELS = saved
    check(engine._load_one is orig, "đã khôi phục _load_one và _REAL_MODELS")


def test_preprocessing_matches_upstream():
    print("== tiền xử lý = val_transform của repo gốc")
    try:
        from torchvision import transforms as T
    except ImportError:
        print("  SKIP: không có torchvision — bỏ qua so sánh trực tiếp")
        return
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # build_dataloader.py:7-10
    val_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])       # build_dataloader.py:20-24
    for size in ((320, 260), (224, 224), (97, 131), (640, 480)):
        img = Image.fromarray(np.random.RandomState(size[0]).randint(0, 255, (size[1], size[0], 3)).astype("uint8"))
        ref = val_transform(img).unsqueeze(0)
        got = engine._to_input(torch, img)
        diff = float((ref - got).abs().max())
        check(tuple(got.shape) == (1, 3, 224, 224) and diff < 1e-4, f"ảnh {size[0]}×{size[1]}: engine._to_input ≈ torchvision (sai lệch lớn nhất {diff:.2e})")
    rgba = Image.new("RGBA", (50, 60), (10, 200, 30, 128))
    check(tuple(engine._to_input(torch, rgba).shape) == (1, 3, 224, 224), "ảnh RGBA được đổi sang RGB, không lỗi")


def test_x3d_layout():
    print("== x_3d: thứ tự ghép [exp, detail, pose, shape, tex] + chuẩn hoá (x − mean)/std")
    dims = {"exp": 50, "detail": 128, "pose": 6, "shape": 100, "tex": 50}
    rng = np.random.RandomState(0)
    stats = {}
    for k, n in dims.items():
        stats[f"{k}_mean"] = (rng.randn(n)).astype("float32")
        stats[f"{k}_std"] = (rng.rand(n) + 0.5).astype("float32")
    c = codes(11)
    x = engine._x3d(torch, c, stats)
    check(tuple(x.shape) == (1, 334), f"x_3d có hình (1, 334) (được {tuple(x.shape)})")
    off = 0
    ok = True
    for k in ("exp", "detail", "pose", "shape", "tex"):
        n = dims[k]
        want = (np.asarray(c[k], dtype="float32") - stats[f"{k}_mean"]) / stats[f"{k}_std"]
        ok &= bool(np.allclose(x[0, off:off + n].numpy(), want, atol=1e-6))
        off += n
    check(ok, "từng khối nằm đúng vị trí và đã chuẩn hoá (theo RAF_DB.py:76-77 / engine.py:27 của repo gốc)")
    swapped = dict(c, exp=c["shape"][:50] + [], shape=c["exp"] + [0.0] * 50)
    check(not np.allclose(engine._x3d(torch, swapped, stats).numpy(), x.numpy()), "đổi chỗ nhóm mã ⇒ vector đổi (kiểm không suy biến)")
    partial = {k: v for k, v in stats.items() if k != "pose_std"}
    try:
        engine._x3d(torch, c, partial)
        check(False, "thiếu pose_std phải lỗi")
    except KeyError as e:
        check("pose_std" in str(e), "file thống kê thiếu khoá ⇒ KeyError nêu đúng khoá (không điền 0)")


def test_real_poster():
    print("== POSTER thật (rafdb_POSTER.pth) trên CPU")
    img = face_image()
    out = engine.predict(img, codes(), models=["poster_rafdb"], attention=True, feature_maps=True, mode="real")
    p = by_id(out, "poster_rafdb")
    check(p["status"] == "ok", f"poster_rafdb status ok ({p.get('error')})")
    if p["status"] != "ok":
        return
    schema = json.loads((ROOT / "backend" / "contracts" / "ma3d_predict.schema.json").read_text(encoding="utf-8"))
    try:
        import jsonschema
        jsonschema.validate(out, schema)
        check(True, "đầu ra đúng ma3d_predict.schema.json")
    except ImportError:
        print("  SKIP: chưa cài jsonschema — bỏ qua kiểm schema")
    except jsonschema.ValidationError as e:
        check(False, f"đầu ra sai schema: {e.message[:150]}")
    probs = p["probs"]
    check(len(probs) == 7 and all(0.0 <= v <= 1.0 for v in probs) and abs(sum(probs) - 1.0) < 1e-5, f"7 xác suất trong [0,1], tổng = {sum(probs):.6f}")
    check(p["top1"]["index"] == max(range(7), key=probs.__getitem__) and p["top1"]["label"] == p["labels"][p["top1"]["index"]], "top1 = argmax, nhãn khớp chỉ số")
    check(p["labels"] == SPECS["poster_rafdb"]["labels"], f"thứ tự nhãn = configs/models.yaml {p['labels']}")
    check("MOCK" not in p["display_name"].upper() and p["display_name"] == SPECS["poster_rafdb"]["display_name"], "tên hiển thị lấy từ config, không phải mock")
    att = p["attention"]
    check(att and att["grid"] == [7, 7] and len(att["values"]) == 49 and min(att["values"]) >= 0 and max(att["values"]) <= 1, "attention 7×7 trong [0,1]")
    fm = p["feature_maps"] or {}
    check({"image_backbone", "landmark_backbone"} <= set(fm) and "fusion_delta" not in fm, f"feature_maps của POSTER: nhánh ảnh + landmark, KHÔNG có fusion_delta ({sorted(fm)})")
    check(all(m["grid"] == [7, 7] and len(m["values"]) == 49 for m in fm.values()), "mọi feature map 7×7")
    check(p["latency_ms"] > 0, f"latency_ms đo được ({p['latency_ms']:.0f} ms)")

    again = engine.predict(img, codes(), models=["poster_rafdb"], mode="real")
    check(by_id(again, "poster_rafdb")["probs"] == probs, "cùng ảnh ⇒ cùng xác suất (tất định, model.eval + no_grad)")
    other = engine.predict(face_image(seed=99), codes(), models=["poster_rafdb"], mode="real")
    check(by_id(other, "poster_rafdb")["probs"] != probs, "ảnh khác ⇒ xác suất khác (không hằng số)")
    nocodes = engine.predict(img, {}, models=["poster_rafdb"], mode="real")
    check(by_id(nocodes, "poster_rafdb")["status"] == "ok" and by_id(nocodes, "poster_rafdb")["probs"] == probs,
          "POSTER không cần mã 3DMM: codes rỗng vẫn cho đúng kết quả")
    off = engine.predict(img, codes(), models=["poster_rafdb"], attention=False, feature_maps=False, mode="real")
    q = by_id(off, "poster_rafdb")
    check(q["attention"] is None and q["feature_maps"] is None and q["probs"] == probs, "tắt attention/feature_maps ⇒ null, xác suất không đổi (hook không làm nhiễu)")


def test_load_once_and_no_mock_fallback():
    print("== nạp một lần, không rơi về mock")
    models_before = engine._get_real_models()
    with count_calls("_load_one") as loads:
        for _ in range(3):
            engine.predict(face_image(), codes(), models=["poster_rafdb", "ma3d_rafdb"], mode="real")
    healthy = not any(h.get("transient") for h in models_before.values())
    if healthy:
        check(loads.calls == [] and engine._get_real_models() is models_before, "3 lần gọi tiếp theo KHÔNG nạp lại mô hình (M2-05 AC5)")
    else:
        print("  SKIP kiểm không-nạp-lại: có mô hình lỗi hết RAM (được phép thử lại theo thiết kế)")
    for mid in ("poster_rafdb", "ma3d_rafdb"):
        h = models_before[mid]
        if "error" in h and out_of_memory(h["error"]):
            print(f"  SKIP {mid}: máy thiếu RAM khi nạp ({h['error'][:80]}…) — đóng bớt ứng dụng rồi chạy lại")
        else:
            check("model" in h, f"{mid} nạp được (strict) ({h.get('error', '')[:80]})")
    mock = engine.predict(face_image(), codes(), models=["poster_rafdb"], mode="mock")
    real = engine.predict(face_image(), codes(), models=["poster_rafdb"], mode="real")
    check(by_id(mock, "poster_rafdb")["probs"] != by_id(real, "poster_rafdb")["probs"] and "MOCK" in by_id(mock, "poster_rafdb")["display_name"].upper(),
          "kết quả real khác hẳn mock (mock có tên MOCK) — không lẫn nhau")
    unknown = engine.predict(face_image(), codes(), models=["khong_co"], mode="real")
    check(by_id(unknown, "khong_co")["error"]["code"] == "model_unavailable", "model_id lạ ⇒ model_unavailable")
    disabled = engine.predict(face_image(), codes(), models=["poster_caers"], mode="real")
    check(by_id(disabled, "poster_caers")["error"]["code"] == "model_unavailable", "mô hình enabled:false (nhãn chưa xác minh) ⇒ model_unavailable, không suy luận")


def test_x3d_with_real_stats():
    print("== _x3d với file stats THẬT (rafdb_emoca_stats.npz): mã = mean ⇒ 0; mã = mean+std ⇒ 1")
    sp = ROOT / ARTIFACTS["raf_emoca_stats"]["path"]
    if not sp.is_file():
        print(f"  SKIP: chưa có {sp.name} (file gitignored, người dùng tự đặt vào model/ma3d/checkpoints/)")
        return
    stats = engine._load_stats(sp)
    keys = engine._X3D_ORDER
    x0 = engine._x3d(torch, {k: stats[f"{k}_mean"].tolist() for k in keys}, stats)
    check(tuple(x0.shape) == (1, 334) and float(x0.abs().max()) < 1e-6, f"mã = mean ⇒ x_3d toàn 0, dạng {tuple(x0.shape)}")
    x1 = engine._x3d(torch, {k: (stats[f"{k}_mean"] + stats[f"{k}_std"]).tolist() for k in keys}, stats)
    check(float((x1 - 1).abs().max()) < 1e-5, "mã = mean + std ⇒ x_3d toàn 1 (đúng (x − mean)/std)")
    check(all(float(stats[f"{k}_std"].min()) > 0 for k in keys), "mọi std > 0 (không chia cho 0)")


def test_ma3d_path():
    print("== MA3D-Net với stats THẬT: đường chạy (mã 3DMM NGẪU NHIÊN ⇒ số không có nghĩa) + chốt chặn mã 3DMM giả")
    handle = engine._get_real_models()["ma3d_rafdb"]
    if "error" in handle:
        if out_of_memory(handle["error"]):
            print(f"  SKIP toàn bộ phần MA3D-Net: máy thiếu RAM khi nạp ({handle['error'][:90]}…) — CHƯA đo được")
            return
        check(False, f"ma3d_rafdb không nạp được: {handle['error'][:150]}")
        return
    if handle["stats"] is None:
        print("  SKIP: registry chưa có stats cho ma3d_rafdb (thiếu rafdb_emoca_stats.npz?) — nhánh này đã được kiểm ở nạp lười")
        return
    img = face_image()
    out = engine.predict(img, codes(), models=["ma3d_rafdb", "poster_rafdb"], attention=True, feature_maps=True, mode="real")
    m = by_id(out, "ma3d_rafdb")
    check(m["status"] == "ok", f"đường chạy MA3D (mm_fusion + x_3d, stats thật) chạy hết, không lỗi hình dạng ({m.get('error')})")
    if m["status"] == "ok":
        check(abs(sum(m["probs"]) - 1.0) < 1e-5, "tổng xác suất = 1 (mã 3DMM ngẫu nhiên ⇒ giá trị KHÔNG có ý nghĩa, chỉ kiểm luồng)")
        check(m["top1"]["index"] == max(range(len(m["probs"])), key=lambda i: m["probs"][i])
              and m["top1"]["label"] == m["labels"][m["top1"]["index"]], "top1 = argmax xác suất, nhãn khớp chỉ số")
        check(m["labels"] == SPECS["ma3d_rafdb"]["labels"], f"thứ tự nhãn = configs/models.yaml {m['labels']}")
        check("fusion_delta" in (m["feature_maps"] or {}), "MA3D có fusion_delta (tác động của 3DMM, ADR-013)")
        again = by_id(engine.predict(img, codes(), models=["ma3d_rafdb"], mode="real"), "ma3d_rafdb")
        check(again["probs"] == m["probs"], "tất định: cùng ảnh + cùng mã ⇒ cùng xác suất")
    check(by_id(out, "poster_rafdb")["status"] == "ok", "một mô hình lỗi/không lỗi không ảnh hưởng mô hình còn lại")
    bad = engine.predict(img, {"exp": [0.0] * 3}, models=["ma3d_rafdb"], mode="real")
    check(by_id(bad, "ma3d_rafdb")["error"]["code"] == "codes_missing", "codes sai/thiếu ⇒ codes_missing (không điền 0)")

    with count_calls("_real_prediction", key=lambda handle, *a, **k: handle["spec"]["id"]) as ran:
        blocked = engine.predict(img, codes(), models=["ma3d_rafdb", "poster_rafdb"], mode="real", codes_real=False)
    mb = by_id(blocked, "ma3d_rafdb")
    check(mb["status"] == "error" and mb["error"]["code"] == "codes_missing" and "giả lập" in mb["error"]["message"],
          f"mã 3DMM từ EMOCA giả ⇒ MA3D-Net từ chối, không suy luận ({mb.get('error', {}).get('message', '')[:50]}…)")
    check(ran.calls == ["poster_rafdb"] and by_id(blocked, "poster_rafdb")["status"] == "ok",
          f"chốt chặn chỉ chặn MA3D: _real_prediction chỉ chạy cho POSTER ({ran.calls})")


def test_torch_hygiene():
    print("== vệ sinh")
    check(torch.load.__module__.startswith("torch"), "torch.load không bị thay thế sau khi nạp (đã khôi phục sau _redirect_backbones)")
    handle = engine._get_real_models()["poster_rafdb"]
    check(not handle["model"].training, "model.eval() (không chạy ở chế độ train)")
    check(not any(p.requires_grad and p.grad is not None for p in handle["model"].parameters()), "không tích luỹ gradient sau suy luận")


def main() -> int:
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()} | suy luận trên CPU")
    test_preprocessing_matches_upstream()
    test_x3d_layout()
    test_x3d_with_real_stats()
    test_transient_load_error_retry()
    test_lazy_loading()
    test_real_poster()
    test_load_once_and_no_mock_fallback()
    test_ma3d_path()
    test_torch_hygiene()
    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
