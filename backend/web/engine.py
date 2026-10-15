"""AI — FER Engine: suy luận POSTER / MA3D-Net bên trong tiến trình web (ADR-012).

`predict(image, codes)` trả dict đúng `contracts/ma3d_predict.schema.json`.
Module này KHÔNG import Streamlit (để sau có thể bọc thành service HTTP). torch chỉ được import
khi dùng chế độ `real` (chế độ `mock` chạy không cần torch).

Chế độ `real` (M2-05, bắt đầu 2026-09-24): registry đọc từ `configs/models.yaml`; mỗi mô hình nạp LƯỜI
(chỉ khi được yêu cầu) và MỘT lần mỗi tiến trình — xem `_get_real_handle`. Hiện chạy được POSTER (`arch: MA3D_no_fusion`, ADR-011). MA3D-Net (`arch: MA3D`) cần
file thống kê `*_emoca_stats.npz` (`stats` trong config) — thiếu thì trả `codes_missing`, không điền 0.
Không bao giờ âm thầm rơi về mock.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from . import mock_data
except ImportError:  # chạy với web/ trên sys.path (streamlit run frontend/app.py)
    import mock_data

_MODES = ("mock", "real")
_ATTN_GRID = (7, 7)

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_CONFIG = _ROOT / "configs" / "models.yaml"
_MA3D_REPO = _ROOT / "model" / "ma3d"
# Tiền xử lý lúc suy luận = val_transform của repo gốc (model/ma3d/build_dataloader.py:20-24).
# Là một phần của checkpoint (bất biến §1.5): KHÔNG đổi nếu không có checkpoint mới + ADR.
_INPUT_SIZE = (224, 224)
_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)
# Thứ tự ghép x_3d = prepare_batch của repo gốc (model/ma3d/engine.py:27).
_X3D_ORDER = ("exp", "detail", "pose", "shape", "tex")
# MA3D.__init__ nạp backbone bằng đường dẫn tương đối (model/ma3d/models/MA3D.py:87,99) với tên *_o1
# không có trên Drive; configs/models.yaml ghi bằng chứng hai file Drive tương đương ⇒ chuyển hướng.
_BACKBONE_REDIRECT = {"checkpoints/mobilefacenet_model_best.pth_o1.tar": "mobilefacenet",
                      "checkpoints/ir50_o1.pth": "ir50"}

# --- Nạp mô hình thật: LƯỜI từng mô hình, mỗi mô hình MỘT lần mỗi tiến trình (cache mức module) ---
# Không dùng st.cache_resource vì module này không import Streamlit (ADR-012).
# {model_id: handle}; có thể mới có một phần (chỉ mô hình đã được yêu cầu). Test được phép tiêm dict.
_REAL_MODELS: dict[str, Any] | None = None
_REAL_MODELS_LOCK = threading.Lock()
# Artifact đã kiểm sha256 trong tiến trình này: backbone dùng chung chỉ băm một lần dù nạp lười nhiều lượt.
_VERIFIED_ARTIFACTS: dict[str, Path] = {}
_TORCH_BANNER_DONE = False
# Dấu hiệu lỗi hết bộ nhớ (môi trường) — không cache vĩnh viễn, cho thử nạp lại.
_NO_STATS_MESSAGE = ("Chưa có file thống kê chuẩn hoá 3DMM (*_emoca_stats.npz, `stats` trong "
                     "configs/models.yaml); không suy luận với mã chưa chuẩn hoá hay điền 0.")
_MOCK_CODES_MESSAGE = ("Mã 3DMM đến từ EMOCA giả lập (MOCK), không phải mã của ảnh này — MA3D-Net không suy luận trên dữ liệu "
                       "giả. Nối EMOCA thật (Cài đặt → Kết nối EMOCA) để có kết quả MA3D-Net.")
_TRANSIENT_MARKERS = ("alloc_cpu", "defaultcpuallocator", "not enough memory", "out of memory",
                      "memoryerror", "paging file", "winerror 1455")


def _read_models_config() -> dict[str, Any]:
    import yaml
    return yaml.safe_load(_MODELS_CONFIG.read_text(encoding="utf-8")) or {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_path(artifacts: dict, key: str, verified: dict[str, Path]) -> Path:
    """Đường dẫn tuyệt đối của artifact, đã kiểm sha256 với config (mỗi file băm một lần)."""
    if key in verified:
        return verified[key]
    art = artifacts.get(key)
    if not art:
        raise RuntimeError(f"artifact '{key}' không có trong configs/models.yaml")
    path = _ROOT / art["path"]
    if not path.is_file():
        raise RuntimeError(f"thiếu file {art['path']}")
    digest = _sha256(path)
    if digest != art["sha256"]:
        raise RuntimeError(f"sha256 của {art['path']} không khớp config ({digest[:12]}… ≠ {art['sha256'][:12]}…)")
    verified[key] = path
    return path


@contextmanager
def _redirect_backbones(torch, paths: dict[str, Path]):
    """Trong lúc dựng MA3D: đổi 2 đường dẫn backbone tương đối sang file đã kiểm; đường dẫn lạ ⇒ lỗi."""
    original = torch.load

    def load(f, *args, **kwargs):
        if not isinstance(f, str) or f not in paths:
            raise RuntimeError(f"MA3D.__init__ nạp file ngoài danh sách đã kiểm: {f!r}")
        return original(str(paths[f]), *args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def _import_ma3d_class():
    os.environ.setdefault("MPLBACKEND", "Agg")  # MA3D.py import matplotlib.pyplot; không mở cửa sổ
    if str(_MA3D_REPO) not in sys.path:
        sys.path.insert(0, str(_MA3D_REPO))
    from models.MA3D import MA3D  # model/ma3d/models/MA3D.py (commit 6959e5a, không sửa)
    return MA3D


def _skip_fusion(torch):
    """ADR-011: POSTER = MA3D bỏ mm_fusion — thay mm_fusion bằng hàm đồng nhất, forward gốc giữ nguyên."""
    class SkipFusion(torch.nn.Module):
        def forward(self, x_ir, x_3d):
            return x_ir
    return SkipFusion()


def _load_stats(path: Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path) as npz:
        return {k: npz[k].astype("float32") for k in npz.files}


def _load_one(torch, MA3D, spec: dict, artifacts: dict, verified: dict[str, Path]) -> dict[str, Any]:
    arch = spec["arch"]
    if arch not in ("MA3D", "MA3D_no_fusion"):
        raise RuntimeError(f"arch '{arch}' chưa hỗ trợ")
    if not spec.get("labels") or len(spec["labels"]) != spec["num_classes"]:
        raise RuntimeError("thứ tự nhãn chưa xác minh (labels null hoặc sai số lớp)")
    backbones = {rel: _artifact_path(artifacts, spec["backbones"][name], verified)
                 for rel, name in _BACKBONE_REDIRECT.items()}
    ckpt_path = _artifact_path(artifacts, spec["ckpt"], verified)
    with _redirect_backbones(torch, backbones):
        model = MA3D(img_size=224, num_classes=spec["num_classes"], type="large")
    try:  # ánh xạ bộ nhớ: không giữ thêm bản sao ~300 MB của checkpoint trong RAM (máy 16 GB dễ hết bộ nhớ)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):  # torch cũ không có mmap / định dạng không hỗ trợ
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt[artifacts[spec["ckpt"]].get("state_key", "model")]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    # strict=True: không bỏ im lặng khoá thiếu/thừa như load_pretrained_weights của repo gốc (01_AI §3).
    model.load_state_dict(state, strict=True)
    print(f"[engine] {spec['id']}: nạp {ckpt_path.name} strict ({len(state)} khoá, không thiếu/thừa)", flush=True)
    del ckpt, state  # nhả sớm bản ánh xạ/bản sao trọng số
    if arch == "MA3D_no_fusion":
        model.mm_fusion = _skip_fusion(torch)
    model.eval()
    stats = None
    if arch == "MA3D" and spec.get("stats"):
        stats = _load_stats(_artifact_path(artifacts, spec["stats"], verified))
    return {"model": model, "spec": spec, "stats": stats}


def _runtime_available() -> bool:
    """torch + mã MA3D import được (không nạp trọng số; import đã có thì gần như miễn phí)."""
    try:
        import torch  # noqa: F401
        _import_ma3d_class()
        return True
    except Exception:
        return False


def _is_transient_load_error(exc: BaseException) -> bool:
    """Lỗi nạp do môi trường tạm thời (hết RAM/commit), không phải lỗi mã hay artifact."""
    if isinstance(exc, MemoryError):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(s in msg for s in _TRANSIENT_MARKERS)


def _load_model(model_id: str) -> dict[str, Any]:
    """Nạp MỘT mô hình `enabled` từ configs/models.yaml (kiểm sha256, strict, model.eval()). Không raise.

    Trả {"model", "spec", "stats"} hoặc {"error", "spec"} (+ "transient": True nếu lỗi môi trường tạm thời).
    """
    global _TORCH_BANNER_DONE
    cfg = _read_models_config()
    artifacts = cfg.get("artifacts") or {}
    spec = next((m for m in cfg.get("models") or [] if m.get("id") == model_id and m.get("enabled")), None)
    if spec is None:
        return {"error": f"'{model_id}' không có hoặc enabled: false trong configs/models.yaml", "spec": {"id": model_id}}
    try:
        import torch
        MA3D = _import_ma3d_class()
    except Exception as exc:  # thiếu torch/einops/matplotlib (hoặc hết RAM khi import torch): báo, không raise
        handle = {"error": f"không nạp được mã MA3D: {type(exc).__name__}: {exc}", "spec": spec}
        if _is_transient_load_error(exc):
            handle["transient"] = True
        return handle
    if not _TORCH_BANNER_DONE:
        print(f"[engine] torch {torch.__version__}, CUDA={torch.cuda.is_available()} (suy luận trên CPU)", flush=True)
        _TORCH_BANNER_DONE = True
    t0 = time.perf_counter()
    try:
        handle = _load_one(torch, MA3D, spec, artifacts, _VERIFIED_ARTIFACTS)
    except Exception as exc:
        handle = {"error": f"{type(exc).__name__}: {exc}", "spec": spec}
        if _is_transient_load_error(exc):
            handle["transient"] = True
            gc.collect()  # nhả phần tensor đã cấp dở trước khi lần sau thử lại
        return handle
    print(f"[engine] {model_id}: sẵn sàng sau {time.perf_counter() - t0:.1f} s", flush=True)
    return handle


def _get_real_handle(model_id: str) -> dict[str, Any]:
    """Handle của một mô hình, nạp LƯỜI lần đầu được yêu cầu rồi giữ trong `_REAL_MODELS` cả tiến trình.

    - Handle đã có trong `_REAL_MODELS` (kể cả do test tiêm vào) được dùng ngay, không nạp lại.
    - Lỗi nạp thường (thiếu file, sha256 sai, khoá state_dict thiếu/thừa, arch lạ) được cache như handle
      lỗi: lần sau báo lại đúng lỗi đó, không băm/nạp lại.
    - Lỗi môi trường tạm thời (MemoryError, `alloc_cpu`/DefaultCPUAllocator, "not enough memory", paging
      file quá nhỏ…) được ghi với "transient": True và ĐƯỢC THỬ NẠP LẠI ở lần gọi sau (khi máy đã có RAM).
    Các lần nạp chạy tuần tự dưới một khoá (không nạp hai mô hình song song — đỡ đỉnh RAM).
    """
    global _REAL_MODELS
    models = _REAL_MODELS
    handle = models.get(model_id) if models is not None else None
    if handle is not None and not handle.get("transient"):
        return handle
    with _REAL_MODELS_LOCK:
        if _REAL_MODELS is None:
            _REAL_MODELS = {}
        handle = _REAL_MODELS.get(model_id)
        if handle is None or handle.get("transient"):
            handle = _load_model(model_id)
            _REAL_MODELS[model_id] = handle
        return handle


def _get_real_models() -> dict[str, Any]:
    """Nạp (lười, qua `_get_real_handle`) MỌI mô hình `enabled` rồi trả `_REAL_MODELS` {model_id: handle}."""
    global _REAL_MODELS
    for spec in _read_models_config().get("models") or []:
        if spec.get("enabled"):
            _get_real_handle(spec["id"])
    with _REAL_MODELS_LOCK:
        if _REAL_MODELS is None:
            _REAL_MODELS = {}
        return _REAL_MODELS


# --- Tiện ích ---

def _error_item(model_id: str, code: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {"model_id": model_id, "status": "error",
            "error": {"code": code, "message": message, "retryable": retryable}}


def _response(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"contract_version": mock_data.CONTRACT_VERSION, "service": "ma3d_svc", "predictions": predictions}


def _check_codes(codes: Any) -> str | None:
    """None nếu `codes` đủ 5 nhóm đúng độ dài và toàn số hữu hạn; ngược lại là lý do (tiếng Việt).

    Tuyệt đối không điền 0 cho nhóm thiếu (01_AI.md §3).
    """
    if not isinstance(codes, dict):
        return f"codes phải là dict 5 nhóm, nhận {type(codes).__name__}"
    problems = []
    for key, dim in mock_data.CODE_DIMS.items():
        vals = codes.get(key)
        if vals is None:
            problems.append(f"thiếu '{key}'")
        elif not isinstance(vals, (list, tuple)) or len(vals) != dim:
            got = len(vals) if isinstance(vals, (list, tuple)) else type(vals).__name__
            problems.append(f"'{key}' cần {dim} phần tử, nhận {got}")
        elif not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals):
            problems.append(f"'{key}' có giá trị không phải số hữu hạn")
    return "; ".join(problems) or None


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _mock_attention(rng: random.Random, spread: float = 4.0) -> dict[str, Any]:
    """Bản đồ 7×7 giả (một đốm Gauss), chuẩn hoá min-max về 0..1, thứ tự hàng-trước."""
    rows, cols = _ATTN_GRID
    cy, cx = rng.uniform(1, rows - 2), rng.uniform(1, cols - 2)
    raw = [math.exp(-((r - cy) ** 2 + (c - cx) ** 2) / spread) for r in range(rows) for c in range(cols)]
    lo, hi = min(raw), max(raw)
    return {"space": "crop224", "grid": [rows, cols], "values": [round((v - lo) / (hi - lo), 4) for v in raw]}


def _mock_feature_maps(spec: dict, rng: random.Random) -> dict[str, Any]:
    """Ảnh trung gian giả theo ADR-013 (cùng hình dạng với attention).

    Bản thật (M2-05): chuẩn L2 theo kênh tại 49 vị trí của đầu ra IR50 (`image_backbone`),
    MobileFaceNet (`landmark_backbone`) và ‖sau − trước mm_fusion‖ (`fusion_delta`, chỉ MA3D).
    """
    maps = {"image_backbone": _mock_attention(rng, spread=8.0),
            "landmark_backbone": _mock_attention(rng, spread=3.0)}
    if spec["uses_3dmm"]:
        maps["fusion_delta"] = _mock_attention(rng, spread=2.0)
    return maps


def _grid_map(values) -> dict[str, Any] | None:
    """49 số → grid_map crop224 7×7, min-max 0..1. Hình dạng sai hoặc hằng số ⇒ None."""
    vals = [float(v) for v in values]
    if len(vals) != 49 or not all(math.isfinite(v) for v in vals):
        return None
    lo, hi = min(vals), max(vals)
    if hi - lo <= 0:
        return None
    return {"space": "crop224", "grid": list(_ATTN_GRID), "values": [round((v - lo) / (hi - lo), 4) for v in vals]}


def _x3d(torch, codes: dict, stats: dict) -> Any:
    """Ghép [exp, detail, pose, shape, tex] (334) và chuẩn hoá từng nhóm (x − mean)/std như RAF_DB.py:76-77."""
    parts = []
    for key in _X3D_ORDER:
        mean, std = stats.get(f"{key}_mean"), stats.get(f"{key}_std")
        if mean is None or std is None:
            raise KeyError(f"file thống kê thiếu '{key}_mean'/'{key}_std'")
        x = torch.tensor(codes[key], dtype=torch.float32)
        parts.append((x - torch.from_numpy(mean).reshape(-1)) / torch.from_numpy(std).reshape(-1))
    return torch.cat(parts).unsqueeze(0)


def _to_input(torch, image: Image.Image):
    """PIL → tensor [1,3,224,224] = Resize((224,224)) → ToTensor → Normalize (val_transform)."""
    import numpy as np
    img = image if image.mode == "RGB" else image.convert("RGB")
    img = img.resize(_INPUT_SIZE, Image.BILINEAR)  # torchvision Resize trên ảnh PIL dùng BILINEAR
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = (x - torch.tensor(_NORM_MEAN).view(3, 1, 1)) / torch.tensor(_NORM_STD).view(3, 1, 1)
    return x.unsqueeze(0)


def _real_prediction(handle: dict, image: Image.Image, codes: Any, codes_problem: str | None,
                     attention: bool, feature_maps: bool) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    spec, model = handle["spec"], handle["model"]
    mid = spec["id"]
    uses_3dmm = spec["arch"] == "MA3D"
    x_3d = None
    if uses_3dmm:
        if codes_problem:
            return _error_item(mid, "codes_missing", f"Mã 3DMM không hợp lệ: {codes_problem}.")
        if handle["stats"] is None:
            return _error_item(mid, "codes_missing", _NO_STATS_MESSAGE)
        try:
            x_3d = _x3d(torch, codes, handle["stats"])
        except (KeyError, ValueError, RuntimeError) as exc:
            return _error_item(mid, "codes_missing", f"Không dựng được x_3d: {exc}.")

    t0 = time.perf_counter()
    x = _to_input(torch, image)
    captured: dict[str, Any] = {}
    hooks = []
    if feature_maps:  # ADR-013: chuẩn L2 theo kênh tại 49 vị trí
        hooks.append(model.ir_layer.register_forward_hook(          # [1, 49, 512]
            lambda m, i, o: captured.__setitem__("image_backbone", o[0].norm(dim=-1))))
        hooks.append(model.face_landback.register_forward_hook(     # (landmark, [1, 512, 7, 7])
            lambda m, i, o: captured.__setitem__("landmark_backbone", o[1][0].norm(dim=0).reshape(-1))))
        if uses_3dmm:
            hooks.append(model.mm_fusion.register_forward_hook(     # ‖sau − trước mm_fusion‖
                lambda m, i, o: captured.__setitem__("fusion_delta", (o[0] - i[0][0]).norm(dim=0).reshape(-1))))
    try:
        with torch.no_grad():
            logits, _, attn_all = model(x, x_3d)
    finally:
        for h in hooks:
            h.remove()

    probs = F.softmax(logits[0].float(), dim=0).tolist()
    labels = list(spec["labels"])
    idx = max(range(len(probs)), key=probs.__getitem__)
    attn = None
    if attention:
        # Như model/ma3d/scripts/attn_vis.py:87,53: lớp cuối, nhánh "s", trung bình theo đầu, hàng 0 bỏ token đầu.
        attn = _grid_map(attn_all[-1]["s"]["img_attn"][0].mean(0)[0, 1:].tolist())
    fmaps = None
    if feature_maps:
        maps = {k: _grid_map(v.tolist()) for k, v in captured.items()}
        if maps.get("image_backbone") and maps.get("landmark_backbone"):
            fmaps = {k: v for k, v in maps.items() if v is not None}
    return {
        "model_id": mid,
        "status": "ok",
        "display_name": spec["display_name"],
        "dataset": spec["dataset"],
        "labels": labels,
        "probs": probs,
        "top1": {"index": idx, "label": labels[idx], "prob": probs[idx]},
        "attention": attn,
        "feature_maps": fmaps,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


def _mock_prediction(spec: dict, digest: bytes, attention: bool, feature_maps: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    labels = list(spec["labels"])
    n = len(labels)
    dominant = digest[0] % n
    # ~1/MOCK_DISAGREE_MODULO số ảnh: MA3D "chọn" lớp khác POSTER để UI thử trạng thái bất đồng.
    if spec["uses_3dmm"] and digest[1] % mock_data.MOCK_DISAGREE_MODULO == 0:
        dominant = (dominant + 1 + digest[2] % (n - 1)) % n
    rng = random.Random(digest + spec["id"].encode("ascii"))
    logits = [rng.gauss(0.0, 1.0) + (3.0 if i == dominant else 0.0) for i in range(n)]
    probs = _softmax(logits)
    idx = max(range(n), key=probs.__getitem__)
    attn = _mock_attention(rng) if attention and spec["supports_attention"] else None
    fmaps = _mock_feature_maps(spec, rng) if feature_maps else None
    return {
        "model_id": spec["id"],
        "status": "ok",
        "display_name": spec["display_name"],
        "dataset": spec["dataset"],
        "labels": labels,
        "probs": probs,
        "top1": {"index": idx, "label": labels[idx], "prob": probs[idx]},
        "attention": attn,
        "feature_maps": fmaps,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


def predict(image: Image.Image, codes: dict, models: list[str] | None = None, *,
            attention: bool = False, feature_maps: bool = True, mode: str = "mock",
            codes_real: bool = True) -> dict[str, Any]:
    """Xác suất cảm xúc cho từng mô hình; một mô hình lỗi không làm hỏng mô hình khác. Không raise.

    image: crop khuôn mặt (ADR-006: `crop_image` của EMOCA, dùng chung cho mọi mô hình).
    codes: 5 nhóm mã FLAME CHƯA chuẩn hoá {shape 100, exp 50, tex 50, pose 6, detail 128}.
           Chỉ bắt buộc với mô hình dùng 3DMM (MA3D); thiếu/sai → mục lỗi `codes_missing`.
    models: danh sách model_id; None → mọi mô hình trong registry.
    attention: trả bản đồ attention 7×7 cho mô hình hỗ trợ (POSTER và MA3D); còn lại `null`.
    feature_maps: trả ảnh trung gian 7×7 (ADR-013): image_backbone, landmark_backbone,
           fusion_delta (chỉ MA3D). False → `null`.
    mode: "mock" | "real". `real`: registry từ configs/models.yaml, models=None → `default_comparison`.
    codes_real: `codes` do EMOCA THẬT sinh ra. False (EMOCA giả lập) ⇒ mode `real` từ chối chạy mô hình dùng 3DMM
           (MA3D) với `codes_missing` — không suy luận thật trên mã giả rồi trình bày như kết quả thật; không nạp mô hình.
    """
    config_error = None
    if mode == "real":
        try:
            cfg = _read_models_config()
        except Exception as exc:
            cfg, config_error = {}, f"không đọc được configs/models.yaml: {type(exc).__name__}: {exc}"
        registry = {m["id"]: m for m in cfg.get("models") or []}
        default_ids = cfg.get("default_comparison") or [i for i, m in registry.items() if m.get("enabled")]
    else:
        registry = {m["id"]: m for m in mock_data.MOCK_MODELS}
        default_ids = list(registry)
    model_ids = list(default_ids) if models is None else list(models)

    if mode not in _MODES:
        return _response([_error_item(mid, "contract_mismatch", f"mode không hợp lệ: {mode!r}; chỉ nhận {_MODES}.")
                          for mid in model_ids])
    if not isinstance(image, Image.Image):
        return _response([_error_item(mid, "contract_mismatch", f"Cần PIL.Image.Image, nhận {type(image).__name__}.")
                          for mid in model_ids])

    codes_problem = _check_codes(codes)
    digest = None
    predictions = []
    for mid in model_ids:
        if config_error:
            predictions.append(_error_item(mid, "model_unavailable", config_error))
            continue
        spec = registry.get(mid)
        if spec is None:
            predictions.append(_error_item(mid, "model_unavailable", f"'{mid}' không có trong registry."))
            continue
        if mode == "real":
            if not spec.get("enabled"):
                predictions.append(_error_item(mid, "model_unavailable",
                                               f"'{mid}' đang enabled: false trong configs/models.yaml."))
                continue
            if spec.get("arch") == "MA3D" and not codes_real:
                predictions.append(_error_item(mid, "codes_missing", _MOCK_CODES_MESSAGE))
                continue
            if (spec.get("arch") == "MA3D" and not spec.get("stats") and mid not in (_REAL_MODELS or {})
                    and _runtime_available()):
                # Chưa có *_emoca_stats.npz ⇒ chắc chắn codes_missing: báo ngay, KHÔNG nạp (~240 MB, ~1.3 s) chỉ
                # để báo lỗi. Handle đã có trong _REAL_MODELS (nạp qua _get_real_models hoặc test tiêm) vẫn được dùng.
                # Thiếu torch/einops ⇒ đi tiếp để báo model_unavailable (lý do gốc), không che bằng codes_missing.
                problem = f"Mã 3DMM không hợp lệ: {codes_problem}." if codes_problem else _NO_STATS_MESSAGE
                predictions.append(_error_item(mid, "codes_missing", problem))
                continue
            handle = _get_real_handle(mid)  # nạp lười: chỉ mô hình được yêu cầu
            if "error" in handle:
                predictions.append(_error_item(mid, "model_unavailable", f"Không nạp được '{mid}': {handle['error']}",
                                               retryable=bool(handle.get("transient"))))
                continue
            try:
                predictions.append(_real_prediction(handle, image, codes, codes_problem, attention, feature_maps))
            except Exception as exc:  # lỗi suy luận của một mô hình không làm hỏng mô hình khác
                predictions.append(_error_item(mid, "model_unavailable",
                                               f"Suy luận '{mid}' lỗi: {type(exc).__name__}: {exc}", retryable=True))
            continue
        if spec["uses_3dmm"] and codes_problem:
            predictions.append(_error_item(mid, "codes_missing", f"Mã 3DMM không hợp lệ: {codes_problem}."))
            continue
        if digest is None:
            img = image if image.mode == "RGB" else image.convert("RGB")
            digest = mock_data.image_digest(img)
        predictions.append(_mock_prediction(spec, digest, attention, feature_maps))
    return _response(predictions)
