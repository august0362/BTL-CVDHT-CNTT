"""emoca_runner — nối EMOCA THẬT cho `emoca_svc` (M1-03, ADR-015).

Hai lớp, cố ý tách để kiểm thử được mà không cần GPU:

* `EmocaRunner` — logic thuần (numpy + Pillow): chọn mặt, đổi hệ toạ độ, dựng response đúng
  `contracts/emoca_analyze.schema.json`, kiểm hình dạng đầu ra. Kiểm thử trên máy thường bằng backend giả
  (`tests/test_emoca_svc_real.py`).
* `EmocaBackend` — phần gọi mã EMOCA/PyTorch (chỉ chạy được trong env `work38`, Python 3.8, GPU CUDA).
  Sao chép đúng luồng của `gdl_apps/EMOCA/demos/test_emoca_on_images.py` + `gdl/datasets/ImageTestDataset.py`
  (FAN → `bbox2point(kpt68)` → scale 1.25 → phép đồng dạng → crop 224 → `encode` → `decode`), nhưng trên ảnh
  trong bộ nhớ: KHÔNG ghi ảnh ra đĩa (ADR-009) và trả luôn `tform` (`crop_transform`, bất biến §1.7).
  **Chưa được kiểm chứng trên EMOCA thật** — notebook `another/notebooks/emoca_colab.ipynb` có bước đo (M0-04/M0-05).

Chạy được trên Python 3.8: không dùng cú pháp `X | Y` ngoài chú thích kiểu.

Hệ toạ độ landmark (đọc từ mã, chờ đo ở M0-05): `predicted_landmarks` là toạ độ chuẩn hoá [-1, 1] trên crop;
`gdl/utils/DecaUtils.py::tensor_vis_landmarks` vẽ chúng bằng `lm * size/2 + size/2` ⇒ crop224 = lm·112 + 112,
rồi đưa về `original_px` bằng nghịch đảo `crop_transform`. Nếu đo thấy miền khác, runner thêm cảnh báo.
"""

from __future__ import annotations

import base64
import io
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

CROP_SIZE = 224
CROP_SCALE = 1.25  # TestData(scale=1.25) — cùng giá trị lúc huấn luyện EMOCA (docs/modules/EMOCA.md §5.1)
CODE_DIMS = {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}
# Khoá trong `codedict` của EMOCA (gdl_apps/EMOCA/utils/io.py::save_codes)
CODE_KEYS = {"shape": "shapecode", "exp": "expcode", "tex": "texcode", "pose": "posecode", "detail": "detailcode"}
# Khoá render trong contract ← khoá trong `visdict` (io.py::save_images)
RENDER_SOURCES = {
    "geometry_coarse": "geometry_coarse",
    "geometry_detail": "geometry_detail",
    "output_coarse": "output_images_coarse",
    "output_detail": "output_images_detail",
}
LANDMARK_NORM_LIMIT = 1.5  # |lm| lớn hơn ngưỡng này ⇒ nghi sai hệ toạ độ, thêm cảnh báo (không chặn)


def wsl_ldpath_active(environ: Dict[str, str]) -> bool:
    """`LD_LIBRARY_PATH` có thư mục thư viện CUDA của WSL (`EMOCA_WSL_LIB_DIR`, mặc định /usr/lib/wsl/lib) không.

    Nguyên nhân gốc lỗi cuDNN trên WSL2 (đo 2026-09-25): cuDNN 8 dlopen `libcuda.so` (không đuôi phiên bản), tệp này chỉ
    có ở /usr/lib/wsl/lib ⇒ thiếu thư mục đó trong LD_LIBRARY_PATH thì cuDNN abort. Script WSL (common.sh::
    apply_wsl_ldpath) tự thêm; trường này cho health biết tiến trình server có được thêm thật không.
    Biến phải có TRƯỚC khi tiến trình khởi động (bộ nạp động đọc lúc đó) nên chỉ báo, không tự sửa.
    """
    target = (environ.get("EMOCA_WSL_LIB_DIR") or "/usr/lib/wsl/lib").rstrip("/")
    return any(p.rstrip("/") == target for p in (environ.get("LD_LIBRARY_PATH") or "").split(":") if p)


def resolve_cudnn(environ: Dict[str, str]) -> Tuple[Optional[bool], str]:
    """Quyết định bật/tắt cuDNN từ biến môi trường. Trả (bật?, lý do); `None` = giữ mặc định của torch (bật).

    Vì sao: trên máy dự án (WSL2, torch 1.11.0+cu115, cudnn 8302, RTX 3050) conv2d CÓ cuDNN lỗi
    CUDNN_STATUS_NOT_INITIALIZED / INTERNAL_ERROR, có lần tiến trình tự huỷ (SIGABRT); không cuDNN thì chạy.
    * `EMOCA_CUDNN=0` → tắt; `=1` → bật; `auto` (mặc định) → đọc `cudnn_ok=` trong tệp `EMOCA_CUDNN_PROBE`
      (services/emoca_svc/wsl/common.sh::cudnn_probe ghi). Có đặt đường dẫn probe mà không đọc được ⇒ tắt cho an toàn
      (giống cudnn_resolve của script); không đặt ⇒ giữ mặc định (vd Colab, cuDNN chạy bình thường).
    * Giá trị khác ⇒ ValueError (không đoán).
    """
    raw = (environ.get("EMOCA_CUDNN") or "auto").strip().lower()
    if raw == "0":
        return False, "EMOCA_CUDNN=0"
    if raw == "1":
        return True, "EMOCA_CUDNN=1"
    if raw != "auto":
        raise ValueError(f"EMOCA_CUDNN={raw!r} không hợp lệ (chỉ nhận auto, 0 hoặc 1).")
    probe = (environ.get("EMOCA_CUDNN_PROBE") or "").strip()
    if not probe:
        return None, "auto: không có EMOCA_CUDNN_PROBE — giữ mặc định của torch"
    try:
        lines = Path(probe).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False, f"auto: không đọc được {probe} — tắt cuDNN cho an toàn"
    ok = next((l.split("=", 1)[1].strip() for l in lines if l.startswith("cudnn_ok=")), "")
    if ok == "1":
        return True, f"auto: {probe} báo cuDNN chạy được"
    if ok == "0":
        return False, f"auto: {probe} báo cuDNN hỏng"
    return False, f"auto: {probe} không có cudnn_ok=0|1 — tắt cuDNN cho an toàn"


# Làm nóng sau khi nạp (M4-01, đo 2026-09-25/26): lần gọi đầu tiên tới server vừa khởi động tốn ~11-27 s (gần hết ở FAN),
# các lần sau ~0,3 s. Chạy trước vài lượt trên ảnh TỔNG HỢP (trong bộ nhớ, không phải ảnh người, không ghi đĩa).
DEFAULT_WARMUP_SIZES = (224, 512, 1024)  # cạnh (px) của ảnh vuông tổng hợp
WARMUP_SIZE_RANGE = (64, 4096)


def resolve_warmup(environ: Dict[str, str]) -> Tuple[bool, List[int]]:
    """`EMOCA_WARMUP` (mặc định 1; 0/false/no/off ⇒ tắt) và `EMOCA_WARMUP_SIZES` ("224,512,1024"). Giá trị sai ⇒
    ValueError (load() đổi thành model_unavailable, giống EMOCA_CUDNN — không đoán)."""
    raw = (environ.get("EMOCA_WARMUP") or "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False, []
    if raw not in ("1", "true", "yes", "on"):
        raise ValueError(f"EMOCA_WARMUP={raw!r} không hợp lệ (1 hoặc 0).")
    sizes_raw = (environ.get("EMOCA_WARMUP_SIZES") or "").strip()
    if not sizes_raw:
        return True, list(DEFAULT_WARMUP_SIZES)
    try:
        sizes = [int(x) for x in sizes_raw.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"EMOCA_WARMUP_SIZES={sizes_raw!r} không hợp lệ (danh sách số nguyên, vd 224,512,1024).")
    lo, hi = WARMUP_SIZE_RANGE
    if not sizes or any(not lo <= x <= hi for x in sizes):
        raise ValueError(f"EMOCA_WARMUP_SIZES={sizes_raw!r}: mỗi cạnh phải trong [{lo}, {hi}].")
    return True, sizes


def synthetic_image(side: int, seed: int = 0) -> np.ndarray:
    """Ảnh RGB uint8 side×side TỔNG HỢP (dải màu + nhiễu, không có mặt người) — chỉ để chạy thử mạng."""
    rng = np.random.RandomState(seed)
    ramp = np.linspace(0, 255, side, dtype=np.float32)
    img = np.stack([np.tile(ramp, (side, 1)), np.tile(ramp[:, None], (1, side)), np.full((side, side), 128.0)], -1)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


DEFAULT_DETECT_MAX_SIDE_PX = 1024  # configs/services.yaml limits.detect_max_side_px ghi đè (server.py truyền vào)


def downscale_for_detect(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    """Bản thu nhỏ (cạnh dài ≤ max_side, LANCZOS) CHỈ để dò mặt; trả (ảnh, sx, sy) với sx = rộng_mới/rộng_gốc.
    Lý do (đo 2026-09-26, ảnh điện thoại 2208×2944): S3FD trên ảnh gốc báo 3 "mặt", khung lớn nhất là tóc/trán ⇒ crop
    sai; thu về 1024 hoặc 600 ⇒ đúng 1 mặt, cùng vị trí tương đối; detect 13,7 s → 2,0 s. max_side ≤ 0 ⇒ không thu."""
    h, w = image.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return image, 1.0, 1.0
    s = max_side / float(max(h, w))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    small = np.asarray(Image.fromarray(image).resize((nw, nh), Image.LANCZOS), dtype=np.uint8)
    return small, nw / float(w), nh / float(h)


class RunnerError(Exception):
    """Lỗi có mã thống nhất của contract (CONTRACTS_ADR Phần A) + mã HTTP gợi ý."""

    def __init__(self, code: str, message: str, retryable: bool = False, status: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status = status


# ---------------------------------------------------------------------------- hàm thuần (numpy)

def apply_homography(matrix: Sequence[Sequence[float]], points: Any) -> np.ndarray:
    """Áp ma trận 3×3 lên N điểm (N×2) → N×2."""
    m = np.asarray(matrix, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1) @ m.T
    return hom[:, :2] / hom[:, 2:3]


def denormalize_landmarks(lm_norm: Any, crop_size: int = CROP_SIZE) -> np.ndarray:
    """[-1, 1] → pixel của crop (đúng công thức của tensor_vis_landmarks trong EMOCA)."""
    lm = np.asarray(lm_norm, dtype=np.float64)
    return lm * (crop_size / 2.0) + crop_size / 2.0


def png_b64(arr: Any) -> Dict[str, str]:
    """Mảng uint8 H×W×3 → {"format": "png", "b64": ...}. Không ghi đĩa."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return {"format": "png", "b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def _floats(x: Any) -> List[float]:
    return [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


def _ms(a: float, b: float) -> float:
    return round((b - a) * 1000.0, 3)


def check_body(body: Dict[str, Any]) -> List[str]:
    """Kiểm nhanh hình dạng response thành công (bản rút gọn của schema; không cần jsonschema).
    Trả danh sách vi phạm (rỗng = ổn). Bắt lỗi âm thầm: sai số chiều, NaN/inf (bất biến §1.5)."""
    problems: List[str] = []

    def finite(name: str, values: Any) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            problems.append(f"{name} chứa NaN/inf")

    for i, face in enumerate(body.get("faces", [])):
        p = f"faces[{i}]"
        for key, n in CODE_DIMS.items():
            v = face.get("codes", {}).get(key)
            if v is None or len(v) != n:
                problems.append(f"{p}.codes.{key} cần {n} số, có {None if v is None else len(v)}")
            else:
                finite(f"{p}.codes.{key}", v)
        for space in ("crop224", "original_px"):
            pts = face.get("landmarks", {}).get(space)
            if pts is None or len(pts) != 68 or any(len(q) != 2 for q in pts):
                problems.append(f"{p}.landmarks.{space} cần 68 điểm (x, y)")
            else:
                finite(f"{p}.landmarks.{space}", pts)
        m = face.get("crop_transform", {}).get("matrix_3x3")
        if m is None or len(m) != 3 or any(len(r) != 3 for r in m):
            problems.append(f"{p}.crop_transform.matrix_3x3 cần 3×3")
        else:
            finite(f"{p}.crop_transform.matrix_3x3", m)
            if abs(np.linalg.det(np.asarray(m, dtype=np.float64))) < 1e-12:
                problems.append(f"{p}.crop_transform.matrix_3x3 suy biến (không khả nghịch)")
        for key in RENDER_SOURCES:
            if key not in face.get("renders", {}):
                problems.append(f"{p}.renders.{key} thiếu")
    return problems


# ---------------------------------------------------------------------------- logic runner

class EmocaRunner:
    """Điều phối một request: phát hiện mặt → crop → suy luận → đổi toạ độ → response.

    `backend` cần có: `detect(image)`, `crop(image, bbox, bbox_type)`, `infer(crop01)`, `info()`; tuỳ chọn
    `release_memory()` (server.py gọi sau lỗi hết VRAM) — xem `EmocaBackend` (thật) và
    `tests/test_emoca_svc_real.py::FakeBackend` (giả).
    Không an toàn đa luồng: người gọi (server.py) phải tuần tự hoá các lời gọi `analyze`.
    """

    def __init__(self, backend: Any, contract_version: str, min_face_px: int = 32,
                 detect_max_side_px: int = DEFAULT_DETECT_MAX_SIDE_PX) -> None:
        self.backend = backend
        self.detect_max_side_px = int(detect_max_side_px)
        self.contract_version = contract_version
        self.min_face_px = int(min_face_px)
        self.requests_served = 0
        self.last_timing_ms: Dict[str, float] = {}
        self.landmark_norm_range_seen: Optional[List[float]] = None  # [min, max] qua mọi request — phục vụ M0-05

    def analyze(self, img: Image.Image, max_faces: int = 1) -> Dict[str, Any]:
        """Trả body thành công đúng contract; raise `RunnerError` với lỗi đã biết."""
        t0 = time.perf_counter()
        image = np.array(img.convert("RGB"), dtype=np.uint8)  # bản sao ghi được
        height, width = image.shape[:2]

        # Dò mặt trên bản thu nhỏ rồi đổi khung về original_px (§1.7); crop + suy luận vẫn trên ảnh GỐC.
        det_image, sx, sy = downscale_for_detect(image, self.detect_max_side_px)
        boxes, bbox_type, _kpts = self.backend.detect(det_image)
        if (sx, sy) != (1.0, 1.0):
            boxes = [[float(b[0]) / sx, float(b[1]) / sy, float(b[2]) / sx, float(b[3]) / sy] + [float(v) for v in b[4:]]
                     for b in boxes]
        t_detect = time.perf_counter()
        if not len(boxes):
            # TestData rơi âm thầm về "chạy trên ảnh nguyên" (ImageTestDataset.py:107-113) ⇒ kết quả vô nghĩa.
            # Ở đây báo no_face rõ ràng (docs/modules/EMOCA.md §8.2).
            raise RunnerError("no_face", "FAN không phát hiện khuôn mặt nào trong ảnh.", False, 422)

        # Mặt lớn nhất trước (tất định); face_index đánh số lại theo thứ tự này.
        def area(b: Sequence[float]) -> float:
            return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))

        face_limit = max(1, min(30, int(max_faces)))
        order = sorted(range(len(boxes)), key=lambda i: -area(boxes[i]))[:face_limit]
        faces: List[Dict[str, Any]] = []
        skipped: List[str] = []
        enc_s = dec_s = 0.0
        for rank, i in enumerate(order):
            bbox = [float(v) for v in boxes[i]]
            try:
                crop01, matrix = self.backend.crop(image, bbox, bbox_type)
                res = self.backend.infer(crop01)
                enc_s += float(res["timing"]["encode"])
                dec_s += float(res["timing"]["decode"])
                faces.append(self._build_face(len(faces), bbox, crop01, matrix, res, n_detected=len(boxes),
                                              n_returned=len(order)))
            except RunnerError as e:
                if rank == 0:  # mặt lớn nhất lỗi ⇒ cả request lỗi như trước (không trả kết quả thiếu mặt chính)
                    raise
                # Mặt PHỤ lỗi (vd quá nhỏ để cắt) không làm hỏng các mặt đã xử lý được: bỏ qua, ghi cảnh báo ở mặt 0.
                # Chỉ bắt lỗi đã biết (RunnerError); lỗi bất ngờ vẫn lan ra để không che lỗi thật (CLAUDE.md §6).
                skipped.append(f"bỏ qua mặt thứ {rank + 1} theo diện tích [{e.code}]: {e.message}")
        if skipped:
            faces[0]["warnings"].extend(skipped)

        t_end = time.perf_counter()
        body: Dict[str, Any] = {
            "contract_version": self.contract_version,
            "service": "emoca_svc",
            "model": dict(self.backend.info()["model"]),
            "image": {"width": int(width), "height": int(height)},
            "faces": faces,
            "timing_ms": {
                "detect": _ms(t0, t_detect),
                "encode": round(enc_s * 1000.0, 3),
                "decode": round(dec_s * 1000.0, 3),
                "total": _ms(t0, t_end),
            },
        }
        problems = check_body(body)
        if problems:  # không gửi dữ liệu sai hình dạng đi tiếp (bất biến §1.5) — báo lỗi thay vì đoán
            raise RunnerError("contract_mismatch", "Kết quả EMOCA không đúng hình dạng contract: "
                              + "; ".join(problems[:5]), False, 500)
        self.requests_served += 1
        self.last_timing_ms = dict(body["timing_ms"])
        return body

    def _build_face(self, face_index: int, bbox: List[float], crop01: Any, matrix: Any, res: Dict[str, Any],
                    n_detected: int, n_returned: int) -> Dict[str, Any]:
        warnings: List[str] = []
        codes = {}
        for key, n in CODE_DIMS.items():
            arr = res["codes"].get(key)
            if arr is None:
                raise RunnerError("codes_missing", f"EMOCA không trả mã '{key}' (mode detail mới có detail).", False, 500)
            codes[key] = _floats(arr)

        lm_norm = np.asarray(res["landmarks_norm"], dtype=np.float64)
        if lm_norm.shape != (68, 2):
            raise RunnerError("contract_mismatch", f"predicted_landmarks có hình {lm_norm.shape}, cần (68, 2).", False, 500)
        lo, hi = float(np.nanmin(lm_norm)), float(np.nanmax(lm_norm))
        seen = self.landmark_norm_range_seen
        self.landmark_norm_range_seen = [lo, hi] if seen is None else [min(seen[0], lo), max(seen[1], hi)]
        if max(abs(lo), abs(hi)) > LANDMARK_NORM_LIMIT:
            warnings.append(f"landmark chuẩn hoá nằm ngoài miền dự kiến [-1, 1] (thấy [{lo:.2f}, {hi:.2f}]) — "
                            "hệ toạ độ chưa được xác minh (M0-05)")

        m = np.asarray(matrix, dtype=np.float64)
        crop_px = denormalize_landmarks(lm_norm)
        original_px = apply_homography(np.linalg.inv(m), crop_px)

        side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if side < self.min_face_px:
            warnings.append(f"mặt nhỏ ({side:.0f} px < {self.min_face_px} px) — kết quả kém tin cậy")
        if face_index == 0 and n_detected > n_returned:
            warnings.append(f"phát hiện {n_detected} mặt, chỉ trả {n_returned} mặt lớn nhất (max_faces)")

        crop_u8 = np.clip(np.asarray(crop01, dtype=np.float64) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return {
            "face_index": face_index,
            "bbox_original_px": [round(v, 3) for v in bbox],  # hộp quanh 68 điểm FAN (chưa nở), left, top, right, bottom
            "crop_transform": {"from": "original_px", "to": "crop224",
                               "matrix_3x3": [[float(v) for v in row] for row in m]},
            "crop_image": png_b64(crop_u8),
            "landmarks": {
                "n_points": 68,
                "crop224": [[round(float(x), 3), round(float(y), 3)] for x, y in crop_px],
                "original_px": [[round(float(x), 3), round(float(y), 3)] for x, y in original_px],
            },
            "codes": codes,
            "renders": {name: png_b64(res["renders"][name]) for name in RENDER_SOURCES},
            "warnings": warnings,
        }

    def info(self) -> Dict[str, Any]:
        out = dict(self.backend.info())
        out["requests_served"] = self.requests_served
        out["last_timing_ms"] = dict(self.last_timing_ms)
        out["landmark_norm_range_seen"] = self.landmark_norm_range_seen
        return out


# ---------------------------------------------------------------------------- backend thật (torch + EMOCA)

class EmocaBackend:
    """Gọi EMOCA thật. Chỉ chạy trong env `work38` (Python 3.8, torch 1.11, PyTorch3D, GPU CUDA).

    Mọi `import` nặng nằm trong `load()` để module này vẫn nạp được trên máy không có torch.
    """

    def __init__(self, models_dir: str, model_name: str = "EMOCA_v2_lr_mse_20", mode: str = "detail",
                 upstream_commit: Optional[str] = None) -> None:
        if mode != "detail":
            # contract bắt buộc codes.detail (128 chiều); chỉ chế độ detail có `detailcode` (DECA.py:889-891)
            raise ValueError("emoca_svc chỉ hỗ trợ mode='detail' (contract cần codes.detail).")
        self.models_dir = str(models_dir)
        self.model_name = model_name
        self.mode = mode
        self._upstream_commit = upstream_commit
        self._loaded = False
        self.load_seconds: Optional[float] = None
        self._cudnn_reason: Optional[str] = None
        self.cudnn_fallbacks = 0  # số lần tắt cuDNN lúc chạy sau lỗi cuDNN (tối đa 1: sau đó cuDNN đã tắt)
        self.warmed = False
        self.warmup_seconds: Optional[float] = None
        self.warmup_sizes: List[int] = []
        self.warmup_error: Optional[str] = None

    # -- nạp --------------------------------------------------------------
    def load(self) -> None:
        t0 = time.perf_counter()
        try:
            cudnn, cudnn_reason = resolve_cudnn(dict(os.environ))
            do_warmup, warmup_sizes = resolve_warmup(dict(os.environ))
        except ValueError as e:
            raise RunnerError("model_unavailable", str(e), False, 503)
        try:
            import torch  # noqa: WPS433 — nạp muộn có chủ đích

            if not torch.cuda.is_available():
                raise RunnerError("model_unavailable",
                                  "Không có GPU CUDA (Colab: Runtime → Change runtime type → GPU).", False, 503)
            # TRƯỚC mọi conv (EMOCA + FAN cùng tiến trình): cờ toàn cục của torch.
            if cudnn is not None:
                torch.backends.cudnn.enabled = cudnn
            self._cudnn_reason = cudnn_reason
            print(f"[emoca_svc real] cuDNN {'BẬT' if torch.backends.cudnn.enabled else 'TẮT'} — {cudnn_reason} · "
                  f"LD_LIBRARY_PATH có thư viện WSL: {wsl_ldpath_active(dict(os.environ))}", flush=True)
            from gdl.datasets.ImageDatasetHelpers import bbox2point
            from gdl.utils.FaceDetector import FAN
            from gdl.utils.lightning_logging import _fix_image
            from gdl_apps.EMOCA.utils.io import decode as emoca_decode
            from gdl_apps.EMOCA.utils.io import torch_img_to_np
            from gdl_apps.EMOCA.utils.load import load_model
            from skimage.transform import estimate_transform, warp
        except ImportError as e:
            raise RunnerError("model_unavailable",
                              f"Thiếu thư viện/mã EMOCA ({e}). Chạy trong env work38 sau khi cài EMOCA "
                              "(another/notebooks/emoca_colab.ipynb).", False, 503)

        self._torch = torch
        self._bbox2point = bbox2point
        self._fix_image = _fix_image
        self._emoca_decode = emoca_decode
        self._torch_img_to_np = torch_img_to_np
        self._estimate_transform = estimate_transform
        self._warp = warp

        try:
            emoca, _conf = load_model(self.models_dir, self.model_name, self.mode)
        except SystemExit:  # load_deca gọi sys.exit(0) khi không thấy checkpoint (load.py:50-52)
            raise RunnerError("model_unavailable",
                              f"Không thấy checkpoint của '{self.model_name}' trong '{self.models_dir}' "
                              "(đã chạy download_assets.sh chưa?).", False, 503)
        emoca.cuda()
        emoca.eval()
        self._emoca = emoca
        self._fan = FAN()  # sfd + 68 điểm, như TestData(face_detector="fan")
        torch.cuda.synchronize()
        self._vram_after_load_mb = torch.cuda.memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        self._device_name = torch.cuda.get_device_name(0)
        if self._upstream_commit is None:
            self._upstream_commit = self._git_commit()
        self.load_seconds = round(time.perf_counter() - t0, 2)
        if do_warmup:
            self.warmup(warmup_sizes)  # SAU reset_peak_memory_stats ⇒ đỉnh VRAM ở health tính cả lượt làm nóng
        self._loaded = True

    def warmup(self, sizes: Sequence[int]) -> None:
        """Chạy trước FAN (dò mặt + 68 điểm trên một khung giả) ở từng cạnh `sizes`, rồi encode/decode một crop 224, trên
        ảnh TỔNG HỢP trong bộ nhớ — để CUDA/cuDNN khởi tạo trước request đầu tiên. Không ghi đĩa, không đếm vào
        requests_served. Lỗi KHÔNG làm hỏng việc khởi động nhưng KHÔNG bị nuốt: ghi `warmup_error` (health) + log."""
        t0 = time.perf_counter()
        self.warmup_sizes = [int(x) for x in sizes]
        self.warmed, self.warmup_error = False, None
        try:
            for side in self.warmup_sizes:
                img = synthetic_image(side)
                self.detect(img)  # bộ dò mặt; ảnh không có mặt ⇒ danh sách rỗng là bình thường
                box = [0.25 * side, 0.2 * side, 0.75 * side, 0.8 * side]  # khung GIẢ ⇒ chạy mạng 68 điểm
                self._cudnn_retry("warmup FAN 68 điểm", self._fan.run, img, True, [box])
            self.infer(synthetic_image(CROP_SIZE).astype(np.float32) / 255.0)
            self._torch.cuda.synchronize()
            self.warmed = True
        except Exception as e:  # noqa: BLE001 — báo qua health/log, server vẫn phục vụ (request thật sẽ báo lỗi riêng)
            self.warmup_error = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200] if str(e).strip() else ''}"
            import gc

            gc.collect()
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        self.warmup_seconds = round(time.perf_counter() - t0, 2)
        print(f"[emoca_svc real] làm nóng {'XONG' if self.warmed else 'LỖI'} sau {self.warmup_seconds} s "
              f"(cạnh {self.warmup_sizes}){' — ' + self.warmup_error if self.warmup_error else ''}", flush=True)

    @staticmethod
    def _git_commit() -> str:
        """Commit EMOCA đang chạy (+ `+patches` nếu mã Python của EMOCA đã bị vá) để tái lập (ADR-015, PROJECT_CONTEXT §3).
        Chỉ xét `gdl/` và `gdl_apps/`: notebook sửa `requirements38.txt` (tệp được git theo dõi) nên xét cả cây sẽ luôn
        ra `+patches` kể cả khi chưa áp bản vá nào."""
        try:
            import gdl

            root = Path(gdl.__file__).resolve().parents[1]
            head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no",
                                    "--", "gdl", "gdl_apps"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return (head or "unknown") + ("+patches" if dirty else "")
        except Exception:  # noqa: BLE001 — chỉ là nhãn tái lập, không được làm hỏng việc khởi động
            return "unknown"

    # -- giao diện cho EmocaRunner -----------------------------------------
    def _cudnn_retry(self, step: str, fn: Any, *args: Any) -> Any:
        """Gọi `fn(*args)`; nếu ném RuntimeError có chữ "cudnn" khi cuDNN đang BẬT ⇒ tắt cuDNN cho cả tiến trình và thử
        LẠI đúng 1 lần. Lý do: trên WSL2 (đo 2026-09-25, đã có LD_LIBRARY_PATH đúng) cuDNN thỉnh thoảng ném
        "Unable to find a valid cuDNN algorithm to run convolution" dù phần lớn các lần chạy được — nguyên nhân chưa rõ.
        Không lặp vô hạn: sau lần tắt, cuDNN đã tắt nên lỗi sau đó được ném thẳng. Lỗi khác: không đổi gì."""
        try:
            return fn(*args)
        except RuntimeError as e:
            torch = self._torch
            if "cudnn" not in str(e).lower() or not torch.backends.cudnn.enabled:
                raise
            first = (str(e).strip().splitlines() or [""])[0][:160]
            torch.backends.cudnn.enabled = False
            self.cudnn_fallbacks += 1
            self._cudnn_reason = f"{self._cudnn_reason}; TẮT lúc chạy sau lỗi cuDNN ở {step}: {first}"
            print(f"[emoca_svc real] lỗi cuDNN ở {step} ({first}) ⇒ tắt cuDNN cho cả tiến trình, thử lại 1 lần",
                  flush=True)
            return fn(*args)

    def detect(self, image: np.ndarray) -> Tuple[List[List[float]], str, Any]:
        """FAN trên ảnh RGB uint8 → (danh sách [left, top, right, bottom], 'kpt68', các bộ 68 điểm)."""
        return self._cudnn_retry("FAN", self._detect_impl, image)

    def _detect_impl(self, image: np.ndarray) -> Tuple[List[List[float]], str, Any]:
        boxes, bbox_type, kpts = self._fan.run(image, with_landmarks=True)
        return [[float(v) for v in b] for b in boxes], bbox_type, kpts

    def crop(self, image: np.ndarray, bbox: Sequence[float], bbox_type: str) -> Tuple[np.ndarray, List[List[float]]]:
        """Y hệt nhánh một mặt của `TestData.__getitem__` (ImageTestDataset.py:96-99, 132-160):
        trả (crop float [0,1] 224×224×3, ma trận 3×3 original_px → crop224)."""
        left, top, right, bottom = bbox
        old_size, center = self._bbox2point(left, right, top, bottom, type=bbox_type)
        size = int(old_size * CROP_SCALE)
        if size < 16:
            raise RunnerError("no_face", f"Mặt quá nhỏ để cắt (cạnh {size} px).", False, 422)
        src = np.array([[center[0] - size / 2, center[1] - size / 2],
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
        dst = np.array([[0, 0], [0, CROP_SIZE - 1], [CROP_SIZE - 1, 0]])
        tform = self._estimate_transform("similarity", src, dst)
        crop = self._warp(image / 255.0, tform.inverse, output_shape=(CROP_SIZE, CROP_SIZE))
        return crop, [[float(v) for v in row] for row in np.asarray(tform.params)]

    def infer(self, crop01: np.ndarray) -> Dict[str, Any]:
        """encode → decode như `gdl_apps/EMOCA/utils/io.py::test`, có đo thời gian từng bước (đồng bộ GPU)."""
        return self._cudnn_retry("encode/decode", self._infer_impl, crop01)

    def _infer_impl(self, crop01: np.ndarray) -> Dict[str, Any]:
        torch = self._torch
        x = torch.tensor(np.ascontiguousarray(crop01.transpose(2, 0, 1))).float().unsqueeze(0).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            vals = self._emoca.encode({"image": x}, training=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        vals, vis = self._emoca_decode(self._emoca, vals, training=False)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        def to_np(t: Any) -> np.ndarray:
            return t.detach().cpu().numpy()

        codes = {}
        for key, src in CODE_KEYS.items():
            if src in vals:
                codes[key] = to_np(vals[src])[0]
        renders = {name: self._fix_image(self._torch_img_to_np(vis[src][0])) for name, src in RENDER_SOURCES.items()}
        return {
            "codes": codes,
            "landmarks_norm": to_np(vals["predicted_landmarks"])[0],
            "renders": renders,
            "timing": {"encode": t1 - t0, "decode": t2 - t1},
        }

    def release_memory(self) -> None:
        """Gọi sau lỗi hết VRAM (server.py): thu gom đối tượng Python rồi trả bộ nhớ cache của PyTorch cho CUDA,
        để các request sau không tiếp tục OOM vì bộ nhớ đệm bị giữ/phân mảnh. Không làm gì khi chưa nạp."""
        import gc

        gc.collect()
        if self._loaded:
            self._torch.cuda.empty_cache()

    def info(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"model": {"name": self.model_name, "mode": self.mode,
                                         "upstream_commit": self._upstream_commit or "unknown"},
                               "loaded": self._loaded, "load_seconds": self.load_seconds,
                               "ld_library_path_wsl": wsl_ldpath_active(dict(os.environ))}
        if self._loaded:
            torch = self._torch
            out.update({
                "device": self._device_name,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "vram_after_load_mb": round(self._vram_after_load_mb, 1),
                "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                # trạng thái THẬT của cờ torch (không phải giá trị biến môi trường) + lý do (EMOCA_CUDNN / probe)
                "cudnn_enabled": bool(torch.backends.cudnn.enabled),
                "cudnn_reason": self._cudnn_reason,
                "cudnn_fallbacks": self.cudnn_fallbacks,
                # làm nóng sau khi nạp (EMOCA_WARMUP): trường THÊM, tuỳ chọn — health không có schema trong contracts/
                "warmed": self.warmed,
                "warmup_seconds": self.warmup_seconds,
                "warmup_sizes": list(self.warmup_sizes),
            })
            if self.warmup_error:
                out["warmup_error"] = self.warmup_error
            out.update(gpu_memory_fields(torch))
        return out


def gpu_memory_fields(torch: Any) -> Dict[str, float]:
    """Số đo VRAM bổ sung cho /v1/health (MiB, 1 chữ số). `max_memory_allocated` bỏ sót CUDA context, cuDNN và phần
    PyTorch giữ trong cache; `nvidia-smi` dưới WDDM (WSL2) báo memory.used = 0. Nên thêm:
    - vram_reserved_mb / vram_reserved_peak_mb: memory_reserved / max_memory_reserved (bộ nhớ cache của PyTorch);
    - gpu_free_mb / gpu_total_mb: cudaMemGetInfo — trống/tổng của cả GPU, gồm mọi thứ ngoài PyTorch.
    Mỗi nhóm bọc try riêng: lỗi thì bỏ trường, không làm sập health."""
    out: Dict[str, float] = {}
    try:
        out["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
        out["vram_reserved_peak_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    except Exception:  # noqa: BLE001 — số đo phụ, không được làm hỏng health
        out.pop("vram_reserved_mb", None)
    getters = [lambda: torch.cuda.cudart().cudaMemGetInfo(0)]
    if hasattr(torch.cuda, "mem_get_info"):
        getters.insert(0, torch.cuda.mem_get_info)
    for get in getters:
        try:
            info = tuple(get())
            # tuỳ bản binding: (free, total) hoặc (mã lỗi, free, total)
            free_b, total_b = (int(info[1]), int(info[2])) if len(info) == 3 else (int(info[0]), int(info[1]))
            if 0 <= free_b <= total_b and total_b > 0:
                out["gpu_free_mb"] = round(free_b / 2**20, 1)
                out["gpu_total_mb"] = round(total_b / 2**20, 1)
                break
        except Exception:  # noqa: BLE001 — thử cách kế tiếp; hết cách thì bỏ trường
            continue
    return out
