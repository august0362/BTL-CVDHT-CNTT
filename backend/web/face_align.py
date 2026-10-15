"""AI — căn chỉnh ảnh vào POSTER/MA3D-Net kiểu RAF-DB aligned từ 68 landmark (ADR-020).

Không import Streamlit, không import torch (chỉ numpy + Pillow + PyYAML). Mọi hàm công khai KHÔNG raise với dữ liệu
đầu vào xấu: trả `(None, lý_do)`.

Luồng: 68 landmark `original_px` (EMOCA) → 5 điểm (tâm hai mắt = TB 36–41 / 42–47, đỉnh mũi 30, khoé miệng 48, 54)
→ phép đồng dạng (Umeyama: xoay + tỉ lệ đều + tịnh tiến, không phản xạ) sang mẫu 5 điểm trong `configs/align.yaml`
→ lấy mẫu song tuyến tính ảnh gốc ra ảnh kích thước mẫu (vd 100×100).

Hệ toạ độ (bất biến §1.7): (0,0) là TÂM pixel trên-trái ở cả ảnh gốc (`original_px`, quy ước skimage/FAN) và ảnh căn
chỉnh (`aligned`). `matrix_3x3` đổi `original_px` → `aligned`: [x', y', 1]ᵀ = M · [x, y, 1]ᵀ.
Ảnh truyền vào phải là CHÍNH ảnh đã gửi cho EMOCA (đã xoay theo EXIF) — landmark tính trên ảnh đó.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_ALIGN_CONFIG = _ROOT / "configs" / "align.yaml"
DEFAULT_TEMPLATE = "rafdb_aligned_100"
# Chỉ số 68 điểm (0-based) của từng điểm trong 5 điểm; thứ tự PHẢI trùng `points` trong configs/align.yaml.
FIVE_POINT_INDICES: tuple[tuple[int, ...], ...] = (tuple(range(36, 42)), tuple(range(42, 48)), (30,), (48,), (54,))
FIVE_POINT_NAMES = ("mat_trai_anh", "mat_phai_anh", "dinh_mui", "mieng_trai_anh", "mieng_phai_anh")
_MIN_EYE_DIST_PX = 2.0  # hai tâm mắt gần hơn mức này ⇒ suy biến (không ước được tỉ lệ/góc tin cậy)


def load_template(name: str = DEFAULT_TEMPLATE, path: str | Path | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Đọc mẫu `name` từ configs/align.yaml. Trả ({"name","size":(w,h),"points":ndarray 5×2,"status","source",
    "enabled"}, None). `enabled` = cờ toàn cục của tệp (đã bật 2026-09-29): người gọi (orchestrator) quyết có căn chỉnh
    hay không; `align_face` tự nó KHÔNG xét cờ này."""
    try:
        import yaml
        cfg = yaml.safe_load(Path(path or _ALIGN_CONFIG).read_text(encoding="utf-8")) or {}
        t = (cfg.get("templates") or {}).get(name)
        if not t:
            return None, f"không có mẫu '{name}' trong {Path(path or _ALIGN_CONFIG).name}"
        w, h = (int(v) for v in t["size"])
        pts = np.asarray(t["points"], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 — lỗi config trả về dạng dữ liệu, không làm hỏng trang
        return None, f"không đọc được mẫu căn chỉnh: {type(exc).__name__}: {exc}"
    if pts.shape != (5, 2) or not np.isfinite(pts).all() or w <= 0 or h <= 0:
        return None, f"mẫu '{name}' sai hình dạng (cần 5 điểm [x, y] và size dương)"
    return {"name": name, "size": (w, h), "points": pts, "status": t.get("status"), "source": t.get("source"),
            "enabled": bool(cfg.get("enabled", False))}, None


def five_points(landmarks68: Any) -> tuple[np.ndarray | None, str | None]:
    """68×2 → 5×2 theo FIVE_POINT_INDICES. Sai hình dạng / không hữu hạn ⇒ (None, lý do)."""
    try:
        lm = np.asarray(landmarks68, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "landmark không phải mảng số"
    if lm.ndim != 2 or lm.shape[0] != 68 or lm.shape[1] < 2:
        return None, f"cần 68 landmark [x, y], nhận hình {lm.shape}"
    lm = lm[:, :2]
    pts = np.stack([lm[list(idx)].mean(axis=0) for idx in FIVE_POINT_INDICES])
    if not np.isfinite(pts).all():
        return None, "landmark có NaN/Inf"
    return pts, None


def umeyama_similarity(src: Any, dst: Any) -> tuple[np.ndarray | None, str | None]:
    """Phép đồng dạng (không phản xạ) bình phương tối thiểu đưa src → dst (Umeyama 1991). Trả ma trận 3×3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 2:
        return None, f"cần hai tập điểm N×2 cùng hình (N ≥ 2), nhận {src.shape} và {dst.shape}"
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        return None, "điểm có NaN/Inf"
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(src)
    if var_s < 1e-9:
        return None, "các điểm nguồn trùng nhau (suy biến)"
    cov = d0.T @ s0 / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0  # chặn phản xạ
    r = u @ np.diag(d) @ vt
    scale = float((sig * d).sum() / var_s)
    if not np.isfinite(scale) or scale <= 0:
        return None, "tỉ lệ ước được không hợp lệ (suy biến)"
    m = np.eye(3)
    m[:2, :2] = scale * r
    m[:2, 2] = mu_d - scale * r @ mu_s
    return m, None


def warp_bilinear(image: Image.Image, matrix: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Ảnh ra kích thước `size` (w, h): pixel ra (x', y') lấy mẫu song tuyến tính ở M⁻¹·(x', y') trên ảnh gốc
    (quy ước tâm pixel). Ngoài ảnh ⇒ đen (0), như warpAffine mặc định."""
    src = np.asarray(image.convert("RGB"), dtype=np.float32)
    h_in, w_in = src.shape[:2]
    w, h = size
    inv = np.linalg.inv(matrix)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    sx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    sy = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    padded = np.zeros((h_in + 2, w_in + 2, 3), dtype=np.float32)
    padded[1:-1, 1:-1] = src

    def at(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        return padded[np.clip(yy + 1, 0, h_in + 1), np.clip(xx + 1, 0, w_in + 1)]

    out = ((1 - fx) * (1 - fy) * at(y0, x0) + fx * (1 - fy) * at(y0, x0 + 1)
           + (1 - fx) * fy * at(y0 + 1, x0) + fx * fy * at(y0 + 1, x0 + 1))
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8), "RGB")


def align_face(image: Image.Image, landmarks68_original_px: Any, template: dict[str, Any] | str | None = None
               ) -> tuple[dict[str, Any] | None, str | None]:
    """Căn chỉnh một mặt về mẫu 5 điểm (ADR-020). Không raise.

    image: PIL ảnh gốc ĐÃ xoay EXIF (chính ảnh đã gửi EMOCA). landmarks68_original_px: 68×2 theo `original_px`.
    template: dict từ `load_template`, tên mẫu trong configs/align.yaml, hoặc None (= DEFAULT_TEMPLATE).
    Trả ({"image": PIL RGB kích thước mẫu, "matrix_3x3": list 3×3 (original_px → aligned), "size": [w, h],
           "template": tên, "template_status": "provisional"|"measured", "residual_px": sai số lớn nhất của 5 điểm
           sau biến đổi so với mẫu}, None) hoặc (None, lý do).
    """
    if not isinstance(image, Image.Image):
        return None, f"cần PIL.Image.Image, nhận {type(image).__name__}"
    if template is None or isinstance(template, str):
        template, err = load_template(template or DEFAULT_TEMPLATE)
        if err:
            return None, err
    pts, err = five_points(landmarks68_original_px)
    if err:
        return None, err
    if float(np.linalg.norm(pts[0] - pts[1])) < _MIN_EYE_DIST_PX:
        return None, "hai tâm mắt gần như trùng nhau (suy biến)"
    m, err = umeyama_similarity(pts, template["points"])
    if err:
        return None, err
    mapped = (m[:2, :2] @ pts.T).T + m[:2, 2]
    residual = float(np.linalg.norm(mapped - template["points"], axis=1).max())
    w, h = template["size"]
    try:
        out = warp_bilinear(image, m, (w, h))
    except (np.linalg.LinAlgError, ValueError, MemoryError) as exc:
        return None, f"không warp được ảnh: {type(exc).__name__}: {exc}"
    return {"image": out, "matrix_3x3": [[float(v) for v in row] for row in m], "size": [w, h],
            "template": template["name"], "template_status": template.get("status"),
            "residual_px": round(residual, 3)}, None


def estimate_template(point_sets: Sequence[Any], size: tuple[int, int]) -> tuple[np.ndarray | None, str | None]:
    """Ước mẫu 5 điểm từ nhiều bộ 5 điểm đo trên ảnh ĐÃ căn chỉnh (vd FAN trên ảnh RAF-DB aligned): trung bình từng
    điểm (các ảnh đã cùng hệ nên không cần Procrustes), rồi đối xứng hoá qua trục dọc giữa ảnh. Dùng để thay mẫu
    `provisional` trong configs/align.yaml. Trả (5×2, None) hoặc (None, lý do)."""
    arr = [np.asarray(p, dtype=np.float64) for p in point_sets]
    arr = [p for p in arr if p.shape == (5, 2) and np.isfinite(p).all()]
    if len(arr) < 1:
        return None, "không có bộ 5 điểm hợp lệ"
    mean = np.mean(arr, axis=0)
    cx = (size[0] - 1) / 2.0
    sym = mean.copy()
    for a, b in ((0, 1), (3, 4)):
        y = (mean[a, 1] + mean[b, 1]) / 2
        dx = ((cx - mean[a, 0]) + (mean[b, 0] - cx)) / 2
        sym[a] = [cx - dx, y]
        sym[b] = [cx + dx, y]
    sym[2, 0] = cx
    return sym, None
