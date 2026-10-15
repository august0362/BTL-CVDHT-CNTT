"""Kiểm web/face_align.py (ADR-020): chọn 5 điểm, Umeyama, warp đúng quy ước tâm pixel, mẫu trong configs/align.yaml,
ca suy biến trả (None, lý do) không raise. Chỉ cần numpy + Pillow + PyYAML (không torch).

Chạy:  python tests/test_face_align.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))
import face_align  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def sim(scale: float, angle_deg: float, tx: float, ty: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    m = np.eye(3)
    m[:2, :2] = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    m[:2, 2] = [tx, ty]
    return m


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (m[:2, :2] @ np.asarray(pts, float).T).T + m[:2, 2]


def landmarks_from_five(five: np.ndarray) -> np.ndarray:
    """68 điểm giả sao cho five_points() trả đúng `five`: mắt = 6 điểm quanh tâm (TB = tâm), mũi 30, miệng 48/54."""
    lm = np.zeros((68, 2))
    ring = np.array([[-3, 0], [-1.5, -1], [1.5, -1], [3, 0], [1.5, 1], [-1.5, 1]])
    lm[36:42] = five[0] + ring
    lm[42:48] = five[1] + ring
    lm[30], lm[48], lm[54] = five[2], five[3], five[4]
    return lm


def dot_image(w: int, h: int, pts: np.ndarray, colors: list[tuple[int, int, int]], sigma: float = 2.5) -> Image.Image:
    """Ảnh nền đen với các chấm Gauss màu khác nhau ở toạ độ thực (quy ước tâm pixel)."""
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3))
    for (x, y), c in zip(pts, colors):
        g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
        img += g[..., None] * np.array(c, float)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def centroid(img: Image.Image, channel: int, near: np.ndarray, r: int = 7) -> np.ndarray:
    a = np.asarray(img, float)[..., channel]
    h, w = a.shape
    x0, y0 = int(round(near[0])), int(round(near[1]))
    ys, xs = np.mgrid[max(0, y0 - r):min(h, y0 + r + 1), max(0, x0 - r):min(w, x0 + r + 1)]
    wgt = a[ys, xs]
    return np.array([(wgt * xs).sum() / wgt.sum(), (wgt * ys).sum() / wgt.sum()])


def test_template_config():
    print("== mẫu trong configs/align.yaml")
    t, err = face_align.load_template()
    check(err is None and t["size"] == (100, 100) and t["points"].shape == (5, 2), f"đọc được rafdb_aligned_100 ({err})")
    if t:
        p = t["points"]
        check(p[0, 0] < p[2, 0] < p[1, 0] and p[3, 0] < p[4, 0] and p[0, 1] < p[2, 1] < p[3, 1],
              "thứ tự 5 điểm hợp lý: mắt trái < mũi < mắt phải (x), mắt < mũi < miệng (y)")
        check(t["status"] in ("provisional", "measured"), f"status = {t['status']}")
        check(t["enabled"] is True, "enabled: true (mẫu đã đo từ RAF-DB aligned, ADR-020)")
    check(face_align.load_template("khong_co")[0] is None, "mẫu không có ⇒ (None, lý do)")


def test_umeyama_recovers():
    print("== Umeyama khôi phục phép đồng dạng đã biết")
    rng = np.random.RandomState(1)
    src = rng.uniform(0, 300, (5, 2))
    for s, a, tx, ty in ((1.0, 0, 0, 0), (0.37, 12.5, -40, 17), (2.2, -33, 5, 90), (0.1, 179, 3, 3)):
        m_true = sim(s, a, tx, ty)
        m, err = face_align.umeyama_similarity(src, apply(m_true, src))
        check(err is None and np.allclose(m, m_true, atol=1e-9), f"s={s}, góc={a}°, t=({tx},{ty}) ⇒ ma trận khớp")
    m_true = sim(0.8, 20, 10, -5)
    noisy = apply(m_true, src) + rng.normal(0, 0.5, src.shape)
    m, _ = face_align.umeyama_similarity(src, noisy)
    dev = float(np.linalg.norm(apply(m, src) - apply(m_true, src), axis=1).max())
    check(dev < 1.0, f"điểm đích nhiễu σ=0,5 px ⇒ ảnh của 5 điểm qua ma trận ước lệch ≤ 1 px so với ma trận thật ({dev:.3f})")
    mirrored = src * [-1, 1]
    m, err = face_align.umeyama_similarity(src, mirrored)
    check(err is None and np.linalg.det(m[:2, :2]) > 0, "đích là ảnh gương ⇒ vẫn không phản xạ (det > 0)")


def test_align_synthetic():
    print("== align_face trên ảnh tổng hợp: điểm rơi đúng mẫu ≤ 0,5 px")
    t, _ = face_align.load_template()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0)]
    chan = [0, 1, 2, 0, 1]
    for s, a, tx, ty in ((3.1, 0, 20, 30), (2.4, 17, 60, 10), (4.0, -25, 150, 80)):
        m_orig_to_aligned = sim(1 / s, -a, 0, 0)
        # dựng ảnh gốc: vị trí 5 điểm = nghịch đảo của mẫu qua phép đồng dạng đã biết
        m_aligned_to_orig = sim(s, a, tx, ty)
        five = apply(m_aligned_to_orig, t["points"])
        w, h = int(five[:, 0].max() + 60), int(five[:, 1].max() + 60)
        img = dot_image(w, h, five, colors, sigma=2.5 * s)
        res, err = face_align.align_face(img, landmarks_from_five(five))
        check(err is None and res["size"] == [100, 100] and res["image"].size == (100, 100),
              f"s={s}, góc={a}°: ảnh ra 100×100 ({err})")
        if err:
            continue
        m = np.array(res["matrix_3x3"])
        check(np.allclose(m, np.linalg.inv(m_aligned_to_orig), atol=1e-6) and res["residual_px"] < 1e-6,
              f"s={s}, góc={a}°: matrix_3x3 = nghịch đảo phép đã dựng, residual {res['residual_px']}")
        errs = [float(np.linalg.norm(centroid(res["image"], c, p) - p)) for c, p in zip(chan, t["points"])]
        check(max(errs) <= 0.5, f"s={s}, góc={a}°: tâm chấm trên ảnh căn chỉnh lệch mẫu tối đa {max(errs):.3f} px (≤ 0,5)")
        del m_orig_to_aligned


def test_degenerate():
    print("== ca suy biến ⇒ (None, lý do), không raise")
    img = Image.new("RGB", (64, 64))
    t, _ = face_align.load_template()
    good = landmarks_from_five(np.array([[20, 25], [44, 25], [32, 36], [24, 48], [40, 48]], float))
    cases = {
        "không phải ảnh PIL": (np.zeros((64, 64, 3)), good),
        "thiếu điểm (67)": (img, good[:67]),
        "sai hình (68×1)": (img, np.zeros((68, 1))),
        "có NaN": (img, np.where(np.arange(68)[:, None] == 30, np.nan, good)),
        "mọi điểm trùng nhau": (img, np.full((68, 2), 10.0)),
        "hai mắt trùng nhau": (img, landmarks_from_five(np.array([[30, 25], [30, 25], [32, 36], [24, 48], [40, 48]], float))),
        "landmark không phải số": (img, "abc"),
    }
    for label, (im, lm) in cases.items():
        try:
            res, err = face_align.align_face(im, lm, t)
            check(res is None and isinstance(err, str) and err, f"{label} ⇒ None, '{err}'")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{label} ⇒ raise {type(exc).__name__}: {exc}")
    res, err = face_align.align_face(img, good, "khong_co")
    check(res is None and "khong_co" in err, "tên mẫu không có ⇒ None + lý do")
    res, err = face_align.align_face(img, good)
    check(err is None and res["image"].size == (100, 100), "ảnh nhỏ hơn mẫu vẫn warp được (phần ngoài ảnh = đen)")


def test_estimate_template():
    print("== estimate_template: TB + đối xứng qua trục giữa")
    t, _ = face_align.load_template()
    rng = np.random.RandomState(3)
    sets = [t["points"] + rng.normal(0, 0.8, (5, 2)) for _ in range(200)]
    est, err = face_align.estimate_template(sets, (100, 100))
    check(err is None and np.abs(est - t["points"]).max() < 0.8, f"200 bộ nhiễu quanh mẫu ⇒ ước lại lệch ≤ 0,8 px ({np.abs(est - t['points']).max():.2f})")
    check(abs(est[0, 1] - est[1, 1]) < 1e-9 and abs((est[0, 0] + est[1, 0]) / 2 - 49.5) < 1e-9 and est[2, 0] == 49.5,
          "kết quả đối xứng: hai mắt cùng y, tâm mắt và mũi trên trục x = 49,5")
    check(face_align.estimate_template([np.zeros((4, 2))], (100, 100))[0] is None, "không có bộ hợp lệ ⇒ None")


def main() -> int:
    test_template_config()
    test_umeyama_recovers()
    test_align_synthetic()
    test_degenerate()
    test_estimate_template()
    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
