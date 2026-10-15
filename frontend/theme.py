"""FRONTEND — hệ thiết kế màu cho frontend/app.py (ADR-012).

Đọc bảng màu từ ảnh trong `frontend/assets/theme/` (mỗi ảnh = 4 màu phẳng), dẫn ra bộ màu SÁNG và TỐI,
và tự chỉnh để đạt độ tương phản WCAG. Không import Streamlit, không import torch, không gọi
mạng (không CDN, không font ngoài) — chạy và kiểm thử được độc lập.

Nguyên tắc thiết kế
- 60-30-10: ~60% nền trung tính nhuốm sắc của bảng màu, ~30% bề mặt (thẻ), ~10% màu nhấn.
- Sắc thái (tâm lý học màu): lạnh = tin cậy/bình tĩnh, ấm = thân thiện/năng lượng, xanh lá = cân bằng.
  Màu NGỮ NGHĨA (thành công/cảnh báo/lỗi) và màu CẢM XÚC giữ sắc cố định qua mọi chủ đề để ý nghĩa
  không đổi; chỉ độ sáng được chỉnh cho đủ tương phản.
- Độc lập trình duyệt: không dùng `prefers-color-scheme`; luôn đặt `color-scheme` tường minh
  (`only light` chặn chế độ tối tự động của trình duyệt như Cốc Cốc/Chrome làm đảo màu).
- Không phụ thuộc riêng màu để truyền thông tin (nhãn chữ + % luôn đi kèm).
"""
from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from string import Template

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Ngưỡng tương phản (WCAG 2.x). "normal" ≈ AAA cho chữ thường; "high" cho người cần tương phản mạnh.
RATIOS = {
    "normal": {"body": 10.0, "muted": 4.6, "ink": 4.6, "ui": 3.0, "on": 4.6, "border": 3.0},
    "high": {"body": 12.0, "muted": 7.0, "ink": 7.0, "ui": 4.5, "on": 7.0, "border": 4.5},
}

# Độ chói tương đối (WCAG) đặt cho nền/bề mặt — theo độ chói chứ không theo độ sáng HLS, để tương phản
# với chữ giống nhau bất kể sắc độ (xanh lá "sáng" hơn xanh dương ở cùng L). (bg, surface, surface2, border)
LUMINANCE_TARGETS = {
    ("light", "normal"): (0.88, 0.97, 0.78, 0.62),
    ("light", "high"): (0.92, 1.00, 0.84, 0.50),
    ("dark", "normal"): (0.007, 0.014, 0.026, 0.060),
    ("dark", "high"): (0.003, 0.008, 0.020, 0.070),
}

# Màu ngữ nghĩa cố định (sắc độ, HLS). Màu cảm xúc: (sắc độ, emoji). Bảng này là token thiết kế
# mặc định; khi 02_CONFIG_METRICS có `configs/labels.yaml` (trường `color`) thì ưu tiên giá trị đó.
SEMANTIC_HUES = {"success": 145, "warning": 38, "danger": 4, "info": 205}
EMOTION_STYLE = {
    "happiness": (46, "😄"), "happy": (46, "😄"), "joy": (46, "😄"),
    "surprise": (188, "😲"), "surprised": (188, "😲"),
    "fear": (268, "😨"), "fearful": (268, "😨"),
    "sadness": (216, "😢"), "sad": (216, "😢"),
    "disgust": (100, "🤢"), "disgusted": (100, "🤢"),
    "anger": (2, "😠"), "angry": (2, "😠"),
    "contempt": (318, "😒"),
    "neutral": (None, "😐"),
}


# --------------------------------------------------------------------------------- màu cơ bản
def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c))) for c in rgb)


def luminance(h: str) -> float:
    def lin(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in hex_to_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def hls(h: str) -> tuple[float, float, float]:
    r, g, b = (c / 255 for c in hex_to_rgb(h))
    return colorsys.rgb_to_hls(r, g, b)  # (h, l, s) trong 0..1


def from_hls(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return rgb_to_hex((r * 255, g * 255, b * 255))


def chroma(h: str) -> float:
    r, g, b = hex_to_rgb(h)
    return (max(r, g, b) - min(r, g, b)) / 255


def mix(a: str, b: str, t: float) -> str:
    """Trộn a→b theo t∈[0,1] trong không gian sRGB."""
    ra, rb = hex_to_rgb(a), hex_to_rgb(b)
    return rgb_to_hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))


def at_luminance(hue: float, sat: float, target: float) -> str:
    """Màu cùng sắc độ/độ bão hoà nhưng có ĐỘ CHÓI tương đối đúng bằng target (tìm nhị phân theo L)."""
    lo, hi = 0.0, 1.0
    for _ in range(28):
        mid = (lo + hi) / 2
        if luminance(from_hls(hue, mid, sat)) < target:
            lo = mid
        else:
            hi = mid
    return from_hls(hue, (lo + hi) / 2, sat)


def _worst(fg: str, bgs: list[str]) -> float:
    return min(contrast(fg, b) for b in bgs)


def ensure_contrast(fg: str, bgs, ratio: float) -> str:
    """Chỉnh ĐỘ SÁNG của fg (giữ sắc độ) cho tới khi tương phản ≥ ratio với MỌI nền trong bgs.

    Thử hướng hợp lý trước (nền sáng → tối đi; nền tối → sáng lên), rồi hướng ngược lại;
    không đạt thì trả bản có tương phản cao nhất.
    """
    bgs = [bgs] if isinstance(bgs, str) else list(bgs)
    if _worst(fg, bgs) >= ratio:
        return fg
    h, l, s = hls(fg)
    mean_lum = sum(luminance(b) for b in bgs) / len(bgs)
    first = -1 if mean_lum > 0.4 else 1
    best, best_val = fg, _worst(fg, bgs)
    for direction in (first, -first):
        for i in range(1, 101):
            cand = from_hls(h, l + direction * i / 100, s)
            val = _worst(cand, bgs)
            if val >= ratio:
                return cand
            if val > best_val:
                best, best_val = cand, val
    return best


# --------------------------------------------------------------------------------- bảng màu
@dataclass(frozen=True)
class Palette:
    id: str
    name: str
    colors: tuple[str, ...]  # 4 màu theo thứ tự diện tích trong ảnh
    source: str
    family: str
    mood: str


def _hue_family(h: str) -> tuple[str, str, str]:
    """(họ màu tiếng Việt, nhiệt độ, sắc thái tâm lý) từ màu chủ đạo."""
    hue, l, s = hls(h)
    deg = hue * 360
    if chroma(h) < 0.07 or s < 0.16:
        return "Xám", "trung tính", "Trung tính · chuyên nghiệp"
    if 12 <= deg < 45 and l < 0.42 and s > 0.15:
        return "Nâu", "ấm", "Vững chãi · ấm cúng"
    table = [
        (15, "Đỏ", "ấm", "Nhiệt huyết · thu hút chú ý"), (45, "Cam", "ấm", "Ấm áp · thân thiện"),
        (68, "Vàng", "ấm", "Lạc quan · tươi vui"), (165, "Xanh lá", "mát", "Cân bằng · tự nhiên"),
        (200, "Xanh ngọc", "mát", "Tươi mát · minh bạch"), (255, "Xanh dương", "lạnh", "Tin cậy · bình tĩnh"),
        (292, "Tím", "lạnh", "Sáng tạo · tinh tế"), (346, "Hồng", "ấm", "Dịu dàng · gần gũi"),
    ]
    for limit, fam, temp, mood in table:
        if deg < limit:
            return fam, temp, mood
    return "Đỏ", "ấm", "Nhiệt huyết · thu hút chú ý"


def _read_colors(path: Path) -> tuple[str, ...]:
    """4 màu chiếm nhiều diện tích nhất trong ảnh (bỏ điểm trong suốt). Ném ValueError nếu không đủ."""
    with Image.open(path) as im:
        im = im.convert("RGBA")
        im.thumbnail((160, 160), Image.NEAREST)  # NEAREST: giữ đúng màu phẳng, không tạo màu trung gian
        px = [p for p in im.getdata() if p[3] >= 128]
    if not px:
        raise ValueError("ảnh trống/trong suốt")
    counts: dict[tuple[int, int, int], int] = {}
    for r, g, b, _ in px:
        counts[(r, g, b)] = counts.get((r, g, b), 0) + 1
    total = len(px)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    picked = [rgb_to_hex(rgb) for rgb, n in ranked if n / total >= 0.02][:4]
    if len(picked) < 2:
        raise ValueError(f"chỉ có {len(picked)} màu đáng kể (cần ≥ 2)")
    return tuple(picked)


def _analyze(colors: tuple[str, ...]) -> dict:
    """Chọn vai trò: brand (nổi + tông vừa), accent (khác sắc nhất), lightest, darkest."""
    def vivid(c: str) -> float:
        return chroma(c) * (1.0 - abs(hls(c)[1] - 0.5))
    brand = max(colors, key=vivid) if max(chroma(c) for c in colors) >= 0.08 else min(colors, key=luminance)
    bh = hls(brand)[0]

    def hue_gap(c: str) -> float:
        d = abs(hls(c)[0] - bh)
        return min(d, 1 - d)
    others = [c for c in colors if c != brand and chroma(c) >= 0.08]
    candidates = [c for c in others if hue_gap(c) * 360 >= 30]
    if candidates:
        accent = max(candidates, key=lambda c: hue_gap(c) * chroma(c))
    else:  # hài hoà tương tự: xoay sắc độ của brand +35°
        h, l, s = hls(brand)
        accent = from_hls(h + 35 / 360, l, s)
    return {"brand": brand, "accent": accent,
            "lightest": max(colors, key=luminance), "darkest": min(colors, key=luminance)}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _pretty_name(stem: str) -> str:
    stem = re.sub(r"Color$", "", stem)
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem).strip() or stem


def load_palettes(folder: Path) -> tuple[list[Palette], list[str]]:
    """Quét `folder`; trả (danh sách bảng màu duy nhất, danh sách vấn đề). Không nuốt lỗi: ảnh hỏng
    được báo trong `problems`. Bảng trùng màu (kể cả khác thứ tự) chỉ giữ một, ưu tiên ảnh đặt tên riêng.
    """
    problems: list[str] = []
    if not folder.is_dir():
        return [], [f"không thấy thư mục {folder}"]
    files = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda p: (p.stem.lower().startswith("color hunt"), p.stem.lower()))
    seen: dict[frozenset, str] = {}
    found: list[Palette] = []
    for path in files:
        try:
            colors = _read_colors(path)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        key = frozenset(colors)
        if key in seen:
            problems.append(f"{path.name}: trùng màu với {seen[key]} (bỏ qua)")
            continue
        seen[key] = path.name
        brand = _analyze(colors)["brand"]
        family, _, mood = _hue_family(brand)
        hunt = path.stem.lower().startswith("color hunt")
        name = f"{family} · {brand.upper()}" if hunt else _pretty_name(path.stem)
        pid = _slug(f"hunt-{brand[1:]}" if hunt else _pretty_name(path.stem))
        found.append(Palette(pid, name, colors, path.name, family, mood))
    ids = [p.id for p in found]
    if len(ids) != len(set(ids)):  # hiếm: hai ảnh cùng brand nhưng khác bộ màu → thêm hậu tố
        fixed, used = [], {}
        for p in found:
            used[p.id] = used.get(p.id, 0) + 1
            fixed.append(p if used[p.id] == 1 else Palette(f"{p.id}-{used[p.id]}", p.name, p.colors, p.source, p.family, p.mood))
        found = fixed
    # Bảng đặt tên riêng (Cold, Winter…) lên trước; bảng Color Hunt xếp theo họ màu để dễ tìm.
    order = {"Đỏ": 0, "Cam": 1, "Nâu": 2, "Vàng": 3, "Xanh lá": 4, "Xanh ngọc": 5, "Xanh dương": 6, "Tím": 7, "Hồng": 8, "Xám": 9}

    def sort_key(p: Palette):
        hunt = p.source.lower().startswith("color hunt")
        return (hunt, order.get(p.family, 99) if hunt else 0, p.name)
    found.sort(key=sort_key)
    return found, problems


DEFAULT_PALETTE = Palette("mac-dinh", "Mặc định · Xanh dương", ("#3368a0", "#66a3bf", "#c8dfdb", "#f2efe7"),
                          "(tích hợp sẵn)", "Xanh dương", "Tin cậy · bình tĩnh")


# --------------------------------------------------------------------------------- token
def _fit_fill(brand: str, surface: str, r: dict, dark_text: str, light_text: str) -> tuple[str, str]:
    """Màu nền nút + màu chữ trên nút. Nền nút ≥ ui:1 với bề mặt; chữ ≥ on:1 với nền nút."""
    fill = ensure_contrast(brand, surface, r["ui"])

    def best_text(f: str) -> tuple[str, float]:
        return max(((t, contrast(t, f)) for t in (light_text, dark_text)), key=lambda x: x[1])
    text, val = best_text(fill)
    if val >= r["on"]:
        return fill, text
    h, l, s = hls(fill)
    best = (fill, text, val)
    for i in range(1, 60):
        for direction in (-1, 1):
            cand = from_hls(h, l + direction * i / 100, s)
            t, v = best_text(cand)
            if v >= r["on"] and contrast(cand, surface) >= r["ui"]:
                return cand, t
            if v > best[2] and contrast(cand, surface) >= r["ui"]:
                best = (cand, t, v)
    return best[0], best[1]


def derive_tokens(colors: tuple[str, ...], mode: str, level: str = "normal") -> dict[str, str]:
    """Bộ token màu đầy đủ cho một bảng màu ở chế độ 'light'|'dark' và mức tương phản 'normal'|'high'."""
    if mode not in ("light", "dark") or level not in RATIOS:
        raise ValueError(f"mode/level không hợp lệ: {mode!r}/{level!r}")
    r = RATIOS[level]
    role = _analyze(colors)
    bh, _, bs = hls(role["brand"])
    dh, _, ds = hls(role["darkest"])
    ns = min(max(bs, 0.15), 0.60)  # độ bão hoà của màu trung tính: giữ sắc thái nhưng nhẹ

    lum_bg, lum_surface, lum_surface2, lum_border = LUMINANCE_TARGETS[(mode, level)]
    bg, surface = at_luminance(bh, ns, lum_bg), at_luminance(bh, ns, lum_surface)
    surface2, border = at_luminance(bh, ns, lum_surface2), at_luminance(bh, ns, lum_border)
    if mode == "light":
        text_seed = from_hls(dh, 0.12, min(ds, 0.5))
        muted_seed = from_hls(bh, 0.38, ns * 0.7)
        border_seed = from_hls(bh, 0.55, ns * 0.5)
        dark_text, light_text = from_hls(bh, 0.10, 0.4), "#ffffff"
        shadow = "0 1px 2px rgba(15,23,42,.06), 0 10px 28px rgba(15,23,42,.08)"
    else:
        text_seed = from_hls(bh, 0.95, min(ns, 0.4))
        muted_seed = from_hls(bh, 0.72, ns * 0.6)
        border_seed = from_hls(bh, 0.50, ns * 0.5)
        dark_text, light_text = from_hls(bh, 0.08, 0.4), "#ffffff"
        shadow = "0 1px 0 rgba(255,255,255,.04) inset, 0 12px 32px rgba(0,0,0,.45)"
    if level == "high":
        shadow = "none"

    plane = [bg, surface, surface2]
    text = ensure_contrast(text_seed, plane, r["body"])
    muted = ensure_contrast(muted_seed, plane, r["muted"])
    border_strong = ensure_contrast(border_seed, [surface, surface2], r["border"])
    if level == "high":
        border = border_strong

    brand_fill, on_brand = _fit_fill(role["brand"], surface, r, dark_text, light_text)
    accent_fill, on_accent = _fit_fill(role["accent"], surface, r, dark_text, light_text)
    tokens = {
        "mode": mode, "level": level,
        "bg": bg, "surface": surface, "surface2": surface2, "border": border, "border_strong": border_strong,
        "text": text, "muted": muted, "shadow": shadow,
        "brand": brand_fill, "on_brand": on_brand, "brand_ink": ensure_contrast(role["brand"], plane, r["ink"]),
        "accent": accent_fill, "on_accent": on_accent, "accent_ink": ensure_contrast(role["accent"], plane, r["ink"]),
        "brand_soft": mix(surface, brand_fill, 0.14 if mode == "light" else 0.22),
        "ring": ensure_contrast(role["brand"], plane, max(r["ui"], 3.0)),
        "stripe": ", ".join(colors), "brand_hue": str(round(bh * 360)),
    }
    for name, hue in SEMANTIC_HUES.items():
        base = from_hls(hue / 360, 0.42 if mode == "light" else 0.62, 0.72)
        tint = surface if level == "high" else mix(surface, base, 0.13 if mode == "light" else 0.18)
        ink = ensure_contrast(base, [tint, surface], r["ink"])
        tokens[f"{name}_ink"] = ink
        tokens[f"{name}_tint"] = tint
        tokens[f"{name}_border"] = ink if level == "high" else mix(surface, ink, 0.45)
    return tokens


def load_label_meta(path: Path) -> tuple[dict[str, dict], list[str]]:
    """Đọc `configs/labels.yaml` (của 02_CONFIG_METRICS, chỉ đọc) → {tên_nhãn: {vi, en, color|None}}.

    Lỗi/thiếu file không ném ra mà trả ({}, [lý do]) để giao diện vẫn chạy với nhãn thô từ dữ liệu.
    """
    if not path.is_file():
        return {}, [f"không thấy {path.name}"]
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        return {}, ["thiếu thư viện PyYAML"]
    except (OSError, yaml.YAMLError) as exc:
        return {}, [f"{path.name}: {exc}"]
    meta: dict[str, dict] = {}
    problems: list[str] = []
    for name, entry in (raw.get("labels") or {}).items():
        entry = entry or {}
        display = entry.get("display") or {}
        color = entry.get("color")
        if color is not None and not (isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color)):
            problems.append(f"nhãn {name}: color không phải #rrggbb ({color!r}), bỏ qua")
            color = None
        meta[str(name).strip().lower()] = {"vi": display.get("vi"), "en": display.get("en"), "color": color}
    return meta, problems


def emotion_style(label: str, index: int, tokens: dict[str, str], color: str | None = None) -> tuple[str, str]:
    """(màu, emoji) cho một nhãn cảm xúc. `color` (từ configs/labels.yaml, nếu đã điền) được ưu tiên;
    nhãn lạ dùng màu xoay quanh sắc độ chủ đạo, không có emoji. Luôn chỉnh độ sáng cho đủ tương phản."""
    hue, emoji = EMOTION_STYLE.get(str(label).strip().lower(), (None, ""))
    if color:
        base = color
    elif str(label).strip().lower() == "neutral":
        base = from_hls(int(tokens["brand_hue"]) / 360, 0.5, 0.12)
    elif hue is None:
        base = from_hls(((int(tokens["brand_hue"]) + 47 * index) % 360) / 360, 0.5, 0.65)
    else:
        base = from_hls(hue / 360, 0.5, 0.78)
    ratio = RATIOS[tokens["level"]]["ui"]
    return ensure_contrast(base, [tokens["surface"], tokens["surface2"]], ratio), emoji


def model_color(index: int, tokens: dict[str, str]) -> str:
    """Màu nhận diện mô hình theo thứ tự trong dữ liệu: brand, accent, rồi xoay sắc độ."""
    if index == 0:
        return tokens["brand"]
    if index == 1:
        return tokens["accent"]
    h = (int(tokens["brand_hue"]) + 60 * index) % 360
    return ensure_contrast(from_hls(h / 360, 0.5, 0.65), [tokens["surface"]], RATIOS[tokens["level"]]["ui"])


# --------------------------------------------------------------------------------- CSS
FONT_SCALE = {"S": "15px", "M": "16px", "L": "18px"}
_FONT_STACK = '"Source Sans", "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif'  # Source Sans do Streamlit tự chứa sẵn (không tải font ngoài)


def build_css(tokens: dict[str, str], font: str = "M", reduce_motion: bool = False) -> str:
    """CSS đầy đủ (chuỗi) cho một bộ token. Không có `@import`, `url(http…)`, hay `prefers-color-scheme`."""
    scheme = "only light" if tokens["mode"] == "light" else "dark"
    motion = "0s" if reduce_motion else ".16s"
    return Template(_css_template()).substitute(
        scheme=scheme, motion=motion, font_size=FONT_SCALE.get(font, "16px"), font_stack=_FONT_STACK,
        radius="18px" if tokens["level"] == "normal" else "12px",
        bw="1px" if tokens["level"] == "normal" else "2px",
        **{k: v for k, v in tokens.items() if k not in ("mode", "level")},
    )


@lru_cache(maxsize=4)
def _read_css(path: str, mtime_ns: int) -> str:
    return Path(path).read_text(encoding="utf-8")


def _css_template() -> str:
    """Mẫu CSS nằm ở frontend/theme.css (dễ sửa, tách khỏi logic màu). Đệm theo thời điểm sửa file để
    chỉnh CSS có hiệu lực ngay, không cần khởi động lại máy chủ."""
    path = Path(__file__).with_name("theme.css")
    return _read_css(str(path), path.stat().st_mtime_ns)
