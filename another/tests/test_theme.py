"""Kiểm thử frontend/theme.py. Chạy: python tests/test_theme.py

Không cần Streamlit. Ảnh giả sinh trong thư mục tạm; không ghi vào dự án.
Phần đo tương phản THẬT trên trang render nằm ở tests/audit_contrast_browser.py (cần trình duyệt).
"""
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))

import theme  # noqa: E402

THEME_DIR = ROOT / "frontend" / "assets" / "theme"
LABELS = ROOT / "backend" / "configs" / "labels.yaml"


def ok(name: str, extra: str = "") -> None:
    print(f"PASS {name}" + (f": {extra}" if extra else ""))


def swatch_png(path: Path, colors, mode="RGB") -> None:
    """Ảnh bảng màu: các dải phẳng ngang (giống ảnh trong frontend/assets/theme)."""
    im = Image.new(mode, (40, 40 * len(colors)))
    for i, c in enumerate(colors):
        im.paste(c, (0, i * 40, 40, (i + 1) * 40))
    im.save(path)


def test_wcag_basics():
    assert abs(theme.contrast("#000000", "#ffffff") - 21.0) < 0.01
    assert abs(theme.contrast("#777777", "#777777") - 1.0) < 1e-9
    # giá trị tham chiếu WCAG: #767676 trên trắng ≈ 4.54
    assert 4.5 < theme.contrast("#767676", "#ffffff") < 4.6
    ok("wcag_basics")


def test_ensure_contrast():
    for fg, bgs, ratio in [("#7aa2f7", ["#ffffff"], 7.0), ("#334455", ["#0b1220"], 7.0), ("#ff9d9d", ["#fbefef", "#ffe2e2"], 10.0),
                           ("#123456", ["#f0f4f9", "#ffffff", "#dae6f3"], 4.6)]:
        out = theme.ensure_contrast(fg, bgs, ratio)
        assert min(theme.contrast(out, b) for b in bgs) >= ratio - 0.01, (fg, bgs, ratio, out)
    assert theme.ensure_contrast("#000000", "#ffffff", 7.0) == "#000000"  # đã đạt thì giữ nguyên
    ok("ensure_contrast")


def test_load_real_palettes():
    palettes, problems = theme.load_palettes(THEME_DIR)
    assert palettes, f"không đọc được bảng màu nào từ {THEME_DIR}: {problems}"
    ids = [p.id for p in palettes]
    assert len(ids) == len(set(ids)), "id trùng"
    assert len({frozenset(p.colors) for p in palettes}) == len(palettes), "còn bảng trùng màu"
    for p in palettes:
        assert 2 <= len(p.colors) <= 4 and all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in p.colors), p
        assert p.family and p.mood and p.name
    ok("load_real_palettes", f"{len(palettes)} bảng; ghi chú: {len(problems)}")


def test_loader_robustness():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        swatch_png(d / "Good.png", [(51, 104, 160), (102, 163, 191), (200, 223, 219), (242, 239, 231)])
        swatch_png(d / "Copy of good.png", [(242, 239, 231), (51, 104, 160), (102, 163, 191), (200, 223, 219)])  # cùng màu, khác thứ tự
        (d / "corrupt.png").write_bytes(b"day khong phai anh png")
        swatch_png(d / "onecolor.png", [(10, 20, 30)])
        Image.new("RGBA", (30, 30), (255, 0, 0, 0)).save(d / "transparent.png")
        (d / "notes.txt").write_text("không phải ảnh", encoding="utf-8")
        palettes, problems = theme.load_palettes(d)
        joined = " | ".join(problems)
        assert len(palettes) == 1 and palettes[0].name in ("Good", "Copy of good"), [p.name for p in palettes]
        dropped = "Good.png" if palettes[0].name == "Copy of good" else "Copy of good.png"
        for name in (dropped, "corrupt.png", "onecolor.png", "transparent.png"):  # giữ đúng một bản trùng, báo bản kia
            assert name in joined, (name, problems)
        assert "notes.txt" not in joined  # file không phải ảnh bị bỏ qua, không báo lỗi
    with tempfile.TemporaryDirectory() as d:  # ảnh đặt tên riêng thắng ảnh Color Hunt dù xếp chữ cái sau
        d = Path(d)
        cols = [(200, 30, 30), (240, 200, 100), (30, 90, 160), (250, 245, 235)]
        swatch_png(d / "Color Hunt Palette c81e1ef0c86420" "5aa0fa.png", cols)
        swatch_png(d / "Zeta.png", cols)
        palettes, problems = theme.load_palettes(d)
        assert [p.name for p in palettes] == ["Zeta"] and any("Color Hunt" in x for x in problems), (palettes, problems)
    missing, probs = theme.load_palettes(ROOT / "src" / "khong_ton_tai")
    assert missing == [] and probs, probs
    ok("loader_robustness")


def test_token_contrast_all():
    palettes, _ = theme.load_palettes(THEME_DIR)
    palettes = palettes + [theme.DEFAULT_PALETTE]
    checked = 0
    for p in palettes:
        for mode in ("light", "dark"):
            for level in ("normal", "high"):
                t = theme.derive_tokens(p.colors, mode, level)
                R = theme.RATIOS[level]
                plane = [t["bg"], t["surface"], t["surface2"]]
                where = (p.id, mode, level)

                def worst(fg, bgs):
                    return min(theme.contrast(fg, b) for b in bgs)
                assert worst(t["text"], plane) >= R["body"] - 0.05, ("text", where)
                assert worst(t["muted"], plane) >= R["muted"] - 0.05, ("muted", where)
                assert worst(t["text"], [t["brand_soft"]]) >= R["muted"], ("text/brand_soft", where)
                assert theme.contrast(t["on_brand"], t["brand"]) >= R["on"] - 0.05, ("on_brand", where)
                assert theme.contrast(t["on_accent"], t["accent"]) >= R["on"] - 0.05, ("on_accent", where)
                assert theme.contrast(t["brand"], t["surface"]) >= R["ui"] - 0.05, ("brand ui", where)
                assert theme.contrast(t["accent"], t["surface"]) >= R["ui"] - 0.05, ("accent ui", where)
                assert worst(t["brand_ink"], plane) >= R["ink"] - 0.05, ("brand_ink", where)
                assert worst(t["accent_ink"], plane) >= R["ink"] - 0.05, ("accent_ink", where)
                assert worst(t["border_strong"], [t["surface"], t["surface2"]]) >= R["border"] - 0.05, ("border", where)
                assert worst(t["ring"], plane) >= max(R["ui"], 3.0) - 0.05, ("ring", where)
                for s in ("success", "warning", "danger", "info"):
                    assert theme.contrast(t[f"{s}_ink"], t[f"{s}_tint"]) >= R["ink"] - 0.05, (s, where)
                    assert theme.contrast(t[f"{s}_ink"], t["surface"]) >= R["ink"] - 0.05, (s, where)
                for i, lab in enumerate(["happiness", "surprise", "fear", "sadness", "disgust", "anger", "neutral", "contempt", "nhãn lạ"]):
                    col, _ = theme.emotion_style(lab, i, t)
                    assert worst(col, [t["surface"], t["surface2"]]) >= R["ui"] - 0.05, ("emotion", lab, where)
                for i in range(4):
                    assert theme.contrast(theme.model_color(i, t), t["surface"]) >= R["ui"] - 0.05, ("model", i, where)
                # sáng/tối đúng nghĩa: nền chói hơn chữ ở chế độ sáng và ngược lại
                assert (theme.luminance(t["bg"]) > theme.luminance(t["text"])) == (mode == "light"), ("mode", where)
                checked += 1
    ok("token_contrast_all", f"{checked} tổ hợp (bảng × sáng/tối × chuẩn/cao)")


def test_tokens_deterministic_and_validation():
    c = ("#3368a0", "#66a3bf", "#c8dfdb", "#f2efe7")
    assert theme.derive_tokens(c, "light") == theme.derive_tokens(c, "light")
    assert theme.derive_tokens(c, "light") != theme.derive_tokens(c, "dark")
    for bad in (("sang", "normal"), ("light", "cao")):
        try:
            theme.derive_tokens(c, *bad)
        except ValueError:
            continue
        raise AssertionError(f"lẽ ra phải ValueError: {bad}")
    ok("tokens_deterministic_and_validation")


def test_css_independent_of_browser():
    c = ("#3368a0", "#66a3bf", "#c8dfdb", "#f2efe7")
    for mode in ("light", "dark"):
        css = theme.build_css(theme.derive_tokens(c, mode), "M", False)
        assert "$" not in css.replace('$="Active"', ""), "còn biến chưa thay"
        for banned in ("prefers-color-scheme", "@import", "http", "url("):
            assert banned not in css, f"CSS chứa '{banned}'"
        assert ("color-scheme: only light" in css) == (mode == "light")
        assert ("color-scheme: dark" in css) == (mode == "dark")
        assert "--brand:" in css and "--bg:" in css
    tokens = theme.derive_tokens(c, "light")
    assert "html { font-size:18px; }" in theme.build_css(tokens, "L")
    assert "--motion:0s" in theme.build_css(tokens, "M", True)
    assert "--motion:.16s" in theme.build_css(tokens, "M", False)
    ok("css_independent_of_browser")


def test_emotion_and_model_colors():
    t = theme.derive_tokens(("#3368a0", "#66a3bf", "#c8dfdb", "#f2efe7"), "dark")
    _, emo = theme.emotion_style("Happiness", 0, t)  # không phân biệt hoa thường
    assert emo == "😄"
    col_unknown, emo_unknown = theme.emotion_style("nhãn chưa biết", 3, t)
    assert emo_unknown == "" and re.fullmatch(r"#[0-9a-f]{6}", col_unknown)
    # màu từ configs/labels.yaml (nếu có) được ưu tiên; nếu đã đủ tương phản thì giữ nguyên
    good = theme.ensure_contrast("#e6b800", [t["surface"], t["surface2"]], 3.0)
    assert theme.emotion_style("happiness", 0, t, good)[0] == good
    assert len({theme.model_color(i, t) for i in range(4)}) == 4, "màu mô hình phải khác nhau"
    ok("emotion_and_model_colors")


def test_family_and_mood():
    assert theme._hue_family("#2196f3")[0] == "Xanh dương"
    assert theme._hue_family("#d32f2f")[0] == "Đỏ"
    assert theme._hue_family("#43a047")[0] == "Xanh lá"
    assert theme._hue_family("#808080")[0] == "Xám"
    assert theme._hue_family("#5d4037")[0] == "Nâu"
    ok("family_and_mood")


def test_label_meta():
    if LABELS.is_file():
        meta, problems = theme.load_label_meta(LABELS)
        assert meta["happiness"]["vi"] == "Vui vẻ" and meta["surprise"]["vi"] == "Ngạc nhiên", meta
        assert problems == [], problems
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "labels.yaml"
        p.write_text("labels:\n  anger: {display: {vi: Giận}, color: 'đỏ'}\n  fear: {display: {vi: Sợ}, color: '#aa00ff'}\n", encoding="utf-8")
        meta, problems = theme.load_label_meta(p)
        assert meta["anger"]["color"] is None and meta["fear"]["color"] == "#aa00ff"
        assert len(problems) == 1 and "anger" in problems[0], problems
        p.write_text("labels: [khong, hop, le", encoding="utf-8")
        meta, problems = theme.load_label_meta(p)
        assert meta == {} and problems, problems
    meta, problems = theme.load_label_meta(Path("khong_ton_tai.yaml"))
    assert meta == {} and problems
    ok("label_meta")


def main():
    test_wcag_basics()
    test_ensure_contrast()
    test_load_real_palettes()
    test_loader_robustness()
    test_token_contrast_all()
    test_tokens_deterministic_and_validation()
    test_css_independent_of_browser()
    test_emotion_and_model_colors()
    test_family_and_mood()
    test_label_meta()
    print("ALL PASS")


if __name__ == "__main__":
    main()
