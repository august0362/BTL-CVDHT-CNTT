"""Đo tương phản THẬT của mọi chữ trên trang đang chạy, trong một trình duyệt Chromium (Cốc Cốc/Chrome/Edge).

Với mỗi phần tử có chữ: lấy màu chữ tính toán (kèm độ trong suốt), leo lên các phần tử cha tìm màu nền thực
(pha các lớp bán trong suốt), tính tỉ lệ tương phản WCAG. Đạt nếu >= 4.5 (chữ thường) hoặc >= 3 (chữ lớn:
>= 24px, hoặc đậm >= 18.66px). Bắt được lỗi mà kiểm tra ở mức bảng màu không thấy (chữ bị quy tắc CSS khác
ghi đè, nền hover/focus mặc định của Streamlit…).

Yêu cầu (chỉ khi cần chạy công cụ này, không thuộc requirements của web/):
  pip install playwright        # chỉ là thư viện điều khiển; dùng trình duyệt có sẵn, không tải thêm
  streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8599

Dùng:
  python tests/audit_contrast_browser.py                        # chủ đề mặc định, sáng/tối × chuẩn/cao
  python tests/audit_contrast_browser.py --all                  # mọi chủ đề trong frontend/assets/theme (lâu)
  python tests/audit_contrast_browser.py --emoca                # thêm trạng thái "Máy chủ EMOCA" (ô nhập, hộp xác nhận)
  python tests/audit_contrast_browser.py --engine-real          # POSTER/MA3D-Net "Thật (CPU)" (server nạp torch ~1 GB RAM)
  python tests/audit_contrast_browser.py --engine-real --no-upload   # chỉ đo nút "Thật (CPU)" + trang trống (không nạp torch)
  python tests/audit_contrast_browser.py --themes cold,fall --browser "C:\\path\\to\\browser.exe"
Thoát mã 1 nếu có chữ không đạt.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))
sys.path.insert(0, str(ROOT / "frontend"))

SETTINGS = ".st-key-ui_settings_open"  # expander "Cài đặt" của frontend/app.py

CANDIDATES = [
    r"C:\Program Files\CocCoc\Browser\Application\browser.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

JS = r"""
() => {
  const parse = (s) => { const m = s.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(',').map(x => parseFloat(x)); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; };
  const over = (top, bottom) => { const a = top[3] + bottom[3] * (1 - top[3]);
    if (a === 0) return [0, 0, 0, 0];
    return [0, 1, 2].map(i => (top[i] * top[3] + bottom[i] * bottom[3] * (1 - top[3])) / a).concat([a]); };
  const lum = (c) => { const f = v => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]); };
  const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  const bgOf = (el) => { const layers = []; for (let e = el; e; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor);
      if (c && c[3] > 0) layers.push(c); if (c && c[3] >= 0.999) break; }
    let acc = [255, 255, 255, 1]; for (let i = layers.length - 1; i >= 0; i--) acc = over(layers[i], acc); return acc; };
  const out = []; const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const n = walker.currentNode; const txt = n.nodeValue.trim(); if (!txt) continue;
    const el = n.parentElement; if (!el || ['SCRIPT', 'STYLE', 'TITLE'].includes(el.tagName)) continue;
    const r = el.getBoundingClientRect(); if (r.width < 2 || r.height < 2) continue;
    const cs = getComputedStyle(el); if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    let op = 1; for (let e = el; e; e = e.parentElement) op *= parseFloat(getComputedStyle(e).opacity);
    if (op < 0.05) continue;
    const fg0 = parse(cs.color); if (!fg0) continue;
    const bg = bgOf(el); const fg = over([fg0[0], fg0[1], fg0[2], fg0[3] * op], bg);
    const size = parseFloat(cs.fontSize); const bold = parseInt(cs.fontWeight) >= 700;
    const large = size >= 24 || (bold && size >= 18.66);
    const key = txt + '|' + cs.color + '|' + bg.join(',');
    if (seen.has(key)) continue; seen.add(key);
    out.push({t: txt.slice(0, 44), r: Math.round(ratio(fg, bg) * 100) / 100, need: large ? 3 : 4.5, size, fg: cs.color, bg: 'rgb(' + bg.slice(0, 3).map(Math.round).join(',') + ')'});
  }
  // Chữ TRONG ô nhập (giá trị đã gõ hoặc placeholder) không phải text node nên bộ duyệt ở trên không thấy.
  document.querySelectorAll('input:not([type=file]):not([type=checkbox]):not([type=radio]), textarea').forEach((el) => {
    const r = el.getBoundingClientRect(); if (r.width < 2 || r.height < 2) return;
    const shown = el.value || el.placeholder; if (!shown) return;
    const cs = getComputedStyle(el);
    const colorStr = el.value ? (cs.webkitTextFillColor || cs.color) : getComputedStyle(el, '::placeholder').color;
    const fg0 = parse(colorStr); if (!fg0) return;
    const bg = bgOf(el); const fg = over(fg0, bg);
    const size = parseFloat(cs.fontSize);
    out.push({t: '[ô nhập' + (el.value ? '' : ', placeholder') + '] ' + (el.type === 'password' ? '••••' : shown).slice(0, 30),
              r: Math.round(ratio(fg, bg) * 100) / 100, need: 4.5, size, fg: colorStr, bg: 'rgb(' + bg.slice(0, 3).map(Math.round).join(',') + ')'});
  });
  return out;
}
"""


def wait_idle(pg, timeout_ms: int = 180000) -> None:
    """Chờ tới khi Streamlit hết trạng thái đang chạy (không còn status widget / spinner)."""
    busy = '[data-testid="stStatusWidget"], [data-testid="stSpinner"]'
    for _ in range(timeout_ms // 200):
        if pg.locator(busy).count() == 0:
            pg.wait_for_timeout(300)
            if pg.locator(busy).count() == 0:
                return
        pg.wait_for_timeout(200)
    raise TimeoutError("Streamlit chạy quá lâu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8599")
    ap.add_argument("--all", action="store_true", help="mọi chủ đề trong frontend/assets/theme")
    ap.add_argument("--themes", help="danh sách id chủ đề, cách nhau bằng dấu phẩy (mặc định: cold)")
    ap.add_argument("--emoca", action="store_true",
                    help="đo cả trạng thái 'Máy chủ EMOCA' (ô địa chỉ/token, hộp xác nhận gửi ảnh; địa chỉ giả .invalid, không gửi gì đi)")
    ap.add_argument("--engine-real", action="store_true",
                    help="đo khi POSTER/MA3D-Net ở chế độ 'Thật (CPU)' (server nạp torch, ~1 GB RAM; chỉ chạy khi đủ RAM)")
    ap.add_argument("--no-upload", action="store_true",
                    help="không tải ảnh: chỉ đo trang trống + bảng Cài đặt (kết hợp --engine-real: đo nút 'Thật (CPU)' mà không nạp torch)")
    ap.add_argument("--browser", default=os.environ.get("BROWSER_EXE"))
    ap.add_argument("--w", type=int, default=1440)
    ap.add_argument("--h", type=int, default=3300)
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    exe = a.browser or next((p for p in CANDIDATES if Path(p).is_file()), None)
    if not exe:
        print("Không thấy trình duyệt Chromium; truyền --browser hoặc đặt BROWSER_EXE.")
        return 2
    if a.all:
        import theme
        themes = [p.id for p in theme.load_palettes(ROOT / "frontend" / "assets" / "theme")[0]]
    else:
        themes = (a.themes or "cold").split(",")

    with tempfile.TemporaryDirectory() as d:  # ảnh thử sinh tại chỗ, không đụng dự án
        img = Path(d) / "face.png"
        Image.linear_gradient("L").resize((320, 320)).convert("RGB").save(img)
        total_bad = 0
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=exe, headless=True)
            for t in themes:
                for m in ("light", "dark"):
                    for c in ("normal", "high"):
                        query = f"t={t}&m={m}&c={c}"
                        pg = b.new_page(viewport={"width": a.w, "height": a.h})
                        pg.goto(f"{a.url}/?{query}")
                        pg.wait_for_selector('[data-testid="stApp"]', timeout=40000)
                        pg.wait_for_timeout(2000)
                        # Cài đặt nằm trong expander "Cài đặt" ở đầu trang (mặc định đóng): mở để đo và để bấm được control.
                        settings = pg.locator(f"{SETTINGS} summary")
                        if settings.count() and pg.locator(f"{SETTINGS} details[open]").count() == 0:
                            settings.click()
                            pg.wait_for_timeout(800)
                        if a.emoca:  # chuyển nguồn EMOCA sang máy chủ từ xa GIẢ: chưa xác nhận nên ảnh không bị gửi đi đâu
                            pg.get_by_text("Máy chủ EMOCA", exact=True).first.click()
                            pg.wait_for_timeout(2500)
                            url_in = pg.locator('input[aria-label="Địa chỉ EMOCA"]')
                            url_in.fill("https://emoca.example.invalid")
                            url_in.press("Enter")
                            pg.wait_for_timeout(1500)
                            pg.locator('input[aria-label="Token"]').fill("token-thu-nghiem")
                            pg.locator('input[aria-label="Token"]').press("Enter")
                            pg.wait_for_timeout(1500)
                        if a.engine_real:  # POSTER/MA3D-Net chạy thật: đổi chế độ rồi chờ Streamlit chạy xong
                            pg.get_by_text("Thật (CPU)", exact=True).first.click()
                            pg.mouse.move(a.w - 5, 5)  # bỏ trạng thái hover để đo màu "đang chọn" thật
                            pg.wait_for_timeout(500)
                            wait_idle(pg)
                        if not a.no_upload:
                            pg.set_input_files('input[type="file"]', str(img))
                            pg.wait_for_timeout(5000)
                            if a.engine_real:  # lần đầu nạp mô hình có thể lâu hơn 5 s
                                pg.get_by_text("engine_real:", exact=False).first.wait_for(timeout=180000)
                                wait_idle(pg)
                        # mở các expander còn đóng (bấm vào expander đang mở sẽ đóng nó lại, vd. bảng Cài đặt)
                        for s in pg.query_selector_all('[data-testid="stMain"] [data-testid="stExpander"] details:not([open]) > summary'):
                            s.click()
                            pg.wait_for_timeout(250)
                        rows = pg.evaluate(JS)
                        pg.close()
                        bad = [r for r in rows if r["r"] < r["need"]]
                        total_bad += len(bad)
                        print(f"[{query}] {len(rows)} kiểu chữ | không đạt: {len(bad)} | nhỏ nhất: {min(r['r'] for r in rows):.2f}")
                        for r in sorted(bad, key=lambda r: r["r"])[:6]:
                            print(f"     {r['r']:5.2f} < {r['need']}  {r['size']:.0f}px  «{r['t']}»  fg={r['fg']} bg={r['bg']}")
                        if a.engine_real:  # nút vừa chọn: in riêng để trả lời "nút đang chọn có đạt không"
                            for r in rows:
                                if r["t"] == "Thật (CPU)":
                                    print(f"     nút «Thật (CPU)»: {r['r']:.2f} (cần {r['need']})  fg={r['fg']} bg={r['bg']}")
            b.close()
    print("TỔNG chữ không đạt:", total_bad)
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
