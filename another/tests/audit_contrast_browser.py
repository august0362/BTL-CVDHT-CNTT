"""Measure WCAG contrast for text on the current fixed-style app in Chromium.

Requires Playwright and a running Streamlit server on 127.0.0.1:8599.
Run: python tests/audit_contrast_browser.py [--browser PATH]
"""
import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))
sys.path.insert(0, str(ROOT / "frontend"))


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
    ap = argparse.ArgumentParser(description="Audit contrast on the current fixed-style page.")
    ap.add_argument("--url", default="http://127.0.0.1:8599")
    ap.add_argument("--browser", default=os.environ.get("BROWSER_EXE"))
    ap.add_argument("--w", type=int, default=1440)
    ap.add_argument("--h", type=int, default=1800)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    exe = a.browser or next((p for p in CANDIDATES if Path(p).is_file()), None)
    if not exe:
        print("Chromium not found; pass --browser or set BROWSER_EXE.")
        return 2
    total_bad = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, headless=True)
        page = browser.new_page(viewport={"width": a.w, "height": a.h})
        page.goto(a.url)
        page.wait_for_selector('[data-testid="stApp"]', timeout=40000)
        page.wait_for_timeout(2000)
        rows = page.evaluate(JS)
        browser.close()
    bad = [row for row in rows if row["r"] < row["need"]]
    total_bad += len(bad)
    print(f"current style: {len(rows)} text styles | failing: {len(bad)} | minimum: {min(row['r'] for row in rows):.2f}")
    for row in sorted(bad, key=lambda row: row["r"])[:6]:
        print(f"  {row['r']:5.2f} < {row['need']}  {row['size']:.0f}px  {row['t']!r}  fg={row['fg']} bg={row['bg']}")
    print("total contrast failures:", total_bad)
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
