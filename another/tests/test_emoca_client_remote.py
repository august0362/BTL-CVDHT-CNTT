"""Kiểm phía gọi EMOCA từ xa: web/emoca_client.py (chế độ real) + web/orchestrator.py, qua HTTP THẬT trên cổng tạm.

Chạy:  python tests/test_emoca_client_remote.py

Máy chủ dùng để thử: mock_server (không token), server.py + FakeBackend (có token), và vài máy chủ "hư" tự dựng
(chậm, trả HTML, sai contract_version). Không GPU, không mạng ngoài. Kiểm được: ánh xạ lỗi mạng → mã lỗi contract,
token không lọt vào thông báo, chặn gửi ảnh ra ngoài khi chưa xác nhận (ADR-015), luồng analyze() đầu-cuối.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "backend" / "web"))
from test_emoca_svc_real import TOKEN, make_runner  # noqa: E402  (cũng nạp services/emoca_svc vào sys.path)

import emoca_client  # noqa: E402
import mock_server  # noqa: E402
import orchestrator  # noqa: E402
import server  # noqa: E402

EMOCA_SCHEMA = json.loads((ROOT / "backend" / "contracts" / "emoca_analyze.schema.json").read_text(encoding="utf-8"))
WEB_SCHEMA = json.loads((ROOT / "backend" / "contracts" / "web_analyze.schema.json").read_text(encoding="utf-8"))
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def valid(body, schema) -> bool:
    try:
        jsonschema.validate(body, schema)
        return True
    except jsonschema.ValidationError as e:
        print("       schema:", e.message[:200])
        return False


def png_bytes(size=(640, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (150, 120, 100)).save(buf, format="PNG")
    return buf.getvalue()


def start(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


def fixed_server(status: int, body: bytes, ctype: str = "text/html", delay: float = 0.0):
    """Máy chủ 'hư': luôn trả đúng một phản hồi cố định (sau `delay` giây)."""
    class H(BaseHTTPRequestHandler):
        def _do(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            time.sleep(delay)
            try:
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # máy khách đã bỏ (thử timeout)

        do_GET = do_POST = _do

        def log_message(self, *a):
            pass
    return ThreadingHTTPServer(("127.0.0.1", 0), H)


def counting_server():
    """mock_server thật + đếm mọi POST tới ở tầng HTTP (để chứng minh ảnh KHÔNG được gửi đi)."""
    hits: list[str] = []
    base_handler = mock_server.make_handler(dict(mock_server.DEFAULTS, max_image_side_px=100_000))

    class Counting(base_handler):
        def do_POST(self):
            hits.append(self.path)
            super().do_POST()

        def log_message(self, *a):
            pass
    return ThreadingHTTPServer(("127.0.0.1", 0), Counting), hits


def test_client():
    print("== emoca_client (mode real) — máy chủ mock, không token")
    img = Image.new("RGB", (640, 480), (90, 90, 90))
    mock = mock_server.make_server(dict(mock_server.DEFAULTS), port=0)
    mock_url = start(mock)
    out = emoca_client.get_analysis(img, mode="real", endpoint={"url": mock_url, "timeout_s": 10})
    check("error" not in out and valid(out, EMOCA_SCHEMA), "gọi mock_server qua HTTP → response đúng emoca_analyze.schema.json")
    check(out["faces"][0]["crop_transform"]["from"] == "original_px" and len(out["faces"][0]["landmarks"]["crop224"]) == 68,
          "mặt có crop_transform + 68 landmark")
    tiny = emoca_client.get_analysis(Image.new("RGB", (20, 20)), mode="real", endpoint={"url": mock_url})
    check(tiny["error"]["code"] == "no_face" and not tiny["error"]["retryable"], "no_face của máy chủ được chuyển tiếp nguyên mã")
    hp = emoca_client.check_service({"url": mock_url})
    check(hp["ok"] and "KHÔNG phải EMOCA thật" in hp["message"] and hp["latency_ms"] is not None,
          f"check_service: nối được, ghi rõ đang là MOCK ({hp['message']})")
    mock.shutdown()
    mock.server_close()

    print("== emoca_client — máy chủ thật (backend giả) có token")
    _, runner = make_runner()
    real = server.build_server(dict(mock_server.DEFAULTS), runner, TOKEN, port=0)
    real_url = start(real)
    ep = {"url": real_url + "/", "token": TOKEN, "timeout_s": 10}  # dấu / cuối được chuẩn hoá
    out = emoca_client.get_analysis(img, mode="real", endpoint=ep)
    check("error" not in out and valid(out, EMOCA_SCHEMA) and out["model"]["name"] == "FAKE-EMOCA", "đúng token → 200, hợp lệ")
    for label, tok in (("thiếu token", ""), ("sai token", "khong-dung")):
        e = emoca_client.get_analysis(img, mode="real", endpoint={"url": real_url, "token": tok})
        msg = e["error"]["message"]
        check(e["error"]["code"] == "service_unavailable" and "401" in msg and not e["error"]["retryable"]
              and TOKEN not in msg and "khong-dung" not in msg, f"{label} → service_unavailable, gợi ý 401, không lộ token ({msg[:70]}…)")
    hp = emoca_client.check_service(ep)
    check(hp["ok"] and hp["info"].get("device") == "fake-gpu" and TOKEN not in json.dumps(hp), "check_service: ok, có thông tin GPU, không có token")
    hp = emoca_client.check_service({"url": real_url, "token": "sai"})
    check(not hp["ok"] and "401" in hp["message"], f"check_service sai token → không ok ({hp['message'][:60]}…)")
    real.shutdown()
    real.server_close()

    print("== emoca_client — ảnh vượt max_image_side_px bị chặn TRƯỚC khi gửi (review D3)")
    srv, hits = counting_server()
    url = start(srv)
    try:
        e = emoca_client.get_analysis(img, mode="real", endpoint={"url": url, "max_image_side_px": 500})
        check(e.get("error", {}).get("code") == "image_too_large" and "CHƯA được gửi" in e["error"]["message"] and hits == [],
              f"640×480 > 500 px → image_too_large, máy chủ nhận {len(hits)} request (phải 0)")
        out = emoca_client.get_analysis(Image.new("RGB", (500, 300), (90, 90, 90)), mode="real",
                                        endpoint={"url": url, "max_image_side_px": 500})
        check("error" not in out and valid(out, EMOCA_SCHEMA) and len(hits) == 1, "500×300 = đúng giới hạn → vẫn gửi (1 request)")
        out = emoca_client.get_analysis(img, mode="real", endpoint={"url": url})
        check("error" not in out and len(hits) == 2, "không có max_image_side_px → không chặn (giữ hành vi cũ)")
    finally:
        srv.shutdown()
        srv.server_close()

    print("== emoca_client — cấu hình sai và máy chủ hỏng")
    for label, url in (("thiếu URL", ""), ("ftp://", "ftp://x.y/z"), ("file://", "file:///etc/passwd"), ("không phải URL", "abc")):
        e = emoca_client.get_analysis(img, mode="real", endpoint={"url": url})
        check(e["error"]["code"] == "model_unavailable", f"{label} → model_unavailable, không gọi mạng")
    check(emoca_client.get_analysis(img, mode="real")["error"]["code"] == "model_unavailable", "không có endpoint → model_unavailable (không rơi về mock)")
    closed = mock_server.make_server(dict(mock_server.DEFAULTS), port=0)
    dead_url = f"http://127.0.0.1:{closed.server_address[1]}"
    closed.server_close()  # cổng đã đóng
    e = emoca_client.get_analysis(img, mode="real", endpoint={"url": dead_url, "timeout_s": 5})
    check(e["error"]["code"] == "service_unavailable" and e["error"]["retryable"], "cổng đóng → service_unavailable, retryable")
    msg = e["error"]["message"]
    check("WSL2" in msg and "run_emoca_server.sh" in msg and "Colab" not in msg,
          f"cổng loopback đóng → gợi ý chạy server WSL2, không nhắc Colab ({msg[:90]}…)")
    hint = emoca_client._unreachable_hint
    check(all("WSL2" in hint(u) for u in ("http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"))
          and all("Colab" in hint(u) and "WSL2" not in hint(u)
                  for u in ("https://abc.trycloudflare.com", "http://192.168.1.5:8765", "http://127example.com", "http://127.0.0.1.evil.com")),
          "_unreachable_hint: loopback (127.x, localhost, ::1) → WSL2; địa chỉ khác → Colab/tunnel")
    slow = fixed_server(200, b"{}", "application/json", delay=1.5)
    t0 = time.perf_counter()
    e = emoca_client.get_analysis(img, mode="real", endpoint={"url": start(slow), "timeout_s": 0.4})
    check(e["error"]["code"] == "timeout" and e["error"]["retryable"] and time.perf_counter() - t0 < 1.4,
          f"máy chủ chậm → timeout sau ~0.4 s ({time.perf_counter() - t0:.2f}s)")
    slow.shutdown()
    cases = [
        ("502 HTML (tunnel không nối được)", fixed_server(502, b"<html>Bad gateway</html>"), "service_unavailable", "502"),
        ("530 HTML (tunnel đã tắt)", fixed_server(530, b"<html>error 1033</html>"), "service_unavailable", "530"),
        ("200 nhưng là HTML", fixed_server(200, b"<html>hello</html>"), "contract_mismatch", ""),
        ("200 JSON không phải emoca_svc", fixed_server(200, b'{"hello": 1}', "application/json"), "contract_mismatch", ""),
        ("contract_version lạ", fixed_server(200, json.dumps({"service": "emoca_svc", "contract_version": "0.1.0", "faces": []}).encode(),
                                             "application/json"), "contract_mismatch", "0.1.0"),
        ("error envelope mã lạ", fixed_server(400, json.dumps({"error": {"code": "la_hoac", "message": "x", "retryable": False}}).encode(),
                                              "application/json"), "contract_mismatch", ""),
    ]
    for label, srv, code, needle in cases:
        e = emoca_client.get_analysis(img, mode="real", endpoint={"url": start(srv), "timeout_s": 5})
        check(e.get("error", {}).get("code") == code and needle in e["error"]["message"], f"{label} → {code}")
        srv.shutdown()

    print("== nén ảnh gửi đi")
    noisy = Image.effect_noise((900, 900), 80).convert("RGB")
    big_png = emoca_client._encode_upload(noisy, 10 * 1024 * 1024)
    check(big_png[:8] == b"\x89PNG\r\n\x1a\n", "ảnh vừa giới hạn → PNG (không mất điểm ảnh)")
    small_limit = len(big_png) - 1
    jpg = emoca_client._encode_upload(noisy, small_limit)
    check(jpg is not None and jpg[:2] == b"\xff\xd8", "PNG vượt giới hạn → chuyển JPEG q95")
    check(emoca_client._encode_upload(noisy, 100) is None, "vẫn vượt → None (báo image_too_large)")


def test_orchestrator():
    print("== orchestrator — luồng đầu-cuối tới emoca_svc thật (backend giả)")
    saved_mode, saved_env = orchestrator._EMOCA_MODE, {}
    for k in ("EMOCA_MODE", "EMOCA_URL", "EMOCA_TOKEN"):
        saved_env[k] = os.environ.pop(k, None)
    _, runner = make_runner()
    real = server.build_server(dict(mock_server.DEFAULTS), runner, TOKEN, port=0)
    url = start(real)
    png = png_bytes()
    try:
        r = orchestrator.analyze(png, emoca={"mode": "real", "url": url, "token": TOKEN})
        check(valid(r, WEB_SCHEMA) and r["status"] == "ok", f"loopback + token: status ok, đúng web_analyze.schema.json ({r['status']})")
        w = " | ".join(r["warnings"])
        check("emoca_real: FAKE-EMOCA (detail), upstream e0be0db+patches" in w and "từ xa" not in w,
              "ghi nguồn EMOCA thật (tên, mode, commit), không nói 'từ xa' vì là loopback")
        check(any(x.startswith("mock_ai: engine") and "emoca_client" not in x for x in r["warnings"]),
              "vẫn ghi rõ engine (POSTER/MA3D) là mock, KHÔNG gắn nhãn mock cho EMOCA")
        check("emoca_roundtrip" in r["timing_ms"] and r["timing_ms"]["emoca"] > 0, f"timing có emoca_roundtrip ({r['timing_ms']})")
        pts = r["faces"][0]["landmarks"]
        check(pts["space"] == "original_px" and len(pts["points"]) == 68, "landmark ở original_px, 68 điểm")
        r = orchestrator.analyze(png, emoca={"mode": "real", "url": url, "token": "sai"})
        check(r["status"] == "partial" and r["error"]["code"] == "service_unavailable" and "emoca_svc_unavailable" in r["warnings"]
              and valid(r, WEB_SCHEMA), "token sai → partial + service_unavailable (trang không hỏng)")
        r = orchestrator.analyze(png, emoca={"mode": "mock"})
        check(r["status"] == "ok" and any("emoca_client" in x for x in r["warnings"]), "override mode=mock → dữ liệu giả, ghi rõ cả emoca_client lẫn engine")
    finally:
        real.shutdown()
        real.server_close()

    print("== orchestrator → emoca_client: giới hạn cạnh ảnh của config được áp trước khi gửi (review D3)")
    srv, hits = counting_server()
    url = start(srv)
    try:
        side = orchestrator.resolve_emoca_settings({"mode": "real", "url": url})[0]["max_image_side_px"]
        r = orchestrator.analyze(png_bytes((side + 4, 20)), emoca={"mode": "real", "url": url})
        check(r["status"] == "error" and r["error"]["code"] == "image_too_large" and valid(r, WEB_SCHEMA) and hits == [],
              f"ảnh {side + 4}×20 (> {side} px của config) → error image_too_large, máy chủ nhận {len(hits)} request (phải 0)")
    finally:
        srv.shutdown()
        srv.server_close()

    print("== orchestrator — chặn gửi ảnh ra ngoài khi chưa xác nhận (ADR-015)")
    calls = []
    real_get = emoca_client.get_analysis

    def spy(image, **kw):
        calls.append(kw)
        return real_get(image, **{**kw, "mode": "mock", "endpoint": None})  # trả dữ liệu mock, không mạng
    emoca_client.get_analysis = spy
    try:
        remote = {"mode": "real", "url": "https://emoca.example.invalid", "token": "abc"}
        r = orchestrator.analyze(png, emoca=remote)
        check(r["status"] == "error" and r["error"]["code"] == "consent_required" and valid(r, WEB_SCHEMA) and not calls,
              "URL từ xa, chưa xác nhận → consent_required, KHÔNG gọi emoca_client (ảnh không rời máy)")
        check("abc" not in json.dumps(r), "token không lọt vào kết quả")
        r = orchestrator.analyze(png, emoca={**remote, "remote_consent": True})
        check(r["status"] == "ok" and len(calls) == 1 and calls[0]["endpoint"]["token"] == "abc"
              and any("máy chủ từ xa emoca.example.invalid" in w for w in r["warnings"]),
              "đã xác nhận → gọi đúng 1 lần với token, cảnh báo ghi rõ máy chủ từ xa")
        calls.clear()
        r = orchestrator.analyze(png, emoca={"mode": "real", "url": "http://192.168.1.50:8765"})
        check(r["error"]["code"] == "consent_required" and not calls, "địa chỉ LAN cũng tính là từ xa")
        for local in ("http://localhost:8765", "http://127.0.0.1:8765", "http://[::1]:8765", "http://127.0.0.2:1"):
            calls.clear()
            r = orchestrator.analyze(png, emoca={"mode": "real", "url": local})
            check(r["status"] != "error" or r["error"]["code"] != "consent_required", f"loopback {local} không cần xác nhận")
        calls.clear()
        r = orchestrator.analyze(png, mock_case="partial", emoca=remote)
        check(r["status"] == "partial" and not calls, "mock_case=partial không gọi mạng nên không đòi xác nhận")
    finally:
        emoca_client.get_analysis = real_get

    print("== orchestrator — thứ tự ưu tiên cấu hình")
    try:
        s, _ = orchestrator.resolve_emoca_settings(None)
        check(s["mode"] == "mock", f"mặc định (config mode=mock) → mock ({s['mode']})")
        os.environ["EMOCA_MODE"], os.environ["EMOCA_URL"], os.environ["EMOCA_TOKEN"] = "real", "http://127.0.0.1:1", "tok-env"
        s, _ = orchestrator.resolve_emoca_settings(None)
        check((s["mode"], s["url"], s["token"]) == ("real", "http://127.0.0.1:1", "tok-env"), "biến môi trường thắng config")
        orchestrator._EMOCA_MODE = "mock"
        s, _ = orchestrator.resolve_emoca_settings(None)
        check(s["mode"] == "mock", "hằng _EMOCA_MODE thắng biến môi trường")
        s, _ = orchestrator.resolve_emoca_settings({"mode": "real", "url": "http://127.0.0.1:2", "token": "tok-ui"})
        check((s["mode"], s["url"], s["token"]) == ("real", "http://127.0.0.1:2", "tok-ui"), "tham số giao diện thắng tất cả")
        st = orchestrator.emoca_status({"mode": "real", "url": "https://x.trycloudflare.com", "token": "bi-mat"})
        check(st["remote"] and st["has_token"] and st["host"] == "x.trycloudflare.com" and "bi-mat" not in json.dumps(st),
              "emoca_status: remote, có token nhưng KHÔNG chứa token")
        os.environ["EMOCA_MODE"] = "reall"
        orchestrator._EMOCA_MODE = None
        s, w = orchestrator.resolve_emoca_settings(None)
        check(s["mode"] == "mock" and any("config_invalid" in x for x in w), "mode sai chính tả → cảnh báo config_invalid, dùng mock")
    finally:
        orchestrator._EMOCA_MODE = saved_mode
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def main() -> int:
    test_client()
    test_orchestrator()
    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
