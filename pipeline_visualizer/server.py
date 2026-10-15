"""Server trực quan hóa Pipeline và cho phép tương tác Test ảnh trực tiếp.
Không cần cài Streamlit! Dùng chuẩn thư viện có sẵn của Python để chạy web UI ngay lập tức.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Thêm backend/web vào sys.path để gọi trực tiếp orchestrator
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
BACKEND_WEB = PROJECT_ROOT / "backend" / "web"
sys.path.insert(0, str(BACKEND_WEB))

import orchestrator  # noqa: E402


class PipelineVisualizerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(CURRENT_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(CURRENT_DIR / "index.html", "rb") as f:
                self.wfile.write(f.read())
            return

        if parsed.path == "/api/samples":
            samples_dir = CURRENT_DIR / "samples"
            samples = []
            if samples_dir.exists():
                for p in sorted(samples_dir.glob("*.jpg")):
                    with open(p, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    samples.append({
                        "id": p.name,
                        "name": p.stem.replace("_", " ").title(),
                        "b64": f"data:image/jpeg;base64,{b64}"
                    })
            self._send_json(samples)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/run_pipeline":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                img_data_url = payload.get("image", "")
                
                # Bóc tách base64
                if "," in img_data_url:
                    img_b64 = img_data_url.split(",", 1)[1]
                else:
                    img_b64 = img_data_url
                img_bytes = base64.b64decode(img_b64)

                # Chạy pipeline qua orchestrator
                # Ở đây dùng emoca mock và engine mock (hoặc fallback) để đảm bảo chạy mượt mà trên mọi máy
                result = orchestrator.analyze(
                    img_bytes,
                    emoca={"mode": "mock"},
                    engine_mode="mock"
                )
                self._send_json(result)
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, status=500)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _send_json(self, data: any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8501):
    server = ThreadingHTTPServer(("127.0.0.1", port), PipelineVisualizerHandler)
    print(f"===========================================================")
    print(f" PIPELINE VISUALIZER DANG CHAY TAI: http://127.0.0.1:{port}")
    print(f"===========================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDang tat server...")
        server.shutdown()


if __name__ == "__main__":
    port = 8501
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
