"""emoca_svc — dịch vụ HTTP EMOCA THẬT (M1-03, ADR-015). Chạy trong env `work38` (Python 3.8, GPU CUDA).

Cùng contract B1 với `mock_server.py` (`POST /v1/analyze`, `GET /v1/health`), cùng tầng HTTP và cùng bước
kiểm ảnh (`mock_server.make_handler`/`check_request`); khác ở chỗ suy luận bằng EMOCA thật qua
`emoca_runner.EmocaRunner`. Mô hình nạp MỘT lần lúc khởi động (M1-03 AC 2), mỗi lúc chỉ một ảnh qua GPU.

Chạy (Colab: xem another/notebooks/emoca_colab.ipynb; WSL2 cũng dùng nguyên file này):
    EMOCA_TOKEN=<chuỗi bí mật> python services/emoca_svc/server.py \\
        --models-dir <emoca>/assets/EMOCA/models --model-name EMOCA_v2_lr_mse_20 [--port 8765]

An toàn:
* Mặc định chỉ nghe 127.0.0.1 (tunnel nối vào cổng cục bộ). Nghe địa chỉ khác mà không có token ⇒ từ chối chạy.
* Có token ⇒ mọi đường dẫn đòi `Authorization: Bearer <token>`. Token đọc từ biến môi trường, không nhận qua dòng lệnh.
* Riêng tư (ADR-009): ảnh chỉ nằm trong bộ nhớ trong một request; không ghi đĩa; log chỉ có method/đường dẫn/mã
  trạng thái và traceback lỗi (không có nội dung ảnh, không có header).

Giới hạn đã biết:
* Tuần tự hoá GPU bằng một khoá: nếu máy khách bỏ cuộc (timeout phía web, mặc định 60 s) thì request đó VẪN chạy tới
  hết và giữ khoá; request gửi lại sẽ xếp hàng sau nó. Không có cách huỷ an toàn một lời gọi encode/decode đang chạy
  trên GPU, nên chỉ ghi nhận ở đây (review ADR-015, D7). Độ trễ thật trên Colab chưa đo.
* Sau lỗi hết VRAM, gọi `backend.release_memory()` (nếu có) để trả bộ nhớ cache của PyTorch; request sau vẫn có thể
  OOM nếu ảnh/GPU thật sự không đủ — chưa đo trên GPU thật.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mock_server as base  # tầng HTTP + kiểm ảnh dùng chung với bản mock
from emoca_runner import DEFAULT_DETECT_MAX_SIDE_PX, EmocaBackend, EmocaRunner, RunnerError

ROOT = Path(__file__).resolve().parents[2]
LOOPBACK = ("127.0.0.1", "::1", "localhost")


class RealAnalyzer:
    """`analyzer(data, cfg, max_faces) -> (http_status, body)` cho `mock_server.make_handler`."""

    def __init__(self, runner: EmocaRunner) -> None:
        self.runner = runner
        self._gpu = threading.Lock()  # EmocaRunner không an toàn đa luồng; GPU nhỏ ⇒ tuần tự (max_concurrent = 1)

    def __call__(self, data: bytes, cfg: Dict[str, Any], max_faces: int) -> Tuple[int, Dict[str, Any]]:
        img, err = base.check_request(data, cfg, max_faces)
        if err is not None:
            return err
        with self._gpu:
            try:
                return 200, self.runner.analyze(img, max_faces)
            except RunnerError as e:
                return e.status, base._error(e.code, e.message, e.retryable)
            except Exception as e:  # noqa: BLE001 — mọi lỗi phải thành mã lỗi có nghĩa, không làm sập tiến trình
                traceback.print_exc()
                oom = "out of memory" in str(e).lower()
                if oom:
                    self._release_memory()
                msg = ("Hết VRAM khi suy luận (CUDA out of memory)." if oom
                       else f"Lỗi nội bộ EMOCA ({type(e).__name__}); xem nhật ký máy chủ.")
                return 503, base._error("service_unavailable", msg, retryable=True)

    def _release_memory(self) -> None:
        """Gọi hook `release_memory()` của backend (EmocaBackend: gc + torch.cuda.empty_cache). Hook lỗi chỉ được ghi
        vào nhật ký, không được che mã lỗi 503 gốc của request."""
        release = getattr(self.runner.backend, "release_memory", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001
                traceback.print_exc()


def build_server(cfg: Dict[str, Any], runner: EmocaRunner, token: Optional[str], port: Optional[int] = None):
    """Ráp ThreadingHTTPServer với bộ phân tích thật (tách ra để kiểm thử với runner giả)."""
    if cfg["host"] not in LOOPBACK and not token:
        raise SystemExit("[LOI] Nghe địa chỉ ngoài loopback mà không có token — từ chối chạy (đặt EMOCA_TOKEN).")
    return base.make_server(cfg, port, analyzer=RealAnalyzer(runner), mode="real", token=token,
                            health_info=runner.info)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emoca_svc THẬT (contract B1, cần GPU + EMOCA)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "services.yaml")
    ap.add_argument("--host", default=None, help="mặc định theo config (127.0.0.1)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--models-dir", required=True, help="thư mục chứa <model-name>/cfg.yaml (assets/EMOCA/models)")
    ap.add_argument("--model-name", default="EMOCA_v2_lr_mse_20")
    ap.add_argument("--mode", default="detail", choices=["detail"])
    ap.add_argument("--token-env", default="EMOCA_TOKEN", help="tên biến môi trường chứa token (mặc định EMOCA_TOKEN)")
    args = ap.parse_args(argv)

    cfg = base.load_config(args.config)
    if args.host:
        cfg["host"] = args.host
    token = os.environ.get(args.token_env) or None
    if token is None:
        print(f"[emoca_svc real] CẢNH BÁO: không có {args.token_env} — không xác thực. "
              "Chỉ chấp nhận khi chạy cục bộ, không mở ra Internet.", file=sys.stderr, flush=True)
    if cfg.get("max_concurrent", 1) > 1:
        print("[emoca_svc real] max_concurrent > 1 chưa hỗ trợ — vẫn chạy tuần tự từng ảnh.", file=sys.stderr, flush=True)

    print(f"[emoca_svc real] nạp {args.model_name} ({args.mode}) từ {args.models_dir} …", flush=True)
    backend = EmocaBackend(args.models_dir, args.model_name, args.mode)
    try:
        backend.load()
    except RunnerError as e:
        print(f"[emoca_svc real] LỖI nạp mô hình [{e.code}]: {e.message}", file=sys.stderr, flush=True)
        return 3
    runner = EmocaRunner(backend, base.CONTRACT_VERSION, cfg["min_face_side_px"],
                         detect_max_side_px=cfg.get("detect_max_side_px", DEFAULT_DETECT_MAX_SIDE_PX))
    info = backend.info()
    print(f"[emoca_svc real] đã nạp trong {info['load_seconds']} s · {info.get('device')} · torch {info.get('torch')} · "
          f"CUDA {info.get('cuda')} · VRAM sau nạp {info.get('vram_after_load_mb')} MB · commit "
          f"{info['model']['upstream_commit']}", flush=True)

    srv = build_server(cfg, runner, token, args.port)
    host, port = srv.server_address[:2]
    print(f"[emoca_svc real] http://{host}:{port}  (contract {base.CONTRACT_VERSION})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
