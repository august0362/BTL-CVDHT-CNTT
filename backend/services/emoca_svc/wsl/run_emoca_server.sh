#!/usr/bin/env bash
# Chay dich vu EMOCA THAT (services/emoca_svc/server.py) trong WSL2, nghe 127.0.0.1:8765. CHUA CHAY THAT (2026-09-25).
# WSL2 chuyen tiep localhost sang Windows => web tren Windows goi http://127.0.0.1:8765 (EMOCA_MODE=real,
# EMOCA_URL=http://127.0.0.1:8765, hoac bang Cai dat cua web). Anh khong roi khoi may (loopback, khong can xac nhan gui anh).
# Chuyen tiep localhost CHUA DO tren may nay — thu truoc (khong can GPU): bash ~/emoca_wsl/bin/smoke_test_emoca.sh --forward-only
#
# Dung (trong Ubuntu):  bash ~/emoca_wsl/bin/run_emoca_server.sh        (Ctrl+C de dung)
# Bien: EMOCA_PORT (8765) | EMOCA_TOKEN tuy chon: dat thi moi request phai co "Authorization: Bearer <token>" (web
#       doc cung bien EMOCA_TOKEN); loopback khong bat buoc token | MODEL_NAME (EMOCA_v2_lr_mse_20) |
#       VRAM_SAMPLE_SEC (5) chu ky lay mau /v1/health | EMOCA_CUDNN auto|0|1 (auto: theo out/cudnn_probe.txt, chua co
#       thi thu truoc; 0 = tat cuDNN — du phong khi cuDNN loi/tu huy tren may nay, xem common.sh::cudnn_probe).
# Log: ~/emoca_wsl/out/server.log (xoay vong khi > 50 MB -> server.log.1).
# VRAM (H1, docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md): lay mau GET /v1/health (python3 cua Ubuntu, khong torch => khong
#      tao CUDA context rieng) -> ~/emoca_wsl/out/server_vram_health.json: min(gpu_free_mb), max(vram_reserved_peak_mb),
#      max(vram_peak_mb); truong server chua bao => "khong co". KHONG dung nvidia-smi memory.used (luon 0 duoi WDDM).
#      4 GB la diem rui ro CHUA DO.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

VRAM_SAMPLE_SEC="${VRAM_SAMPLE_SEC:-5}"
require_wsl2
require_gpu
require_env
require_assets
cudnn_resolve   # export EMOCA_CUDNN=0|1 + EMOCA_CUDNN_PROBE => server.py (emoca_runner.py) doc
if (echo > "/dev/tcp/127.0.0.1/$EMOCA_PORT") 2>/dev/null; then
  die "Cong $EMOCA_PORT dang co tien trinh khac nghe (server cu? mock_server tren Windows?). Dat EMOCA_PORT khac hoac tat no."
fi
if [[ -z "${EMOCA_TOKEN:-}" ]]; then
  log "Khong co EMOCA_TOKEN — khong xac thuc; chap nhan vi chi nghe 127.0.0.1 (khong mo ra mang)."
fi

LOG="$OUT_DIR/server.log"
if [[ -f "$LOG" ]] && (( $(stat -c %s "$LOG") > 52428800 )); then mv -f "$LOG" "$LOG.1"; fi

# Lay mau VRAM qua /v1/health cho toi khi script nay thoat (theo doi pid $$). Ghi de JSON sau moi mau.
VRAM_JSON="$OUT_DIR/server_vram_health.json"
python3 - "$EMOCA_PORT" "$VRAM_JSON" "$$" "$VRAM_SAMPLE_SEC" <<'PY' &
import json, os, sys, time, urllib.request
port, out, pid, every = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
token = os.environ.get("EMOCA_TOKEN")
NA = "khong co (server khong bao truong nay)"
agg = {"source": "GET /v1/health (torch trong tien trinh server; KHONG phai nvidia-smi)", "interval_s": every,
       "started": time.strftime("%F %T"), "samples": 0, "errors": 0, "min_gpu_free_mb": None, "gpu_total_mb": None,
       "max_vram_reserved_peak_mb": None, "max_vram_reserved_mb": None, "max_vram_peak_mb": None,
       "vram_after_load_mb": None, "requests_served": None, "last_sample": None}
def upd(key, val, fn):
    if isinstance(val, (int, float)):
        agg[key] = val if agg[key] is None else fn(agg[key], val)
def dump():
    view = {k: (NA if v is None and (k.startswith(("min_", "max_")) or k in ("gpu_total_mb", "vram_after_load_mb")) else v)
            for k, v in agg.items()}
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(view, f, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
while True:
    try:
        os.kill(pid, 0)
    except OSError:
        break
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/health",
                                     headers={"Authorization": "Bearer " + token} if token else {})
        with urllib.request.urlopen(req, timeout=2) as r:
            h = json.loads(r.read())
        if h.get("loaded"):
            agg["samples"] += 1
            agg["last_sample"] = time.strftime("%H:%M:%S")
            agg["requests_served"] = h.get("requests_served")
            upd("min_gpu_free_mb", h.get("gpu_free_mb"), min)
            upd("gpu_total_mb", h.get("gpu_total_mb"), max)
            upd("max_vram_reserved_peak_mb", h.get("vram_reserved_peak_mb"), max)
            upd("max_vram_reserved_mb", h.get("vram_reserved_mb"), max)
            upd("max_vram_peak_mb", h.get("vram_peak_mb"), max)
            upd("vram_after_load_mb", h.get("vram_after_load_mb"), max)
            dump()
    except Exception:
        agg["errors"] += 1
    time.sleep(every)
dump()
PY
SMP_PID=$!
trap 'kill "$SMP_PID" 2>/dev/null || true' EXIT

log "server.py: 127.0.0.1:$EMOCA_PORT, mo hinh $MODEL_NAME, cuDNN $([[ $EMOCA_CUDNN == 1 ]] && echo bat || echo TAT), log $LOG, VRAM -> $VRAM_JSON (moi ${VRAM_SAMPLE_SEC} s)"
log "Tu Windows: curl http://127.0.0.1:$EMOCA_PORT/v1/health   (co token thi them -H \"Authorization: Bearer ...\")"
set +u
# shellcheck disable=SC1091
source "$CONDA_DIR/bin/activate" "$EMOCA_ENV"
set -u
apply_wsl_ldpath   # /usr/lib/wsl/lib o dau LD_LIBRARY_PATH (cuDNN can libcuda.so) — lai sau activate
log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
cd "$SVC_DIR"
EMOCA_CUDNN="$EMOCA_CUDNN" EMOCA_CUDNN_PROBE="$EMOCA_CUDNN_PROBE" PYTHONUNBUFFERED=1 python -u server.py --config "$SVC_DIR/configs/services.yaml" --host 127.0.0.1 \
  --port "$EMOCA_PORT" --models-dir "$MODELS_DIR" --model-name "$MODEL_NAME" 2>&1 | tee -a "$LOG"
