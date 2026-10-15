#!/usr/bin/env bash
# Khởi động container EMOCA: kiểm/tải tài sản FLAME+EMOCA (chỉ khi người dùng đã đồng ý giấy phép) rồi chạy server.py.
# Tài sản nằm trong volume gắn ở $EMOCA_DIR/assets => chỉ tải lần đầu.
set -euo pipefail

EMOCA_DIR="${EMOCA_DIR:-/opt/emoca}"
MODEL_NAME="${MODEL_NAME:-EMOCA_v2_lr_mse_20}"
EMOCA_PORT="${EMOCA_PORT:-8765}"
MODELS_DIR="$EMOCA_DIR/assets/EMOCA/models"
TEXTURE="$EMOCA_DIR/assets/FLAME/texture/FLAME_albedo_from_BFM.npz"
FLAME_ALBEDO_DRIVE_ID="${FLAME_ALBEDO_DRIVE_ID:-1poa4GTzjyArYJ7LhbQN1Gqp2U2zjwiD7}"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

assets_ok() {
  [[ -s "$MODELS_DIR/$MODEL_NAME/cfg.yaml" ]] \
    && [[ -n "$(find "$MODELS_DIR/$MODEL_NAME" -name '*.ckpt' -size +1M -print -quit 2>/dev/null)" ]] \
    && [[ -n "$(find "$EMOCA_DIR/assets/FLAME" -type f -size +1M -print -quit 2>/dev/null)" ]]
}

# Như texture_required trong wsl/common.sh: cần texture trừ khi mọi stage của cfg.yaml đặt use_texture: false.
texture_required() {
  python - "$MODELS_DIR/$MODEL_NAME/cfg.yaml" <<'PY'
import sys, yaml
root = yaml.compose(open(sys.argv[1], encoding="utf-8"))
vals = []
for _, st in getattr(root, "value", []):
    if not isinstance(st, yaml.MappingNode):
        continue
    model = next((v for k, v in st.value if k.value == "model" and isinstance(v, yaml.MappingNode)), None)
    if model is not None:
        vals.append(next((str(v.value).lower() for k, v in model.value if k.value == "use_texture"), "missing"))
sys.exit(1 if vals and all(v == "false" for v in vals) else 0)
PY
}

if ! assets_ok; then
  if [[ "${ACCEPT_FLAME_EMOCA_LICENSES:-0}" != "1" ]]; then
    cat <<'MSG'
================================================================================
CHƯA CÓ TÀI SẢN FLAME/EMOCA. Hai bộ này có giấy phép nghiên cứu phi thương mại; bạn phải TỰ đăng ký và đồng ý:
  https://flame.is.tue.mpg.de    và    https://emoca.is.tue.mpg.de
Sau đó đặt ACCEPT_FLAME_EMOCA_LICENSES=1 trong file .env (cạnh docker-compose.yml) rồi chạy lại:
  docker compose --profile real up
================================================================================
MSG
    exit 10
  fi
  log "Tải tài sản FLAME/EMOCA vào volume (vài GB, chỉ lần đầu) ..."
  cd "$EMOCA_DIR/gdl_apps/EMOCA/demos"
  set +o pipefail
  (yes || true) | bash download_assets.sh
  set -o pipefail
  assets_ok || { log "[LỖI] tải xong vẫn thiếu $MODELS_DIR/$MODEL_NAME hoặc assets/FLAME"; exit 1; }
fi

if [[ ! -s "$TEXTURE" ]] && texture_required; then
  log "Tải texture FLAME_albedo_from_BFM.npz (Google Drive) ..."
  mkdir -p "$(dirname "$TEXTURE")"
  gdown "https://drive.google.com/uc?id=$FLAME_ALBEDO_DRIVE_ID" -O "$TEXTURE" || true
  if [[ ! -s "$TEXTURE" ]]; then
    rm -f "$TEXTURE"
    log "[LỖI] tải texture thất bại. Tải tay FLAME_albedo_from_BFM.npz rồi chép vào volume:"
    log "  docker compose cp FLAME_albedo_from_BFM.npz emoca:$TEXTURE"
    exit 11
  fi
fi

# Docker Desktop (WSL2) đặt libcuda của driver Windows ở /usr/lib/wsl/lib; cuDNN cần thấy libcuda.so (wsl/common.sh).
if [[ -d /usr/lib/wsl/lib ]]; then export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
# Không chạy phép thử cuDNN như setup WSL => mặc định tắt cuDNN (an toàn, chậm hơn chút). Đặt EMOCA_CUDNN=1 để bật.
export EMOCA_CUDNN="${EMOCA_CUDNN:-0}"

# Nghe loopback: container web dùng chung network namespace (network_mode: service:emoca) => ảnh không rời máy,
# server.py không đòi token (ADR-015/018).
cd /app/backend/services/emoca_svc
log "server.py 127.0.0.1:$EMOCA_PORT, mô hình $MODEL_NAME, EMOCA_CUDNN=$EMOCA_CUDNN"
exec python -u server.py --config /app/backend/configs/services.yaml --host 127.0.0.1 --port "$EMOCA_PORT" \
  --models-dir "$MODELS_DIR" --model-name "$MODEL_NAME"
