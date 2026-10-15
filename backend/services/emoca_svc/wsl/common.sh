# shellcheck shell=bash
# Phan dung chung cho setup_emoca_wsl.sh / smoke_test_emoca.sh / run_emoca_server.sh (source, khong chay truc tiep).
# Chay BEN TRONG Ubuntu WSL2. Moi duong dan lam viec nam trong he file goc cua WSL ($HOME), KHONG o /mnt/c
# (cham, CRLF, duong dan co dau cach + tieng Viet) — duoc chan ngay khi source (M6).
#
# Bien moi truong ghi de duoc (mac dinh trong ngoac):
#   EMOCA_WSL_HOME   thu muc lam viec: svc/, bin/, out/, state/          (~/emoca_wsl)
#   EMOCA_DIR        ban clone EMOCA upstream                             (~/emoca)
#   CONDA_DIR        Miniconda                                            (~/miniconda3)
#   EMOCA_ENV        ten env conda                                        (work38)
#   EMOCA_COMMIT     commit upstream ma bo patch viet cho                 (e0be0db)
#   MODEL_NAME       mo hinh EMOCA                                        (EMOCA_v2_lr_mse_20)
#   EMOCA_PORT       cong server                                          (8765)
#   EMOCA_HOST_DRIVE o Windows chua dia ao ext4.vhdx cua distro           (/mnt/c) — gioi han dung luong THAT
#   EMOCA_CUDNN      auto | 0 | 1 — cuDNN cho EMOCA                       (auto = theo out/cudnn_probe.txt)
#   EMOCA_CUDNN_REPROBE=1  chay lai phep thu cuDNN du da co out/cudnn_probe.txt
#   EMOCA_WSL_LIB_DIR thu vien CUDA cua driver Windows trong WSL, dat vao DAU LD_LIBRARY_PATH  (/usr/lib/wsl/lib)
#   EMOCA_REQUIRE_TEXTURE auto | 0 | 1 — co bat buoc FLAME_albedo_from_BFM.npz  (auto = theo use_texture trong cfg.yaml)
#   EMOCA_CFG_PYTHON python co PyYAML de doc cfg.yaml (mac dinh: python3 he thong, roi python cua env $EMOCA_ENV)
#
# Hoi phuc sau khi WSL tat dot ngot (docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md, H4): tep moi tao chua kip ghi xuong dia
# co the thanh 0 byte. Vi vay: `stage` sync truoc/sau khi ghi moc, moc 0 byte coi nhu chua co, va cac ham *_ok ben
# duoi kiem bang CHAY THAT (khong chi -x/-f/-d).

EMOCA_WSL_HOME="${EMOCA_WSL_HOME:-$HOME/emoca_wsl}"
EMOCA_DIR="${EMOCA_DIR:-$HOME/emoca}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
EMOCA_ENV="${EMOCA_ENV:-work38}"
EMOCA_COMMIT="${EMOCA_COMMIT:-e0be0db}"
MODEL_NAME="${MODEL_NAME:-EMOCA_v2_lr_mse_20}"
EMOCA_PORT="${EMOCA_PORT:-8765}"
EMOCA_HOST_DRIVE="${EMOCA_HOST_DRIVE:-/mnt/c}"
EMOCA_CUDNN="${EMOCA_CUDNN:-auto}"
EMOCA_WSL_LIB_DIR="${EMOCA_WSL_LIB_DIR:-/usr/lib/wsl/lib}"
EMOCA_REQUIRE_TEXTURE="${EMOCA_REQUIRE_TEXTURE:-auto}"

log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die()  { printf '[LOI] %s\n' "$*" >&2; exit "${DIE_CODE:-1}"; }

# M6: khong lam viec tren o Windows.
for _v in EMOCA_WSL_HOME EMOCA_DIR CONDA_DIR; do
  case "${!_v}/" in
    /mnt/*) die "$_v=${!_v} nam tren o Windows (/mnt/...). Dat vao he file cua WSL, vd \$HOME/... (mac dinh)." ;;
  esac
done
unset _v
case "$EMOCA_CUDNN" in auto|0|1) ;; *) die "EMOCA_CUDNN=$EMOCA_CUDNN khong hop le (chi nhan auto, 0 hoac 1)." ;; esac
case "$EMOCA_REQUIRE_TEXTURE" in auto|0|1) ;; *) die "EMOCA_REQUIRE_TEXTURE=$EMOCA_REQUIRE_TEXTURE khong hop le (chi nhan auto, 0 hoac 1)." ;; esac

SVC_DIR="$EMOCA_WSL_HOME/svc"          # noi giai nen notebooks/emoca_svc_bundle.zip
BIN_DIR="$EMOCA_WSL_HOME/bin"          # ban sao cac script nay (de goi ngan: bash ~/emoca_wsl/bin/...)
OUT_DIR="$EMOCA_WSL_HOME/out"          # log, so do thoi gian, phien ban, ket qua smoke test
STATE_DIR="$EMOCA_WSL_HOME/state"      # moc da-xong cua tung chang (idempotent)
ENV_PREFIX="$CONDA_DIR/envs/$EMOCA_ENV"
MODELS_DIR="$EMOCA_DIR/assets/EMOCA/models"
TEXTURE="$EMOCA_DIR/assets/FLAME/texture/FLAME_albedo_from_BFM.npz"
TIMINGS="$OUT_DIR/timings.tsv"
CUDNN_PROBE="$OUT_DIR/cudnn_probe.txt"   # ket qua phep thu cuDNN (key=value); emoca_runner.py doc qua EMOCA_CUDNN_PROBE

mkdir -p "$OUT_DIR" "$STATE_DIR"

# Day du lieu xuong dia (WSL co the tat dot ngot — H4). Khong co `sync` thi bo qua, khong lam hong script.
durable_sync() { sync 2>/dev/null || true; }

# Khoa tai san (docs/qa/REVIEW_PREFETCH_2026-09-25.md, H2): prefetch_assets.sh va chang download_assets cua
# setup_emoca_wsl.sh KHONG duoc chay dong thoi (hai curl/wget cung ghi mot tep => du lieu trung/hong).
# assets_lock TEN: giu khoa toi khi tien trinh (hoac subshell cua `stage`) goi no thoat. Dang co nguoi giu => thoat 11.
# Uu tien flock (util-linux, co san tren Ubuntu). Khong co flock (vd Git Bash khi chay test tren Windows) => thu muc
# khoa chua "pid boot_id"; khoa cua tien trinh da chet hoac cua lan khoi dong truoc (WSL tat dot ngot) bi coi la cu.
# $ASSETS_LOCK.who = ai dang giu (ten, pid, luc); xoa khi tra khoa bang trap EXIT NOI vao trap san co (khong de).
ASSETS_LOCK="$STATE_DIR/assets.lock"

# _assets_lock_holder [PID_CHU_KHOA]: mo ta nguoi giu tu .who. Pid trong .who khong con chay, hoac khac pid chu khoa
# (nhanh thu muc khoa) => ghi ro "lan giu gan nhat (co the da cu)" (.who co the con tu lan truoc / chu khoa khong qua ham nay).
_assets_lock_holder() {
  local owner="${1:-}" line wpid
  line="$(cat "$ASSETS_LOCK.who" 2>/dev/null || true)"
  if [[ -z "$line" ]]; then echo "khong ro ai giu${owner:+, pid $owner}"; return 0; fi
  wpid="$(awk '{ for (i = 1; i < NF; i++) if ($i == "pid") { print $(i + 1); exit } }' <<< "$line")"
  if [[ -n "$wpid" && ( -z "$owner" || "$wpid" == "$owner" ) ]] && kill -0 "$wpid" 2>/dev/null; then
    echo "$line"
  else
    echo "lan giu gan nhat (co the da cu): $line"
  fi
}

# Tra khoa khi thoat: xoa .who CUA MINH (khong xoa .who cua nguoi khac) va thu muc khoa (nhanh khong-flock).
# Nhanh flock: khoa tu nha khi fd 9 dong luc tien trinh thoat.
_assets_lock_release() {
  if [[ -n "${ASSETS_LOCK_WHO_LINE:-}" && "$(cat "$ASSETS_LOCK.who" 2>/dev/null)" == "$ASSETS_LOCK_WHO_LINE" ]]; then
    rm -f "$ASSETS_LOCK.who"
  fi
  if [[ "${ASSETS_LOCK_DIR_OWNED:-0}" == 1 ]]; then rm -rf "$ASSETS_LOCK.d"; fi
  return 0
}

# Noi _assets_lock_release vao trap EXIT san co. Trong subshell (vd `stage`), `trap -p` van in trap cua CHA (khong chay
# o subshell) => chi noi khi dang o tien trinh chinh, tranh chay nham don dep cua cha.
_assets_lock_trap() {
  ASSETS_LOCK_PREV_TRAP=""
  if [[ "$BASHPID" == "$$" ]]; then
    eval "set -- $(trap -p EXIT)"
    ASSETS_LOCK_PREV_TRAP="${3:-}"
  fi
  trap '_assets_lock_release; eval "$ASSETS_LOCK_PREV_TRAP"' EXIT
}

assets_lock() {
  local who="${1:-$(basename "$0")}" d="$ASSETS_LOCK.d" boot pid="" oboot=""
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$ASSETS_LOCK"
    flock -n 9 || DIE_CODE=11 die "Dang co tien trinh khac giu khoa tai san ($(_assets_lock_holder)). Doi no xong (hoac dung no) roi chay lai. Khoa: $ASSETS_LOCK"
  else
    boot="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo -)"
    if ! mkdir "$d" 2>/dev/null; then
      read -r pid oboot < "$d/owner" 2>/dev/null || true
      if [[ -n "$pid" && "$oboot" == "$boot" ]] && kill -0 "$pid" 2>/dev/null; then
        DIE_CODE=11 die "Dang co tien trinh khac giu khoa tai san ($(_assets_lock_holder "$pid")). Doi no xong (hoac dung no) roi chay lai. Khoa: $d"
      fi
      log "Khoa tai san cu (pid ${pid:-?} khong con chay) — chiem lai: $d"
      rm -rf "$d"; mkdir "$d" || DIE_CODE=11 die "Khong tao duoc khoa $d"
    fi
    echo "$BASHPID $boot" > "$d/owner"
    ASSETS_LOCK_DIR_OWNED=1
  fi
  ASSETS_LOCK_WHO_LINE="$who pid $BASHPID $(date '+%F %T')"
  echo "$ASSETS_LOCK_WHO_LINE" > "$ASSETS_LOCK.who"
  _assets_lock_trap
}

# NGUYEN NHAN GOC loi cuDNN (do tren may du an bang log cua cudnn_probe, 2026-09-25): cuDNN 8 dlopen "libcuda.so" (KHONG
# duoi phien ban); trong WSL2 tep nay chi co o /usr/lib/wsl/lib, con ld cache chi co libcuda.so.1 => cuDNN abort
# (CUDNN_STATUS_NOT_INITIALIZED / SIGABRT). Sua: dat $EMOCA_WSL_LIB_DIR vao DAU LD_LIBRARY_PATH (da do: conv2d CO cuDNN
# 0.73 ms/lan so voi 1.12 ms khi tat). Giu cac muc san co (chi bo ban trung lap cua chinh thu muc nay).
# EMOCA_NO_WSL_LDPATH=1 => KHONG them (chi de cudnn_probe chung minh nguyen nhan). Thu muc khong ton tai => khong them.
apply_wsl_ldpath() {
  local d="$EMOCA_WSL_LIB_DIR" out="" p
  local -a parts=()
  IFS=: read -r -a parts <<< "${LD_LIBRARY_PATH:-}" || true
  for p in "${parts[@]}"; do
    if [[ -n "$p" && "$p" != "$d" ]]; then out="${out:+$out:}$p"; fi
  done
  if [[ "${EMOCA_NO_WSL_LDPATH:-0}" != 1 && -d "$d" ]]; then out="$d${out:+:$out}"; fi
  if [[ -n "$out" ]]; then export LD_LIBRARY_PATH="$out"; else unset LD_LIBRARY_PATH; fi
}
apply_wsl_ldpath   # ngay khi source: moi tien trinh con (server.py, python cua env) thua huong

# Chay mot lenh trong env conda (giong `source activate work38; ...` cua notebook). Subshell: khong lam ban shell goi.
# Script activate cua conda dung bien chua dat => tat `set -u` trong luc activate.
envrun() {
  (
    set +u
    # shellcheck disable=SC1091
    source "$CONDA_DIR/bin/activate" "$EMOCA_ENV" || exit 97
    set -u
    apply_wsl_ldpath   # lai SAU activate (activate.d cua env co the doi LD_LIBRARY_PATH)
    "$@"
  )
}

# Xoa moc cua cac chang (ten key, cho phep glob) — dung khi chang truoc duoc lam lai nen chang sau phai chay lai.
invalidate_markers() {
  local k
  for k in "$@"; do
    # shellcheck disable=SC2086
    rm -f "$STATE_DIR"/$k.done
  done
}

# stage TEN LENH...: chay mot chang, ghi thoi gian that vao $TIMINGS (ngay gio, script, chang, giay, trang thai).
# Chang da co moc (KHONG rong) trong $STATE_DIR thi bo qua (xoa moc hoac dat EMOCA_WSL_REDO=1 de chay lai).
# Moc 0 byte (WSL tat dot ngot) bi coi nhu chua co. Moc chi ghi SAU khi du lieu cua chang da sync xuong dia.
# GOI `stage` NHU MOT LENH DON (khong dat trong if / && / || / $(...)): o cac ngu canh do bash tat `set -e` cho
# toan bo ben trong, ke ca subshell, nen loi giua chang se KHONG dung chang lai. tests/test_wsl_scripts.py kiem dieu nay.
stage() {
  local name="$1"; shift
  local key="${name//[^A-Za-z0-9_.-]/_}"
  local mark="$STATE_DIR/$key.done"
  if [[ -e "$mark" && ! -s "$mark" ]]; then
    log "[canh bao] moc cua $name rong (0 byte — co the do WSL tat dot ngot) => chay lai chang nay"
    rm -f "$mark"
  fi
  if [[ -s "$mark" && "${EMOCA_WSL_REDO:-0}" != "1" ]]; then
    log "[bo qua] $name (da xong luc $(cat "$mark"))"
    return 0
  fi
  log "[bat dau] $name"
  local t0 rc
  t0=$(date +%s)
  # KHONG viet `"$@" || rc=$?`: trong ngu canh `||` bash tat `set -e` cho ca ham ben trong => loi giua chang bi nuot.
  set +e
  ( set -euo pipefail; "$@" )
  rc=$?
  set -e
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$(basename "$0")" "$name" "$dt" \
    "$([[ $rc == 0 ]] && echo ok || echo "FAIL($rc)")" >> "$TIMINGS"
  if [[ $rc != 0 ]]; then
    log "[HONG] $name sau ${dt} s (ma $rc). Xem log o tren; sua xong chay lai script — cac chang da xong se duoc bo qua."
    exit "$rc"
  fi
  durable_sync                                   # du lieu cua chang xuong dia TRUOC khi co moc
  date '+%F %T' > "$mark.tmp"
  mv -f "$mark.tmp" "$mark"
  durable_sync
  log "[xong] $name — ${dt} s"
}

# ---- kiem bang CHAY THAT (H4): tra 0 khi dung duoc, khong in gi
conda_ok()      { [[ -s "$CONDA_DIR/bin/conda" ]] && "$CONDA_DIR/bin/conda" --version >/dev/null 2>&1; }
mamba_ok()      { [[ -s "$CONDA_DIR/bin/mamba" ]] && "$CONDA_DIR/bin/mamba" --version >/dev/null 2>&1; }
env_python_ok() {
  [[ -s "$ENV_PREFIX/bin/python" ]] \
    && "$ENV_PREFIX/bin/python" -c 'import sys, json, ssl; sys.exit(0 if sys.version_info[:2] == (3, 8) else 1)' >/dev/null 2>&1
}
# Chi repo chinh: --ignore-submodules=all vi submodule video (spectre va con cua no) co the hong sau khi WSL sap ma
# khong can cho suy luan anh; emonet/SwinTransformer/face-parsing duoc kiem rieng o chang pull_submodules.
repo_ok() {
  git -C "$EMOCA_DIR" rev-parse --verify -q "$EMOCA_COMMIT^{commit}" >/dev/null 2>&1 \
    && git -C "$EMOCA_DIR" status --porcelain --ignore-submodules=all >/dev/null 2>&1
}
# In tung van de cua tai san (rong = dung duoc). Kiem noi dung, khong chi su ton tai (M1 + H4).
assets_problems() {
  local d="$MODELS_DIR/$MODEL_NAME"
  [[ -s "$d/cfg.yaml" ]] || echo "thieu hoac rong: $d/cfg.yaml"
  [[ -n "$(find "$d" -name '*.ckpt' -size +1M -print -quit 2>/dev/null)" ]] || echo "khong co checkpoint *.ckpt > 1 MB trong $d"
  [[ -n "$(find "$EMOCA_DIR/assets/FLAME" -type f -size +1M -print -quit 2>/dev/null)" ]] \
    || echo "assets/FLAME khong co tep nao > 1 MB (FLAME chua giai nen?)"
  [[ -n "$(find "$EMOCA_DIR" -path '*EMOCA_test_example_data/images/*' -type f -size +0 -print -quit 2>/dev/null)" ]] \
    || echo "khong thay anh mau EMOCA_test_example_data/images/**"
  local z
  z="$(find "$d" "$EMOCA_DIR/assets/FLAME" -type f -empty -print 2>/dev/null | head -3)"
  [[ -z "$z" ]] || echo "tep 0 byte (WSL tat dot ngot?): $(echo "$z" | tr '\n' ' ')"
}

require_wsl2() {
  grep -qi microsoft /proc/version 2>/dev/null || die "Khong phai WSL (khong thay 'microsoft' trong /proc/version)."
  [[ "$(uname -r)" == *WSL2* || "$(uname -r)" == *microsoft-standard* ]] \
    || die "Can WSL2 (GPU/CUDA khong co tren WSL1). Tren Windows: wsl --set-version <distro> 2"
  case "$PWD/" in /mnt/*) log "Luu y: dang dung o $PWD (o Windows). Script chi LAM VIEC trong $EMOCA_WSL_HOME, $EMOCA_DIR.";; esac
}

# nvidia-smi trong WSL2 do driver Windows cung cap (/usr/lib/wsl/lib). KHONG cai driver NVIDIA ben trong Ubuntu.
# Da do tren may du an: `nvidia-smi` KHONG doi so bi segfault; -L va --query-gpu chay duoc. memory.used luon 0 duoi
# WDDM => KHONG dung de do VRAM (dung /v1/health cua server: gpu_free_mb, vram_reserved_peak_mb).
require_gpu() {
  local smi lst
  smi="$(command -v nvidia-smi || true)"
  [[ -z "$smi" && -x /usr/lib/wsl/lib/nvidia-smi ]] && smi=/usr/lib/wsl/lib/nvidia-smi
  [[ -n "$smi" ]] || die "Khong thay nvidia-smi trong WSL. Cap nhat driver NVIDIA tren Windows (ban co ho tro WSL), roi 'wsl --shutdown'."
  lst="$("$smi" -L 2>&1)" || die "nvidia-smi -L loi: $(printf '%s' "$lst" | head -3)"
  grep -q GPU <<<"$lst" || die "nvidia-smi khong liet ke GPU nao: $(printf '%s' "$lst" | head -3)"
  NVSMI="$smi"
  log "GPU: $("$smi" --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
}

require_env() {
  env_python_ok || die "Env $EMOCA_ENV ($ENV_PREFIX) chua co hoac hong (python khong chay). Chay setup_emoca_wsl.sh truoc."
  [[ -s "$SVC_DIR/server.py" ]] || die "Chua co ma dich vu o $SVC_DIR. Chay setup_emoca_wsl.sh truoc."
}

# texture_required: tra 0 = PHAI co $TEXTURE, 1 = khong can; dat TEXTURE_REASON (de log). Can cu (upstream e0be0db):
# FLAMETex chi duoc dung khi DECA.uses_texture() (gdl/models/DECA.py:2846-2849: config.use_texture, THIEU khoa => True);
# load_model lay cfg[stage] cua cfg.yaml (gdl_apps/EMOCA/utils/load.py:168-176) — tex_path chi duoc GAN (load.py:142).
# auto: doc model.use_texture cua MOI stage trong $MODELS_DIR/$MODEL_NAME/cfg.yaml bang PyYAML ($EMOCA_CFG_PYTHON, hoac
# python3 he thong, hoac python cua env). Chi khi MOI stage ghi ro false => khong can. Thieu cfg / khong doc duoc /
# thieu khoa / true => can (an toan).
texture_required() {
  local cfg="$MODELS_DIR/$MODEL_NAME/cfg.yaml" py="" c out
  case "$EMOCA_REQUIRE_TEXTURE" in
    1) TEXTURE_REASON="dat tay EMOCA_REQUIRE_TEXTURE=1"; return 0 ;;
    0) TEXTURE_REASON="dat tay EMOCA_REQUIRE_TEXTURE=0"; return 1 ;;
  esac
  if [[ ! -s "$cfg" ]]; then TEXTURE_REASON="auto: khong co $cfg => can texture (an toan)"; return 0; fi
  for c in ${EMOCA_CFG_PYTHON:+"$EMOCA_CFG_PYTHON"} python3 "$ENV_PREFIX/bin/python"; do
    if "$c" -c 'import yaml' >/dev/null 2>&1; then py="$c"; break; fi
  done
  if [[ -z "$py" ]]; then TEXTURE_REASON="auto: khong co python nao co PyYAML de doc $cfg => can texture (an toan)"; return 0; fi
  out="$("$py" - "$cfg" 2>&1 <<'PY'
import sys, yaml
root = yaml.compose(open(sys.argv[1], encoding="utf-8"))
if not isinstance(root, yaml.MappingNode):
    print("loi: cfg.yaml khong phai mapping"); sys.exit(0)
res = []
for k, st in root.value:
    if not isinstance(st, yaml.MappingNode):
        continue
    model = next((v for kk, v in st.value if kk.value == "model" and isinstance(v, yaml.MappingNode)), None)
    if model is None:
        continue
    node = next(((kk, v) for kk, v in model.value if kk.value == "use_texture"), None)
    if node is None:
        res.append(f"{k.value}=thieu")
    else:
        res.append(f"{k.value}={str(node[1].value).lower()}@{node[0].start_mark.line + 1}")
if res and all("=false@" in r for r in res):
    print("OFF " + ",".join(r.split("@")[1] for r in res) + " " + " ".join(res))
else:
    print(" ".join(res) if res else "loi: cfg.yaml khong co muc nao chua model")
PY
)" || out="loi: $out"
  if [[ "$out" == "OFF "* ]]; then
    read -r _ lines out <<<"$out"
    TEXTURE_REASON="auto: use_texture=false o $cfg:$lines ($out)"
    return 1
  fi
  TEXTURE_REASON="auto: $cfg: ${out:-rong} => can texture"
  return 0
}

require_assets() {
  local p
  p="$(assets_problems)"
  [[ -z "$p" ]] || die "Tai san EMOCA chua dung duoc (setup_emoca_wsl.sh, can ACCEPT_FLAME_EMOCA_LICENSES=1):
$p"
  if [[ ! -s "$TEXTURE" ]]; then
    if texture_required; then
      die "Thieu texture $TEXTURE (xem huong dan tai tay trong setup_emoca_wsl.sh). Ly do can: $TEXTURE_REASON"
    fi
    log "[canh bao] bo qua texture: $TEXTURE chua co nhung khong can — $TEXTURE_REASON"
  fi
}

# ---- cuDNN (do tren may du an, env work38: torch 1.11.0+cu115, cudnn 8302, RTX 3050 sm_86): matmul va conv2d KHONG
# cuDNN chay; conv2d CO cuDNN loi CUDNN_STATUS_NOT_INITIALIZED / INTERNAL_ERROR, co lan tien trinh tu huy (exit 134,
# SIGABRT). Nguyen nhan goc DA TIM RA: thieu /usr/lib/wsl/lib trong LD_LIBRARY_PATH (xem apply_wsl_ldpath o tren) —
# da sua mac dinh. Du phong van giu: probe VAN hong du da co LD_LIBRARY_PATH => tat cuDNN (EMOCA_CUDNN=auto -> 0).
# Da do sau khi sua: phan lon lan chay OK (0.60-0.66 ms) nhung CHAP CHON "Unable to find a valid cuDNN algorithm to run
# convolution" (2 lan, nguyen nhan chua ro) => luot "on" thu toi da CUDNN_PROBE_ATTEMPTS lan, 1 lan OK la dat; server
# con tu tat cuDNN + thu lai 1 lan khi gap loi cuDNN luc chay (emoca_runner.py::_cudnn_retry).
# KHONG them torch/lib vao LD_LIBRARY_PATH (da do: gay loi "Unable to find a valid cuDNN algorithm" nhat quan).
CUDNN_PROBE_TIMEOUT="${CUDNN_PROBE_TIMEOUT:-300}"
CUDNN_PROBE_ATTEMPTS="${CUDNN_PROBE_ATTEMPTS:-3}"

# _cudnn_err LOG MA: tom tat loi cua mot lan thu (dong loi dau tien + SIGABRT/timeout).
_cudnn_err() {
  local e="" pat
  # Uu tien loi cu the (nguyen nhan goc) truoc dong "...Error: ..." chung chung dung truoc no tren cung dong.
  for pat in 'libcuda[.]so[^ ]*: cannot open shared object file' 'CUDNN_STATUS_[A-Z_]+' '[A-Za-z]*Error: .{0,120}'; do
    e="$(grep -m1 -oE "$pat" "$1" 2>/dev/null | head -1 || true)"
    if [[ -n "$e" ]]; then break; fi
  done
  case "$2" in 124|137) e="${e:-timeout ${CUDNN_PROBE_TIMEOUT} s}" ;; 134) e="${e:+$e; }SIGABRT (exit 134)" ;; esac
  printf '%s' "$e"
}

# cudnn_probe: thu conv2d + BatchNorm (giong lop dau ResNet cua EMOCA), moi lan trong TIEN TRINH CON rieng co timeout
# (tien trinh con co the tu huy ma khong keo script theo). KHONG die. Ghi $CUDNN_PROBE (key=value). Thu tu:
#   on         CO cuDNN, CO sua LD_LIBRARY_PATH (cach server/smoke chay that): toi da CUDNN_PROBE_ATTEMPTS lan, moi lan
#              mot tien trinh con MOI, dung o lan OK dau tien => cudnn_ok=1 (ghi attempts_on, failures_on)
#   off        KHONG cuDNN (du phong)
#   no_ldpath  CO cuDNN, KHONG co $EMOCA_WSL_LIB_DIR trong LD_LIBRARY_PATH — chan doan (gay abort), chay CUOI, chi khi
#              EMOCA_CUDNN_DIAG=1 hoac moi lan "on" deu hong (va thu muc ton tai)
cudnn_probe() {
  local rc_on=0 rc_off=0 rc_raw="" ms_on="" ms_off ms_raw="" err_on="" err_raw="" ok_on=0 wsl_lib=0 py="$OUT_DIR/cudnn_probe.py"
  local attempts_on=0 failures_on=0 errs_on="" diag=0 max_on="$CUDNN_PROBE_ATTEMPTS" rc_try e_try
  local log_on="$OUT_DIR/cudnn_probe_on.log" log_off="$OUT_DIR/cudnn_probe_off.log"
  local log_raw="$OUT_DIR/cudnn_probe_on_no_ldpath.log"
  cat > "$py" <<'PY'
import sys, time
import torch
mode = sys.argv[1]
torch.backends.cudnn.enabled = mode == "on"
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available(), "torch khong thay GPU"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", mode, flush=True)
if mode == "on":
    print("cudnn version", torch.backends.cudnn.version(), flush=True)
x = torch.randn(1, 3, 224, 224, device="cuda")
conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).cuda().eval()
bn = torch.nn.BatchNorm2d(64).cuda().eval()
conv2 = torch.nn.Conv2d(64, 64, 3, padding=1).cuda().eval()
with torch.no_grad():
    for _ in range(3):
        y = conv2(torch.relu(bn(conv(x))))
    torch.cuda.synchronize()
    n = 20
    t = time.perf_counter()
    for _ in range(n):
        y = conv2(torch.relu(bn(conv(x))))
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000 / n
assert bool(torch.isfinite(y).all()), "ket qua NaN/Inf"
print("RESULT ok ms_per_iter=%.2f" % ms, flush=True)
PY
  [[ "$max_on" =~ ^[1-9][0-9]*$ ]] || max_on=3
  [[ -d "$EMOCA_WSL_LIB_DIR" ]] && wsl_lib=1
  : > "$log_on"
  while (( attempts_on < max_on )); do
    attempts_on=$((attempts_on + 1))
    log "Thu cuDNN CO sua LD_LIBRARY_PATH (${LD_LIBRARY_PATH:-rong}) — lan ${attempts_on}/${max_on}, tien trinh con moi, timeout ${CUDNN_PROBE_TIMEOUT} s ..."
    rc_try=0
    envrun timeout -k 10 "$CUDNN_PROBE_TIMEOUT" python "$py" on > "$log_on.try" 2>&1 || rc_try=$?
    { printf '=== lan %s (ma %s) ===\n' "$attempts_on" "$rc_try"; cat "$log_on.try"; } >> "$log_on"
    rc_on=$rc_try
    ms_on="$(sed -n 's/^RESULT ok ms_per_iter=//p' "$log_on.try" | head -1 || true)"
    if [[ $rc_try == 0 && -n "$ms_on" ]]; then
      ok_on=1
      break
    fi
    failures_on=$((failures_on + 1))
    e_try="$(_cudnn_err "$log_on.try" "$rc_try")"
    err_on="$e_try"
    errs_on="${errs_on:+$errs_on | }lan ${attempts_on}: ${e_try:-ma $rc_try}"
    log "  lan ${attempts_on} hong (ma ${rc_try}${e_try:+: $e_try})"
  done
  rm -f "$log_on.try"
  log "Thu conv2d KHONG cuDNN (tien trinh con) ..."
  envrun timeout -k 10 "$CUDNN_PROBE_TIMEOUT" python "$py" off > "$log_off" 2>&1 || rc_off=$?
  ms_off="$(sed -n 's/^RESULT ok ms_per_iter=//p' "$log_off" | head -1 || true)"
  if [[ $wsl_lib == 1 && ( "${EMOCA_CUDNN_DIAG:-0}" == 1 || $ok_on == 0 ) ]]; then
    diag=1
    rc_raw=0
    log "Chan doan: cuDNN KHONG co $EMOCA_WSL_LIB_DIR trong LD_LIBRARY_PATH (co the tu huy) ..."
    EMOCA_NO_WSL_LDPATH=1 envrun timeout -k 10 "$CUDNN_PROBE_TIMEOUT" python "$py" on > "$log_raw" 2>&1 || rc_raw=$?
    ms_raw="$(sed -n 's/^RESULT ok ms_per_iter=//p' "$log_raw" | head -1 || true)"
    err_raw="$(_cudnn_err "$log_raw" "$rc_raw")"
  fi
  {
    printf '# phep thu cuDNN — %s — services/emoca_svc/wsl/common.sh::cudnn_probe (log: %s, %s, %s)\n' \
      "$(date '+%F %T')" "$log_on" "$log_off" "$log_raw"
    printf '# cudnn_ok / *_cudnn_on: CO sua LD_LIBRARY_PATH (dung cach server/smoke chay that)\n'
    printf 'cudnn_ok=%s\n' "$ok_on"
    printf 'ld_library_path_wsl=%s\n' "$wsl_lib"
    printf 'ld_library_path=%s\n' "${LD_LIBRARY_PATH:-}"
    printf 'attempts_on=%s\nfailures_on=%s\n' "$attempts_on" "$failures_on"
    printf 'rc_cudnn_on=%s\nrc_cudnn_off=%s\n' "$rc_on" "$rc_off"
    printf 'error_cudnn_on=%s\n' "${err_on:-}"
    printf 'errors_on=%s\n' "${errs_on:-}"
    printf 'conv_ms_cudnn_on=%s\nconv_ms_cudnn_off=%s\n' "${ms_on:-khong co}" "${ms_off:-khong co}"
    if [[ $diag == 1 ]]; then
      printf '# *_no_ldpath: CO cuDNN nhung KHONG co %s trong LD_LIBRARY_PATH (chung minh nguyen nhan)\n' "$EMOCA_WSL_LIB_DIR"
      printf 'cudnn_ok_no_ldpath=%s\n' "$([[ $rc_raw == 0 && -n "$ms_raw" ]] && echo 1 || echo 0)"
      printf 'rc_cudnn_on_no_ldpath=%s\n' "$rc_raw"
      printf 'error_cudnn_on_no_ldpath=%s\n' "$err_raw"
      printf 'conv_ms_cudnn_on_no_ldpath=%s\n' "${ms_raw:-khong co}"
    elif [[ $wsl_lib == 1 ]]; then
      printf 'cudnn_ok_no_ldpath=khong chay (cuDNN dat; dat EMOCA_CUDNN_DIAG=1 de chay chan doan)\n'
    else
      printf 'cudnn_ok_no_ldpath=khong ap dung (khong co %s)\n' "$EMOCA_WSL_LIB_DIR"
    fi
    printf 'recommend=%s\n' "$([[ $ok_on == 1 ]] && echo 'EMOCA_CUDNN=1 (hoac auto)' || echo 'EMOCA_CUDNN=0')"
    printf 'latency_emoca=chua do (smoke_test_emoca.sh phan B ghi vao out/latency_cudnn.tsv, kem trang thai cuDNN)\n'
  } > "$CUDNN_PROBE.tmp"
  mv -f "$CUDNN_PROBE.tmp" "$CUDNN_PROBE"
  durable_sync
  if [[ $diag == 1 ]]; then
    log "Chan doan KHONG co $EMOCA_WSL_LIB_DIR: ma ${rc_raw}${err_raw:+ ($err_raw)} — log $log_raw"
  fi
  if [[ $ok_on == 1 ]]; then
    log "cuDNN: DAT o lan ${attempts_on}/${max_on} (${failures_on} lan hong truoc do; conv ${ms_on} ms/lan; khong cuDNN ${ms_off:-khong co} ms/lan) — ghi $CUDNN_PROBE"
  else
    log "cuDNN: HONG ca ${attempts_on} lan du da sua LD_LIBRARY_PATH (${errs_on}) => dung EMOCA_CUDNN=0. Log: $log_on"
  fi
  if [[ $rc_off != 0 || -z "$ms_off" ]]; then
    log "[CANH BAO] conv2d KHONG cuDNN cung loi (ma $rc_off) — EMOCA se khong chay duoc tren GPU nay. Log: $log_off"
  else
    log "conv2d KHONG cuDNN: chay duoc (${ms_off} ms/lan)"
  fi
  return 0
}

# cudnn_resolve: doi EMOCA_CUDNN thanh 0/1 THAT SU dung va export (kem EMOCA_CUDNN_PROBE) cho server.py (emoca_runner.py
# doc) va test_emoca_on_images.py (smoke boc bang runpy). auto: doc $CUDNN_PROBE; chua co hoac EMOCA_CUDNN_REPROBE=1 =>
# chay cudnn_probe truoc; doc khong duoc => tat cuDNN cho an toan (da gap SIGABRT). Goi sau require_env.
cudnn_resolve() {
  local ok
  if [[ "$EMOCA_CUDNN" == auto ]]; then
    if [[ ! -s "$CUDNN_PROBE" || "${EMOCA_CUDNN_REPROBE:-0}" == 1 ]]; then cudnn_probe; fi
    ok="$(sed -n 's/^cudnn_ok=//p' "$CUDNN_PROBE" 2>/dev/null | head -1 || true)"
    case "$ok" in
      1) EMOCA_CUDNN=1; CUDNN_REASON="auto: phep thu cuDNN dat ($CUDNN_PROBE)" ;;
      0) EMOCA_CUDNN=0; CUDNN_REASON="auto: phep thu cuDNN HONG ($CUDNN_PROBE)" ;;
      *) EMOCA_CUDNN=0; CUDNN_REASON="auto: khong doc duoc $CUDNN_PROBE => tat cuDNN cho an toan" ;;
    esac
  else
    CUDNN_REASON="dat tay EMOCA_CUDNN=$EMOCA_CUDNN"
  fi
  EMOCA_CUDNN_PROBE="$CUDNN_PROBE"
  export EMOCA_CUDNN EMOCA_CUDNN_PROBE
  log "cuDNN cho EMOCA: $([[ $EMOCA_CUDNN == 1 ]] && echo BAT || echo TAT) — $CUDNN_REASON"
}
