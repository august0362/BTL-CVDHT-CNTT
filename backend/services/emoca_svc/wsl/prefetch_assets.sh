#!/usr/bin/env bash
# Tai TRUOC tai san EMOCA bang nhieu ket noi song song (curl -r theo doan nho + hang doi), thay buoc tai TUAN TU cua
# download_assets.sh (wget, 1 ket noi/tep). Ly do (do 2026-09-25): download.is.tue.mpg.de 4.5-15 KB/s moi ket noi,
# 12 ket noi ~270 KB/s tong; server nginx, Accept-Ranges: bytes. Chay xong ma assets_problems (common.sh) rong =>
# setup_emoca_wsl.sh bo qua chang download_assets. KHONG can sudo, KHONG can aria2c.
#
# Dung (trong Ubuntu WSL2):
#   ACCEPT_FLAME_EMOCA_LICENSES=1 bash "$(wslpath 'C:\...\BTL')/services/emoca_svc/wsl/prefetch_assets.sh"
# Ma thoat: 0 xong (assets_problems rong) | 10 chua dong y giay phep | 11 dang co prefetch/setup khac giu khoa tai san
#           | khac: loi (xem log).
#
# Bien (them cac bien o common.sh):
#   ACCEPT_FLAME_EMOCA_LICENSES=1  CHINH BAN da dang ky + dong y dieu khoan FLAME/EMOCA (script khong dong y thay ban)
#   PREFETCH_CONN (12)        so ket noi dong thoi (so worker cua hang doi)
#   PREFETCH_CHUNK_MB (4)     kich thuoc moi doan; PREFETCH_CHUNK_BYTES ghi de theo byte (chi de test)
#   PREFETCH_TRIES (6)        so lan thu moi doan (moi lan tai tiep tu byte da co); PREFETCH_ROUNDS = ten cu, van nhan
#   PREFETCH_FACEREC auto|0|1 FaceRecognition.zip: auto = theo cfg.yaml cua mo hinh (xem duoi)
#   EMOCA_ASSET_CACHE         noi giu zip (~/emoca_wsl/cache) — zip KHONG nam trong ~/emoca/assets
#   EMOCA_ASSET_BASE_URL      (https://download.is.tue.mpg.de/emoca/assets)
#
# Cach tai (docs/qa/REVIEW_PREFETCH_2026-09-25.md): tep chia thanh doan PREFETCH_CHUNK_MB, ghi thanh danh sach
# <tep>.parts/segments (dong dau "# <Content-Length> <doan>", moi dong "<ten tep doan> <byte dau> <byte cuoi>", theo
# thu tu byte). `xargs -P $PREFETCH_CONN` chay hang doi: worker nao xong lay ngay doan ke tiep (M4 — 12 doan co dinh +
# vong co rao chan lam so ket noi hoat dong giam dan ve cuoi tep). Moi lan tai ghi vao <doan>.cur, CHI noi vao doan khi
# HTTP 206 (M2); du lieu khac/doan qua co chuyen sang <tep>.parts.bad/ (NGOAI thu muc doan, H1). Ghep va dem byte theo
# danh sach segments, khong theo glob (H1). Ghep xong sai kich thuoc hoac unzip -tq hong => bo ca bo doan sang
# .stale (lan sau tai lai tu dau, M1). Bo doan KIEU CU (<tep>.parts/meta = "<total> <n>", n doan bang nhau) duoc giu:
# phan da tai cua moi doan cu giu nguyen, phan con thieu chia thanh doan con part.NNNN.sMMMMMM.
# Khoa tai san dung chung voi setup_emoca_wsl.sh (common.sh: assets_lock, H2).
#
# Tai gi — doc tu ma upstream (e0be0db), noi giai nen y nhu download_assets.sh (cd + unzip):
#   EMOCA/models/$MODEL_NAME.zip  -> assets/EMOCA/models   checkpoint + cfg.yaml (emoca_runner.py: MODEL_NAME)
#   FLAME.zip                     -> assets                replace_asset_dirs (gdl_apps/EMOCA/utils/load.py:141-150)
#   data/EMOCA_test_example_data.zip -> data               anh mau cho smoke test
#   FaceRecognition.zip           -> assets  CHI KHI cfg.yaml detail.model.idw > 0 (hoac khong doc duoc): DECA.__init__
#       luon goi _init_deep_losses (gdl/models/DECA.py:2774,2780), idw > 0 => VGGFace2Loss nap
#       assets/FaceRecognition/resnet50_ft_weight.pkl (DECA.py:2799-2806, load.py:152) ngay khi dung mo hinh.
# KHONG tai:
#   assets/DECA.zip (DECA/data/deca_model.tar): chi nap trong _load_old_checkpoint khi resume_training (DECA.py:3009);
#       load_deca dat cfg.model.resume_training = False (load.py:47) => suy luan khong can.
#   EMOCA/models/DECA.zip, EMOCA.zip, EMOCA_v2_mp.zip, EMOCA_v2_lr_cos_1.5.zip: mo hinh khac, khong dung.
# So do: out/prefetch_timings.tsv (giay, byte, KB/s THAT cua lan nay, ket_noi_tb = so ket noi dang nhan du lieu trung
# binh), them 1 dong/tep vao out/timings.tsv; log out/prefetch_<ngay_gio>.log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

if [[ "${ACCEPT_FLAME_EMOCA_LICENSES:-}" != "1" ]]; then
  cat >&2 <<'EOF'
DUNG: chua dat ACCEPT_FLAME_EMOCA_LICENSES=1. Dang ky + dong y dieu khoan o https://flame.is.tue.mpg.de va
https://emoca.is.tue.mpg.de, roi chay lai:
EOF
  printf '    ACCEPT_FLAME_EMOCA_LICENSES=1 bash %q\n' "$0" >&2   # %q: duong dan co dau cach/Unicode van dan lai duoc (L1)
  exit 10
fi

CACHE_DIR="${EMOCA_ASSET_CACHE:-$EMOCA_WSL_HOME/cache}"
BASE_URL="${EMOCA_ASSET_BASE_URL:-https://download.is.tue.mpg.de/emoca/assets}"
CONN="${PREFETCH_CONN:-12}"
TRIES="${PREFETCH_TRIES:-${PREFETCH_ROUNDS:-6}}"
CHUNK_MB="${PREFETCH_CHUNK_MB:-4}"
FACEREC="${PREFETCH_FACEREC:-auto}"
PROGRESS_SEC="${PREFETCH_PROGRESS_SEC:-60}"
PREFETCH_TSV="$OUT_DIR/prefetch_timings.tsv"
[[ "$CONN" =~ ^[1-9][0-9]*$ ]] || die "PREFETCH_CONN=$CONN khong hop le (so nguyen >= 1)."
[[ "$TRIES" =~ ^[1-9][0-9]*$ ]] || die "PREFETCH_TRIES=$TRIES khong hop le (so nguyen >= 1)."
[[ "$CHUNK_MB" =~ ^[1-9][0-9]*$ ]] || die "PREFETCH_CHUNK_MB=$CHUNK_MB khong hop le (so nguyen >= 1)."
CHUNK="${PREFETCH_CHUNK_BYTES:-$(( CHUNK_MB * 1024 * 1024 ))}"
[[ "$CHUNK" =~ ^[1-9][0-9]*$ ]] || die "PREFETCH_CHUNK_BYTES=$CHUNK khong hop le (so nguyen >= 1)."
case "$FACEREC" in auto|0|1) ;; *) die "PREFETCH_FACEREC=$FACEREC khong hop le (auto, 0 hoac 1)." ;; esac
command -v curl >/dev/null || die "Thieu curl (sudo apt-get install -y curl)."
command -v unzip >/dev/null || die "Thieu unzip (sudo apt-get install -y unzip)."
command -v xargs >/dev/null || die "Thieu xargs (sudo apt-get install -y findutils)."
mkdir -p "$CACHE_DIR"

LOGFILE="$OUT_DIR/prefetch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
# H2: mot tien trinh tai san tai mot thoi diem (prefetch nay, hoac chang download_assets cua setup_emoca_wsl.sh).
assets_lock prefetch_assets.sh
log "prefetch_assets.sh — $CONN ket noi, doan $CHUNK byte, $TRIES lan thu/doan, cache $CACHE_DIR, log $LOGFILE"

# --fail: loi HTTP => ma khac 0. --speed-limit/--speed-time: ket noi treo => cat, worker thu lai tu byte da co.
CURL_OPTS=(-sS -L --fail --connect-timeout 30 --speed-limit 256 --speed-time 120)

fsize() { stat -c %s "$1" 2>/dev/null || echo 0; }

# M3: `curl -I -L` in header cua MOI buoc chuyen huong; chi lay Content-Length cua phan hoi CUOI (dat lai o moi dong
# trang thai HTTP/...). Phan hoi cuoi khong co Content-Length => rong (khong lay nham cua trang 301/302).
remote_size() {
  curl "${CURL_OPTS[@]}" -I "$1" | tr -d '\r' \
    | awk '/^HTTP\// { n = "" } tolower($1) == "content-length:" { n = $2 } END { if (n != "") print n }'
}

# fetch_seg TEN_DOAN BYTE_DAU BYTE_CUOI — chay trong worker cua xargs (bash con: bien PF_* duoc export).
# Tai tiep tu byte da co; CHI noi khi HTTP 206 (M2); thu toi PF_TRIES lan; tra 0 khi doan du byte.
fetch_seg() {
  local seg="$1" start="$2" end="$3" f want have try code
  f="$PF_PARTS/$seg"; want=$(( end - start + 1 ))
  for (( try = 1; try <= PF_TRIES; try++ )); do
    have="$(stat -c %s "$f" 2>/dev/null || echo 0)"
    (( have == want )) && return 0
    if (( have > want )); then mkdir -p "$PF_BAD"; mv -f "$f" "$PF_BAD/$seg.oversize.$(date +%s%N)"; have=0; fi
    # shellcheck disable=SC2086  # PF_CURL_OPTS: cac co khong chua dau cach
    code="$(curl $PF_CURL_OPTS -r "$(( start + have ))-$end" -o "$f.cur" -w '%{http_code}' "$PF_URL" 2>>"$PF_PARTS/curl.err")" \
      || true
    if [[ "$code" == 206 ]]; then
      [[ -f "$f.cur" ]] && cat "$f.cur" >> "$f"          # dut giua chung (ma 18/28) van la 206: giu phan da nhan
    elif [[ -s "$f.cur" ]]; then
      mkdir -p "$PF_BAD"; mv -f "$f.cur" "$PF_BAD/$seg.http$code.$(date +%s%N)"   # server bo qua Range (200)...
    fi
    rm -f "$f.cur"
  done
  have="$(stat -c %s "$f" 2>/dev/null || echo 0)"
  (( have == want ))
}
export -f fetch_seg

# emit_segments TEP TIEN_TO TU DEN: noi cac dong "<TIEN_TO><6 chu so> a b" (doan CHUNK byte, phu [TU, DEN]) vao TEP.
emit_segments() {
  local out="$1" pre="$2" a="$3" to="$4" k=0 b
  while (( a <= to )); do
    b=$(( a + CHUNK - 1 )); (( b <= to )) || b=$to
    echo "$pre$(printf '%06d' "$k") $a $b" >> "$out"
    a=$(( b + 1 )); k=$(( k + 1 ))
  done
}

# make_plan THU_MUC_DOAN TOTAL TEN: ghi THU_MUC_DOAN/segments. Giu bo doan kieu cu (meta "<total> <n>") neu khop total.
make_plan() {
  local parts="$1" total="$2" name="$3" tmp="$1/segments.tmp" mt="" mn="" extra="" i chunk_old start end want have f kept=0
  echo "# $total $CHUNK" > "$tmp"
  if [[ -s "$parts/meta" ]]; then read -r mt mn extra < "$parts/meta" || true; fi
  if [[ "$mt" == "$total" && "$mn" =~ ^[1-9][0-9]*$ && -z "$extra" ]]; then
    chunk_old=$(( (total + mn - 1) / mn ))
    for (( i = 0; i < mn; i++ )); do
      start=$(( i * chunk_old )); (( start < total )) || break
      end=$(( start + chunk_old - 1 )); (( end < total )) || end=$(( total - 1 ))
      want=$(( end - start + 1 )); f="part.$(printf '%04d' "$i")"; have="$(fsize "$parts/$f")"
      if (( have > want )); then
        mkdir -p "$parts.bad"; mv -f "$parts/$f" "$parts.bad/$f.oversize.$(date +%s%N)"; have=0
      fi
      if (( have > 0 )); then echo "$f $start $(( start + have - 1 ))" >> "$tmp"; kept=$(( kept + have )); fi
      emit_segments "$tmp" "$f.s" $(( start + have )) "$end"
    done
    log "  $name: bo doan kieu cu ($mn doan) — giu $kept byte da tai, phan con thieu chia doan $CHUNK byte"
  else
    if compgen -G "$parts/*" >/dev/null; then mv "$parts" "$parts.stale.$(date +%s)"; mkdir -p "$parts"; fi
    emit_segments "$tmp" "part." 0 $(( total - 1 ))
  fi
  mv -f "$tmp" "$parts/segments"
}

# plan_stats THU_MUC_DOAN: in "<byte da co, moi doan tinh toi da do dai cua no> <so doan chua du>" theo segments.
plan_stats() {
  local parts="$1" f s e want have got=0 left=0
  while read -r f s e; do
    [[ "$f" == "#" ]] && continue
    want=$(( e - s + 1 )); have="$(fsize "$parts/$f")"
    (( have > want )) && have=$want
    got=$(( got + have )); (( have == want )) || left=$(( left + 1 ))
  done < "$parts/segments"
  echo "$got $left"
}

# record TEN TOTAL BYTE GIAY KB_S TRANG_THAI KET_NOI_TB: 1 dong vao $PREFETCH_TSV va $TIMINGS.
record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" "$3" "$4" "$5" "$CONN" "$6" "$7" >> "$PREFETCH_TSV"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$(basename "$0")" "prefetch_$1" "${4%.*}" "$6" >> "$TIMINGS"
}

# fetch REL: tai $BASE_URL/REL ve $CACHE_DIR/<ten> (song song theo doan, tai tiep duoc), kiem Content-Length + unzip -tq.
# Ghi 1 dong vao $PREFETCH_TSV va $TIMINGS. Tra ma khac 0 khi hong (khong die: de ghi so do truoc).
fetch() {
  local rel="$1" name url dest parts total pt="" b0 b1 left t0 t1 secs kbps st xpid last ntodo samples=0 act_sum=0 act avg="-"
  local f s e
  name="$(basename "$rel")"; url="$BASE_URL/$rel"; dest="$CACHE_DIR/$name"; parts="$dest.parts"
  total="$(remote_size "$url" || true)"
  if ! [[ "$total" =~ ^[0-9]+$ ]] || (( total == 0 )); then
    echo "[LOI] khong lay duoc Content-Length cua $url" >&2
    record "$name" 0 0 0 0 "FAIL(content-length)" "-"
    return 1
  fi
  if [[ "$(fsize "$dest")" == "$total" ]] && unzip -tq "$dest" >/dev/null 2>&1; then
    log "[cache] $name: da co ($total byte, khop Content-Length, unzip -tq OK) — khong tai lai"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$total" 0 0 "-" "$CONN" "cache" "-" >> "$PREFETCH_TSV"
    return 0
  fi
  [[ -e "$dest" ]] && mv -f "$dest" "$dest.stale.$(date +%s)"   # khac kich thuoc/hong: giu lai, khong xoa
  mkdir -p "$parts"
  if [[ -s "$parts/segments" ]]; then read -r _ pt _ < "$parts/segments" || true; fi
  if [[ "$pt" != "$total" ]]; then
    [[ -z "$pt" ]] || { mv "$parts" "$parts.stale.$(date +%s)"; mkdir -p "$parts"; }   # tep tren server da doi
    make_plan "$parts" "$total" "$name"
  fi
  read -r b0 left < <(plan_stats "$parts")
  : > "$parts/todo"
  while read -r f s e; do
    [[ "$f" == "#" ]] && continue
    [[ "$(fsize "$parts/$f")" == $(( e - s + 1 )) ]] || echo "$f $s $e" >> "$parts/todo"
  done < "$parts/segments"
  ntodo="$(wc -l < "$parts/todo")"; ntodo="${ntodo// /}"
  log "[tai] $name: $total byte, $(( $(wc -l < "$parts/segments") - 1 )) doan, $ntodo doan can tai$( (( b0 > 0 )) && echo ", tai tiep tu $b0 byte da co")"
  t0="$(date +%s.%N)"
  if (( ntodo > 0 )); then
    export PF_URL="$url" PF_PARTS="$parts" PF_BAD="$parts.bad" PF_TRIES="$TRIES" PF_CURL_OPTS="${CURL_OPTS[*]}"
    xargs -P "$CONN" -L 1 bash -c 'fetch_seg "$@"' _ < "$parts/todo" &
    xpid=$!
    last=$SECONDS
    while kill -0 "$xpid" 2>/dev/null; do
      sleep 1
      act="$( (compgen -G "$parts/*.cur" || true) | wc -l)"
      samples=$(( samples + 1 )); act_sum=$(( act_sum + ${act// /} ))
      if (( SECONDS - last >= PROGRESS_SEC )); then
        last=$SECONDS
        read -r b1 left < <(plan_stats "$parts")
        log "  $name: $b1/$total byte, con $left doan, ${act// /} ket noi dang nhan"
      fi
    done
    wait "$xpid" || true   # xargs tra 123 khi co doan hong: da dem lai ben duoi
  fi
  t1="$(date +%s.%N)"
  read -r b1 left < <(plan_stats "$parts")
  secs="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')"
  kbps="$(awk -v x=$(( b1 - b0 )) -v a="$t0" -v b="$t1" 'BEGIN { d = b - a; printf "%.1f", (d > 0 ? x / 1024 / d : 0) }')"
  (( samples == 0 )) || avg="$(awk -v s="$act_sum" -v n="$samples" 'BEGIN { printf "%.1f", s / n }')"
  st=ok
  if (( left > 0 )); then
    st="FAIL(thieu $(( total - b1 )) byte, $left doan)"
    echo "[LOI] $name: sau $TRIES lan thu moi doan van thieu $(( total - b1 )) byte ($left doan). Loi curl cuoi:" >&2
    tail -5 "$parts/curl.err" >&2 || true
  else
    : > "$dest.tmp"
    while read -r f s e; do
      [[ "$f" == "#" ]] && continue
      cat "$parts/$f" >> "$dest.tmp"
    done < "$parts/segments"
    if [[ "$(fsize "$dest.tmp")" != "$total" ]]; then
      st="FAIL(ghep $(fsize "$dest.tmp") != $total)"
    elif ! unzip -tq "$dest.tmp" >/dev/null 2>&1; then
      st="FAIL(unzip -tq)"
    else
      mv -f "$dest.tmp" "$dest"
      durable_sync
      rm -rf "$parts"   # doan tam cua chinh script; zip da ghep + kiem
    fi
    if [[ "$st" != ok ]]; then
      # M1: du byte ma ghep/kiem hong => bo doan co du lieu sai; giu lai (khong xoa), lan sau tai lai tu dau.
      rm -f "$dest.tmp"
      mv "$parts" "$parts.stale.$(date +%s)"
      echo "[LOI] $name: $st — da chuyen bo doan sang $parts.stale.* ; chay lai se tai lai tu dau" >&2
    fi
  fi
  record "$name" "$total" "$(( b1 - b0 ))" "$secs" "$kbps" "$st" "$avg"
  log "[$st] $name: $(( b1 - b0 )) byte moi trong $secs s = $kbps KB/s ($CONN ket noi, trung binh $avg dang nhan)"
  [[ "$st" == ok ]]
}

# extract ZIP DIR: unzip vao $EMOCA_DIR/DIR nhu download_assets.sh (cd DIR; unzip). Moc theo ten + kich thuoc zip.
extract() {
  local zip="$1" dir="$EMOCA_DIR/$2" mark name
  name="$(basename "$zip")"; mark="$STATE_DIR/prefetch_unzip_${name//[^A-Za-z0-9_.-]/_}.done"
  if [[ "$(cat "$mark" 2>/dev/null)" == "$name $(fsize "$zip")" && "${EMOCA_WSL_REDO:-0}" != 1 ]]; then
    log "[bo qua] giai nen $name (da giai nen tu zip nay)"; return 0
  fi
  mkdir -p "$dir"
  local t0=$SECONDS
  if ! (cd "$dir" && unzip -o -q "$zip"); then echo "[LOI] giai nen $name vao $dir hong" >&2; return 1; fi
  durable_sync
  echo "$name $(fsize "$zip")" > "$mark.tmp"; mv -f "$mark.tmp" "$mark"
  log "[xong] giai nen $name -> $dir ($(( SECONDS - t0 )) s)"
}

# In detail.model.idw trong cfg.yaml (OmegaConf, thut 2 dau cach); rong neu khong thay.
cfg_detail_idw() {
  awk '
    /^[^ #]/ { top = $1; sub(/:.*/, "", top); model = 0; next }
    top == "detail" && /^  [^ ]/ { model = ($1 == "model:"); next }
    top == "detail" && model && /^    idw:/ { print $2; exit }
  ' "$1" 2>/dev/null || true
}

# Zip do (tai tuan tu bi dung giua chung) nam trong assets/EMOCA/models: CHUYEN sang cache/stale (khong xoa) de
# download_assets cua setup khong bao "zip hong". An toan vi dang giu khoa tai san (setup khong the dang tai).
tidy_models_dir() {
  local z moved=""
  [[ -d "$MODELS_DIR" ]] || return 0
  for z in "$MODELS_DIR"/*.zip "$MODELS_DIR"/wget-log*; do
    [[ -e "$z" ]] || continue
    if [[ "$z" == *.zip ]] && unzip -tq "$z" >/dev/null 2>&1; then continue; fi
    mkdir -p "$CACHE_DIR/stale"
    mv -f "$z" "$CACHE_DIR/stale/$(basename "$z").$(date +%s)"
    moved="$moved $(basename "$z")"
  done
  [[ -z "$moved" ]] || log "Chuyen tep do/hong khoi $MODELS_DIR sang $CACHE_DIR/stale:$moved"
}

[[ -s "$PREFETCH_TSV" ]] \
  || printf 'ngay_gio\ttep\tbyte_tep\tbyte_tai_lan_nay\tgiay\tKB_s\tket_noi\ttrang_thai\tket_noi_tb\n' > "$PREFETCH_TSV"
T_ALL=$SECONDS
tidy_models_dir
fail=0
# Mo hinh truoc: cfg.yaml cua no quyet dinh co can FaceRecognition khong.
if ! { fetch "EMOCA/models/$MODEL_NAME.zip" && extract "$CACHE_DIR/$MODEL_NAME.zip" assets/EMOCA/models; }; then fail=1; fi
if ! { fetch FLAME.zip && extract "$CACHE_DIR/FLAME.zip" assets; }; then fail=1; fi
if ! { fetch data/EMOCA_test_example_data.zip && extract "$CACHE_DIR/EMOCA_test_example_data.zip" data; }; then fail=1; fi

need_fr=0
case "$FACEREC" in
  1) need_fr=1; log "FaceRecognition: PREFETCH_FACEREC=1 => tai" ;;
  0) log "FaceRecognition: PREFETCH_FACEREC=0 => bo qua" ;;
  auto)
    idw="$(cfg_detail_idw "$MODELS_DIR/$MODEL_NAME/cfg.yaml")"
    if [[ -z "$idw" ]]; then
      need_fr=1; log "FaceRecognition: khong doc duoc detail.model.idw trong cfg.yaml => tai cho an toan"
    elif awk -v v="$idw" 'BEGIN { exit !(v + 0 > 0) }'; then
      need_fr=1; log "FaceRecognition: cfg.yaml detail.model.idw=$idw > 0 => VGGFace2Loss nap resnet50_ft_weight.pkl => tai"
    else
      log "FaceRecognition: cfg.yaml detail.model.idw=$idw => khong nap id loss => bo qua"
    fi ;;
esac
if (( need_fr )); then
  if ! { fetch FaceRecognition.zip && extract "$CACHE_DIR/FaceRecognition.zip" assets; }; then fail=1; fi
fi
log "Bo qua (khong can cho suy luan anh, xem dau script): assets/DECA.zip, EMOCA/models/DECA.zip, EMOCA.zip, EMOCA_v2_mp.zip, EMOCA_v2_lr_cos_1.5.zip"

probs="$(assets_problems)"
log "Tong thoi gian: $(( SECONDS - T_ALL )) s. So do: $PREFETCH_TSV"
if [[ -s "$TEXTURE" ]]; then log "Texture: co $TEXTURE"; else log "Texture: CHUA co $TEXTURE (setup_emoca_wsl.sh tai bang gdown sau buoc tai san)"; fi
if [[ -n "$probs" ]]; then
  echo "[LOI] assets_problems chua rong:" >&2; printf '  %s\n' "$probs" >&2
  exit 1
fi
(( fail == 0 )) || exit 1
log "assets_problems: rong — setup_emoca_wsl.sh se bo qua buoc tai tai san"
