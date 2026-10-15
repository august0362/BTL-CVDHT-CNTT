#!/usr/bin/env bash
# Smoke test EMOCA trong WSL2 sau setup_emoca_wsl.sh. CHUA CHAY THAT (viet 2026-09-25, chi kiem tinh).
#   A. M0-04 (notebook muc 6): test_emoca_on_images.py cua upstream tren anh mau CONG KHAI cua EMOCA; moi anh phai co
#      5 PNG (inputs, geometry_coarse, geometry_detail, out_im_coarse, out_im_detail) + 5 NPY dai 100/50/50/6/128.
#   B. M1-03 (notebook muc 8): chay server.py TAM o cong rieng, POST vai anh mau, kiem schema + do dai codes + 68 landmark
#      o hai he toa do, khong nap lai mo hinh, anh xam -> no_face 422.
#      VRAM (H1, docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md): lay mau GET /v1/health moi SMOKE_SAMPLE_SEC giay (python3 cua
#      Ubuntu, KHONG torch => khong tao CUDA context rieng): min(gpu_free_mb), max(vram_reserved_peak_mb), max(vram_peak_mb)
#      -> smoke_vram_health.json. Truong server chua bao => ghi "khong co", khong bia. KHONG dung nvidia-smi memory.used:
#      duoi WDDM no luon 0 (da do tren may du an). RTX 3050 4 GB: CHUA DO, day la lan do dau.
#   C. M3: thu chuyen tiep localhost WSL2 -> Windows (KHONG can GPU): http.server tam o 127.0.0.1:SMOKE_FWD_PORT trong WSL,
#      goi tu phia Windows bang curl.exe qua interop. Chay rieng phan nay:  bash smoke_test_emoca.sh --forward-only
#
# Dung (trong Ubuntu):  bash ~/emoca_wsl/bin/smoke_test_emoca.sh [--forward-only]
# Bien: SMOKE_MAX_IMAGES (5) so anh mau cho phan A | SMOKE_PORT (8766) cong server tam | SMOKE_SKIP_SERVER=1 bo phan B |
#       SMOKE_SAMPLE_SEC (1) chu ky lay mau /v1/health | SMOKE_FWD_PORT (8799) cong thu chuyen tiep | SMOKE_SKIP_FORWARD=1 |
#       EMOCA_CUDNN auto|0|1 (common.sh::cudnn_resolve; 0 => phan A boc test_emoca_on_images.py bang runpy + tat cuDNN,
#       KHONG sua upstream; phan B server.py tu tat). Do tre kem trang thai cuDNN -> out/latency_cudnn.tsv.
# Ket qua: ~/emoca_wsl/out/m0_04/, m0_04_log.txt, smoke_server.log, smoke_measurements.json, smoke_vram_health.json,
#          smoke_forward.txt, timings.tsv, cudnn_probe.txt, latency_cudnn.tsv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

SMOKE_MAX_IMAGES="${SMOKE_MAX_IMAGES:-5}"
SMOKE_PORT="${SMOKE_PORT:-8766}"
SMOKE_SAMPLE_SEC="${SMOKE_SAMPLE_SEC:-1}"
SMOKE_FWD_PORT="${SMOKE_FWD_PORT:-8799}"

port_busy() { (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# ---------------------------------------------------------------- C. chuyen tiep localhost WSL2 -> Windows (M3)
# Tra 0 = Windows goi duoc server nghe 127.0.0.1 trong WSL; 1 = khong; 2 = khong thu duoc tu dong (in huong dan tay).
check_forwarding() {
  local curl_win="" c code rc=2 tmpd srv
  for c in curl.exe /mnt/c/Windows/System32/curl.exe; do
    if command -v "$c" >/dev/null 2>&1; then curl_win="$c"; break; fi
  done
  if port_busy "$SMOKE_FWD_PORT"; then log "Cong $SMOKE_FWD_PORT dang ban — dat SMOKE_FWD_PORT khac."; return 2; fi
  tmpd="$(mktemp -d)"
  echo ok > "$tmpd/index.html"
  python3 -m http.server --bind 127.0.0.1 --directory "$tmpd" "$SMOKE_FWD_PORT" >/dev/null 2>&1 &
  srv=$!
  sleep 2
  if [[ -n "$curl_win" ]]; then
    code="$("$curl_win" -s -o NUL -m 10 -w '%{http_code}' "http://127.0.0.1:$SMOKE_FWD_PORT/" 2>/dev/null | tr -d '\r' || true)"
    if [[ "$code" == 200 ]]; then rc=0; else rc=1; fi
    log "M3: Windows ($curl_win) -> http://127.0.0.1:$SMOKE_FWD_PORT trong WSL: HTTP '${code:-khong co}' => $([[ $rc == 0 ]] && echo DAT || echo KHONG DAT)"
  else
    log "M3: khong goi duoc curl.exe cua Windows (interop tat?) — thu tay trong 60 s: PowerShell Windows chay"
    log "    curl.exe http://127.0.0.1:$SMOKE_FWD_PORT/     (phai in 'ok')"
    sleep 60
  fi
  kill "$srv" 2>/dev/null || true
  wait "$srv" 2>/dev/null || true
  rm -rf "$tmpd"
  if (( rc == 1 )); then
    log "    Khong dat: xem .wslconfig (localhostForwarding=true, hoac networkingMode=mirrored) roi 'wsl --shutdown' tu Windows."
  fi
  printf '%s\tforward_127.0.0.1:%s\t%s\n' "$(date '+%F %T')" "$SMOKE_FWD_PORT" \
    "$(case $rc in 0) echo DAT;; 1) echo KHONG_DAT;; *) echo CHUA_THU_TU_DONG;; esac)" >> "$OUT_DIR/smoke_forward.txt"
  return "$rc"
}

if [[ "${1:-}" == "--forward-only" ]]; then
  require_wsl2
  fwd_rc=0
  check_forwarding || fwd_rc=$?
  exit "$fwd_rc"
fi

require_wsl2
require_gpu
require_env
require_assets
cudnn_resolve   # export EMOCA_CUDNN=0|1 (+ EMOCA_CUDNN_PROBE) cho phan A va server tam o phan B
# M4: server chinh dang chay => hai EMOCA tren GPU 4 GB (va nvidia-smi khong cho thay). Dung server chinh truoc.
if port_busy "$EMOCA_PORT"; then
  die "Cong $EMOCA_PORT dang co tien trinh nghe (run_emoca_server.sh?). Tat no truoc khi smoke test — hai ban EMOCA tren GPU 4 GB se tranh VRAM va lam sai so do."
fi

# ---------------------------------------------------------------- A. M0-04
# Tim anh mau bang glob (download_assets.sh giai nen vao ~/emoca/data/..., khong phai assets/data/... nhu mac dinh demo).
mapfile -t found < <(find "$EMOCA_DIR" -path '*EMOCA_test_example_data/images/*' -type f \( -iname '*.jpg' -o -iname '*.png' \) | sort)
(( ${#found[@]} )) || die "Khong thay anh mau EMOCA_test_example_data/images/** duoi $EMOCA_DIR (download_assets.sh da tai data chua?)."
EX_DIR="$(dirname "${found[0]}")"
IN_DIR="$OUT_DIR/m0_04_input"
rm -rf "$IN_DIR" "$OUT_DIR/m0_04" && mkdir -p "$IN_DIR"
n=0
for f in "${found[@]}"; do
  [[ "$(dirname "$f")" == "$EX_DIR" ]] || continue
  ln -s "$f" "$IN_DIR/$(basename "$f")"
  n=$((n + 1)); (( n >= SMOKE_MAX_IMAGES )) && break
done
log "M0-04: $n anh tu $EX_DIR"

run_m0_04() {
  cd "$EMOCA_DIR/gdl_apps/EMOCA/demos"
  # --save_codes khai bao type=bool (loi upstream: "False" cung thanh True) => chi truyen khi muon bat; bo --save_mesh.
  # KHONG sua upstream: boc bang runpy; EMOCA_CUDNN=0 => tat cuDNN TRUOC khi chay demo (cung tien trinh, co hieu luc
  # cho moi conv cua EMOCA + FAN). sys.argv[0] giu ten demo nhu khi chay truc tiep.
  envrun python -c '
import os, runpy, sys
if os.environ.get("EMOCA_CUDNN") == "0":
    import torch
    torch.backends.cudnn.enabled = False
    print("[smoke] cuDNN TAT (EMOCA_CUDNN=0): torch.backends.cudnn.enabled = False", flush=True)
sys.argv = ["test_emoca_on_images.py"] + sys.argv[1:]
runpy.run_path("test_emoca_on_images.py", run_name="__main__")
' --input_folder "$IN_DIR" --output_folder "$OUT_DIR/m0_04" \
    --model_name "$MODEL_NAME" --path_to_models "$MODELS_DIR" --save_codes True 2>&1 | tee "$OUT_DIR/m0_04_log.txt"
  # TestData in 'please check the test path' roi exit() ma 0 khi thu muc anh sai.
  if grep -q "please check the test path" "$OUT_DIR/m0_04_log.txt"; then
    echo "[LOI] test_emoca_on_images.py khong thay thu muc anh $IN_DIR" >&2; return 1
  fi
}
EMOCA_WSL_REDO=1 stage m0_04_test_emoca_on_images run_m0_04

envrun python - "$OUT_DIR/m0_04/$MODEL_NAME" <<'PY'
import glob, os, sys
import numpy as np
root = sys.argv[1]
want_png = ["inputs", "geometry_coarse", "geometry_detail", "out_im_coarse", "out_im_detail"]
want_npy = {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}
rows = []
for d in sorted(glob.glob(root + "/*")):
    miss = [n for n in want_png if not os.path.isfile(f"{d}/{n}.png")]
    bad = {}
    for n, L in want_npy.items():
        p = f"{d}/{n}.npy"
        size = int(np.load(p).reshape(-1).size) if os.path.isfile(p) else None
        if size != L:
            bad[n] = size
    rows.append((os.path.basename(d), not miss and not bad, miss, bad))
ok = [r for r in rows if r[1]]
print(f"M0-04: {len(ok)}/{len(rows)} anh du 5 PNG + 5 NPY dung do dai")
for r in rows:
    if not r[1]:
        print("  THIEU/SAI:", r)
if not rows or len(ok) != len(rows):
    sys.exit("M0-04 CHUA DAT")
print("M0-04 (1)-(3) DAT. Tieu chi (4) 'anh dung nhin dung khuon mat' can MAT NGUOI: xem cac PNG trong", root)
PY

# ---------------------------------------------------------------- B. server qua HTTP + VRAM
if [[ "${SMOKE_SKIP_SERVER:-0}" == "1" ]]; then
  log "Bo qua phan B (SMOKE_SKIP_SERVER=1)."
else
if port_busy "$SMOKE_PORT"; then die "Cong $SMOKE_PORT dang ban — dat SMOKE_PORT khac."; fi

TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"   # chi dung trong lan kiem nay, khong in ra
VRAM_JSON="$OUT_DIR/smoke_vram_health.json"
rm -f "$VRAM_JSON"
SRV_PID=""; SMP_PID=""
cleanup() {
  if [[ -n "$SMP_PID" ]]; then kill "$SMP_PID" 2>/dev/null || true; wait "$SMP_PID" 2>/dev/null || true; fi
  if [[ -n "$SRV_PID" ]]; then kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

t0=$(date +%s)
(
  set +u
  # shellcheck disable=SC1091
  source "$CONDA_DIR/bin/activate" "$EMOCA_ENV" || { echo "[LOI] khong activate duoc env $EMOCA_ENV" >&2; exit 97; }
  apply_wsl_ldpath   # /usr/lib/wsl/lib o dau LD_LIBRARY_PATH (cuDNN can libcuda.so) — lai sau activate
  cd "$SVC_DIR"
  EMOCA_TOKEN="$TOKEN" PYTHONUNBUFFERED=1 exec python -u server.py --config "$SVC_DIR/configs/services.yaml" \
    --host 127.0.0.1 --port "$SMOKE_PORT" --models-dir "$MODELS_DIR" --model-name "$MODEL_NAME"
) > "$OUT_DIR/smoke_server.log" 2>&1 &
SRV_PID=$!

# Bo lay mau VRAM qua /v1/health (H1). python3 cua Ubuntu (stdlib), khong torch. Ghi de JSON sau MOI mau (tep tam +
# os.replace) nen dung giua chung van con so do. Chay toi khi server thoat hoac bi kill.
EMOCA_TOKEN="$TOKEN" python3 - "$SMOKE_PORT" "$VRAM_JSON" "$SRV_PID" "$SMOKE_SAMPLE_SEC" <<'PY' &
import json, os, sys, time, urllib.request
port, out, pid, every = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
token = os.environ.get("EMOCA_TOKEN")
NA = "khong co (server khong bao truong nay)"
agg = {"source": "GET /v1/health (torch trong tien trinh server; KHONG phai nvidia-smi)", "interval_s": every,
       "samples": 0, "errors": 0, "min_gpu_free_mb": None, "gpu_total_mb": None, "max_vram_reserved_peak_mb": None,
       "max_vram_reserved_mb": None, "max_vram_peak_mb": None, "vram_after_load_mb": None,
       "first_sample": None, "last_sample": None}
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
            now = time.strftime("%H:%M:%S")
            agg["first_sample"] = agg["first_sample"] or now
            agg["last_sample"] = now
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
log "server tam (pid $SRV_PID) o 127.0.0.1:$SMOKE_PORT — log $OUT_DIR/smoke_server.log; lay mau VRAM qua /v1/health moi ${SMOKE_SAMPLE_SEC} s"

client_rc=0
EMOCA_TOKEN="$TOKEN" envrun python - "$SMOKE_PORT" "$SVC_DIR/contracts/emoca_analyze.schema.json" "$IN_DIR" \
  "$OUT_DIR/smoke_measurements.json" "$SRV_PID" "$OUT_DIR/latency_cudnn.tsv" <<'PY' || client_rc=$?
import base64, glob, io, json, os, sys, time, urllib.error, urllib.request
import jsonschema
from PIL import Image

port, schema_path, in_dir, out_path, srv_pid, lat_path = sys.argv[1:7]
base, token = f"http://127.0.0.1:{port}", os.environ["EMOCA_TOKEN"]
NA = "khong co"

def call(path, body=None, timeout=180):
    h = {"Authorization": "Bearer " + token}
    if body is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=body, method="POST" if body is not None else "GET", headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

t0, health = time.time(), None
while time.time() - t0 < 900:
    try:
        os.kill(int(srv_pid), 0)
    except OSError:
        sys.exit("server.py da thoat truoc khi san sang — xem smoke_server.log")
    try:
        st, body = call("/v1/health", timeout=5)
        if st == 200:
            health = body
            break
    except (urllib.error.URLError, ConnectionError, OSError):
        pass
    time.sleep(3)
if not health:
    sys.exit("server khong len sau 15 phut — xem smoke_server.log")
print(f"cuDNN trong server: {health.get('cudnn_enabled', NA)} ({health.get('cudnn_reason', NA)})")
print(f"server san sang sau {time.time() - t0:.0f} s | nap mo hinh {health.get('load_seconds')} s | "
      f"VRAM sau nap {health.get('vram_after_load_mb', NA)} MB | GPU trong {health.get('gpu_free_mb', NA)} / "
      f"{health.get('gpu_total_mb', NA)} MB | {health.get('device')}")

schema = json.load(open(schema_path))
imgs = sorted(glob.glob(in_dir + "/*"))[:3]
rows = []
for p in imgs:
    payload = json.dumps({"image_b64": base64.b64encode(open(p, "rb").read()).decode("ascii"),
                          "options": {"max_faces": 1}}).encode()
    for k in range(3):
        t = time.perf_counter()
        st, body = call("/v1/analyze", payload)
        ms = (time.perf_counter() - t) * 1000
        assert st == 200, f"{os.path.basename(p)}: HTTP {st} {str(body)[:300]}"
        jsonschema.validate(body, schema)
        f = body["faces"][0]
        assert {n: len(v) for n, v in f["codes"].items()} == {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}
        assert len(f["landmarks"]["crop224"]) == 68 and len(f["landmarks"]["original_px"]) == 68
        rows.append({"image": os.path.basename(p), "run": k + 1, "roundtrip_ms": round(ms), **body["timing_ms"]})
print("PASS: dung emoca_analyze.schema.json, codes 100/50/50/6/128, 68 landmark o crop224 + original_px")
for r in rows:
    print(f"  {r['image']:<28} lan {r['run']}  khu hoi {r['roundtrip_ms']:>6} ms  total {r.get('total', 0):>7.0f} ms")
warm = sorted(r["total"] for r in rows if r["run"] >= 2)
cudnn = health.get("cudnn_enabled")
cudnn_txt = NA if cudnn is None else ("BAT" if cudnn else "TAT")
if warm:
    print(f"Do tre phia server, cuDNN {cudnn_txt} (bo lan 1 moi anh, n={len(warm)}): p50 {warm[len(warm) // 2]:.0f} ms, "
          f"max {warm[-1]:.0f} ms")
# Do tre kem trang thai cuDNN (du phong tat cuDNN cham hon bao nhieu?). Khong co lan am => "chua do", khong bia.
rt = sorted(r["roundtrip_ms"] for r in rows if r["run"] >= 2)
new_file = not os.path.isfile(lat_path) or os.path.getsize(lat_path) == 0
with open(lat_path, "a", encoding="utf-8") as f:
    if new_file:
        f.write("ngay_gio\tcudnn\tly_do\tn_lan_am\tp50_server_ms\tmax_server_ms\tp50_khu_hoi_ms\n")
    f.write("\t".join([time.strftime("%F %T"), cudnn_txt, str(health.get("cudnn_reason", NA)), str(len(warm)),
                       f"{warm[len(warm) // 2]:.0f}" if warm else "chua do", f"{warm[-1]:.0f}" if warm else "chua do",
                       f"{rt[len(rt) // 2]:.0f}" if rt else "chua do"]) + "\n")
print("Da ghi do tre + trang thai cuDNN vao", lat_path)

_, h2 = call("/v1/health")
assert h2["load_seconds"] == health["load_seconds"], "mo hinh bi nap lai giua cac lan goi"
buf = io.BytesIO()
Image.new("RGB", (320, 320), (128, 128, 128)).save(buf, "PNG")
st, body = call("/v1/analyze", json.dumps({"image_b64": base64.b64encode(buf.getvalue()).decode("ascii")}).encode())
assert st == 422 and body["error"]["code"] == "no_face", f"anh xam phai la no_face 422, duoc {st} {body}"
print("PASS: khong nap lai mo hinh; anh xam -> no_face 422")
print(f"VRAM (torch, sau cac lan goi): sau nap {h2.get('vram_after_load_mb', NA)} MB | dinh allocated "
      f"{h2.get('vram_peak_mb', NA)} MB | dinh reserved {h2.get('vram_reserved_peak_mb', NA)} MB | "
      f"GPU trong {h2.get('gpu_free_mb', NA)} / {h2.get('gpu_total_mb', NA)} MB")
print("Mien landmark chuan hoa da thay (M0-05, du kien [-1, 1]):", h2.get("landmark_norm_range_seen"))
json.dump({"rows": rows, "health": h2, "cudnn_enabled": h2.get("cudnn_enabled"), "cudnn_reason": h2.get("cudnn_reason")},
          open(out_path, "w"), ensure_ascii=False, indent=1)
PY

stop_t=$(( $(date +%s) - t0 ))
cleanup; trap - EXIT
printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$(basename "$0")" m1_03_server_http "$stop_t" \
  "$([[ $client_rc == 0 ]] && echo ok || echo "FAIL($client_rc)")" >> "$TIMINGS"
if [[ -s "$VRAM_JSON" ]]; then
  log "VRAM qua /v1/health trong suot phan B ($VRAM_JSON):"
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); [print("   ", k, "=", d[k]) for k in ("samples","min_gpu_free_mb","gpu_total_mb","max_vram_reserved_peak_mb","max_vram_peak_mb","vram_after_load_mb")]' "$VRAM_JSON"
else
  log "VRAM qua /v1/health: khong co mau nao (server chua san sang lan nao?)."
fi
(( client_rc == 0 )) || die "Phan B (server qua HTTP) CHUA DAT — xem $OUT_DIR/smoke_server.log"
fi

# ---------------------------------------------------------------- C. chuyen tiep localhost (M3) — chi bao, khong lam hong smoke
if [[ "${SMOKE_SKIP_FORWARD:-0}" != "1" ]]; then
  fwd_rc=0
  check_forwarding || fwd_rc=$?
  (( fwd_rc == 0 )) || log "M3 chua dat/chua thu tu dong (ma $fwd_rc) — web tren Windows co the khong goi duoc 127.0.0.1:$EMOCA_PORT."
fi
log "Smoke test DAT. So do: $OUT_DIR/smoke_measurements.json, $OUT_DIR/smoke_vram_health.json, $OUT_DIR/smoke_forward.txt, $TIMINGS"
