#!/usr/bin/env bash
# Dung EMOCA that trong Ubuntu WSL2 (env conda work38 + ban clone EMOCA + tai san) cho services/emoca_svc/server.py.
# Lenh dung lai another/notebooks/emoca_colab.ipynb (muc 0-5), bo phan Colab. Idempotent: chay lai thi bo qua chang da xong.
# Ban sua theo docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md (H1-H4, M1-M10). CHUA CHAY THAT tren WSL2.
#
# Dung (trong Ubuntu WSL2, KHONG can cd vao thu muc du an):
#   ACCEPT_FLAME_EMOCA_LICENSES=1 bash "$(wslpath 'C:\...\BTL')/backend/services/emoca_svc/wsl/setup_emoca_wsl.sh"
# Lan chay dau khong co ACCEPT_FLAME_EMOCA_LICENSES=1: dung env xong roi DUNG truoc khi tai tai san (ma thoat 10).
#
# SAU KHI WSL SAP / TAT DOT NGOT: chay lai bang DUONG DAN DU AN (lenh tren), KHONG dung ~/emoca_wsl/bin/... — ban sao
# trong bin co the thanh 0 byte va `bash tep_rong` thoat 0 ma khong lam gi. Script tu kiem tung chang bang CHAY THAT
# (conda/mamba/python/git), don phan hong (doi ten sang *.broken.<epoch>, KHONG xoa) roi lam lai.
#
# Bien moi truong (them cac bien o common.sh):
#   ACCEPT_FLAME_EMOCA_LICENSES=1  CHINH BAN da dang ky + dong y dieu khoan o flame.is.tue.mpg.de va
#                                  emoca.is.tue.mpg.de. Script KHONG tu dong y thay ban; khong dat => khong tai.
#   BUNDLE_ZIP=duong/dan.zip       goi ma do pack_for_colab.py tao. Mac dinh another/notebooks/emoca_svc_bundle.zip;
#                                  thieu hoac cu hon ma nguon => tu dong goi lai vao ~/emoca_wsl/ (khong ghi vao /mnt/c).
#   FLAME_ALBEDO_DRIVE_ID=...      id Google Drive cua FLAME_albedo_from_BFM.npz (mac dinh nhu notebook).
#   EMOCA_WSL_REDO=1               chay lai moi chang (bo qua moc da-xong).
#   EMOCA_HOST_DRIVE=/mnt/e        o Windows chua dia ao cua distro neu da chuyen khoi C (mac dinh /mnt/c).
#
# Ket qua: ~/miniconda3/envs/work38, ~/emoca (e0be0db + patch 0001/0002), ~/emoca_wsl/{svc,bin,out,state}.
# So do that: ~/emoca_wsl/out/timings.tsv (giay tung chang), setup_*.log, env_work38.yml, pip_freeze_work38.txt,
# pip_check_work38.txt, du.txt, download_assets.log, cudnn_probe.txt (cuDNN dung duoc khong; loi => EMOCA_CUDNN=0).
# Ma thoat: 0 xong | 10 chua dong y giay phep | 11 thieu texture | 12 can ban tu chay lenh sudo (in san) |
#           khac: loi (xem log).
# TODO(M2, sau lan dung thanh cong dau tien): ghim Miniconda (ten tep + sha256) va `mamba=<ban da chay>` thay cho
#           Miniconda3-latest / mamba moi nhat — lay tu out/env_work38.yml va log cua lan do.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

FLAME_ALBEDO_DRIVE_ID="${FLAME_ALBEDO_DRIVE_ID:-1poa4GTzjyArYJ7LhbQN1Gqp2U2zjwiD7}"
YML_SRC="$HERE/environment_work38.yml"
LOGFILE="$OUT_DIR/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
log "setup_emoca_wsl.sh — log day du: $LOGFILE"

# Thu muc goc du an (chua pack_for_colab.py). Chay tu ban sao ~/emoca_wsl/bin thi doc lai tu lan dau.
PROJECT_ROOT=""
if [[ -f "$HERE/../pack_for_colab.py" ]]; then
  PROJECT_ROOT="$(cd "$HERE/../../.." && pwd)"
  printf '%s\n' "$PROJECT_ROOT" > "$STATE_DIR/project_root"
elif [[ -s "$STATE_DIR/project_root" ]]; then
  PROJECT_ROOT="$(cat "$STATE_DIR/project_root")"
  [[ -f "$PROJECT_ROOT/services/emoca_svc/pack_for_colab.py" ]] || PROJECT_ROOT=""
fi

# The local environment specification lives in the top-level environments/ group.
# A copy is still staged beside this script in ~/emoca_wsl/bin for standalone runs.
if [[ -n "$PROJECT_ROOT" && -s "$PROJECT_ROOT/../environments/emoca_wsl/environment_work38.yml" ]]; then
  YML_SRC="$PROJECT_ROOT/../environments/emoca_wsl/environment_work38.yml"
elif [[ ! -s "$YML_SRC" && -s "$HERE/../../../../environments/emoca_wsl/environment_work38.yml" ]]; then
  YML_SRC="$HERE/../../../../environments/emoca_wsl/environment_work38.yml"
fi

# ---------------------------------------------------------------- 0. kiem truoc (khong ton gi)
require_wsl2
require_gpu

# Dung luong (M8): $HOME nam tren dia ao ext4.vhdx (tang dan toi ~1 TB) nen df $HOME gan nhu luon bao con nhieu.
# Gioi han THAT la o Windows chua tep .vhdx ($EMOCA_HOST_DRIVE, chi doc bang df). 30 GB la UOC LUONG, chua do.
disk_report() {
  local home_gb host_gb="" eff
  home_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
  [[ "$home_gb" =~ ^[0-9]+$ ]] || home_gb=0
  if mountpoint -q "$EMOCA_HOST_DRIVE" 2>/dev/null; then
    host_gb=$(df -BG --output=avail "$EMOCA_HOST_DRIVE" | tail -1 | tr -dc '0-9')
    [[ "$host_gb" =~ ^[0-9]+$ ]] || host_gb=""
  fi
  log "Trong: dia ao WSL ($HOME) ${home_gb} GB | o Windows chua dia ao ($EMOCA_HOST_DRIVE) ${host_gb:-khong doc duoc} GB"
  eff="$home_gb"
  if [[ -n "$host_gb" ]] && (( host_gb < eff )); then eff="$host_gb"; fi
  if [[ -z "$host_gb" ]]; then
    log "CANH BAO: khong doc duoc $EMOCA_HOST_DRIVE — dat EMOCA_HOST_DRIVE=/mnt/<o chua distro>; so cua \$HOME KHONG phan anh o that."
  fi
  if (( eff < 30 )); then
    log "CANH BAO: chi con ${eff} GB (nguong 30 GB la uoc luong) — env conda + tai san EMOCA co the khong du."
  fi
}
disk_report

# H5 (da do 2026-09-25): Windows khong co pagefile => han muc commit = RAM vat ly; con trong 0.71 GB lam wsl.exe/VM
# chet (E_UNEXPECTED, OutOfMemoryException) giua luc mamba "Solving environment". Kiem qua interop (powershell.exe cua
# Windows, chi DOC), ghi ra out/windows_memory.txt. CHI CANH BAO, khong dung; interop hong thi bo qua em.
windows_memory_report() {
  local ps="" c out free_kb="" total_kb="" auto_pf="" n_pf=""
  for c in powershell.exe /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe; do
    if command -v "$c" >/dev/null 2>&1; then ps="$c"; break; fi
  done
  if [[ -z "$ps" ]]; then log "Bo qua kiem bo nho Windows (khong goi duoc powershell.exe qua interop)."; return 0; fi
  # shellcheck disable=SC2016
  out="$(timeout 60 "$ps" -NoProfile -NonInteractive -Command \
    '$o = Get-CimInstance Win32_OperatingSystem; $s = Get-CimInstance Win32_ComputerSystem; $p = @(Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue); "{0} {1} {2} {3}" -f $o.FreeVirtualMemory, $o.TotalVirtualMemorySize, [int]$s.AutomaticManagedPagefile, $p.Count' \
    2>/dev/null | tr -d '\r')" || { log "Bo qua kiem bo nho Windows (powershell.exe loi/qua 60 s)."; return 0; }
  read -r free_kb total_kb auto_pf n_pf <<<"$out" || true
  if ! [[ "$free_kb" =~ ^[0-9]+$ && "$total_kb" =~ ^[0-9]+$ && "$n_pf" =~ ^[0-9]+$ ]]; then
    log "Bo qua kiem bo nho Windows (khong doc duoc ket qua: '${out:0:80}')."; return 0
  fi
  local free_gb total_gb
  free_gb=$(awk -v k="$free_kb" 'BEGIN{printf "%.2f", k/1048576}')
  total_gb=$(awk -v k="$total_kb" 'BEGIN{printf "%.2f", k/1048576}')
  printf '%s\t%s\tcommit_con_trong_GB=%s\tcommit_limit_GB=%s\tpagefile_tu_quan_ly=%s\tso_pagefile=%s\n' \
    "$(date '+%F %T')" "${1:-}" "$free_gb" "$total_gb" "$auto_pf" "$n_pf" >> "$OUT_DIR/windows_memory.txt"
  log "Bo nho Windows (${1:-}): commit con trong ${free_gb} GB / han muc ${total_gb} GB | pagefile: ${n_pf} tep, tu quan ly=${auto_pf}"
  if (( n_pf == 0 )) || (( free_kb < 3 * 1048576 )); then
    cat <<MSG
================================================================================
CANH BAO BO NHO WINDOWS: commit con trong ${free_gb} GB, pagefile: ${n_pf} tep.
Lan dung truoc da chet vi Windows het han muc commit (WSL bao E_UNEXPECTED / OutOfMemoryException) du RAM con trong.
Truoc khi chay chang nang (mamba "Solving environment"):
  - dong trinh duyet / ung dung nang tren Windows;
  - bat pagefile: System Properties > Advanced > Performance Settings > Advanced > Virtual memory >
    "Automatically manage paging file size" (can khoi dong lai Windows);
  - hoac giam memory= trong C:\\Users\\<ban>\\.wslconfig.
Script VAN chay tiep (chi canh bao).
================================================================================
MSG
  fi
}
windows_memory_report "dau setup"

# Goi he thong (M9, M10): "install ok installed" moi tinh la da cai (dpkg -s tra 0 ca khi goi dang do / con config).
APT_PKGS=(wget git git-lfs unzip ca-certificates build-essential python3 libgl1 libglib2.0-0)
pkg_installed() { dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q "install ok installed"; }
missing=()
for p in "${APT_PKGS[@]}"; do pkg_installed "$p" || missing+=("$p"); done
if (( ${#missing[@]} )); then
  log "Goi he thong con thieu hoac cai dang do: ${missing[*]}"
  if ! sudo -n true 2>/dev/null && [[ ! -t 0 ]]; then
    cat <<MSG
================================================================================
CAN BAN TU CHAY (sudo can mat khau, ma script dang chay khong co terminal de hoi). Trong Ubuntu:
    sudo dpkg --configure -a
    sudo apt-get update && sudo apt-get install -y ${missing[*]}
roi chay lai script nay.
================================================================================
MSG
    exit 12
  fi
  apt_install() { sudo dpkg --configure -a; sudo apt-get update; sudo apt-get install -y "$@"; }
  stage "apt_${#missing[@]}_goi" apt_install "${missing[@]}"
  rm -f "$STATE_DIR"/apt_*.done   # kiem lai bang dpkg-query moi lan, khong dua vao moc
fi

# ---------------------------------------------------------------- 1. ma du an (goi zip) -> ~/emoca_wsl/svc
# Kiem goi: sha256 tung tep khop MANIFEST, dung commit EMOCA, .sh khong CR; va (neu thay du an) khong cu hon ma nguon.
check_bundle() {  # $1 = zip; in "OK", "STALE <ly do>" hoac "BAD <ly do>"
  python3 - "$1" "$EMOCA_COMMIT" "$PROJECT_ROOT" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
zp, commit, root = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    z = zipfile.ZipFile(zp)
    man = json.loads(z.read("MANIFEST.json"))
except Exception as e:
    print(f"BAD khong doc duoc goi/MANIFEST: {e}"); sys.exit()
for f in man["files"]:
    try:
        data = z.read(f["path"])
    except Exception as e:
        print(f"BAD khong doc duoc {f['path']} trong goi: {e}"); sys.exit()
    if hashlib.sha256(data).hexdigest() != f["sha256"]:
        print(f"BAD sha256 lech MANIFEST: {f['path']}"); sys.exit()
    if f["path"].endswith(".sh") and b"\r" in data:
        print(f"BAD {f['path']} con CR"); sys.exit()
if man.get("upstream_commit_expected") != commit:
    print(f"BAD goi viet cho EMOCA {man.get('upstream_commit_expected')}, script dat {commit}"); sys.exit()
if root:
    r = Path(root)
    src = lambda p: r / p if p.split("/")[0] in ("contracts", "configs") else r / "services/emoca_svc" / p
    for f in man["files"]:
        s = src(f["path"])
        if not s.is_file():
            print(f"STALE tep trong goi khong con trong du an: {f['path']}"); sys.exit()
        if hashlib.sha256(s.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != f["sha256"]:
            print(f"STALE {f['path']} da doi sau khi dong goi"); sys.exit()
    packed = {f["path"] for f in man["files"]}
    for d, arc in (("services/emoca_svc/patches", "patches"), ("services/emoca_svc/patches/optional", "patches/optional")):
        for p in sorted((r / d).glob("0*.patch")):
            if f"{arc}/{p.name}" not in packed:
                print(f"STALE patch moi chua co trong goi: {arc}/{p.name}"); sys.exit()
print("OK")
PY
}

prepare_bundle() {
  local zip="${BUNDLE_ZIP:-}" verdict
  if [[ -z "$zip" && -n "$PROJECT_ROOT" ]]; then zip="$PROJECT_ROOT/../another/notebooks/emoca_svc_bundle.zip"; fi
  verdict="BAD khong co goi"
  if [[ -n "$zip" && -f "$zip" ]]; then verdict="$(check_bundle "$zip")"; fi
  if [[ "$verdict" != OK && -z "${BUNDLE_ZIP:-}" && -n "$PROJECT_ROOT" ]]; then
    log "Goi ${zip:-?}: $verdict — dong goi lai bang pack_for_colab.py vao $EMOCA_WSL_HOME (khong ghi vao du an)."
    zip="$EMOCA_WSL_HOME/emoca_svc_bundle.zip"
    python3 "$PROJECT_ROOT/services/emoca_svc/pack_for_colab.py" --out "$zip"
    verdict="$(check_bundle "$zip")"
  fi
  if [[ "$verdict" != OK ]]; then
    if [[ -s "$SVC_DIR/MANIFEST.json" && -z "${BUNDLE_ZIP:-}" && -z "$PROJECT_ROOT" ]]; then
      log "Khong thay du an; giu ma da giai nen o $SVC_DIR."
      return 0
    fi
    die "Goi ma khong dung duoc: $verdict. Tren Windows: python backend/services/emoca_svc/pack_for_colab.py, hoac dat BUNDLE_ZIP."
  fi
  local tmp="$SVC_DIR.new"
  rm -rf "$tmp" && mkdir -p "$tmp"
  python3 -m zipfile -e "$zip" "$tmp"
  chmod +x "$tmp"/patches/*.sh
  rm -rf "$SVC_DIR.old"
  if [[ -d "$SVC_DIR" ]]; then mv "$SVC_DIR" "$SVC_DIR.old"; fi
  mv "$tmp" "$SVC_DIR" && rm -rf "$SVC_DIR.old"
  durable_sync
  log "Ma du an: $SVC_DIR (tu $zip, $(python3 -c 'import json,sys;m=json.load(open(sys.argv[1]));print(len(m["files"]),"tep, contract",m["contract_version"])' "$SVC_DIR/MANIFEST.json"))"
}
prepare_bundle

# Ban sao script vao ~/emoca_wsl/bin (H2). Nguon la thu muc wsl/ cua DU AN (neu biet), khong phai $HERE, de ban bin
# luon theo kip du an ke ca khi dang chay tu bin. Thay bang `mv` (inode moi) — khong ghi de tai cho tep ma mot tien
# trinh bash khac co the dang doc. KHONG BAO GIO thay tep dang chay: neu chinh no khac ban du an thi dung lai.
BIN_FILES=(common.sh setup_emoca_wsl.sh smoke_test_emoca.sh run_emoca_server.sh environment_work38.yml)
refresh_bin() {
  local src_dir="$HERE" self f src dst
  if [[ -n "$PROJECT_ROOT" && -d "$PROJECT_ROOT/services/emoca_svc/wsl" ]]; then
    src_dir="$PROJECT_ROOT/services/emoca_svc/wsl"
  fi
  self="$(readlink -f "${BASH_SOURCE[0]}")"
  mkdir -p "$BIN_DIR"
  for f in "${BIN_FILES[@]}"; do
    src="$src_dir/$f"
    if [[ "$f" == "environment_work38.yml" && -n "$PROJECT_ROOT" ]]; then
      src="$PROJECT_ROOT/../environments/emoca_wsl/$f"
    fi
    dst="$BIN_DIR/$f"
    if [[ ! -s "$src" ]]; then die "Thieu hoac rong: $src"; fi
    if [[ "$src" -ef "$dst" ]]; then continue; fi
    if [[ -e "$dst" ]] && cmp -s <(tr -d '\r' < "$src") "$dst"; then continue; fi
    if [[ -e "$dst" && "$(readlink -f "$dst")" == "$self" ]]; then
      die "Ban dang chay $dst nhung ban trong du an da doi (hoac ban nay hong). Chay lai bang duong dan du an:
    bash \"$src\""
    fi
    tr -d '\r' < "$src" > "$dst.tmp.$$"
    chmod +x "$dst.tmp.$$"
    mv -f "$dst.tmp.$$" "$dst"
    log "Cap nhat $dst tu $src_dir"
  done
  durable_sync
}
refresh_bin

# ---------------------------------------------------------------- 2. Miniconda + cau hinh kenh (notebook o 8, 11)
# H4: kiem bang CHAY THAT. Thu muc hong: chi doi ten (khong xoa) khi co tep danh dau do CHINH script nay tao;
# Miniconda co san cua nguoi dung thi khong dung toi ma dung lai, kem huong dan.
install_miniconda() {
  if conda_ok; then log "Da co $CONDA_DIR (conda chay duoc)"; return 0; fi
  if [[ -e "$CONDA_DIR" ]]; then
    if [[ -f "$CONDA_DIR/.emoca_wsl_owned" ]]; then
      local bad
      bad="$CONDA_DIR.broken.$(date +%s)"
      log "Miniconda o $CONDA_DIR hong (conda khong chay) — doi ten sang $bad (KHONG xoa; tu xoa khi chac chan)."
      mv "$CONDA_DIR" "$bad"
      invalidate_markers 'mamba_env_update_*' 'pip_*'
    else
      echo "[LOI] $CONDA_DIR ton tai nhung conda khong chay, va khong co $CONDA_DIR/.emoca_wsl_owned (khong chac do script nay cai) => khong tu dong toi." >&2
      echo "      Kiem tra roi tu doi ten:  mv \"$CONDA_DIR\" \"$CONDA_DIR.broken\"   (hoac dat CONDA_DIR khac), sau do chay lai." >&2
      echo "      Neu chinh script nay da cai (lan dung truoc khi co tep danh dau): touch \"$CONDA_DIR/.emoca_wsl_owned\" roi chay lai." >&2
      return 1
    fi
  fi
  local inst="$EMOCA_WSL_HOME/miniconda.sh.part" size
  rm -f "$inst"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$inst"
  size=$(stat -c %s "$inst")
  (( size > 50000000 )) || { echo "[LOI] tep cai Miniconda chi ${size} byte — tai dang do?" >&2; return 1; }
  bash "$inst" -b -p "$CONDA_DIR"
  rm -f "$inst"
  conda_ok || { echo "[LOI] cai xong nhung $CONDA_DIR/bin/conda khong chay" >&2; return 1; }
  durable_sync
  date '+%F %T' > "$CONDA_DIR/.emoca_wsl_owned"
  durable_sync
}
EMOCA_WSL_REDO=1 stage miniconda install_miniconda
export PATH="$CONDA_DIR/bin:$PATH"
log "conda $(conda --version | awk '{print $2}') (Miniconda3-latest: CHUA ghim phien ban, giong notebook — TODO M2)"

# Chay lai moi lan (idempotent): ~/.condarc la tep ghi de, sau khi WSL sap co the rong ma van la YAML hop le.
conda_channels() {
  conda config --set channel_priority strict
  conda config --add channels defaults
  conda config --add channels conda-forge
  if conda tos --help >/dev/null 2>&1; then
    # M5: day la dong y THAY nguoi dung dieu khoan su dung kenh Anaconda (khong phai giay phep FLAME/EMOCA) — giong
    # notebook o 11. Khong dong y thi dung lai (Ctrl+C) va dung kenh conda-forge/Miniforge (lech notebook, chua thu).
    log "LUU Y: dang chap nhan dieu khoan su dung kenh Anaconda (repo.anaconda.com/pkgs/main, /pkgs/r) thay ban, nhu notebook."
    conda tos accept --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --channel https://repo.anaconda.com/pkgs/r
  else
    log "conda nay khong co lenh 'tos' — bo qua (ban conda cu khong doi chap nhan dieu khoan kenh)."
  fi
  conda config --show channels | grep -q conda-forge || { echo "[LOI] ~/.condarc khong giu duoc kenh" >&2; return 1; }
}
EMOCA_WSL_REDO=1 stage conda_channels conda_channels

# ---------------------------------------------------------------- 3. EMOCA upstream: clone moi vao ~/emoca, ghim commit, patch
# H4: repo hong (clone dang do, doi tuong 0 byte) => doi ten sang *.broken.<epoch> va clone lai. assets/ va data/ KHONG
# thuoc upstream (git ls-files = 0) nen duoc chuyen sang ban clone moi; tinh dung cua chung kiem rieng o chang tai san.
clone_emoca() {
  local bad="" d
  if [[ -e "$EMOCA_DIR" ]] && ! repo_ok; then
    bad="$EMOCA_DIR.broken.$(date +%s)"
    log "Repo $EMOCA_DIR hong/dang do — doi ten sang $bad (KHONG xoa), clone lai."
    mv "$EMOCA_DIR" "$bad"
    invalidate_markers pull_submodules pip_install_e
  fi
  if [[ ! -e "$EMOCA_DIR" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -c core.autocrlf=false clone https://github.com/radekd91/emoca.git "$EMOCA_DIR"
    if [[ -n "$bad" ]]; then
      for d in assets data; do
        if [[ -d "$bad/$d" && ! -e "$EMOCA_DIR/$d" ]]; then
          mv "$bad/$d" "$EMOCA_DIR/$d"
          log "Giu lai $d/ tu ban hong (se kiem lai o chang tai san)."
        fi
      done
    fi
  fi
  git -C "$EMOCA_DIR" -c advice.detachedHead=false checkout "$EMOCA_COMMIT"
  durable_sync
}
EMOCA_WSL_REDO=1 stage clone_emoca clone_emoca
[[ "$(git -C "$EMOCA_DIR" rev-parse HEAD)" == "$(git -C "$EMOCA_DIR" rev-parse "$EMOCA_COMMIT^{commit}")" ]] \
  || die "$EMOCA_DIR dang o $(git -C "$EMOCA_DIR" rev-parse --short HEAD), can $EMOCA_COMMIT (git -C ~/emoca checkout $EMOCA_COMMIT)."

# apply.sh tu kiem commit va idempotent — chay moi lan. Neu tep da va bi hong (vd 0 byte sau khi WSL sap) thi patch
# khong ap/go duoc: khoi phuc cac tep DA THEO DOI ve ban goc cua commit (khong dung assets/, data/) roi ap lai.
if ! bash "$SVC_DIR/patches/apply.sh" "$EMOCA_DIR"; then
  log "apply.sh loi — thu khoi phuc tep da theo doi trong $EMOCA_DIR roi ap lai. Cac tep se bi dat lai:"
  git -C "$EMOCA_DIR" status --short --untracked-files=no --ignore-submodules=all | head -20
  git -C "$EMOCA_DIR" checkout -- .
  bash "$SVC_DIR/patches/apply.sh" "$EMOCA_DIR"
fi

# Chay lai moi lan: `git submodule update --init` idempotent va nhanh khi da co.
pull_submodules() {
  cd "$EMOCA_DIR"
  local rc=0 s
  GIT_LFS_SKIP_SMUDGE=1 bash pull_submodules.sh || rc=$?
  for s in external/emonet external/SwinTransformer external/face-parsing.PyTorch; do
    [[ -n "$(ls -A "$s" 2>/dev/null)" ]] && git -C "$s" rev-parse --verify -q HEAD >/dev/null \
      || { echo "[LOI] submodule $s rong hoac hong" >&2; return 1; }
  done
  # Lan chay 2025-11-20: loi git-lfs o submodule spectre (khong dung khi suy luan anh) — chi canh bao.
  (( rc == 0 )) || log "CANH BAO: pull_submodules.sh ma $rc (thuong la spectre/git-lfs; khong can cho suy luan anh)."
}
EMOCA_WSL_REDO=1 stage pull_submodules pull_submodules

# ---------------------------------------------------------------- 4. env work38 (notebook o 12-14)
install_mamba() {
  if mamba_ok; then log "Da co mamba"; return 0; fi
  if [[ -e "$CONDA_DIR/bin/mamba" ]]; then
    # Tep mamba co ma khong chay (hong): conda coi la da cai nen `conda install` se khong lam gi => cai de.
    conda install --force-reinstall mamba -n base -c conda-forge -y
  else
    conda install mamba -n base -c conda-forge -y
  fi
  mamba_ok || { echo "[LOI] cai mamba xong van khong chay duoc lenh mamba" >&2; return 1; }
}
EMOCA_WSL_REDO=1 stage mamba install_mamba

# H4: env hong (python 0 byte, conda-meta rong) => doi ten, bo cache goi co the hong, tao lai, va xoa moc cac chang
# phu thuoc (env update + pip) de chung chay lai.
create_env() {
  if ! env_python_ok; then
    if [[ -e "$ENV_PREFIX" ]]; then
      local bad
      bad="$ENV_PREFIX.broken.$(date +%s)"
      log "Env $ENV_PREFIX hong (python khong chay) — doi ten sang $bad (KHONG xoa), bo cache goi, tao lai."
      mv "$ENV_PREFIX" "$bad"
      conda clean --packages --tarballs -y
    fi
    invalidate_markers 'mamba_env_update_*' 'pip_*'
    "$CONDA_DIR/bin/mamba" create -n "$EMOCA_ENV" python=3.8 -y
    env_python_ok || { echo "[LOI] tao env xong nhung $ENV_PREFIX/bin/python khong chay" >&2; return 1; }
  fi
  conda config --set channel_priority flexible
}
EMOCA_WSL_REDO=1 stage create_env create_env

RESOLVED_YML="$OUT_DIR/environment_work38.resolved.yml"
sed -E "s#^prefix: .*#prefix: $ENV_PREFIX#" "$YML_SRC" > "$RESOLVED_YML"
yml_sha="$(sha256sum "$YML_SRC" | cut -c1-12)"
if [[ ! -s "$STATE_DIR/mamba_env_update_$yml_sha.done" || "${EMOCA_WSL_REDO:-0}" == "1" ]]; then
  windows_memory_report "truoc mamba env update"
fi
# Moc gan voi sha cua YAML: doi YAML => chang nay chay lai.
# mamba o env base: goi bang duong dan tuyet doi (activate work38 co the bo base/bin khoi PATH).
stage "mamba_env_update_$yml_sha" envrun "$CONDA_DIR/bin/mamba" env update -n "$EMOCA_ENV" --file "$RESOLVED_YML" --prune -y
envrun conda list torch
envrun conda list cudatoolkit

# ---------------------------------------------------------------- 5. pip (notebook o 15-19, 24)
# Notebook sua requirements38.txt NGAY TRONG repo; o day ghi ban da sua ra ngoai de ~/emoca khong ban (chi patch sua upstream).
REQ_WSL="$OUT_DIR/requirements38.wsl.txt"
sed -E 's/^(onnx|youtube-dl|mediapipe|mmcv-full|pyzmq)/# \1/' "$EMOCA_DIR/requirements38.txt" > "$REQ_WSL"

pip_requirements() {
  cd "$EMOCA_DIR"
  envrun pip install --upgrade pip==24.0
  envrun pip install -r "$REQ_WSL"
  envrun pip install Cython==0.29
}
stage pip_requirements38 pip_requirements

pip_extras() {
  envrun pip install onnx==1.10.2 onnxruntime-gpu 'youtube-dl~=2021.6.6' mediapipe pyzmq==26.2.1
  envrun pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html
}
stage pip_onnx_mediapipe_mmcv pip_extras

pip_gdl() { cd "$EMOCA_DIR" && envrun pip install -e .; }
stage pip_install_e pip_gdl

# M7 (lech notebook co chu y): notebook o 24 go opencv-contrib-python-headless KHONG co -y (tren Colab nhieu kha nang bi
# huy vi khong co stdin). O day -y => go that. So pip freeze voi lan 2025 neu co khac biet.
pip_fixups() {
  envrun pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless
  envrun pip install opencv-python-headless==4.5.5.64
  envrun pip install git+https://github.com/mattloper/chumpy
  envrun pip install tensorboard==2.9.1
}
stage pip_opencv_chumpy_tensorboard pip_fixups

# Lech notebook CO CHU Y (do 2026-09-25 tren may du an): `mediapipe` KHONG ghim (dong pip_extras giu y notebook) => pip
# lay 1.0.1 (wheel py3-none, khong khai Requires-Python) => `import gdl.models.DECA` loi tren Python 3.8:
# mediapipe/tasks/python/components/containers/category.py:93 `-> list[Category]` (TypeError: 'type' object is not
# subscriptable). Ban cua upstream (requirements38.txt:72, mediapipe==0.8.10) KHONG con tren PyPI. 0.10.11 = wheel cp38
# cuoi cung, protobuf <4 (env: 3.14.0). --no-deps: khong keo lai opencv-contrib-python (pip_fixups da go) / jax / torch
# (flatbuffers, sounddevice da co tu lan cai truoc). Da do: import gdl.utils.MediaPipeLandmarkLists + load_model chay.
pip_mediapipe_pin() {
  envrun pip install --no-deps mediapipe==0.10.11
  envrun python -c "import mediapipe; from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS; print('mediapipe', mediapipe.__version__, len(FACEMESH_CONTOURS))"
}
stage pip_mediapipe_0.10.11 pip_mediapipe_pin

# ---------------------------------------------------------------- 6. kiem env TRUOC khi tai tai san nang (fail-fast)
# H3: khong chi import — chay kernel that tren GPU nay (RTX 3050 = sm_86; ban dung 2025 chi chay tren T4 = sm_75).
# conv2d o day chay KHONG cuDNN (du phong phai dung duoc); cuDNN duoc thu RIENG trong tien trinh con (cudnn_probe,
# common.sh) vi tren may du an no loi/tu huy (SIGABRT) — loi cuDNN KHONG dung setup, chi ghi out/cudnn_probe.txt.
log "Kiem env: nvidia-smi, torch.cuda, conv2d (khong cuDNN), pytorch3d CUDA, face_alignment, gdl"
"$NVSMI" --query-gpu=name,memory.total,driver_version --format=csv
probe_rc=0
envrun python - <<'PY' || probe_rc=$?
import torch, pytorch3d, face_alignment, numpy, cv2
assert torch.cuda.is_available(), "torch khong thay GPU (torch.cuda.is_available() = False)"
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0),
      "| VRAM", round(torch.cuda.get_device_properties(0).total_memory / 2**20), "MB")
print("pytorch3d", pytorch3d.__version__, "| face_alignment", face_alignment.__version__, "| numpy", numpy.__version__,
      "| cv2", cv2.__version__)
cap = torch.cuda.get_device_capability(0)
arch = torch.cuda.get_arch_list()
print("compute capability", "%d.%d" % cap, "| torch arch_list", arch)
if "sm_%d%d" % cap not in arch:
    print("CANH BAO: torch khong co san sm_%d%d — phai dua vao PTX (JIT), co the cham hoac loi" % cap)
torch.backends.cudnn.enabled = False  # cuDNN thu rieng (cudnn_probe) — co the lam tien trinh tu huy
x = torch.randn(1, 3, 64, 64, device="cuda")
w = torch.randn(8, 3, 3, 3, device="cuda")
y = torch.nn.functional.conv2d(x, w)
torch.cuda.synchronize()
assert torch.isfinite(y).all(), "conv2d tren cuda ra NaN/Inf"
print("conv2d cuda (KHONG cuDNN) OK", tuple(y.shape))
from pytorch3d.ops import knn_points
p = torch.rand(1, 256, 3, device="cuda")
nn = knn_points(p, p, K=2)
torch.cuda.synchronize()
assert float(nn.dists[..., 0].abs().max()) < 1e-5, "pytorch3d knn_points tren cuda cho ket qua sai"
print("pytorch3d knn_points cuda OK")
import gdl  # pip install -e . cua EMOCA
print("gdl tu", gdl.__file__)
# import day du chuoi ma demo/server dung (bat loi tuong thich thu vien, vd mediapipe 1.0.1 tren Python 3.8, 2026-09-25)
from gdl_apps.EMOCA.utils.load import load_model  # noqa: F401 — keo gdl.models.DECA, mediapipe
print("gdl_apps.EMOCA.utils.load OK")
PY
(( probe_rc == 0 )) || die "Env $EMOCA_ENV chua dung duoc (ma $probe_rc) — DUNG truoc khi tai tai san. Gui phan log ben tren cho nhom. Neu WSL vua sap: EMOCA_WSL_REDO=1 hoac xoa $STATE_DIR/pip_*.done roi chay lai."

# cuDNN: tien trinh con rieng, KHONG die; ket luan ghi $CUDNN_PROBE (cudnn_ok=0 => server/smoke tu tat cuDNN khi
# EMOCA_CUDNN=auto). Luon chay lai o moi lan setup (driver/env co the vua doi).
cudnn_probe

# pip check: ghi lai, KHONG dung — da biet co xung dot do go opencv-contrib ma mediapipe khai bao (giong notebook).
pip_check_rc=0
envrun pip check > "$OUT_DIR/pip_check_work38.txt" 2>&1 || pip_check_rc=$?
if (( pip_check_rc != 0 )); then
  log "pip check: $(grep -c . "$OUT_DIR/pip_check_work38.txt") dong xung dot (ghi o $OUT_DIR/pip_check_work38.txt, khong dung):"
  head -10 "$OUT_DIR/pip_check_work38.txt"
else
  log "pip check: khong co xung dot."
fi

record_versions() {
  envrun conda env export > "$OUT_DIR/env_work38.yml"
  envrun pip freeze > "$OUT_DIR/pip_freeze_work38.txt"
  git -C "$EMOCA_DIR" rev-parse HEAD > "$OUT_DIR/emoca_commit.txt"
  { date '+%F %T'
    du -sh "$ENV_PREFIX" "$CONDA_DIR/pkgs" "$EMOCA_DIR" "$EMOCA_DIR/assets" "$EMOCA_DIR/data" 2>/dev/null || true
    echo "dia ao WSL (\$HOME):"; df -h "$HOME" | tail -1
    echo "o Windows chua dia ao ($EMOCA_HOST_DRIVE):"; df -h "$EMOCA_HOST_DRIVE" 2>/dev/null | tail -1 || true
  } | tee "$OUT_DIR/du.txt"
  durable_sync
}
record_versions

# ---------------------------------------------------------------- 7. tai san co giay phep (notebook o 22, 25)
if [[ "${ACCEPT_FLAME_EMOCA_LICENSES:-}" != "1" ]]; then
  cat <<'MSG'
================================================================================
DUNG TRUOC KHI TAI TAI SAN: chua dat ACCEPT_FLAME_EMOCA_LICENSES=1.
FLAME (https://flame.is.tue.mpg.de) va EMOCA (https://emoca.is.tue.mpg.de) yeu cau CHINH BAN dang ky va dong y
dieu khoan. Script nay khong tu dong y thay ban. Sau khi da dong y, chay lai (cac chang da xong se duoc bo qua):
    ACCEPT_FLAME_EMOCA_LICENSES=1 bash ~/emoca_wsl/bin/setup_emoca_wsl.sh
================================================================================
MSG
  exit 10
fi

# M1: download_assets.sh cua upstream khong co `set -e` va thoat bang ma cua `cd` cuoi => loi wget/unzip bi che.
# Kiem noi dung (assets_problems o common.sh), grep log tim loi, va unzip -tq cac zip no de lai.
download_assets() {
  local probs
  assets_lock setup_emoca_wsl.sh:download_assets   # H2: khong chay cung prefetch_assets.sh
  probs="$(assets_problems)"
  if [[ -z "$probs" ]]; then log "Da co tai san hop le o $EMOCA_DIR/assets"; return 0; fi
  log "Tai san chua du/hong — tai (lai):"
  printf '  %s\n' "$probs"
  cd "$EMOCA_DIR/gdl_apps/EMOCA/demos"
  # download_assets.sh hoi 2 cau y/n ve giay phep; tra loi y CHI vi nguoi dung da tu dat ACCEPT_FLAME_EMOCA_LICENSES=1.
  (yes || true) | envrun bash download_assets.sh 2>&1 | tee "$OUT_DIR/download_assets.log"
  durable_sync
  if grep -nE 'ERROR [0-9]{3}|failed: |cannot find zipfile|End-of-central-directory|unzip: +cannot|No space left on device' \
      "$OUT_DIR/download_assets.log"; then
    echo "[LOI] download_assets.sh co loi (dong o tren, log day du: $OUT_DIR/download_assets.log)" >&2; return 1
  fi
  local z bad_zip=0
  while IFS= read -r z; do
    if ! unzip -tq "$z" >/dev/null 2>&1; then echo "[LOI] zip hong: $z" >&2; bad_zip=1; fi
  done < <(find "$EMOCA_DIR/assets" "$EMOCA_DIR/data" -maxdepth 3 -name '*.zip' -type f 2>/dev/null)
  (( bad_zip == 0 )) || return 1
  probs="$(assets_problems)"
  [[ -z "$probs" ]] || { echo "[LOI] tai xong van chua dung duoc:" >&2; printf '  %s\n' "$probs" >&2; return 1; }
}
EMOCA_WSL_REDO=1 stage download_assets download_assets

# Texture: > 1 MB VA numpy doc duoc (bat tep cut sau khi WSL sap).
texture_ok() {
  [[ -f "$TEXTURE" ]] && (( $(stat -c %s "$TEXTURE") > 1000000 )) \
    && envrun python -c "import numpy, sys; z = numpy.load(sys.argv[1]); [z[k].shape for k in z.files]" "$TEXTURE" >/dev/null 2>&1
}
# Texture chi bat buoc khi mo hinh dung texture (common.sh::texture_required; EMOCA_REQUIRE_TEXTURE=auto|0|1). Khong can
# => KHONG goi gdown, chi canh bao (2026-09-25: EMOCA_v2_lr_mse_20 co use_texture: false; file Drive mac dinh bi khoa).
TEXTURE_SKIPPED=0
if ! texture_ok && ! texture_required; then
  TEXTURE_SKIPPED=1
  log "[canh bao] bo qua texture vi khong can: $TEXTURE_REASON. $TEXTURE chua co — mo hinh dung texture (use_texture: true)"
  log "           se can tai tay (xem huong dan khi EMOCA_REQUIRE_TEXTURE=1)."
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$(basename "$0")" texture_skip 0 "ok(khong can)" >> "$TIMINGS"
elif ! texture_ok; then
  mkdir -p "$(dirname "$TEXTURE")"
  t0=$(date +%s)
  # gdown co san trong env (requirements38: gdown~=3.12.2); dang URL chay duoc ca ban gdown cu lan moi.
  envrun gdown "https://drive.google.com/uc?id=$FLAME_ALBEDO_DRIVE_ID" -O "$TEXTURE" || true
  durable_sync
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$(basename "$0")" texture_gdown "$(( $(date +%s) - t0 ))" \
    "$(texture_ok && echo ok || echo FAIL)" >> "$TIMINGS"
  if ! texture_ok; then
    rm -f "$TEXTURE"
    cat <<MSG
================================================================================
TAI TEXTURE THAT BAI (Drive het luot tai / file khong con chia se / sai id FLAME_ALBEDO_DRIVE_ID / tep hong).
Tai tay FLAME_albedo_from_BFM.npz (kiem quyen su dung texture BFM, docs/modules/EMOCA.md muc 7.4), vi du ve
C:\\Users\\<ban>\\Downloads tren Windows, roi trong Ubuntu:
    cp /mnt/c/Users/<ban>/Downloads/FLAME_albedo_from_BFM.npz "$TEXTURE"
    ACCEPT_FLAME_EMOCA_LICENSES=1 bash ~/emoca_wsl/bin/setup_emoca_wsl.sh
Hoac dung id Drive khac: FLAME_ALBEDO_DRIVE_ID=<id> ACCEPT_FLAME_EMOCA_LICENSES=1 bash ~/emoca_wsl/bin/setup_emoca_wsl.sh
================================================================================
MSG
    exit 11
  fi
fi
if [[ $TEXTURE_SKIPPED == 0 ]]; then log "Texture: $TEXTURE ($(stat -c %s "$TEXTURE") byte, numpy doc duoc)"; fi

# ---------------------------------------------------------------- 8. tong ket
record_versions
log "Bang thoi gian (giay) — ~/emoca_wsl/out/timings.tsv:"
column -t -s $'\t' "$TIMINGS" 2>/dev/null || cat "$TIMINGS"
cat <<MSG
================================================================================
XONG phan dung. Tiep theo (trong Ubuntu):
  bash ~/emoca_wsl/bin/smoke_test_emoca.sh     # M0-04 (5 PNG + 5 NPY) + goi thu server qua HTTP, do VRAM qua /v1/health
  bash ~/emoca_wsl/bin/run_emoca_server.sh     # server that o http://127.0.0.1:$EMOCA_PORT (Windows goi cung dia chi)
Log: $LOGFILE
================================================================================
MSG
