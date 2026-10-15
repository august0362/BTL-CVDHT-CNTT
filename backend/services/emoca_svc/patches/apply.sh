#!/usr/bin/env bash
# Ap bo patch len upstream EMOCA (model/emoca). Idempotent: patch da ap thi bo qua.
#
# Dung:  bash apply.sh [--optional] [DUONG_DAN_EMOCA]
#   --optional        ap them optional/*.patch (chi can cho demo video)
#   DUONG_DAN_EMOCA   mac dinh: ../../../model/emoca (tinh tu thu muc chua script nay)
#
# Dich du kien: ban checkout LF (Linux/Colab). Tren Windows: git -c core.autocrlf=false clone ...
# Bien moi truong: EMOCA_PATCH_FORCE=1 de bo qua kiem tra commit upstream.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_COMMIT="e0be0db"

WITH_OPTIONAL=0
if [[ "${1:-}" == "--optional" ]]; then WITH_OPTIONAL=1; shift; fi
TARGET="$(cd "${1:-$HERE/../../../model/emoca}" && pwd)"

if ! git -C "$TARGET" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[LOI] '$TARGET' khong phai repo git (can de kiem tra commit upstream)." >&2
  exit 2
fi
HEAD_SHORT="$(git -C "$TARGET" rev-parse --short=7 HEAD)"
if [[ "$HEAD_SHORT" != "$EXPECTED_COMMIT" && "${EMOCA_PATCH_FORCE:-0}" != "1" ]]; then
  echo "[LOI] upstream o $HEAD_SHORT, patch viet cho $EXPECTED_COMMIT. Dat EMOCA_PATCH_FORCE=1 de bo qua." >&2
  exit 3
fi

shopt -s nullglob
patches=("$HERE"/0*.patch)
if (( WITH_OPTIONAL )); then patches+=("$HERE"/optional/0*.patch); fi
if (( ${#patches[@]} == 0 )); then
  echo "[LOI] khong thay patch nao trong $HERE" >&2
  exit 4
fi

err="$(mktemp)"
trap 'rm -f "$err"' EXIT

for p in "${patches[@]}"; do
  name="$(basename "$p")"
  if git -C "$TARGET" apply -R --check "$p" 2>/dev/null; then
    echo "[da ap ] $name"
  elif git -C "$TARGET" apply --check "$p" 2>"$err"; then
    git -C "$TARGET" apply "$p"
    echo "[ap    ] $name"
  else
    echo "[LOI ] $name khong ap duoc len $TARGET:" >&2
    cat "$err" >&2
    echo "Goi y: checkout co CRLF (Windows)? Clone lai bang: git -c core.autocrlf=false clone ..." >&2
    exit 1
  fi
done

echo "Xong ($HEAD_SHORT). Go patch: bash \"$HERE/revert.sh\" \"$TARGET\""
