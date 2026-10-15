#!/usr/bin/env bash
# Go moi patch da ap (ke ca optional) khoi upstream EMOCA, theo thu tu nguoc. Idempotent.
#
# Dung:  bash revert.sh [DUONG_DAN_EMOCA]
#   DUONG_DAN_EMOCA   mac dinh: ../../../model/emoca (tinh tu thu muc chua script nay)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$(cd "${1:-$HERE/../../../model/emoca}" && pwd)"

if ! git -C "$TARGET" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[LOI] '$TARGET' khong phai repo git." >&2
  exit 2
fi

shopt -s nullglob
# Thu tu ap: required roi optional. Go theo thu tu nguoc lai.
patches=("$HERE"/0*.patch "$HERE"/optional/0*.patch)

for (( i=${#patches[@]}-1; i>=0; i-- )); do
  p="${patches[$i]}"
  name="$(basename "$p")"
  if git -C "$TARGET" apply -R --check "$p" 2>/dev/null; then
    git -C "$TARGET" apply -R "$p"
    echo "[go    ] $name"
  else
    echo "[chua ap] $name"
  fi
done

changed="$(git -C "$TARGET" status --short | wc -l | tr -d ' ')"
echo "Xong. Con $changed muc thay doi trong $TARGET (0 = sach)."
