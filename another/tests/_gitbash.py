"""Tìm Git Bash cho các test chạy script bash trên Windows (module phụ trợ, run_all.py bỏ qua file bắt đầu bằng `_`).

KHÔNG dùng cửa vào WSL: `C:\\Windows\\System32\\bash.exe` và `...\\Microsoft\\WindowsApps\\bash.exe` (có sau khi cài WSL,
ADR-018) — bash trong WSL không đọc được đường dẫn `C:/...` của Windows nên test đỏ giả. Thứ tự ưu tiên: vị trí cài Git
chuẩn → cạnh `git` trên PATH → `bash` trên PATH. Không thấy ⇒ None (test gọi phải SKIP rõ ràng, không FAIL).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_EXCLUDED = ("system32", "windowsapps")


def _git_dirs() -> list[Path]:
    dirs = [Path(os.environ[v]) / "Git" for v in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)") if os.environ.get(v)]
    dirs.append(Path(r"C:\Program Files\Git"))
    if os.environ.get("LOCALAPPDATA"):
        dirs.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Git")
    git = shutil.which("git")
    if git:  # ...\Git\cmd\git.exe hoặc ...\Git\bin\git.exe hoặc ...\Git\mingw64\bin\git.exe
        dirs += [Path(git).resolve().parent.parent, Path(git).resolve().parent.parent.parent]
    return dirs


def git_bash() -> str | None:
    cands = [str(d / sub / "bash.exe") for d in _git_dirs() for sub in ("bin", "usr/bin")]
    cands.append(shutil.which("bash"))
    for c in cands:
        if c and Path(c).is_file() and not any(x in c.lower() for x in _EXCLUDED):
            return c
    return None
