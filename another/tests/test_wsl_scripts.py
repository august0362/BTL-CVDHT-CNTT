"""Kiểm tĩnh bộ script dựng/chạy EMOCA trong WSL2 (services/emoca_svc/wsl/) — không cần WSL, không cần GPU.

Chạy:  python tests/test_wsl_scripts.py

Đo được ở máy thường: xuống dòng LF, `bash -n` qua Git Bash, không còn dấu vết Colab/tunnel, chốt giấy phép đứng trước
mọi bước tải tài sản và không tự đặt cờ thay người dùng, kiểm env (GPU/torch/pytorch3d) đứng trước tải tài sản, YAML
môi trường khớp từng dòng với notebook, các lệnh/phiên bản/tham số khớp notebook + server.py + configs/services.yaml,
logic kiểm gói zip (chạy thật trên gói do pack_for_colab.py tạo), hàm `stage` dừng đúng khi lỗi giữa chặng và bỏ qua
chặng đã xong (chạy thật bằng Git Bash). KHÔNG đo được ở đây: chạy thật trong Ubuntu WSL2 (dựng env, GPU, tải tài sản,
server) — chưa có bằng chứng.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
WSL = ROOT / "backend" / "services" / "emoca_svc" / "wsl"
SVC = ROOT / "backend" / "services" / "emoca_svc"
NOTEBOOK = ROOT / "another" / "notebooks" / "emoca_colab.ipynb"
SCRIPTS = ["common.sh", "setup_emoca_wsl.sh", "smoke_test_emoca.sh", "run_emoca_server.sh"]
YML = ROOT / "environments" / "emoca_wsl" / "environment_work38.yml"
# WSL_SCRIPT_SUFFIX=.next ⇒ kiểm bản `<tên>.next` (nếu có) thay cho bản thường — dùng khi bản đang chạy thật chưa được
# thay (bash đọc script dần dần nên không ghi đè tại chỗ; điều phối đổi tên .next → tên thường bằng mv sau lần dựng).
SUFFIX = os.environ.get("WSL_SCRIPT_SUFFIX", "")
# Bản script đã sửa theo docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md mang dấu này; kiểm H2-H5 chỉ áp cho bản đó.
REVIEW_TAG = "REVIEW_WSL_SCRIPTS_2026-09-25"
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def path(name: str) -> Path:
    if name == "environment_work38.yml":
        return YML
    if SUFFIX and (WSL / (name + SUFFIX)).is_file():
        return WSL / (name + SUFFIX)
    return WSL / name


def text(name: str) -> str:
    return path(name).read_text(encoding="utf-8")


def reviewed(name: str) -> bool:
    return REVIEW_TAG in text(name)


def code_lines(s: str) -> str:
    """Bỏ dòng chú thích bash (giữ nguyên nội dung heredoc/lệnh)."""
    return "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))


def notebook_code() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def notebook_yaml() -> str:
    src = next(c for c in notebook_code() if "yaml_content = " in c)
    start = src.index("\n", src.index("'''") + 3) + 1
    return src[start:src.index("'''", start)]


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gitbash import git_bash  # noqa: E402  (Git Bash, không phải cửa vào WSL — dùng chung với test_colab_bundle.py)


def test_files_and_line_endings():
    if SUFFIX:
        used = [path(n).name for n in SCRIPTS + ["environment_work38.yml"]]
        print(f"  (WSL_SCRIPT_SUFFIX={SUFFIX}: kiểm {used})")
        check(any(n.endswith(SUFFIX) for n in used), f"có ít nhất một tệp *{SUFFIX} để kiểm")
    for name in SCRIPTS + ["environment_work38.yml"]:
        p = path(name)
        check(p.is_file(), f"có {p.relative_to(ROOT).as_posix()}")
        if p.is_file():
            data = p.read_bytes()
            check(b"\r" not in data, f"{name}: LF, không có CR")
    for name in SCRIPTS:
        if name != "common.sh":
            check(text(name).startswith("#!/usr/bin/env bash\n"), f"{name}: shebang bash")
            check("set -euo pipefail" in text(name), f"{name}: set -euo pipefail")


def test_bash_syntax():
    bash = git_bash()
    if not bash:
        print("  SKIP không thấy Git Bash — bỏ qua bash -n")
        return
    for name in SCRIPTS:
        r = subprocess.run([bash, "-n", path(name).name], cwd=WSL, capture_output=True, text=True)
        check(r.returncode == 0, f"bash -n {path(name).name}" + (f": {r.stderr.strip()[:300]}" if r.returncode else ""))


def test_no_colab_or_tunnel():
    for name in SCRIPTS + ["environment_work38.yml"]:
        low = text(name).lower()
        bad = [w for w in ("google.colab", "cloudflared", "trycloudflare", "/content/", "ngrok") if w in low]
        check(not bad, f"{name}: không có dấu vết Colab/tunnel {bad or ''}")
    for name in SCRIPTS:
        body = code_lines(text(name))
        check("0.0.0.0" not in body, f"{name}: không nghe 0.0.0.0")


def test_license_gate():
    s = text("setup_emoca_wsl.sh")
    body = code_lines(s)
    gate = body.find('if [[ "${ACCEPT_FLAME_EMOCA_LICENSES:-}" != "1" ]]')
    check(gate >= 0, "setup: có chốt ACCEPT_FLAME_EMOCA_LICENSES phải đúng bằng \"1\"")
    after_gate = body[gate:]
    check(re.search(r"^\s*exit 10\s*$", after_gate[:1500], re.M) is not None, "setup: chưa đồng ý ⇒ thoát (mã 10) ngay tại chốt")
    for marker in ("download_assets.sh", "gdown", "(yes || true)"):
        pos = body.find(marker)
        check(pos > gate >= 0, f"setup: '{marker}' chỉ xuất hiện SAU chốt giấy phép")
    # Gán đứng riêng / export (dòng hướng dẫn "ACCEPT_FLAME_EMOCA_LICENSES=1 bash ..." trong thông báo thì được phép).
    assign = [l for l in s.splitlines()
              if re.match(r"^\s*(export\s+ACCEPT_FLAME_EMOCA_LICENSES|ACCEPT_FLAME_EMOCA_LICENSES=\S*\s*$)", l)
              or re.search(r"(declare|readonly|export)\s+.*ACCEPT_FLAME_EMOCA_LICENSES", l)]
    check(not assign, f"setup: không tự đặt cờ giấy phép thay người dùng {assign or ''}")
    for name in SCRIPTS:
        if name != "setup_emoca_wsl.sh":
            b = code_lines(text(name))
            check(not re.search(r"bash\s+download_assets\.sh|gdown\s|wget\s", b), f"{name}: không tải tài sản")
    nb_id = re.search(r'FLAME_ALBEDO_DRIVE_ID = "([^"]+)"', "\n".join(notebook_code())).group(1)
    check(f'FLAME_ALBEDO_DRIVE_ID="${{FLAME_ALBEDO_DRIVE_ID:-{nb_id}}}"' in s,
          "setup: FLAME_ALBEDO_DRIVE_ID mặc định như notebook, ghi đè được bằng biến môi trường")
    check("cp /mnt/c/Users/" in s and "FLAME_albedo_from_BFM.npz" in s and "exit 11" in s,
          "setup: gdown hỏng ⇒ hướng dẫn tải tay texture rồi thoát mã 11")


def test_fail_fast_order():
    body = code_lines(text("setup_emoca_wsl.sh"))
    pos = {k: body.find(k) for k in ("require_gpu", "torch.cuda.is_available()", "import torch, pytorch3d, face_alignment",
                                     "mamba_env_update", "pip_install_e", "download_assets.sh")}
    check(all(v >= 0 for v in pos.values()), f"setup: có đủ các bước {[k for k, v in pos.items() if v < 0] or ''}")
    check(pos["require_gpu"] < pos["mamba_env_update"], "setup: kiểm nvidia-smi trước khi dựng env")
    check(pos["pip_install_e"] < pos["torch.cuda.is_available()"] < pos["download_assets.sh"],
          "setup: kiểm torch.cuda + import pytorch3d/face_alignment SAU dựng env, TRƯỚC tải tài sản")
    check("die \"Env $EMOCA_ENV chua dung duoc" in body, "setup: kiểm env hỏng ⇒ dừng (die), không đi tiếp")
    for k in ("timings.tsv", "du -sh"):
        check(k in text("common.sh") + body, f"có ghi số đo: {k}")
    check('"$ENV_PREFIX"' in body and '"$EMOCA_DIR/assets"' in body, "setup: du -sh cả env lẫn assets")
    # H1: memory.used của nvidia-smi luôn 0 dưới WDDM (đo trên máy dự án) ⇒ đo VRAM qua /v1/health của server.
    for name in ("smoke_test_emoca.sh", "run_emoca_server.sh"):
        b = code_lines(text(name))
        check("/v1/health" in b and all(k in b for k in ("gpu_free_mb", "vram_reserved_peak_mb", "vram_peak_mb")),
              f"{name}: lấy mẫu VRAM qua /v1/health (gpu_free_mb, vram_reserved_peak_mb, vram_peak_mb)")
        check("memory.used" not in b, f"{name}: không dùng nvidia-smi memory.used (luôn 0 dưới WDDM)")
        # Bộ lấy mẫu = heredoc `python3 - ... <<'PY'` (python3 của Ubuntu). Chỉ kiểm khối đó: phần A của smoke có
        # `import torch` hợp lệ (bọc demo EMOCA bằng runpy để tắt cuDNN, chạy trong env work38).
        samplers = [b[i:b.find("\nPY\n", i)] for i in (m.start() for m in re.finditer(r"python3 - [^\n]*<<'PY'", b))]
        check(samplers and all("/v1/health" in sp and "import torch" not in sp for sp in samplers) and "khong co" in b,
              f"{name}: bộ lấy mẫu không nạp torch; trường vắng ghi 'khong co' (không bịa)")
    for name in SCRIPTS:
        check(not re.search(r"(^|[;&|(]\s*)(\"\$NVSMI\"|nvidia-smi)\s*$", code_lines(text(name)), re.M),
              f"{name}: không gọi nvidia-smi trần (segfault trên máy dự án)")


def test_yaml_matches_notebook():
    nb = notebook_yaml().splitlines()
    ours = [l for l in YML.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    check(ours[-1] == "prefix: ~/miniconda3/envs/work38", "yml: prefix ~/miniconda3/envs/work38")
    check(nb[-1] == "prefix: /content/miniconda/envs/work38", "notebook yml: prefix /content/miniconda/envs/work38 (như lúc trích)")
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(ours[:-1], nb[:-1])) if a != b]
    check(len(ours) == len(nb) and not diff,
          f"yml khớp từng dòng với notebook trừ prefix ({len(ours)} vs {len(nb)} dòng){' lệch: ' + str(diff[:3]) if diff else ''}")
    for pin in ("python=3.8.20", "pytorch=1.11.0=py3.8_cuda11.5_cudnn8.3.2_0", "pytorch3d=0.6.2=py38_cu115_pyt1110",
                "cudatoolkit=11.5.1"):
        check(any(pin in l for l in ours), f"yml ghim {pin}")
    setup = text("setup_emoca_wsl.sh")
    check('sed -E "s#^prefix: .*#prefix: $ENV_PREFIX#"' in setup, "setup: thay prefix bằng đường dẫn tuyệt đối của env")


def test_commands_mirror_notebook():
    nbc = "\n".join(notebook_code())
    s = text("setup_emoca_wsl.sh")
    common = text("common.sh")
    commit_nb = re.search(r'EMOCA_COMMIT = "([^"]+)"', nbc).group(1)
    commit_apply = re.search(r'EXPECTED_COMMIT="([^"]+)"', (SVC / "patches" / "apply.sh").read_text(encoding="utf-8")).group(1)
    check(f'EMOCA_COMMIT="${{EMOCA_COMMIT:-{commit_nb}}}"' in common and commit_nb == commit_apply,
          f"commit EMOCA {commit_nb} = notebook = patches/apply.sh")
    model_nb = re.search(r'MODEL_NAME = "([^"]+)"', nbc).group(1)
    check(f'MODEL_NAME="${{MODEL_NAME:-{model_nb}}}"' in common, f"MODEL_NAME mặc định {model_nb} như notebook")
    pairs = [
        ("Miniconda3-latest-Linux-x86_64.sh", "Miniconda"),
        ("conda tos accept --channel https://repo.anaconda.com/pkgs/main", "conda tos accept (main)"),
        ("conda tos accept --channel https://repo.anaconda.com/pkgs/r", "conda tos accept (r)"),
        ("GIT_LFS_SKIP_SMUDGE=1", "clone không kéo LFS"),
        ("https://github.com/radekd91/emoca.git", "repo EMOCA"),
        ("bash pull_submodules.sh", "pull_submodules.sh"),
        ("conda install mamba -n base -c conda-forge -y", "cài mamba"),
        ("create -n \"$EMOCA_ENV\" python=3.8 -y", "mamba create python=3.8"),
        ("env update -n \"$EMOCA_ENV\" --file \"$RESOLVED_YML\" --prune -y", "mamba env update --prune"),
        ("pip install --upgrade pip==24.0", "pip==24.0"),
        ("pip install Cython==0.29", "Cython==0.29"),
        ("pip install onnx==1.10.2 onnxruntime-gpu 'youtube-dl~=2021.6.6' mediapipe pyzmq==26.2.1", "onnx/mediapipe/pyzmq"),
        ("pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html", "mmcv-full cu115/torch1.11"),
        ("pip install -e .", "pip install -e ."),
        ("pip install opencv-python-headless==4.5.5.64", "opencv-python-headless==4.5.5.64"),
        ("pip install git+https://github.com/mattloper/chumpy", "chumpy từ git"),
        ("pip install tensorboard==2.9.1", "tensorboard==2.9.1"),
        ('bash "$SVC_DIR/patches/apply.sh" "$EMOCA_DIR"', "áp patch bằng apply.sh trong gói"),
    ]
    for needle, label in pairs:
        core = needle.replace(' -n "$EMOCA_ENV"', "").split(" -f ")[0].replace("'", "")
        in_nb = any(tok in nbc for tok in (needle, core, needle.replace('"$EMOCA_ENV"', "work38")))
        check(needle in s, f"setup có: {label}")
        if label not in ("áp patch bằng apply.sh trong gói", "repo EMOCA", "mamba env update --prune",
                         "mamba create python=3.8", "conda tos accept (main)", "conda tos accept (r)"):
            check(in_nb, f"notebook cũng có: {label}")
    for pkg in ("onnx", "youtube-dl", "mediapipe", "mmcv-full", "pyzmq"):
        check(f'line.startswith("{pkg}")' in nbc, f"notebook comment dòng {pkg} trong requirements38")
    check("s/^(onnx|youtube-dl|mediapipe|mmcv-full|pyzmq)/# \\1/" in s, "setup: comment đúng 5 nhóm dòng đó (ra bản sao, không sửa repo)")
    check("git -c core.autocrlf=false clone" in s and '"$EMOCA_DIR"' in s, "setup: clone MỚI vào ~/emoca với LF")
    check('EMOCA_DIR="${EMOCA_DIR:-$HOME/emoca}"' in common and 'CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"' in common
          and 'EMOCA_WSL_HOME="${EMOCA_WSL_HOME:-$HOME/emoca_wsl}"' in common, "thư mục làm việc trong $HOME của WSL (không /mnt/c)")
    check("model/emoca" not in code_lines(s), "setup: không dùng bản model/emoca của Windows")


def test_server_invocation():
    tree = ast.parse((SVC / "server.py").read_text(encoding="utf-8"))
    flags = {a.value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
             for a in n.args[:1] if isinstance(a, ast.Constant)}
    for name in ("run_emoca_server.sh", "smoke_test_emoca.sh"):
        body = code_lines(text(name))
        lines = body[body.find("python -u server.py"):].splitlines()
        cmd = []
        for l in lines:  # lệnh server.py kéo dài tới dòng đầu tiên không kết thúc bằng "\\"
            cmd.append(l)
            if not l.rstrip().endswith("\\"):
                break
        used = set(re.findall(r"(--[a-z][a-z-]+)", "\n".join(cmd)))
        check(used and used <= flags, f"{name}: cờ server.py {sorted(used)} đều có trong argparse")
        check("--host 127.0.0.1" in body, f"{name}: server nghe 127.0.0.1")
        check('--config "$SVC_DIR/configs/services.yaml"' in body, f"{name}: đọc configs/services.yaml trong gói")
    import yaml  # configs/ dùng PyYAML (có trong mọi env của dự án)
    port = yaml.safe_load((ROOT / "backend" / "configs" / "services.yaml").read_text(encoding="utf-8"))["emoca_svc"]["port"]
    check(f'EMOCA_PORT="${{EMOCA_PORT:-{port}}}"' in text("common.sh"), f"cổng mặc định {port} = configs/services.yaml")
    run = code_lines(text("run_emoca_server.sh"))
    check("EMOCA_TOKEN=" not in run, "run_emoca_server.sh: token chỉ lấy từ môi trường người dùng (tuỳ chọn), không đặt cứng")
    smoke = text("smoke_test_emoca.sh")
    for n in ("inputs", "geometry_coarse", "geometry_detail", "out_im_coarse", "out_im_detail"):
        check(f'"{n}"' in smoke, f"smoke: kiểm {n}.png")
    check('{"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}' in smoke, "smoke: độ dài NPY/codes 100/50/50/6/128")
    check("please check the test path" in smoke, "smoke: bắt lỗi thư mục ảnh sai (TestData thoát mã 0)")


def _bundle_checker() -> str:
    s = text("setup_emoca_wsl.sh")
    start = s.index("<<'PY'", s.index("check_bundle() {")) + len("<<'PY'\n")
    return s[start:s.index("\nPY\n", start)]


def test_bundle_check_logic():
    sys.path.insert(0, str(SVC))
    import pack_for_colab

    code = _bundle_checker()
    commit = re.search(r'EXPECTED_COMMIT="([^"]+)"', (SVC / "patches" / "apply.sh").read_text(encoding="utf-8")).group(1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "check.py"
        script.write_text(code, encoding="utf-8")

        def verdict(zip_path: Path, root: str) -> str:
            r = subprocess.run([sys.executable, str(script), str(zip_path), commit, root], capture_output=True, text=True,
                               encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            return (r.stdout.strip() or r.stderr.strip()[-300:])

        data, manifest = pack_for_colab.build_bundle()
        good = tmp / "good.zip"
        good.write_bytes(data)
        check(verdict(good, str(ROOT / "backend")) == "OK", f"gói mới tạo từ dự án ⇒ OK ({verdict(good, str(ROOT / "backend"))})")
        check(verdict(good, "") == "OK", "không biết dự án ⇒ chỉ kiểm MANIFEST ⇒ OK")
        check(verdict(good, str(ROOT / "backend")).startswith("OK") and
              subprocess.run([sys.executable, str(script), str(good), "abc1234", ""], capture_output=True, text=True)
              .stdout.startswith("BAD"), "commit khác ⇒ BAD")

        bad = tmp / "tampered.zip"
        with zipfile.ZipFile(good) as zi, zipfile.ZipFile(bad, "w") as zo:
            for it in zi.infolist():
                b = zi.read(it.filename)
                zo.writestr(it, b + b"# sua\n" if it.filename == "server.py" else b)
        check(verdict(bad, "").startswith("BAD sha256"), "sửa tệp trong gói mà không cập nhật MANIFEST ⇒ BAD")

        fake = tmp / "proj"
        for f in manifest["files"]:
            p = f["path"]
            src = ROOT / "backend" / p if p.split("/")[0] in ("contracts", "configs") else SVC / p
            dst = fake / p if p.split("/")[0] in ("contracts", "configs") else fake / "services/emoca_svc" / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        check(verdict(good, str(fake)) == "OK", "bản sao dự án giống gói ⇒ OK")
        (fake / "services/emoca_svc/server.py").write_bytes((SVC / "server.py").read_bytes() + b"\n# doi\n")
        check(verdict(good, str(fake)).startswith("STALE server.py"), "mã nguồn đổi sau khi đóng gói ⇒ STALE")
        (fake / "services/emoca_svc/server.py").write_bytes((SVC / "server.py").read_bytes())
        (fake / "services/emoca_svc/patches/0999-moi.patch").write_text("x\n", encoding="utf-8")
        check(verdict(good, str(fake)).startswith("STALE patch moi"), "có patch mới chưa vào gói ⇒ STALE")


def test_stage_behaviour():
    bash = git_bash()
    if not bash:
        print("  SKIP không thấy Git Bash — bỏ qua chạy thử hàm stage")
        return
    common_file = path("common.sh").name
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp).as_posix()
        head = (f"set -euo pipefail\nsource ./{common_file}\n"
                "half() { false; echo KHONG_DUOC_IN; }\nokay() { echo chang_ok; }\n")
        env = {**os.environ, "HOME": home, "EMOCA_WSL_HOME": home + "/w", "EMOCA_WSL_REDO": "0"}

        def run(body: str):
            r = subprocess.run([bash, "-c", head + body], cwd=WSL, capture_output=True, text=True, env=env,
                               encoding="utf-8", errors="replace")
            return r.returncode, r.stdout + r.stderr

        rc, out = run("stage chang_tot okay\nstage chang_tot okay\n")
        check(rc == 0, f"chạy thử stage bằng Git Bash (mã {rc}) {out[-300:] if rc else ''}")
        check(out.count("chang_ok") == 1 and "[bo qua] chang_tot" in out, "stage: chặng đã xong ⇒ lần sau bỏ qua (idempotent)")
        # Đúng cách dùng trong script: `stage` là một lệnh đơn.
        rc, out = run("stage chang_hong half\necho SAU_STAGE\n")
        check(rc == 1 and "KHONG_DUOC_IN" not in out and "SAU_STAGE" not in out,
              f"stage: lệnh lỗi giữa chặng ⇒ dừng chặng và cả script, mã 1 (không nuốt lỗi) [mã {rc}]")
        tsv = (Path(tmp) / "w" / "out" / "timings.tsv").read_text(encoding="utf-8")
        check(re.search(r"\tchang_tot\t\d+\tok", tsv) is not None and re.search(r"\tchang_hong\t\d+\tFAIL\(1\)", tsv) is not None,
              "stage: ghi timings.tsv (tên chặng, giây, ok/FAIL)")
        check(not (Path(tmp) / "w" / "state" / "chang_hong.done").exists(), "stage: chặng hỏng không có mốc đã-xong")
        if not reviewed("common.sh"):
            print(f"  SKIP {common_file} là bản trước rà soát — bỏ qua kiểm H4/M6 của stage")
            return
        # H4: mốc 0 byte (WSL tắt đột ngột) coi như chưa có ⇒ chạy lại chặng.
        (Path(tmp) / "w" / "state" / "chang_rong.done").write_bytes(b"")
        rc, out = run("stage chang_rong okay\n")
        check(rc == 0 and "chang_ok" in out and "0 byte" in out, f"stage: mốc 0 byte ⇒ chạy lại chặng [mã {rc}]")
        check((Path(tmp) / "w" / "state" / "chang_rong.done").stat().st_size > 0, "stage: mốc mới ghi có nội dung")
        # M6: thư mục làm việc trên ổ Windows bị chặn ngay khi source.
        for var in ("EMOCA_WSL_HOME", "EMOCA_DIR", "CONDA_DIR"):
            r = subprocess.run([bash, "-c", f"set -euo pipefail\nsource ./{common_file}\necho KHONG_DUOC_IN\n"], cwd=WSL,
                               capture_output=True, text=True, env={**env, var: "/mnt/c/x"}, encoding="utf-8",
                               errors="replace")
            check(r.returncode != 0 and "KHONG_DUOC_IN" not in r.stdout and "/mnt" in r.stderr,
                  f"common: {var}=/mnt/c/... ⇒ dừng (M6)")


def test_review_fixes():
    """H2-H5, M1, M5, M8-M10 của docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md — kiểm tĩnh trên bản đã sửa."""
    smoke = code_lines(text("smoke_test_emoca.sh"))
    check('port_busy "$EMOCA_PORT"' in smoke and "die" in smoke, "smoke: server chính (EMOCA_PORT) đang nghe ⇒ dừng (M4)")
    check("--forward-only" in smoke and "curl.exe" in smoke and "--bind 127.0.0.1" in smoke,
          "smoke: có bước thử chuyển tiếp localhost WSL2→Windows không cần GPU (M3)")
    check("exit 97" in smoke, "smoke: activate env lỗi ⇒ báo rõ (LOW)")
    if not (reviewed("setup_emoca_wsl.sh") and reviewed("common.sh")):
        print("  SKIP setup/common là bản trước rà soát (bản đang chạy) — kiểm bản sửa bằng WSL_SCRIPT_SUFFIX=.next")
        return
    s, c = code_lines(text("setup_emoca_wsl.sh")), code_lines(text("common.sh"))
    # H2
    check("refresh_bin" in s and 'readlink -f "${BASH_SOURCE[0]}"' in s and 'mv -f "$dst.tmp.$$" "$dst"' in s,
          "setup: cập nhật ~/emoca_wsl/bin bằng mv (inode mới), không ghi đè tệp đang chạy (H2)")
    check("Chay lai bang duong dan du an" in s, "setup: tệp đang chạy khác bản dự án ⇒ dừng, bảo chạy bằng đường dẫn dự án (H2)")
    # H3
    for k in ("get_device_capability", "get_arch_list", "conv2d", "knn_points", "pip check"):
        check(k in s, f"setup: probe có {k} (H3)")
    probe = s.find("import torch, pytorch3d, face_alignment")
    check(0 <= probe < s.find("download_assets.sh"), "setup: probe GPU chạy TRƯỚC khi tải tài sản")
    # H4
    check("durable_sync" in c and '-s "$mark"' in c and 'mv -f "$mark.tmp" "$mark"' in c,
          "common: sync + mốc không rỗng + ghi mốc nguyên tử (H4)")
    for fn, probe_cmd in (("conda_ok", "--version"), ("mamba_ok", "--version"), ("env_python_ok", "-c"),
                          ("repo_ok", "--ignore-submodules=all")):
        m = re.search(re.escape(fn) + r"\(\)\s*\{", c)
        body = ""
        if m:  # hàm một dòng `f() { ...; }` hoặc nhiều dòng kết thúc bằng dòng "}"
            first = c[m.end():c.find("\n", m.end())]
            body = first if first.rstrip().endswith("}") else c[m.end():c.find("\n}", m.end())]
        check(probe_cmd in body,
              f"common: {fn} kiểm bằng chạy thật ({probe_cmd}) (H4)")
    for fn in ("conda_ok", "mamba_ok", "env_python_ok", "repo_ok"):
        check(fn in s, f"setup dùng {fn} (H4)")
    check(not re.search(r"\[\[\s*-x\s+\"\$CONDA_DIR/bin/(conda|mamba)\"\s*\]\]\s*\|\|", s)
          and '-x "$ENV_PREFIX/bin/python" ]] ||' not in s and '! -d "$EMOCA_DIR/.git"' not in s,
          "setup: bỏ các phép kiểm chỉ-tồn-tại -x/-d cũ (H4)")
    check(".emoca_wsl_owned" in s and ".broken.$(date +%s)" in s, "setup: dời phần hỏng sang *.broken.<epoch>, cần tệp đánh dấu (H4)")
    check(not re.search(r"rm -rf\s+\"?\$(CONDA_DIR|ENV_PREFIX|EMOCA_DIR)\"?(\s|$)", s),
          "setup: KHÔNG rm -rf Miniconda/env/repo (H4)")
    check("--ignore-submodules=all" in s, "setup: kiểm git bỏ qua submodule (spectre hỏng không chặn) (H4)")
    check("SAU KHI WSL SAP" in text("setup_emoca_wsl.sh"), "setup: chú thích đầu tệp hướng dẫn chạy lại bằng đường dẫn dự án (H4)")
    # H5
    check("windows_memory_report" in s and "FreeVirtualMemory" in s and "Win32_PageFileUsage" in s,
          "setup: kiểm commit/pagefile của Windows qua interop (H5)")
    h5 = s[s.find("windows_memory_report() {"):s.find("windows_memory_report \"dau setup\"")]
    check(h5 and "die" not in h5 and "exit" not in h5, "setup: kiểm bộ nhớ Windows chỉ cảnh báo, không dừng (H5)")
    # M1
    check("assets_problems" in s and "unzip -tq" in s and "download_assets.log" in s and "-size +1M" in c,
          "setup: kiểm nội dung tài sản sau download_assets.sh (M1)")
    # M5, M8, M9, M10
    check("chap nhan dieu khoan" in s, "setup: in rõ việc conda tos accept vào log (M5)")
    check('EMOCA_HOST_DRIVE="${EMOCA_HOST_DRIVE:-/mnt/c}"' in c and 'df -BG --output=avail "$EMOCA_HOST_DRIVE"' in s,
          "setup: kiểm dung lượng theo ổ Windows chứa vhdx (M8)")
    check("sudo -n true" in s and re.search(r"^\s*exit 12\s*$", s, re.M) is not None, "setup: sudo không tương tác ⇒ in lệnh, mã 12 (M9)")
    check("install ok installed" in s and "dpkg -s" not in s, "setup: dpkg-query 'install ok installed' (M10)")
    check("TODO(M2" in text("setup_emoca_wsl.sh"), "setup: M2 (ghim Miniconda/mamba) ghi TODO")


def test_cudnn_fallback():
    """Dự phòng TẮT cuDNN (đo trên máy dự án: conv2d CÓ cuDNN lỗi CUDNN_STATUS_* / SIGABRT, KHÔNG cuDNN thì chạy)."""
    s, c = code_lines(text("setup_emoca_wsl.sh")), code_lines(text("common.sh"))
    smoke, run = code_lines(text("smoke_test_emoca.sh")), code_lines(text("run_emoca_server.sh"))
    probe = s[s.find("import torch, pytorch3d, face_alignment"):s.find("\nPY\n", s.find("import torch, pytorch3d, face_alignment"))]
    check(0 <= probe.find("torch.backends.cudnn.enabled = False") < probe.find("conv2d("),
          "setup: probe chính chạy conv2d KHÔNG cuDNN (lỗi cuDNN không làm dừng setup)")
    die_env = s.find('die "Env $EMOCA_ENV chua dung duoc')
    check(die_env < s.find("\ncudnn_probe\n") < s.find("download_assets.sh"),
          "setup: gọi cudnn_probe sau kiểm env, trước tải tài sản")
    check('EMOCA_CUDNN="${EMOCA_CUDNN:-auto}"' in c and 'CUDNN_PROBE="$OUT_DIR/cudnn_probe.txt"' in c,
          "common: EMOCA_CUDNN mặc định auto, kết quả ở out/cudnn_probe.txt")
    body = c[c.find("cudnn_probe() {"):c.find("\n}\n", c.find("cudnn_probe() {"))]
    runs = re.findall(r"envrun timeout -k 10 \"\$CUDNN_PROBE_TIMEOUT\" python \"\$py\" \w+ > [^\n]*", body)
    check(len(runs) == 3 and all(re.search(r"\|\| rc_(try|off|raw)=\$\?$", r) for r in runs) and "die" not in body,
          "common: cuDNN thử trong tiến trình con có timeout, bắt mã lỗi, KHÔNG die")
    for name, b in (("run_emoca_server.sh", run), ("smoke_test_emoca.sh", smoke)):
        check(0 <= b.find("require_env") < b.find("\ncudnn_resolve") < b.find("python -u server.py"),
              f"{name}: cudnn_resolve sau require_env, trước khi chạy server.py (export EMOCA_CUDNN)")
    check('EMOCA_CUDNN="$EMOCA_CUDNN"' in run, "run_emoca_server.sh: truyền EMOCA_CUDNN cho server.py")
    check('runpy.run_path("test_emoca_on_images.py", run_name="__main__")' in smoke
          and "python test_emoca_on_images.py" not in smoke and 'os.environ.get("EMOCA_CUDNN") == "0"' in smoke,
          "smoke A: bọc demo upstream bằng runpy, tắt cuDNN khi EMOCA_CUDNN=0 (không sửa upstream)")
    check("latency_cudnn.tsv" in smoke and "chua do" in smoke and "cudnn_enabled" in smoke,
          "smoke B: ghi độ trễ kèm trạng thái cuDNN; chưa đo thì 'chua do'")
    # Nguyên nhân gốc (đo 2026-09-25): thiếu /usr/lib/wsl/lib trong LD_LIBRARY_PATH ⇒ cuDNN không mở được libcuda.so.
    check('EMOCA_WSL_LIB_DIR="${EMOCA_WSL_LIB_DIR:-/usr/lib/wsl/lib}"' in c and re.search(r"^apply_wsl_ldpath\s", c, re.M),
          "common: mặc định /usr/lib/wsl/lib, áp ngay khi source")
    env_fn = c[c.find("envrun() {"):c.find("\n}\n", c.find("envrun() {"))]
    check(0 < env_fn.find('source "$CONDA_DIR/bin/activate"') < env_fn.find("apply_wsl_ldpath") < env_fn.find('"$@"'),
          "common: envrun áp LD_LIBRARY_PATH SAU activate, trước lệnh")
    for name, b in (("run_emoca_server.sh", run), ("smoke_test_emoca.sh", smoke)):
        act = b.find('source "$CONDA_DIR/bin/activate"')
        check(0 <= act < b.find("apply_wsl_ldpath", act) < b.find("python -u server.py", act),
              f"{name}: áp LD_LIBRARY_PATH sau activate, trước server.py")
    check("EMOCA_NO_WSL_LDPATH=1 envrun" in body and "cudnn_ok_no_ldpath" in body,
          "cudnn_probe: thêm lượt KHÔNG LD_LIBRARY_PATH để chứng minh nguyên nhân")
    check(0 < body.find('python "$py" on > "$log_on.try"') < body.find('python "$py" off')
          < body.find("EMOCA_NO_WSL_LDPATH=1 envrun") and 'CUDNN_PROBE_ATTEMPTS="${CUDNN_PROBE_ATTEMPTS:-3}"' in c
          and "EMOCA_CUDNN_DIAG" in body,
          "cudnn_probe: lượt 'on' (thử lại tối đa 3) TRƯỚC, 'off', chẩn đoán no_ldpath CUỐI (chỉ khi DIAG=1 hoặc on hỏng)")
    check("torch/lib" not in c.replace("KHONG them torch/lib", ""), "common: không thêm torch/lib vào LD_LIBRARY_PATH (đo: gây lỗi)")

    bash = git_bash()
    if not bash:
        print("  SKIP không thấy Git Bash — bỏ qua chạy thử cudnn_probe/cudnn_resolve")
        return
    common_file = path("common.sh").name
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp).as_posix()
        out_dir = Path(tmp) / "w" / "out"
        env = {**os.environ, "HOME": home, "EMOCA_WSL_HOME": home + "/w"}
        env.pop("EMOCA_CUDNN", None)
        wsl_lib = Path(tmp) / "wsl_lib"   # thay /usr/lib/wsl/lib
        wsl_lib.mkdir()
        lib = wsl_lib.as_posix()
        if re.match(r"^[A-Za-z]:/", lib):
            lib = f"/{lib[0].lower()}{lib[2:]}"
        env["EMOCA_WSL_LIB_DIR"] = lib
        env.pop("EMOCA_NO_WSL_LDPATH", None)
        # envrun GIẢ thay env conda, mô phỏng NGUYÊN NHÂN GỐC đã đo: thiếu thư mục thư viện WSL trong LD_LIBRARY_PATH ⇒
        # cuDNN không mở được libcuda.so và tự huỷ (134). ON_MODE: fixed (có LD_LIBRARY_PATH là chạy) | broken (vẫn
        # hỏng dù đã sửa) | noresult (thoát 0 mà không in RESULT) | flaky (lần "on" có sửa ĐẦU TIÊN lỗi chập chờn như đã
        # đo, lần sau chạy — đếm qua tệp FAKE_CNT). "off" (không cuDNN) luôn chạy. Mỗi lần gọi ghi thứ tự vào FAKE_ORDER.
        env["FAKE_CNT"] = (Path(tmp) / "fake_cnt").as_posix()
        env["FAKE_ORDER"] = (Path(tmp) / "fake_order").as_posix()
        fake = r'''envrun() (
  apply_wsl_ldpath
  m="${!#}"
  if [[ $m == on && ":${LD_LIBRARY_PATH:-}:" != *":$EMOCA_WSL_LIB_DIR:"* ]]; then tag=no_ldpath; else tag=$m; fi
  echo "$tag" >> "$FAKE_ORDER"
  if [[ $m == on && $tag == on && $ON_MODE == flaky ]]; then
    n=$(( $(cat "$FAKE_CNT" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$FAKE_CNT"
    if (( n == 1 )); then echo "RuntimeError: Unable to find a valid cuDNN algorithm to run convolution"; exit 1; fi
  fi
  exec_fake "$m"
)
exec_fake() (
  apply_wsl_ldpath
  m="${!#}"
  if [[ $m == on ]]; then
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$EMOCA_WSL_LIB_DIR:"* ]]; then
      echo "Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory"
      exit 134
    fi
    case "$ON_MODE" in
      broken) echo "RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"; exit 134 ;;
      noresult) echo "RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"; exit 0 ;;
    esac
    echo 'RESULT ok ms_per_iter=0.73'
  else
    echo 'RESULT ok ms_per_iter=1.12'
  fi
)
'''

        def run(body: str, **extra):
            for f in (env["FAKE_CNT"], env["FAKE_ORDER"]):
                Path(f).unlink(missing_ok=True)
            r = subprocess.run([bash, "-c", f"set -euo pipefail\nsource ./{common_file}\n{fake}{body}"], cwd=WSL,
                               capture_output=True, text=True, env={**env, "ON_MODE": "fixed", **extra},
                               encoding="utf-8", errors="replace")
            return r.returncode, r.stdout + r.stderr

        def probe_txt() -> str:
            p = out_dir / "cudnn_probe.txt"
            return p.read_text(encoding="utf-8") if p.exists() else ""

        # apply_wsl_ldpath: đặt thư mục WSL lên ĐẦU, giữ mục sẵn có, không lặp; không có thư mục thì không thêm.
        rc, out = run('echo "SRC=${LD_LIBRARY_PATH:-}"\n', LD_LIBRARY_PATH=f"/a:{lib}:/b")
        check(rc == 0 and f"SRC={lib}:/a:/b\n" in out, f"source common.sh ⇒ LD_LIBRARY_PATH={lib}:/a:/b (lên đầu, giữ /a /b, không lặp) [{out.strip()[-120:]}]")
        rc, out = run('unset LD_LIBRARY_PATH; apply_wsl_ldpath; echo "L=$LD_LIBRARY_PATH"\n'
                      'LD_LIBRARY_PATH=/a; apply_wsl_ldpath; apply_wsl_ldpath; echo "L2=$LD_LIBRARY_PATH"\n'
                      'EMOCA_NO_WSL_LDPATH=1 apply_wsl_ldpath; echo "L3=${LD_LIBRARY_PATH:-}"\n'
                      'EMOCA_WSL_LIB_DIR=/khong/co; LD_LIBRARY_PATH=/a; apply_wsl_ldpath; echo "L4=$LD_LIBRARY_PATH"\n')
        check(rc == 0 and f"L={lib}\n" in out and f"L2={lib}:/a\n" in out and "L3=/a\n" in out and "L4=/a\n" in out,
              f"apply_wsl_ldpath: rỗng → chỉ thư mục WSL; gọi 2 lần không lặp; EMOCA_NO_WSL_LDPATH=1 → bỏ; thư mục không có → giữ nguyên [{out.strip()[-160:]}]")

        def order() -> list:
            p = Path(env["FAKE_ORDER"])
            return p.read_text(encoding="utf-8").split() if p.exists() else []

        # Cách sửa có tác dụng (như đo thật): có LD_LIBRARY_PATH thì chạy ngay lần 1 ⇒ EMOCA_CUDNN=1; KHÔNG chạy chẩn đoán.
        rc, out = run('cudnn_resolve\necho "E=$EMOCA_CUDNN"\nbash -c \'echo "CON=$EMOCA_CUDNN P=$EMOCA_CUDNN_PROBE"\'\n')
        t = probe_txt()
        check(rc == 0 and "E=1" in out and "CON=1 P=" in out and order() == ["on", "off"],
              f"auto, chưa có probe ⇒ chạy probe; cuDNN chạy lần 1 ⇒ EMOCA_CUDNN=1 export; không chẩn đoán ({order()}) [mã {rc}] {out[-300:] if rc else ''}")
        check(all(k in t for k in ("cudnn_ok=1", "ld_library_path_wsl=1", "attempts_on=1", "failures_on=0", "rc_cudnn_on=0",
                                   "conv_ms_cudnn_on=0.73", "conv_ms_cudnn_off=1.12", "cudnn_ok_no_ldpath=khong chay",
                                   "recommend=EMOCA_CUDNN=1", "latency_emoca=chua do")),
              "cudnn_probe.txt: cudnn_ok=1, attempts_on=1, failures_on=0, 0.73 vs 1.12 ms, chẩn đoán 'khong chay'")
        # Lỗi CHẬP CHỜN (đã đo): lần 1 "Unable to find a valid cuDNN algorithm", lần 2 chạy ⇒ vẫn cudnn_ok=1.
        rc, out = run('EMOCA_CUDNN_REPROBE=1 cudnn_resolve\necho "E=$EMOCA_CUDNN"\n', ON_MODE="flaky")
        t = probe_txt()
        check(rc == 0 and "E=1" in out and order() == ["on", "on", "off"]
              and all(k in t for k in ("cudnn_ok=1", "attempts_on=2", "failures_on=1", "cudnn_ok_no_ldpath=khong chay"))
              and "lan 1: RuntimeError: Unable to find a valid cuDNN algorithm" in t,
              f"chập chờn: lần 1 lỗi, lần 2 (tiến trình con mới) chạy ⇒ cudnn_ok=1, attempts_on=2, failures_on=1 ({order()})")
        # Chẩn đoán theo yêu cầu: EMOCA_CUDNN_DIAG=1 ⇒ chạy lượt không LD_LIBRARY_PATH, CUỐI cùng.
        rc, out = run("cudnn_probe\n", EMOCA_CUDNN_DIAG="1")
        t = probe_txt()
        check(rc == 0 and order() == ["on", "off", "no_ldpath"]
              and all(k in t for k in ("cudnn_ok=1", "cudnn_ok_no_ldpath=0", "rc_cudnn_on_no_ldpath=134"))
              and re.search(r"^error_cudnn_on_no_ldpath=libcuda\.so: cannot open shared object file; SIGABRT", t, re.M),
              f"EMOCA_CUDNN_DIAG=1 ⇒ chẩn đoán chạy CUỐI: không LD_LIBRARY_PATH → libcuda.so + SIGABRT ({order()})")
        # Dự phòng: hỏng cả 3 lần dù đã sửa ⇒ tắt cuDNN; chẩn đoán tự chạy (cuối).
        rc, out = run('EMOCA_CUDNN_REPROBE=1 cudnn_resolve\necho "E=$EMOCA_CUDNN"\n', ON_MODE="broken")
        t = probe_txt()
        check(rc == 0 and "E=0" in out and order() == ["on", "on", "on", "off", "no_ldpath"]
              and all(k in t for k in ("cudnn_ok=0", "attempts_on=3", "failures_on=3", "rc_cudnn_on=134",
                                       "recommend=EMOCA_CUDNN=0", "cudnn_ok_no_ldpath=0"))
              and re.search(r"^error_cudnn_on=CUDNN_STATUS_INTERNAL_ERROR; SIGABRT", t, re.M),
              f"hỏng cả 3 lần dù có LD_LIBRARY_PATH ⇒ cudnn_ok=0, EMOCA_CUDNN=0, tự chẩn đoán cuối, không die ({order()}) [mã {rc}]")
        rc, out = run("cudnn_probe\n", ON_MODE="flaky", CUDNN_PROBE_ATTEMPTS="1")
        check(rc == 0 and "cudnn_ok=0" in probe_txt() and "attempts_on=1" in probe_txt(),
              "CUDNN_PROBE_ATTEMPTS=1 + lỗi chập chờn ⇒ cudnn_ok=0 (số lần thử chỉnh được)")
        rc, out = run("cudnn_probe\necho SAU_PROBE\n", ON_MODE="noresult")
        check(rc == 0 and "SAU_PROBE" in out and "cudnn_ok=0" in probe_txt(),
              "cuDNN thoát 0 nhưng không in RESULT ⇒ vẫn coi là hỏng (cudnn_ok=0)")
        rc, out = run("cudnn_probe\n", EMOCA_WSL_LIB_DIR=(Path(tmp) / "khong_co").as_posix())
        check(rc == 0 and "cudnn_ok_no_ldpath=khong ap dung" in probe_txt() and "ld_library_path_wsl=0" in probe_txt(),
              "không có thư mục thư viện WSL ⇒ bỏ lượt chứng minh, ghi 'khong ap dung'")
        (out_dir / "cudnn_probe.txt").write_text("cudnn_ok=1\n", encoding="utf-8")
        rc, out = run('cudnn_resolve\necho "E=$EMOCA_CUDNN"\n')
        check(rc == 0 and "E=1" in out and "RESULT" not in out, "auto + probe sẵn cudnn_ok=1 ⇒ 1, không thử lại")
        rc, out = run('cudnn_resolve\necho "E=$EMOCA_CUDNN"\n', EMOCA_CUDNN="0")
        check(rc == 0 and "E=0" in out, "đặt tay EMOCA_CUDNN=0 thắng probe cudnn_ok=1")
        (out_dir / "cudnn_probe.txt").write_text("rac\n", encoding="utf-8")
        rc, out = run('cudnn_resolve\necho "E=$EMOCA_CUDNN"\n')
        check(rc == 0 and "E=0" in out, "probe không đọc được cudnn_ok ⇒ tắt cuDNN cho an toàn")
        rc, out = run('echo KHONG_DUOC_IN\n', EMOCA_CUDNN="yes")
        check(rc != 0 and "KHONG_DUOC_IN" not in out and "EMOCA_CUDNN" in out, "EMOCA_CUDNN sai ⇒ dừng khi source common.sh")


def test_stage_calls_are_plain():
    for name in SCRIPTS:
        for i, l in enumerate(text(name).splitlines(), 1):
            st = l.strip()
            if st.startswith("#") or not re.search(r"(^|[\s;&|(])stage\s+\S", st) or st.startswith("stage() {"):
                continue
            plain = re.match(r"^(\w+=\S+\s+)*stage\s", st) is not None and not re.search(r"(&&|\|\||;)", st)
            check(plain, f"{name}:{i}: `stage` gọi như lệnh đơn (set -e còn hiệu lực)")


FAKE_CURL = r'''#!/usr/bin/env bash
# curl GIẢ cho test prefetch: phục vụ tệp trong $FAKE_SRV theo URL .../assets/<rel>; -I in header; -r a-b trả đúng đoạn
# byte (vào -o nếu có); -w in mã HTTP. Chế độ lỗi (theo TÊN tệp, mỗi chế độ một lần nhờ mkdir làm cờ):
#   FAKE_FAIL_ONCE=<tên>          lần tải đoạn ĐẦU TIÊN chỉ trả một nửa rồi thoát 18 (đứt giữa chừng), mã 206
#   FAKE_IGNORE_RANGE_ONCE=<tên>  lần tải đầu trả 200 + CẢ tệp (máy chủ/redirect bỏ qua Range)
#   FAKE_FAIL_GET=<tên>           mọi lần tải (không phải HEAD) lỗi 404 — hỏng vĩnh viễn
#   FAKE_HEAD_REDIRECT=<tên>|all  -I in thêm một phản hồi 301 có Content-Length: 162 đứng trước;
#   FAKE_HEAD_CHUNKED=1           ... và phản hồi cuối KHÔNG có Content-Length (chunked)
head=0; range=""; url=""; out=""; wfmt=""
while (($#)); do
  case "$1" in
    -I) head=1 ;;
    -r) range="$2"; shift ;;
    -o) out="$2"; shift ;;
    -w) wfmt="$2"; shift ;;
    --connect-timeout|--speed-limit|--speed-time) shift ;;
    -*) ;;
    *) url="$1" ;;
  esac
  shift
done
rel="${url#*/assets/}"; f="$FAKE_SRV/$rel"; name="${rel##*/}"
echo "$rel ${range:-HEAD}" >> "$FAKE_LOG"
code() { [[ -z "$wfmt" ]] || printf '%s' "$1"; }
emit() { if [[ -n "$out" ]]; then cat > "$out"; else cat; fi; }
fail404() { echo "curl: (22) The requested URL returned error: 404" >&2; code 404; exit 22; }
[[ -f "$f" ]] || fail404
size=$(stat -c %s "$f")
if ((head)); then
  if [[ "${FAKE_HEAD_REDIRECT:-}" == "$name" || "${FAKE_HEAD_REDIRECT:-}" == all ]]; then
    printf 'HTTP/1.1 301 Moved Permanently\r\nContent-Length: 162\r\nLocation: https://mirror/%s\r\n\r\n' "$rel"
    if [[ "${FAKE_HEAD_CHUNKED:-0}" == 1 ]]; then printf 'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n'; exit 0; fi
  fi
  printf 'HTTP/1.1 200 OK\r\nContent-Length: %s\r\nAccept-Ranges: bytes\r\n\r\n' "$size"; exit 0
fi
[[ "${FAKE_FAIL_GET:-}" == "$name" ]] && fail404
s=${range%-*}; e=${range#*-}; n=$((e - s + 1))
if [[ "${FAKE_IGNORE_RANGE_ONCE:-}" == "$name" ]] && mkdir "$FAKE_SRV/.ignored" 2>/dev/null; then
  emit < "$f"; code 200; exit 0
fi
if [[ "${FAKE_FAIL_ONCE:-}" == "$name" ]] && mkdir "$FAKE_SRV/.failed" 2>/dev/null; then
  tail -c +$((s + 1)) "$f" | head -c $((n / 2)) | emit; code 206; exit 18
fi
tail -c +$((s + 1)) "$f" | head -c "$n" | emit; code 206
'''
PREFETCH_CHUNK = 300_000   # PREFETCH_CHUNK_BYTES trong test: FLAME.zip (~1.2 MB) ⇒ 5 đoạn


def _prefetch_fixture(srv: Path, idw: str) -> None:
    """Máy chủ tài sản GIẢ: cùng đường dẫn tương đối như download.is.tue.mpg.de/emoca/assets."""
    def mk(rel: str, members: dict[str, bytes]) -> None:
        p = srv / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
            for k, v in members.items():
                z.writestr(k, v)
    cfg = (f"coarse:\n  model:\n    idw: 0.5\n  learning:\n    lr: 1\n"
           f"detail:\n  inout:\n    name: x\n  model:\n    n_detail: 128\n    idw: {idw}\n    mrfwr: 0\n").encode()
    mk("EMOCA/models/EMOCA_v2_lr_mse_20.zip", {"EMOCA_v2_lr_mse_20/cfg.yaml": cfg,
                                               "EMOCA_v2_lr_mse_20/detail/checkpoints/deca-epoch=10.ckpt": os.urandom(1_300_000)})
    mk("FLAME.zip", {"FLAME/geometry/generic_model.pkl": os.urandom(1_200_000), "FLAME/mask/uv_face_mask.png": b"png"})
    mk("data/EMOCA_test_example_data.zip", {"EMOCA_test_example_data/images/affectnet_test_examples/a.png": b"img"})
    mk("FaceRecognition.zip", {"FaceRecognition/resnet50_ft_weight.pkl": b"w" * 1000})
    mk("DECA.zip", {"DECA/data/deca_model.tar": b"x"})


def test_prefetch_assets():
    """prefetch_assets.sh (2026-09-25, sửa theo docs/qa/REVIEW_PREFETCH_2026-09-25.md): hàng đợi đoạn nhỏ + xargs -P,
    chỉ nối khi 206, tải tiếp khi đứt, ghép theo danh sách đoạn, bỏ bộ đoạn hỏng, khoá tài sản dùng chung với setup,
    giữ bộ đoạn kiểu cũ — chạy THẬT bằng Git Bash với curl GIẢ (không mạng, không tài sản thật), trong thư mục làm việc
    có dấu cách + chữ có dấu (T5)."""
    s = text("prefetch_assets.sh")
    check(path("prefetch_assets.sh").read_bytes().count(b"\r") == 0 and s.startswith("#!/usr/bin/env bash\n")
          and "set -euo pipefail" in s, "prefetch: LF, shebang bash, set -euo pipefail")
    body = code_lines(s)
    gate = body.find('if [[ "${ACCEPT_FLAME_EMOCA_LICENSES:-}" != "1" ]]')
    check(0 <= gate < body.find("curl ") and re.search(r"^\s*exit 10\s*$", body[gate:gate + 600], re.M) is not None,
          "prefetch: chốt giấy phép (thoát 10) đứng TRƯỚC mọi lời gọi curl")
    check(gate < body.find("assets_lock prefetch_assets.sh") < body.find("tidy_models_dir\n"),
          "prefetch: giữ khoá tài sản (H2) sau chốt giấy phép, trước mọi thao tác trên tài sản")
    check(not re.search(r"(export\s+|^\s*)ACCEPT_FLAME_EMOCA_LICENSES=\S*\s*$", s, re.M), "prefetch: không tự đặt cờ giấy phép")
    check("-r \"$(( start + have ))-$end\"" in body and "unzip -tq" in body and '"$(fsize "$dest.tmp")" != "$total"' in body,
          "prefetch: curl -r theo đoạn (tải tiếp từ byte đã có), so kích thước ghép với Content-Length, unzip -tq")
    check("-w '%{http_code}'" in body and '[[ "$code" == 206 ]]' in body, "prefetch: chỉ nối vào đoạn khi HTTP 206 (M2)")
    check("xargs -P \"$CONN\"" in body and not re.search(r'cat "\$parts"/part\.\*', body),
          "prefetch: hàng đợi xargs -P (M4); ghép theo danh sách đoạn, không theo glob part.* (H1)")
    check("/^HTTP\\// { n = \"\" }" in body, "prefetch: Content-Length lấy của phản hồi CUỐI sau redirect (M3)")
    check("printf '    ACCEPT_FLAME_EMOCA_LICENSES=1 bash %q\\n' \"$0\"" in body, "prefetch: lệnh gợi ý quote bằng %q (L1)")
    for rel, where in (('EMOCA/models/$MODEL_NAME.zip', "assets/EMOCA/models"), ("FLAME.zip", "assets"),
                       ("data/EMOCA_test_example_data.zip", "data"), ("FaceRecognition.zip", "assets")):
        check(re.search(re.escape(rel) + r'.*extract "[^"]+" ' + re.escape(where) + r";", body) is not None,
              f"prefetch: {rel} giải nén vào {where} (như download_assets.sh)")
    check(not re.search(r"fetch\s+\S*DECA", body), "prefetch: KHÔNG tải DECA.zip (chỉ dùng khi resume_training)")
    c, st = code_lines(text("common.sh")), code_lines(text("setup_emoca_wsl.sh"))
    check("assets_lock() {" in c and "flock -n 9" in c and c.count("DIE_CODE=11 die") >= 2,
          "common.sh: assets_lock — flock -n, dự phòng thư mục khoá, bận ⇒ thoát 11 (H2)")
    dl = st.find("download_assets() {")
    check(0 <= dl < st.find("assets_lock setup_emoca_wsl.sh:download_assets", dl) < st.find("assets_problems", dl),
          "setup: chặng download_assets giữ CÙNG khoá tài sản trước khi kiểm/tải (H2)")
    bash = git_bash()
    if not bash:
        print("  SKIP không thấy Git Bash — bỏ qua chạy thử prefetch_assets.sh")
        return
    has_flock = subprocess.run([bash, "-c", "command -v flock"], capture_output=True).returncode == 0

    def posix(p: Path) -> str:
        q = p.as_posix()
        return "/" + q[0].lower() + q[2:] if re.match(r"^[A-Za-z]:/", q) else q

    def scenario(idw: str, **extra: str):
        top = Path(tempfile.mkdtemp(prefix="prefetch_"))
        tmp = top / "thư mục có dấu cách"   # T5: mọi đường dẫn làm việc có dấu cách + chữ có dấu
        srv, fbin = tmp / "srv", tmp / "fakebin"
        srv.mkdir(parents=True); fbin.mkdir()
        _prefetch_fixture(srv, idw)
        (fbin / "curl").write_bytes(FAKE_CURL.encode())
        env = dict(os.environ, EMOCA_WSL_HOME=posix(tmp / "wsl"), EMOCA_DIR=posix(tmp / "emoca"),
                   CONDA_DIR=posix(tmp / "conda"), FAKE_SRV=posix(srv), FAKE_LOG=posix(tmp / "curl.log"),
                   EMOCA_WSL_LIB_DIR=posix(tmp / "khong_co"), PREFETCH_CONN="4", PREFETCH_PROGRESS_SEC="1",
                   PREFETCH_CHUNK_BYTES=str(PREFETCH_CHUNK), PATH_PREFIX=posix(fbin), **extra)
        env.pop("ACCEPT_FLAME_EMOCA_LICENSES", None)

        def run(accept: bool = True, **more: str):
            e = dict(env, **({"ACCEPT_FLAME_EMOCA_LICENSES": "1"} if accept else {}), **more)
            r = subprocess.run([bash, "-c", 'export PATH="$PATH_PREFIX:$PATH"; bash "$0"', path("prefetch_assets.sh").as_posix()],
                               env=e, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
            return r.returncode, r.stdout + r.stderr

        def calls() -> list[str]:
            f = tmp / "curl.log"
            return f.read_text(encoding="utf-8").splitlines() if f.exists() else []
        return top, tmp, srv, run, calls

    def gets(c: list[str], name: str) -> list[str]:
        return [l.split()[1] for l in c if l.split()[0].endswith(name) and l.split()[1] != "HEAD"]

    def tsv_rows(tmp: Path) -> list[list[str]]:
        return [l.split("\t") for l in (tmp / "wsl/out/prefetch_timings.tsv").read_text(encoding="utf-8").splitlines()[1:]]

    top, tmp, srv, run, calls = scenario("0.2", FAKE_FAIL_ONCE="FLAME.zip")
    try:
        rc, out = run(accept=False)
        check(rc == 10 and not calls(), f"không có ACCEPT_FLAME_EMOCA_LICENSES=1 ⇒ thoát 10, không gọi curl [mã {rc}]")
        hint = next((l for l in out.splitlines() if "ACCEPT_FLAME_EMOCA_LICENSES=1 bash " in l), "")
        q = hint.split("ACCEPT_FLAME_EMOCA_LICENSES=1 bash ", 1)[-1]
        back = subprocess.run([bash, "-c", 'eval "printf %s $1"', "_", q], capture_output=True, text=True, encoding="utf-8").stdout
        check(bool(hint) and back == path("prefetch_assets.sh").as_posix(),
              f"L1: lệnh gợi ý dán lại được — đường dẫn có dấu cách/Unicode quote đúng ({q[:60]}…)")
        (tmp / "emoca/assets/EMOCA/models").mkdir(parents=True)
        (tmp / "emoca/assets/EMOCA/models/EMOCA.zip").write_bytes(b"PK\x03\x04partial")   # zip dở của lần wget bị dừng
        rc, out = run()
        c = calls()
        emoca = tmp / "emoca"
        check(rc == 0 and "assets_problems: rong" in out, f"tải + giải nén xong ⇒ assets_problems rỗng, mã 0 [mã {rc}] {out[-600:] if rc else ''}")
        check((emoca / "assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg.yaml").is_file()
              and (emoca / "assets/FLAME/geometry/generic_model.pkl").stat().st_size == 1_200_000
              and (emoca / "data/EMOCA_test_example_data/images/affectnet_test_examples/a.png").is_file(),
              "giải nén đúng chỗ: assets/EMOCA/models/<model>, assets/FLAME, data/EMOCA_test_example_data")
        check((emoca / "assets/FaceRecognition/resnet50_ft_weight.pkl").is_file() and "idw=0.2 > 0" in out,
              "cfg.yaml detail.model.idw=0.2 ⇒ tải + giải nén FaceRecognition.zip")
        check(not any(l.startswith("DECA.zip") or "models/DECA.zip" in l for l in c), "không gọi tới DECA.zip")
        flame = gets(c, "FLAME.zip")
        size = (srv / "FLAME.zip").stat().st_size
        seg_starts = set(range(0, size, PREFETCH_CHUNK))
        starts = {int(r.split("-")[0]) for r in flame}
        check(len(seg_starts) == 5 and len(flame) == 6 and seg_starts <= starts and len(starts - seg_starts) == 1,
              f"FLAME.zip: 5 đoạn {PREFETCH_CHUNK} byte qua hàng đợi + 1 lần tải TIẾP từ giữa đoạn bị đứt ({sorted(flame)})")
        cache = tmp / "wsl/cache"
        check((cache / "FLAME.zip").read_bytes() == (srv / "FLAME.zip").read_bytes() and not list(cache.glob("*.parts")),
              "zip ghép giống từng byte bản gốc; đã dọn thư mục đoạn tạm")
        check(not (emoca / "assets/EMOCA/models/EMOCA.zip").exists() and list((cache / "stale").glob("EMOCA.zip.*")),
              "zip dở EMOCA.zip trong assets/EMOCA/models được CHUYỂN sang cache/stale (không xoá)")
        tsv = (tmp / "wsl/out/prefetch_timings.tsv").read_text(encoding="utf-8").splitlines()
        rows = tsv_rows(tmp)
        check(tsv[0] == "ngay_gio\ttep\tbyte_tep\tbyte_tai_lan_nay\tgiay\tKB_s\tket_noi\ttrang_thai\tket_noi_tb" and len(rows) == 4
              and all(len(r) == 9 and r[7] == "ok" and r[2] == r[3] for r in rows),
              f"prefetch_timings.tsv: 4 tệp ok, byte tải = byte tệp, có cột ket_noi_tb ({[r[1] for r in rows]})")
        check(sum("prefetch_" in l for l in (tmp / "wsl/out/timings.tsv").read_text(encoding="utf-8").splitlines()) == 4,
              "timings.tsv: thêm 1 dòng/tệp")
        check(not (tmp / "wsl/state/assets.lock.d").exists() or has_flock, "khoá tài sản được trả khi script thoát")
        n_before = len(calls())
        rc, out = run()
        again = calls()[n_before:]
        check(rc == 0 and again and all(l.endswith(" HEAD") for l in again) and "[bo qua] giai nen" in out,
              f"chạy lại: chỉ HEAD (zip trong cache khớp), không tải/giải nén lại [mã {rc}]")

        # H2: đang có tiến trình khác giữ khoá tài sản ⇒ thoát 11, không gọi curl.
        state = tmp / "wsl/state"
        pidf = tmp / "holder.pid"
        if has_flock:
            holder = subprocess.Popen([bash, "-c", 'exec 9>"$1"; flock 9; echo $$ > "$2"; sleep 60',
                                       "_", posix(state / "assets.lock"), posix(pidf)])
        else:
            holder = subprocess.Popen([bash, "-c", 'sleep 60 & echo $! > "$1"; wait', "_", posix(pidf)])
        hpid = ""
        try:
            for _ in range(100):
                if pidf.exists() and pidf.read_text().strip():
                    break
                time.sleep(0.1)
            hpid = pidf.read_text().strip()
            if not has_flock:
                boot = subprocess.run([bash, "-c", "cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo -"],
                                      capture_output=True, text=True).stdout.strip()
                (state / "assets.lock.d").mkdir()
                (state / "assets.lock.d/owner").write_bytes(f"{hpid} {boot}\n".encode())   # LF: bash đọc "pid boot_id"
            n_before = len(calls())
            rc, out = run()
            check(rc == 11 and "giu khoa tai san" in out and len(calls()) == n_before,
                  f"H2: khoá tài sản đang bị giữ ({'flock' if has_flock else 'thư mục khoá, pid sống'}) ⇒ thoát 11, không gọi curl [mã {rc}]")
        finally:
            if hpid:
                subprocess.run([bash, "-c", f"kill {hpid} 2>/dev/null; true"])
            holder.kill()
            holder.wait()
        if not has_flock:
            (state / "assets.lock.d/owner").write_bytes(b"999999 -\n")   # tiến trình đã chết / lần khởi động trước
            rc, out = run()
            check(rc == 0 and "Khoa tai san cu" in out and not (state / "assets.lock.d").exists(),
                  f"H2: khoá của tiến trình đã chết ⇒ chiếm lại, chạy tiếp, trả khoá khi xong [mã {rc}]")
    finally:
        shutil.rmtree(top, ignore_errors=True)

    # T1 (M2 + H1): máy chủ bỏ qua Range một lần (200 + cả tệp) ⇒ KHÔNG nối vào đoạn, cất vào .parts.bad (ngoài thư mục đoạn), thử lại.
    top, tmp, srv, run, calls = scenario("0.0", FAKE_IGNORE_RANGE_ONCE="FLAME.zip")
    try:
        rc, out = run()
        cache = tmp / "wsl/cache"
        bad = list((cache / "FLAME.zip.parts.bad").glob("*.http200.*"))
        check(rc == 0 and (cache / "FLAME.zip").read_bytes() == (srv / "FLAME.zip").read_bytes() and len(bad) == 1
              and bad[0].stat().st_size == (srv / "FLAME.zip").stat().st_size and len(gets(calls(), "FLAME.zip")) == 6,
              f"T1: phản hồi 200 bỏ qua Range bị cất ra .parts.bad, đoạn tải lại, zip đúng từng byte [mã {rc}]")
    finally:
        shutil.rmtree(top, ignore_errors=True)

    # Bộ đoạn KIỂU CŨ (meta "<total> 2") đang tải dở + đoạn cũ quá cỡ (H1): giữ phần đã có, chia phần thiếu thành đoạn con.
    top, tmp, srv, run, calls = scenario("0.0")
    try:
        data = (srv / "FLAME.zip").read_bytes()
        parts = tmp / "wsl/cache/FLAME.zip.parts"
        parts.mkdir(parents=True)
        (parts / "meta").write_bytes(f"{len(data)} 2\n".encode())   # LF như bash ghi
        (parts / "part.0000").write_bytes(data[:100_000])
        old = (len(data) + 1) // 2
        (parts / "part.0001").write_bytes(os.urandom(old + 10))   # quá cỡ: server từng bỏ qua Range
        rc, out = run()
        flame = gets(calls(), "FLAME.zip")
        starts = sorted(int(r.split("-")[0]) for r in flame)
        cache = tmp / "wsl/cache"
        check(rc == 0 and (cache / "FLAME.zip").read_bytes() == data and starts and starts[0] == 100_000
              and "bo doan kieu cu (2 doan)" in out and list((cache / "FLAME.zip.parts.bad").glob("part.0001.oversize.*")),
              f"bộ đoạn kiểu cũ: giữ 100000 byte đã tải (không tải lại từ 0), đoạn quá cỡ cất ra .parts.bad, zip đúng [mã {rc}] {starts[:4]}")
    finally:
        shutil.rmtree(top, ignore_errors=True)

    # T2 (M1): đủ byte nhưng sai nội dung ⇒ FAIL(unzip -tq), bỏ bộ đoạn sang .stale; chạy lại tải lại từ đầu và xong.
    top, tmp, srv, run, calls = scenario("0.0")
    try:
        data = bytearray((srv / "FLAME.zip").read_bytes())
        data[len(data) // 2] ^= 0xFF
        parts = tmp / "wsl/cache/FLAME.zip.parts"
        parts.mkdir(parents=True)
        half = (len(data) + 1) // 2
        (parts / "meta").write_bytes(f"{len(data)} 2\n".encode())   # LF như bash ghi
        (parts / "part.0000").write_bytes(bytes(data[:half]))
        (parts / "part.0001").write_bytes(bytes(data[half:]))
        rc, out = run()
        cache = tmp / "wsl/cache"
        check(rc == 1 and "FAIL(unzip -tq)" in out and not gets(calls(), "FLAME.zip") and not parts.exists()
              and list(cache.glob("FLAME.zip.parts.stale.*")) and not (cache / "FLAME.zip.tmp").exists(),
              f"T2: đoạn đủ byte nhưng hỏng ⇒ FAIL(unzip -tq), bộ đoạn chuyển .stale, xoá .tmp [mã {rc}]")
        rc, out = run()
        check(rc == 0 and (cache / "FLAME.zip").read_bytes() == (srv / "FLAME.zip").read_bytes(),
              f"T2: chạy lại ⇒ tải lại từ đầu, không kẹt [mã {rc}]")
    finally:
        shutil.rmtree(top, ignore_errors=True)

    # T3 (M3): redirect 301 có Content-Length rồi 200 chunked ⇒ KHÔNG lấy nhầm 162; redirect có Content-Length ở cuối ⇒ đúng.
    top, tmp, srv, run, calls = scenario("0.0", FAKE_HEAD_REDIRECT="FLAME.zip", FAKE_HEAD_CHUNKED="1")
    try:
        rc, out = run()
        rows = {r[1]: r for r in tsv_rows(tmp)}
        check(rc != 0 and "khong lay duoc Content-Length" in out and not gets(calls(), "FLAME.zip")
              and rows.get("FLAME.zip", [""] * 8)[7] == "FAIL(content-length)",
              f"T3: 301 (Content-Length 162) + 200 chunked ⇒ báo thiếu Content-Length, không tải 162 byte, TSV FAIL [mã {rc}]")
        rc, out = run(FAKE_HEAD_REDIRECT="all", FAKE_HEAD_CHUNKED="0")
        check(rc == 0 and (tmp / "wsl/cache/FLAME.zip").read_bytes() == (srv / "FLAME.zip").read_bytes(),
              f"T3: 301 + 200 có Content-Length ⇒ lấy kích thước của phản hồi cuối, tải đúng [mã {rc}]")
    finally:
        shutil.rmtree(top, ignore_errors=True)

    # T4: tải hỏng vĩnh viễn (HEAD được, mọi GET 404) ⇒ thử đủ PREFETCH_TRIES lần, dòng FAIL, mã 1; tệp khác vẫn xong.
    top, tmp, srv, run, calls = scenario("0.2", FAKE_FAIL_GET="FaceRecognition.zip", PREFETCH_TRIES="2")
    try:
        rc, out = run()
        rows = {r[1]: r for r in tsv_rows(tmp)}
        check(rc == 1 and rows.get("FaceRecognition.zip", [""] * 8)[7].startswith("FAIL(thieu")
              and all(rows.get(n, [""] * 8)[7] == "ok" for n in ("EMOCA_v2_lr_mse_20.zip", "FLAME.zip", "EMOCA_test_example_data.zip"))
              and len(gets(calls(), "FaceRecognition.zip")) == 2 and "[LOI] FaceRecognition.zip" in out,
              f"T4: tải hỏng vĩnh viễn ⇒ 2 lần thử, TSV FAIL(thieu…), 3 tệp khác ok, mã 1 [mã {rc}]")
    finally:
        shutil.rmtree(top, ignore_errors=True)

    top, tmp, srv, run, calls = scenario("0.0")
    try:
        rc, out = run()
        check(rc == 0 and not any(l.startswith("FaceRecognition.zip") for l in calls())
              and not (tmp / "emoca/assets/FaceRecognition").exists(),
              f"cfg.yaml detail.model.idw=0.0 ⇒ KHÔNG tải FaceRecognition.zip [mã {rc}]")
    finally:
        shutil.rmtree(top, ignore_errors=True)

    top, tmp, srv, run, calls = scenario("0.2")
    try:
        (srv / "FLAME.zip").unlink()
        rc, out = run()
        check(rc != 0 and "assets_problems" in out, f"thiếu FLAME.zip trên máy chủ ⇒ mã khác 0, báo assets_problems [mã {rc}]")
    finally:
        shutil.rmtree(top, ignore_errors=True)


def test_assets_lock_release():
    """assets_lock (LOW a của docs/qa/REVIEW_PREFETCH_2026-09-25.md, đo trong WSL 2026-09-26: .who cũ làm thông báo
    chỉ sai người giữ): trả khoá ⇒ xoá .who của mình; NỐI vào trap EXIT sẵn có (không đè); bị chặn mà .who không khớp
    người đang giữ ⇒ ghi "lan giu gan nhat (co the da cu)"."""
    c = code_lines(text("common.sh"))
    check("_assets_lock_release; eval \"$ASSETS_LOCK_PREV_TRAP\"" in c and 'rm -f "$ASSETS_LOCK.who"' in c
          and "lan giu gan nhat (co the da cu)" in c, "common.sh: assets_lock nối trap EXIT, xoá .who khi trả khoá, báo .who cũ")
    bash = git_bash()
    if not bash:
        print("  SKIP không thấy Git Bash — bỏ qua chạy thử assets_lock")
        return
    has_flock = subprocess.run([bash, "-c", "command -v flock"], capture_output=True).returncode == 0
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "EMOCA_WSL_HOME": (Path(tmp) / "w").as_posix(), "EMOCA_DIR": (Path(tmp) / "emoca").as_posix(),
               "CONDA_DIR": (Path(tmp) / "conda").as_posix()}
        state = Path(tmp) / "w/state"
        who = state / "assets.lock.who"

        def run(body: str):
            r = subprocess.run([bash, "-c", f"set -euo pipefail\nsource ./{path('common.sh').name}\n{body}"], cwd=WSL,
                               capture_output=True, text=True, env=env, encoding="utf-8", errors="replace", timeout=60)
            return r.returncode, r.stdout + r.stderr

        rc, out = run("trap 'echo TRAP_CU_VAN_CHAY' EXIT\nassets_lock thu_nghiem\ncat \"$ASSETS_LOCK.who\"\n")
        check(rc == 0 and "thu_nghiem pid " in out and "TRAP_CU_VAN_CHAY" in out and not who.exists()
              and not (state / "assets.lock.d").exists(),
              f"trả khoá: .who của mình bị xoá, trap EXIT sẵn có VẪN chạy (nối, không đè) [mã {rc}] {out.strip()[-160:]}")

        pidf = Path(tmp) / "holder.pid"
        if has_flock:
            holder = subprocess.Popen([bash, "-c", 'exec 9>"$1"; flock 9; echo $$ > "$2"; sleep 60',
                                       "_", (state / "assets.lock").as_posix(), pidf.as_posix()])
        else:
            holder = subprocess.Popen([bash, "-c", 'sleep 60 & echo $! > "$1"; wait', "_", pidf.as_posix()])
        hpid = ""
        try:
            for _ in range(100):
                if pidf.exists() and pidf.read_text().strip():
                    break
                time.sleep(0.1)
            hpid = pidf.read_text().strip()
            if not has_flock:
                boot = subprocess.run([bash, "-c", "cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo -"],
                                      capture_output=True, text=True).stdout.strip()
                (state / "assets.lock.d").mkdir()
                (state / "assets.lock.d/owner").write_bytes(f"{hpid} {boot}\n".encode())
            who.write_bytes(b"prefetch_assets.sh pid 999999 2026-09-26 08:53:42\n")   # .who còn từ lần giữ trước
            rc, out = run("assets_lock lan_hai\necho KHONG_DUOC_IN\n")
            check(rc == 11 and "KHONG_DUOC_IN" not in out
                  and "lan giu gan nhat (co the da cu): prefetch_assets.sh pid 999999" in out
                  and who.read_bytes().startswith(b"prefetch_assets.sh pid 999999"),
                  f"bị chặn, .who cũ (pid đã chết) ⇒ thoát 11, báo 'lan giu gan nhat (co the da cu)', không xoá .who của người khác [mã {rc}]")
            who.write_bytes(f"holder pid {hpid} 2026-09-26 09:00:00\n".encode())
            rc, out = run("assets_lock lan_ba\n")
            check(rc == 11 and f"(holder pid {hpid} " in out and "co the da cu" not in out,
                  f".who khớp người đang giữ ⇒ báo đúng người giữ, không ghi 'co the da cu' [mã {rc}]")
        finally:
            if hpid:
                subprocess.run([bash, "-c", f"kill {hpid} 2>/dev/null; true"])
            holder.kill()
            holder.wait()


def test_mediapipe_pin():
    """Đo 2026-09-25: mediapipe không ghim ⇒ 1.0.1, `import gdl.models.DECA` lỗi trên Python 3.8 (list[...])."""
    s = code_lines(text("setup_emoca_wsl.sh"))
    fix, pin = s.find("stage pip_opencv_chumpy_tensorboard pip_fixups"), s.find("stage pip_mediapipe_0.10.11 pip_mediapipe_pin")
    check(0 <= fix < pin and "envrun pip install --no-deps mediapipe==0.10.11" in s,
          "setup: ghim mediapipe==0.10.11 --no-deps SAU pip_fixups (không kéo lại opencv-contrib)")
    check(pin < s.find("from gdl_apps.EMOCA.utils.load import load_model") < s.find('die "Env $EMOCA_ENV chua dung duoc'),
          "setup: kiểm env import load_model (gdl.models.DECA + mediapipe) trước khi tải tài sản")


def test_texture_required():
    """Texture chỉ bắt buộc khi cfg.yaml dùng texture (2026-09-25: EMOCA_v2_lr_mse_20 có use_texture: false ở mọi stage;
    DECA.uses_texture() mặc định True khi thiếu khoá — gdl/models/DECA.py:2846-2849)."""
    c = text("common.sh")
    s = code_lines(text("setup_emoca_wsl.sh"))
    check('EMOCA_REQUIRE_TEXTURE="${EMOCA_REQUIRE_TEXTURE:-auto}"' in c and "texture_required() {" in c,
          "common: EMOCA_REQUIRE_TEXTURE mặc định auto, có hàm texture_required")
    blk = s[s.find("TEXTURE_SKIPPED=0"):]
    check(0 <= blk.find("! texture_required") < blk.find("elif ! texture_ok") < blk.find("gdown") < blk.find("exit 11"),
          "setup: không cần texture ⇒ nhánh cảnh báo TRƯỚC, gdown + mã 11 chỉ ở nhánh cần texture")
    ra = c[c.find("require_assets() {"):]
    ra = ra[:ra.find("\n}\n")]
    check("texture_required" in ra and "die" in ra and "canh bao" in ra,
          "common: require_assets (smoke/server) chỉ dừng vì thiếu texture khi texture_required")
    bash = git_bash()
    if not bash:
        print("  SKIP không thấy Git Bash — bỏ qua chạy thử texture_required")
        return
    with tempfile.TemporaryDirectory() as tmp:
        emoca = Path(tmp) / "emoca"
        cfg = emoca / "assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg.yaml"
        cfg.parent.mkdir(parents=True)
        env = {**os.environ, "EMOCA_WSL_HOME": (Path(tmp) / "w").as_posix(), "EMOCA_DIR": emoca.as_posix(),
               "CONDA_DIR": (Path(tmp) / "conda").as_posix(), "EMOCA_CFG_PYTHON": Path(sys.executable).as_posix()}
        for k in ("EMOCA_REQUIRE_TEXTURE", "EMOCA_CUDNN"):
            env.pop(k, None)

        def run(body: str, **extra):
            r = subprocess.run([bash, "-c", f"set -euo pipefail\nsource ./{path('common.sh').name}\n{body}"], cwd=WSL,
                               capture_output=True, text=True, env={**env, **extra}, encoding="utf-8", errors="replace")
            return r.returncode, r.stdout + r.stderr

        probe = 'if texture_required; then echo "CAN|$TEXTURE_REASON"; else echo "KHONG|$TEXTURE_REASON"; fi\n'
        both_false = ("coarse:\n  model:\n    n_shape: 100\n    use_texture: false\n"
                      "detail:\n  inout:\n    name: x\n  model:\n    use_texture: false\n    idw: 0\n")
        cases = [
            (both_false, {}, "KHONG", "cfg.yaml:4,9", "use_texture: false ở mọi stage ⇒ không cần, lý do ghi dòng cfg.yaml:4,9"),
            (both_false.replace("use_texture: false\n    idw", "use_texture: true\n    idw"), {}, "CAN", "detail=true@9",
             "một stage use_texture: true ⇒ cần"),
            (both_false.replace("    use_texture: false\n    idw", "    idw"), {}, "CAN", "detail=thieu",
             "thiếu khoá use_texture ở một stage (mặc định upstream = True) ⇒ cần"),
            ("coarse: [khong, hop, le\n", {}, "CAN", "loi", "cfg.yaml hỏng cú pháp ⇒ cần (an toàn)"),
            (None, {}, "CAN", "khong co", "không có cfg.yaml ⇒ cần (an toàn)"),
            (None, {"EMOCA_REQUIRE_TEXTURE": "0"}, "KHONG", "EMOCA_REQUIRE_TEXTURE=0", "EMOCA_REQUIRE_TEXTURE=0 ⇒ không cần dù thiếu cfg"),
            (both_false, {"EMOCA_REQUIRE_TEXTURE": "1"}, "CAN", "EMOCA_REQUIRE_TEXTURE=1", "EMOCA_REQUIRE_TEXTURE=1 thắng cfg use_texture: false"),
        ]
        for content, extra, want, reason, msg in cases:
            if content is None:
                cfg.unlink(missing_ok=True)
            else:
                cfg.write_text(content, encoding="utf-8")
            rc, out = run(probe, **extra)
            line = next((l for l in out.splitlines() if l.startswith(("CAN|", "KHONG|"))), "")
            check(rc == 0 and line.startswith(want + "|") and reason in line, f"texture_required: {msg} [{line[:150] or out[-200:]}]")
        rc, out = run("echo KHONG_DUOC_IN\n", EMOCA_REQUIRE_TEXTURE="yes")
        check(rc != 0 and "KHONG_DUOC_IN" not in out and "EMOCA_REQUIRE_TEXTURE" in out, "EMOCA_REQUIRE_TEXTURE sai ⇒ dừng khi source")
        # require_assets (smoke/server): tài sản khác coi như đủ (assets_problems giả rỗng), texture KHÔNG có.
        ra_body = 'assets_problems() { :; }\nrequire_assets\necho QUA_REQUIRE_ASSETS\n'
        cfg.write_text(both_false, encoding="utf-8")
        rc, out = run(ra_body)
        check(rc == 0 and "QUA_REQUIRE_ASSETS" in out and "bo qua texture" in out,
              f"require_assets: thiếu texture + use_texture false ⇒ chỉ cảnh báo, chạy tiếp [mã {rc}]")
        cfg.write_text(both_false.replace("use_texture: false\n    idw", "use_texture: true\n    idw"), encoding="utf-8")
        rc, out = run(ra_body)
        check(rc != 0 and "QUA_REQUIRE_ASSETS" not in out and "Thieu texture" in out,
              f"require_assets: thiếu texture + use_texture true ⇒ dừng như cũ [mã {rc}]")


def main() -> int:
    for t in (test_files_and_line_endings, test_bash_syntax, test_no_colab_or_tunnel, test_license_gate,
              test_fail_fast_order, test_yaml_matches_notebook, test_commands_mirror_notebook, test_server_invocation,
              test_bundle_check_logic, test_stage_behaviour, test_stage_calls_are_plain, test_review_fixes,
              test_cudnn_fallback, test_prefetch_assets, test_assets_lock_release, test_texture_required,
              test_mediapipe_pin):
        print(f"[{t.__name__}]")
        t()
    if failures:
        print(f"\nFAIL {len(failures)} kiểm")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
