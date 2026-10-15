"""Kiểm gói triển khai Colab (ADR-015): backend/services/emoca_svc/pack_for_colab.py + another/notebooks/emoca_colab.ipynb.

Chạy:  python tests/test_colab_bundle.py

Đo được ở máy thường: gói zip tất định, không kèm tài sản có giấy phép, xuống dòng LF, patch áp được lên bản LF sạch của
EMOCA, notebook là JSON hợp lệ, mọi ô Python biên dịch được, tham số notebook khớp `server.py` và script upstream,
cờ giấy phép mặc định False. KHÔNG đo được ở đây: chạy thật trên Colab (dựng env, GPU, tunnel) — chưa có bằng chứng.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
SVC = ROOT / "backend" / "services" / "emoca_svc"
sys.path.insert(0, str(SVC))
import pack_for_colab  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gitbash import git_bash  # noqa: E402

NOTEBOOK = ROOT / "another" / "notebooks" / "emoca_colab.ipynb"
OLD_NOTEBOOK = ROOT / "another" / "notebooks" / "emoca_setup_v2_work38_ok.ipynb"
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def src(cell) -> str:
    return "".join(cell["source"])


def python_part(code: str) -> str:
    """Bỏ các dòng đặc thù IPython (`!lệnh`, `%magic`, phần nối dòng của lệnh shell) để biên dịch phần Python."""
    lines = code.splitlines()
    if lines and lines[0].startswith("%%"):  # %%writefile: phần còn lại là nội dung tệp
        lines = lines[1:]
    out, skip_cont = [], False
    for ln in lines:
        if skip_cont:
            skip_cont = ln.rstrip().endswith("\\")
            continue
        if ln.lstrip().startswith(("!", "%")):
            skip_cont = ln.rstrip().endswith("\\")
            out.append("pass" if ln.startswith(("!", "%")) else ln[: len(ln) - len(ln.lstrip())] + "pass")
            continue
        out.append(ln)
    return "\n".join(out)


def test_bundle():
    print("== gói zip")
    data1, manifest = pack_for_colab.build_bundle()
    data2, _ = pack_for_colab.build_bundle()
    check(data1 == data2, "tất định: hai lần đóng gói ra cùng từng byte")
    zf = zipfile.ZipFile(io.BytesIO(data1))
    names = set(zf.namelist())
    need = {"server.py", "emoca_runner.py", "mock_server.py", "patches/apply.sh", "patches/revert.sh",
            "patches/0001-numpy-remove-np-int-alias.patch", "patches/0002-face-alignment-landmarkstype-compat.patch",
            "contracts/emoca_analyze.schema.json", "configs/services.yaml", "MANIFEST.json"}
    check(need <= names, f"đủ tệp cần thiết (thiếu: {sorted(need - names)})")
    man = json.loads(zf.read("MANIFEST.json"))
    check(all(hashlib.sha256(zf.read(f["path"])).hexdigest() == f["sha256"] for f in man["files"]), "MANIFEST khớp sha256 từng tệp")
    check(man["contract_version"] == "0.3.0" and man["upstream_commit_expected"] == "e0be0db", f"manifest: contract {man['contract_version']}, upstream {man['upstream_commit_expected']}")
    check(all(b"\r" not in zf.read(n) for n in names), "không còn CRLF trong tệp nào (script .sh CRLF hỏng trên Linux/Colab)")
    check(not any(Path(n).suffix.lower() in pack_for_colab.DENY_SUFFIXES for n in names), "không có tệp đuôi tài sản có giấy phép/nặng")
    check(max(i.file_size for i in zf.infolist()) < pack_for_colab.MAX_BYTES, "mọi tệp < 2 MiB (chỉ là mã)")
    check(all((i.external_attr >> 16) & 0o111 for i in zf.infolist() if i.filename.endswith(".sh")), ".sh có bit thực thi")
    check(all(i.date_time == pack_for_colab.FIXED_TIME for i in zf.infolist()), "dấu thời gian cố định")
    secret = re.compile(r"(?i)(token|password|secret)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]")
    check(not any(secret.search(zf.read(n).decode("utf-8", "replace")) for n in names if n.endswith((".py", ".yaml", ".sh"))),
          "không có token/mật khẩu nhúng cứng")
    for n in ("server.py", "emoca_runner.py", "mock_server.py"):
        try:
            ast.parse(zf.read(n).decode("utf-8"), feature_version=(3, 8))
            ok = True
        except SyntaxError:
            ok = False
        check(ok, f"{n} trong gói hợp lệ với Python 3.8 (env work38)")
    # Từ chối tệp bị cấm
    fake_root = Path(tempfile.mkdtemp())
    try:
        for rel in [s for s, _ in pack_for_colab.FILES] + ["services/emoca_svc/patches/0001-x.patch"]:
            p = fake_root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x\n", encoding="utf-8")
        (fake_root / "services/emoca_svc/patches/0002-big.patch").write_bytes(b"x" * (pack_for_colab.MAX_BYTES + 1))
        (fake_root / "services/emoca_svc/mock_server.py").write_text('CONTRACT_VERSION = "9"\n', encoding="utf-8")
        (fake_root / "services/emoca_svc/patches/apply.sh").write_text('EXPECTED_COMMIT="abc"\n', encoding="utf-8")
        try:
            pack_for_colab.build_bundle(fake_root)
            check(False, "tệp .patch quá 2 MiB phải bị từ chối")
        except ValueError as e:
            check("lớn" in str(e), "tệp quá cỡ bị từ chối")
    finally:
        shutil.rmtree(fake_root, ignore_errors=True)
    return data1


def test_patches_apply(bundle: bytes):
    print("== patch trong gói áp được lên EMOCA (bản LF sạch)")
    emoca = ROOT / "backend" / "model" / "emoca"
    bash, git = git_bash(), shutil.which("git")  # Git Bash, không phải cửa vào WSL (tests/_gitbash.py)
    if not (emoca / ".git").exists() or not bash or not git:
        print("  SKIP không có model/emoca (git) hoặc Git Bash/git — bỏ qua")
        return
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run([git, "-c", "core.autocrlf=false", "clone", "-q", "--no-checkout", str(emoca), str(tmp / "emoca")], check=True, capture_output=True)
        subprocess.run([git, "-c", "core.autocrlf=false", "-C", str(tmp / "emoca"), "checkout", "-q", "e0be0db"], check=True, capture_output=True)
        zipfile.ZipFile(io.BytesIO(bundle)).extractall(tmp / "svc")
        target = (tmp / "emoca").as_posix()
        run = lambda *a: subprocess.run([bash, *a], capture_output=True, text=True, cwd=tmp)  # noqa: E731
        r = run((tmp / "svc" / "patches" / "apply.sh").as_posix(), target)
        check(r.returncode == 0 and r.stdout.count("[ap    ]") == 2, f"apply.sh: mã 0, áp 2 patch bắt buộc ({r.stdout.strip().splitlines()[:2]} {r.stderr[:120]})")
        deca = (tmp / "emoca" / "gdl/utils/DecaUtils.py").read_bytes().replace(b"\r", b"")
        fan = (tmp / "emoca" / "gdl/utils/FaceDetector.py").read_bytes().replace(b"\r", b"")
        check(b"astype(np.int32)" in deca and b"astype(np.int)\n" not in deca[deca.find(b"def load_local_mask"):deca.find(b"def load_local_mask") + 1200],
              "0001: np.int → np.int32 đã có trong load_local_mask")
        check(b"TWO_D" in fan, "0002: FaceDetector hỗ trợ LandmarksType.TWO_D")
        r = run((tmp / "svc" / "patches" / "apply.sh").as_posix(), target)
        check(r.returncode == 0 and "[da ap ]" in r.stdout and "[ap    ]" not in r.stdout, "áp lần hai: idempotent (báo 'đã áp')")
        r = run((tmp / "svc" / "patches" / "revert.sh").as_posix(), target)
        deca = (tmp / "emoca" / "gdl/utils/DecaUtils.py").read_bytes().replace(b"\r", b"")
        check(r.returncode == 0 and b"astype(np.int)" in deca, "revert.sh gỡ được (np.int trở lại)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_notebook():
    print("== notebook")
    check(NOTEBOOK.is_file(), f"có {NOTEBOOK.relative_to(ROOT)} (sinh bằng script trong phiên; giữ tệp này làm nguồn)")
    if not NOTEBOOK.is_file():
        return
    raw = NOTEBOOK.read_text(encoding="utf-8")
    nb = json.loads(raw)
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    check(nb["nbformat"] == 4 and nb["metadata"].get("accelerator") == "GPU", "nbformat 4, đặt accelerator GPU")
    check(all(isinstance(c["source"], list) for c in cells), "mọi ô có source dạng danh sách dòng")
    check(all(c.get("outputs") == [] and c.get("execution_count") is None for c in code_cells), "ô code sạch kết quả (không rò token/đầu ra)")

    bad = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            compile(python_part(src(c)), f"ô {i}", "exec")
        except SyntaxError as e:
            bad.append((i, e.msg, e.lineno))
    check(not bad, f"mọi ô code biên dịch được sau khi bỏ dòng IPython ({len(code_cells)} ô; lỗi: {bad})")

    text = "\n".join(src(c) for c in cells)
    check(re.search(r"^ACCEPT_FLAME_EMOCA_LICENSES = False\b", text, re.M) is not None, "cờ giấy phép mặc định False (người dùng tự đổi)")
    for c in code_cells:
        s = src(c)
        if "download_assets.sh" in s and "yes |" in s:
            check("assert ACCEPT_FLAME_EMOCA_LICENSES" in s and s.index("assert ACCEPT_FLAME_EMOCA_LICENSES") < s.index("yes |"),
                  "'yes | download_assets.sh' chỉ chạy sau assert cờ giấy phép")
    check(len(re.findall(r"yes \|", text)) == 1, "chỉ MỘT chỗ trả lời tự động 'y' (ô tải tài sản đã có chốt)")
    check(re.search(r"--token\b", text) is None and "EMOCA_TOKEN=TOKEN" in text, "token truyền cho server bằng biến môi trường, không qua dòng lệnh")
    check(re.search(r"TOKEN = secrets\.token_urlsafe", text) is not None and not re.search(r"TOKEN\s*=\s*[\"'][A-Za-z0-9_\-]{16,}[\"']", text),
          "token sinh ngẫu nhiên mỗi phiên, không nhúng cứng")

    # Tham số notebook ↔ server.py
    m = re.search(r'python -u server\.py "\n?\s*f?"?(.*?)\)', text, re.S)
    server_src = (SVC / "server.py").read_text(encoding="utf-8")
    flags = set(re.findall(r"--[a-z][a-z-]+", text[text.find("python -u server.py"): text.find("python -u server.py") + 400]))
    defined = set(re.findall(r'add_argument\("(--[a-z-]+)"', server_src))
    check(flags and flags <= defined, f"cờ dòng lệnh notebook dùng cho server.py đều có trong server.py ({sorted(flags)})")
    # Tham số notebook ↔ script upstream
    up = ROOT / "backend" / "model" / "emoca" / "gdl_apps" / "EMOCA" / "demos" / "test_emoca_on_images.py"
    if up.is_file():
        up_flags = set(re.findall(r"add_argument\('(--[a-z_]+)'", up.read_text(encoding="utf-8")))
        used = set(re.findall(r"--[a-z_]+", text[text.find("python test_emoca_on_images.py"): text.find("python test_emoca_on_images.py") + 300]))
        check(used and used <= up_flags, f"cờ notebook dùng cho test_emoca_on_images.py có trong script upstream ({sorted(used)})")
    # Đường dẫn HTTP ↔ mock_server/server
    mock_src = (SVC / "mock_server.py").read_text(encoding="utf-8")
    check("/v1/health" in text and "/v1/analyze" in text and '"/v1/health"' in mock_src and '"/v1/analyze"' in mock_src, "đường dẫn /v1/health, /v1/analyze khớp dịch vụ")

    # Trung thành với notebook đã kiểm chứng
    if OLD_NOTEBOOK.is_file():
        old = json.loads(OLD_NOTEBOOK.read_text(encoding="utf-8"))
        old_yaml = src(old["cells"][10])
        check(old_yaml in text, "environment.yml sao chép NGUYÊN VĂN từ notebook đã chạy được")
        for i in (2, 3, 13, 14, 15, 17, 18, 21, 24, 26, 27, 28, 30, 32, 34):
            check(src(old["cells"][i]).strip() in text, f"ô {i} của notebook cũ có nguyên văn trong notebook mới")

    # Thứ tự các bước
    order = ["Miniconda", "git clone", "environment.yml", "mamba env update", "pip install -e .", "yes | bash download_assets.sh",
             "bash /content/svc/patches/apply.sh", "python test_emoca_on_images.py", "python -u server.py", "cloudflared", "make_archive"]
    pos = [text.find(k) for k in order]
    check(all(p >= 0 for p in pos) and pos == sorted(pos), f"thứ tự bước hợp lý (vị trí: {pos})")
    check("git checkout {EMOCA_COMMIT}" in text and 'EMOCA_COMMIT = "e0be0db"' in text, "ghim commit e0be0db trước khi áp patch")
    check(re.search(r"os\.environ\['PATH'\] = \"/content/miniconda/bin:\"", text) is not None, "PATH có miniconda (cần cho `source activate work38`)")
    check("google.colab" in text and "files.upload()" in text, "tải bundle bằng ô Upload của Colab")

    print("== notebook: các sửa theo review ADR-015 (docs/qa/REVIEW_ADR015_EMOCA_COLAB.md)")
    check("assets/data/EMOCA_test_example_data" not in text.replace("không phải assets/data/", ""),
          "D1: không đặt cứng EX_DIR = assets/data/... (download_assets.sh giải nén vào /content/emoca/data/...)")
    ex_def = [i for i, c in enumerate(code_cells) if re.search(r"^EX_DIR\s*=", src(c), re.M)]
    dl = next((i for i, c in enumerate(code_cells) if "yes | bash download_assets.sh" in src(c)), -1)
    first_use = next((i for i, c in enumerate(code_cells)
                      if i not in ex_def and ("{EX_DIR}" in src(c) or "EX_DIR +" in src(c))), -1)
    check(len(ex_def) == 1 and dl < ex_def[0] < first_use and "glob.glob(" in src(code_cells[ex_def[0]])
          and "recursive=True" in src(code_cells[ex_def[0]]) and "assert found" in src(code_cells[ex_def[0]]),
          f"D1: EX_DIR được TÌM bằng glob (recursive) + assert, sau ô tải tài sản, trước lần dùng đầu (ô code {ex_def}, tải {dl}, dùng {first_use})")
    smoke = next((src(c) for c in code_cells if "python test_emoca_on_images.py" in src(c)), "")
    check("please check the test path" in smoke and "raise" in smoke,
          "D1: ô M0-04 báo LỖI khi test_emoca_on_images.py in 'please check the test path' (TestData thoát mã 0)")
    tunnel_at = text.find("cloudflared")
    post_via_url = re.search(r'call\(URL, "/v1/analyze"', text)
    check(post_via_url is not None and post_via_url.start() > tunnel_at and "jsonschema.validate(body, schema)" in text[post_via_url.start():],
          "D2: sau khi mở tunnel, POST thử /v1/analyze QUA URL tunnel và kiểm schema")
    check(re.search(r'print\([^)\n]*,\s*TOKEN\)', text) is not None and "clear_output()" in text
          and text.find("clear_output()") > text.find('print("Token:", TOKEN)'),
          "L6: token chỉ hiện ở một ô, rồi clear_output() sau khi người dùng xác nhận đã sao chép")
    check(len(re.findall(r'print\([^\n]*\bTOKEN\)', text)) == 1, "L6: chỉ MỘT chỗ in token đầy đủ")
    check(text.rfind("_proc.terminate()") > text.find("make_archive") and "tunnel" in text[text.rfind("_proc.terminate()") - 300:],
          "L6: có ô cuối tắt server + tunnel")
    check("CHƯA được ghim phiên bản" in src(cells[0]), "L7: ô đầu ghi rõ Miniconda/mamba/tunnel chưa ghim phiên bản")


def main() -> int:
    bundle = test_bundle()
    test_patches_apply(bundle)
    test_notebook()
    print()
    if failures:
        print(f"FAIL — {len(failures)} lỗi")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
