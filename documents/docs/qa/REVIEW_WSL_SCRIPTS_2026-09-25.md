# Rà soát script dựng EMOCA trong WSL2 — 2026-09-25

- **Vai trò:** 01_AI/QA, giao việc từ phiên điều phối `btl-bb`.
- **Phạm vi:** `backend/services/emoca_svc/wsl/{common.sh, setup_emoca_wsl.sh, smoke_test_emoca.sh, run_emoca_server.sh, environments/emoca_wsl/environment_work38.yml}`.
- **Cách làm:** đọc từng dòng, CHỈ ĐỌC. Đối chiếu với `another/notebooks/emoca_colab.ipynb` (ô 2–48), `backend/services/emoca_svc/{server.py, emoca_runner.py, mock_server.py, pack_for_colab.py, patches/apply.sh}`, `another/tests/test_wsl_scripts.py` và mã upstream `backend/model/emoca@e0be0db` (`setup.py`, `pull_submodules.sh`, `gdl_apps/EMOCA/demos/{download_assets.sh, test_emoca_on_images.py}`, `requirements38.txt`, `gdl/models/IO.py`).
- **Giới hạn:** không chạy gì trong WSL.
- **Số dòng** dẫn theo bản script TRƯỚC khi sửa (bản đang chạy lúc rà soát).
- **Nhãn mỗi mục:** "đã xác minh" = đọc mã hoặc số đo điều phối cung cấp; "chưa kiểm" = cần WSL/GPU thật.
- **Kiểm tĩnh:** `bash -n` (Git Bash) cả 4 script OK; không tệp nào có CR.

**Kết luận:** không có BLOCKER. 4 HIGH, 7 MED (cộng các mục MED về hồi phục trong H4), nhiều LOW.

## Xác minh 4 điểm do điều phối đo trên máy thật

1. **`nvidia-smi` không đối số bị segfault.** Đã xác minh: script không gọi dạng trần.
   - Các chỗ gọi: `common.sh:92` (`-L`), `common.sh:94` (name,memory.total,driver_version), `setup:240` (thêm memory.used), `smoke:103` và `run:31` (memory.used,memory.total).
   - Riêng `setup:240` là lệnh đứng riêng dưới `set -e`: nếu segfault thì script chết. Điều phối đo được mã 0.
   - **Tài liệu:** `documents/docs/RUNBOOK_DEMO.md:155` bảo gõ `nvidia-smi` trần → đổi thành `nvidia-smi -L` (việc của DOCS).
2. **VRAM** → H1.
3. **Dung lượng.** Đã xác minh: `setup:46-48` và `record_versions` `:260` đọc `df "$HOME"`. Đây là `ext4.vhdx` động (~1 TB) nên cảnh báo không bao giờ bật. Giới hạn thật là ổ Windows chứa `.vhdx`, không nhất thiết là C (RUNBOOK:165 gợi ý chuyển sang E). Xếp MED (M8).
4. **Nhánh apt** (`setup:50-58`) → M9, M10 và LOW.
   - Đã xác minh là đúng: key `apt_N_goi` kèm `rm -f apt_*.done`, và `stage` nằm trong thân `if` nên `set -e` vẫn có hiệu lực.

## HIGH

- **H1 — Số VRAM sai.**
  - Vị trí: `smoke:102-105` và `:188-190`, `run:28-33`, `setup:240`; lời hứa trong tài liệu ở `smoke:7`, `run:9-10`, RUNBOOK:195.
  - Vấn đề: `memory.used` luôn bằng 0 dưới WDDM (điều phối đã đo) ⇒ "đỉnh 0 / 4096" sẽ bị ghi vào nghiệm thu M0-06.
  - Số của server cũng thiếu: `vram_peak_mb` = `max_memory_allocated` (`emoca_runner.py:423`). Nó không tính CUDA context/cuDNN, không tính phần reserved và không tính ~800 MB Windows đang giữ.
  - Đề xuất:
    - `/v1/health` báo `gpu_free_mb`/`gpu_total_mb` (`torch.cuda.mem_get_info`) và `vram_reserved_peak_mb`.
    - Lấy mẫu bằng `GET /v1/health`, KHÔNG dùng một tiến trình torch riêng (nó tự tạo context ~300 MB).
    - Sửa `another/tests/test_wsl_scripts.py:142`.
  - Đã xác minh bằng đọc mã.
- **H2 — Bản sao trong `~/emoca_wsl/bin` bị cũ.**
  - Vị trí: `setup:126-132`. Khi chạy từ bin, `-ef` bỏ qua việc chép lại.
  - Hệ quả: sau khi sửa dự án, lệnh `bash ~/emoca_wsl/bin/...` (`setup:271,302-303,318-319`; RUNBOOK:181,191,200) vẫn chạy bản cũ mà không báo gì.
  - Đề xuất: chép đè mọi tệp khác bằng `mv` (inode mới). KHÔNG ghi đè tệp đang chạy (bash đọc script dần dần; tệp được `source` thì đã đọc hết). Nếu chính tệp đang chạy khác bản dự án thì dừng và bảo chạy lại bằng đường dẫn dự án.
  - Đã xác minh.
- **H3 — Probe không chạy kernel CUDA nào.**
  - Vị trí: `setup:242-251`, chỉ `import`.
  - Bản dựng 2025 chỉ chạy trên T4 (sm_75). RTX 3050 là sm_86, và torch 1.11/cu115 cùng pytorch3d 0.6.2 chưa ai thử trên kiến trúc này.
  - Hệ quả: nếu thiếu kernel thì phải tới smoke mới lỗi, sau khi đã tải vài GB tài sản.
  - Đề xuất: in capability/arch_list, chạy conv2d trên cuda, chạy `pytorch3d.ops.knn_points` trên cuda, chạy `pip check`.
  - Chưa kiểm trên GPU thật.
- **H4 — Không hồi phục được sau khi VM tắt đột ngột.**
  - Bối cảnh: sự cố thật lúc 11:33:57 — `wslservice.exe` STACK_OVERFLOW (WSL 2.7.14.0) để lại `~/miniconda3` toàn tệp 0 byte.
  - Nguyên nhân: ext4 cấp phát trễ ⇒ tệp MỚI tạo mà chưa ghi xuống đĩa sẽ thành 0 byte.
  - (a) Các phép kiểm "đã có" chỉ xem tệp có tồn tại, không chạy thử. Tệp 0 byte vẫn giữ bit `+x`, nên các chặng bị bỏ qua:
    - `setup:136` `-x bin/conda`
    - `:187` `-x bin/mamba`
    - `:194` `-x python`
    - `:161` `-d .git`
    - `:278` `-f cfg.yaml`
  - (b) Tệp mốc `.done` (`common.sh:75`) có thể còn mà rỗng, trong khi dữ liệu của chặng là 0 byte.
  - (c) `EMOCA_WSL_REDO=1` không cứu được vì các phép kiểm ở (a) vẫn bỏ qua việc cài.
  - Đề xuất:
    - `sync` trước và sau khi ghi mốc; mốc 0 byte coi như chưa có.
    - Kiểm bằng CHẠY THẬT (`conda --version`, `mamba --version`, `python -c`, `git rev-parse/status`).
    - Miniconda hỏng: KHÔNG `rm -rf`. Chỉ `mv` sang `.broken.<epoch>` khi có tệp đánh dấu `.emoca_wsl_owned` do script tạo; không có thì `die` kèm hướng dẫn. Không dùng `-u`, vì nó giữ lại `pkgs/` và `conda-meta` hỏng.
    - Tải installer ra tệp tạm và kiểm kích thước.
  - Từng chặng khi sập giữa chừng:
    - **apt** (MED): gói ở trạng thái half-configured vẫn qua `dpkg -s`; cần `sudo dpkg --configure -a`.
    - **Gói svc:** tự lành (giải nén vào `.new` rồi `mv`).
    - **Miniconda** (HIGH): như trên.
    - **`~/.condarc`** (MED): tệp 0 byte vẫn là YAML hợp lệ nên kênh mất mà không báo ⇒ chạy lại chặng kênh mỗi lần.
    - **clone** (MED, KẸT): `.git` dở hoặc có "object file is empty" làm checkout lỗi mãi. Tệp đã vá bị 0 byte làm `apply.sh` exit 1 mãi.
      - Cách xử lý: repo hỏng thì dời sang tên khác rồi clone lại. `assets/` và `data/` KHÔNG được upstream theo dõi (đã xác minh: `git ls-files` = 0) nên chuyển lại được.
      - Chỉ tệp đã vá bị hỏng thì `git checkout -- .` rồi áp lại patch.
    - **Submodule** (LOW-MED): `git submodule update` thường tự lành; phép kiểm `:178` không bắt được tệp 0 byte.
    - **env** (HIGH): `conda-meta/*.json` 0 byte dẫn tới JSONDecodeError; `pkgs/` hỏng bị dùng lại.
      - Cách xử lý: dời env đi, `conda clean --packages --tarballs`, tạo lại, xoá mốc env update và pip.
    - **pip** (MED): `dist-info` hoặc `.py` 0 byte ⇒ thêm `pip check`; khi nghi ngờ thì làm lại env.
    - **Tài sản** (MED): `cfg.yaml` 0 byte vẫn qua phép kiểm → M1.
    - **texture:** gần như tự lành (ngưỡng > 1 MB); tệp cụt mà vẫn > 1 MB thì lọt ⇒ thêm `numpy.load`.
    - **smoke:** tự lành (M0-04 luôn chạy lại).
  - Ghi chú vận hành: sau khi WSL sập, chạy lại bằng ĐƯỜNG DẪN DỰ ÁN, vì bản trong bin có thể thành 0 byte và `bash` tệp rỗng thoát 0 mà không làm gì.
  - Ngoài phạm vi script: cân nhắc `wsl --update` (người dùng quyết).

## MED

- **M1 — Không phát hiện tải tài sản hỏng.**
  - Vị trí: `setup:277-284`. Upstream `download_assets.sh` không có `set -e` và thoát theo mã của `cd` cuối cùng ⇒ lỗi `wget`/`unzip` bị che. Script chỉ kiểm `cfg.yaml`.
  - Đề xuất:
    - Kiểm `cfg.yaml` không rỗng.
    - Kiểm có `*.ckpt` > 1 MB (upstream tìm `rglob("*.ckpt")`, `gdl/models/IO.py:34`).
    - Kiểm `assets/FLAME` có tệp > 1 MB và có ảnh mẫu trong `data/`.
    - `unzip -tq` các zip còn lại.
    - Grep log tìm lỗi.
- **M2 — Không ghim phiên bản Miniconda3-latest và mamba** (`setup:138,188`; notebook cũng vậy). Ghim sau lần dựng thành công đầu tiên — CHỈ ghi TODO.
- **M3 — Chưa đo localhost WSL2 → Windows** (`run:43`). Đề xuất một bước thử không cần GPU: `python3 -m http.server --bind 127.0.0.1` trong WSL, gọi từ Windows.
- **M4 — Smoke có thể tranh VRAM với server chính.** Smoke dùng cổng 8766 và không kiểm 8765 ⇒ có thể có hai EMOCA trên 4 GB; `memory.used`=0 nên không ai thấy.
- **M5 — `conda tos accept` tự đồng ý điều khoản kênh Anaconda thay người dùng** (`setup:150-152`, giống notebook ô 11). Ít nhất phải in rõ vào log.
- **M6 — Không chặn đường dẫn làm việc trên ổ Windows.** `EMOCA_WSL_HOME`/`EMOCA_DIR`/`CONDA_DIR` có thể trỏ vào `/mnt/*` (`common.sh:15-17`).
- **M7 — Lệch notebook ở bước gỡ opencv.** Notebook ô 24 gỡ `opencv-contrib-python-headless` KHÔNG có `-y` (nhiều khả năng bị huỷ trên Colab); script dùng `-y` (`setup:231`). Ghi làm lệch có chủ đích; so `pip freeze`.
- **M8 — Kiểm dung lượng sai ổ** (điểm 3). Thêm `EMOCA_HOST_DRIVE` (mặc định `/mnt/c`) và `df` chỉ đọc.
- **M9 — `sudo` cần TTY.** `sudo apt-get` (`setup:55`) chạy không tương tác thì lỗi. Đề xuất: `sudo -n true`; nếu không có TTY thì in đúng lệnh cho người dùng và thoát mã 12.
- **M10 — `dpkg -s` trả 0 cho gói ở trạng thái `rc`/half-configured** (`setup:52`). Dùng `dpkg-query -W -f='${Status}'` và so với "install ok installed".

## LOW

- `common.sh:92`: `nvidia-smi -L | grep -q` dưới pipefail có thể bị SIGPIPE ⇒ `die` oan.
- `setup:102/107`: MANIFEST liệt kê tệp không có trong zip ⇒ KeyError và traceback thay vì báo BAD.
- `setup:167` (cùng `apply.sh:23`): `--short=7` có thể dài hơn 7 ký tự ⇒ nên so hash đầy đủ.
- `smoke:93-100`: `source activate` trong subshell nền có thể thất bại âm thầm.
- `run:29-44`: `server.log` và CSV cứ nối dài mãi.
- `git-lfs` được cài nhưng không chạy `git lfs install` ⇒ `GIT_LFS_SKIP_SMUDGE` không có tác dụng (vô hại cho suy luận ảnh).
- Tệp texture cụt mà vẫn > 1 MB thì lọt phép kiểm.

## Đã kiểm và ổn

- **Chốt giấy phép** (`setup:265-275`):
  - Chỉ nhận đúng giá trị `"1"`, không lưu vào state, không script nào tự đặt biến.
  - Việc tải texture nằm sau chốt.
  - `(yes||true) |` trả lời đúng 2 câu `read -p` của `download_assets.sh`.
  - Quy trình: chỉ NGƯỜI DÙNG gõ `ACCEPT_FLAME_EMOCA_LICENSES=1`, agent không được đặt thay.
- **`rm -rf`:** mọi đường dẫn đều dẫn xuất từ `EMOCA_WSL_HOME`, biến này dùng `${:-}` cùng `set -u`.
- **`/mnt/c` chỉ đọc:** `pack_for_colab.py` chỉ import stdlib nên không sinh `__pycache__`, và ghi `--out` vào `~/emoca_wsl`.
- **Hàm `stage`** đúng, và mọi lời gọi là lệnh đơn.
- **Tham số `server.py`** khớp `server.py:88-94`. Token chỉ đi qua biến môi trường, không vào log (`mock_server.py:238`).
- **Tham số M0-04** khớp `test_emoca_on_images.py:37-53`. `setup.py:18` có `gdl_apps` nên `emoca_runner` import được.
- **Thứ tự pip/mamba** khớp notebook. Các chỗ lệch có chủ đích:
  - patch áp sớm (vô hại vì `pip -e` là editable);
  - bước opencv/chumpy/tensorboard chạy trước khi tải tài sản;
  - `GIT_LFS_SKIP_SMUDGE` cả khi kéo submodule;
  - `gdown` dạng URL;
  - bỏ `--save_mesh`.
- `requirements38.txt:36` có `numpy>=1.23.1`, nâng bản của YAML (1.21.6) giống notebook — đây là lý do cần patch 0001.
- **YAML** trùng từng dòng với notebook, chỉ khác `prefix`.

## Bổ sung từ điều phối (đã đo trên máy thật)

- **H5 (HIGH)** — nguyên nhân lần hỏng thứ hai.
  - Windows không có pagefile ⇒ hạn mức commit = RAM vật lý 15,7 GB. Commit còn trống chỉ 0,71 GB dù RAM trống 4 GB.
  - Lỗi quan sát được: `wsl.exe` báo `E_UNEXPECTED` / `OutOfMemoryException` ở bước mamba "Solving environment".
  - Đề xuất: kiểm tiền đề qua interop (`powershell.exe`: `Win32_OperatingSystem.FreeVirtualMemory`, pagefile). Chỉ cảnh báo, không dừng.
- **H4 bổ sung:** mọi phép kiểm git dùng `--ignore-submodules=all`. Submodule spectre (phần video, không cần cho ảnh) có object 0 byte sau lần sập.

## Trạng thái sửa (2026-09-25)

**Nguyên tắc:** `setup_emoca_wsl.sh` và `common.sh` đang được chạy thật ⇒ bản sửa nằm trong **`setup_emoca_wsl.sh.next`** và **`common.sh.next`** (tệp mới). Điều phối đổi tên bằng `mv` sau lần dựng. KHÔNG chạy trực tiếp bản `.next`: nó `source common.sh`, nên phải hoán đổi cả hai cùng lúc. `smoke_test_emoca.sh`, `run_emoca_server.sh` và `another/tests/test_wsl_scripts.py` được sửa trực tiếp; `smoke`/`run` chạy được với cả `common.sh` cũ lẫn mới.

| Mục | Trạng thái | Ở đâu |
|---|---|---|
| H1 | ĐÃ SỬA (phía script) | smoke/run lấy mẫu `GET /v1/health` bằng python3 stdlib (không torch) → `smoke_vram_health.json` / `server_vram_health.json`: min `gpu_free_mb`, max `vram_reserved_peak_mb`/`vram_peak_mb`; trường vắng ⇒ "khong co". Bỏ `nvidia-smi memory.used` ở mọi nơi. Các trường mới của `/v1/health` do phiên btl-06 thêm vào `emoca_runner.py`. |
| H2 | ĐÃ SỬA | `setup.next` `refresh_bin`: nguồn = `wsl/` của dự án; thay bằng `mv` (inode mới); tệp đang chạy mà khác bản dự án ⇒ `die`, bảo chạy lại bằng đường dẫn dự án. |
| H3 | ĐÃ SỬA | probe: capability, arch_list, conv2d cuda, `pytorch3d.ops.knn_points` cuda; `pip check` ghi vào `out/pip_check_work38.txt` (không dừng: đã biết xung đột mediapipe/opencv-contrib). |
| H4 | ĐÃ SỬA | `common.next`: `durable_sync`, mốc `-s`, ghi mốc nguyên tử, các hàm `conda_ok`/`mamba_ok`/`env_python_ok`/`repo_ok`/`assets_problems`. `setup.next`: chặng nhẹ luôn tự kiểm (`EMOCA_WSL_REDO=1 stage`); phần hỏng được dời sang `*.broken.<epoch>` (Miniconda chỉ khi có `.emoca_wsl_owned`); env hỏng ⇒ `conda clean` rồi tạo lại và xoá mốc env/pip; repo hỏng ⇒ clone lại và chuyển `assets/`/`data/` sang; patch hỏng ⇒ `git checkout -- .` rồi áp lại; `--ignore-submodules=all`. Chú thích đầu tệp nói rõ cách chạy lại sau khi WSL sập. |
| H5 | ĐÃ SỬA | `setup.next` `windows_memory_report` (đầu setup và trước mamba env update) → `out/windows_memory.txt`; chỉ cảnh báo. |
| M1 | ĐÃ SỬA | `assets_problems` + grep `out/download_assets.log` + `unzip -tq`. |
| M2 | TODO | ghi ở đầu `setup.next`; ghim sau lần dựng thành công đầu tiên. |
| M3 | ĐÃ SỬA | `smoke_test_emoca.sh --forward-only` (http.server 127.0.0.1 trong WSL, gọi bằng `curl.exe` của Windows qua interop; không có interop thì in hướng dẫn thử tay); phần C của smoke. |
| M4 | ĐÃ SỬA | smoke dừng nếu `EMOCA_PORT` đang có tiến trình nghe. |
| M5 | ĐÃ SỬA | log nói rõ việc chấp nhận điều khoản kênh Anaconda thay người dùng. |
| M6 | ĐÃ SỬA | `common.next` dừng nếu thư mục làm việc nằm trong `/mnt/*`. |
| M7 | GHI CHÚ | chú thích trong `setup.next` (lệch có chủ đích). |
| M8 | ĐÃ SỬA | `disk_report`: `EMOCA_HOST_DRIVE` (mặc định `/mnt/c`), `du.txt` ghi cả hai ổ. |
| M9 | ĐÃ SỬA | `sudo -n true`; không có TTY ⇒ in lệnh cho người dùng (kèm `dpkg --configure -a`), mã thoát 12. |
| M10 | ĐÃ SỬA | `dpkg-query` so với "install ok installed". |
| LOW | ĐÃ SỬA phần rẻ | SIGPIPE ở `require_gpu`; KeyError ở `check_bundle`; so hash đầy đủ; activate lỗi trong smoke ⇒ mã 97; xoay vòng `server.log`; texture kiểm bằng `numpy.load`. Để lại: `git lfs install`; `apply.sh` `--short=7` (tệp không nằm trong phạm vi giao). |

**Kiểm chứng (Git Bash trên Windows, không WSL):**
- `bash -n` cả 4 script và 2 tệp `.next`: OK.
- `python another/tests/test_wsl_scripts.py`: ALL PASS (150 ok, 2 SKIP có chủ đích — bản đang chạy).
- `WSL_SCRIPT_SUFFIX=.next python another/tests/test_wsl_scripts.py`: ALL PASS (186 ok).
- `python another/tests/run_all.py`: 12 PASS, 1 SKIP (nhóm nặng), 0 FAIL.

**CHƯA KIỂM (cần WSL/GPU thật):** mọi đường chạy thật của các phần sửa — hồi phục sau sập, `refresh_bin`, probe CUDA, interop `powershell.exe`/`curl.exe`, bộ lấy mẫu `/v1/health` (cần các trường mới của btl-06), `unzip -tq`, `sudo -n`.
