# RUNBOOK_DEMO — Chạy demo MA3D-Net Web trên máy Windows

> Task **M4-04** (`documents/docs/context/ROADMAP_BACKLOG.md`). Viết ngày 2026-09-24 bởi vai trò `DOCS`, dựa trên mã và tài liệu hiện có.
> Người đọc: thành viên nhóm **chưa biết gì về dự án**, có Windows + Python.
> Mục nào ghi **(chưa kiểm)** là chưa có ai chạy thử theo đúng các bước đó; đừng coi là "chạy được".

**Tóm tắt một dòng:** web Streamlit chạy trên máy bạn; POSTER/MA3D-Net chạy trong tiến trình web (giả lập hoặc CPU thật); EMOCA (landmark + 3DMM) chạy giả lập, hoặc chạy thật — **đường chính:** trong WSL2 trên chính máy bạn, web gọi `http://127.0.0.1:8765`, ảnh không rời máy (ADR-018, mục 4-wsl); **dự phòng:** trên Google Colab, web gọi tới qua tunnel có token (ADR-015, mục 4). POSTER/MA3D-Net luôn chạy cục bộ (ADR-012).

```
ĐƯỜNG CHÍNH (ADR-018) — mọi thứ trên máy bạn, ảnh không rời máy
Máy bạn (Windows)                                        WSL2 Ubuntu-22.04 (cùng máy, GPU RTX 3050)
┌──────────────────────────────────────────────┐        ┌────────────────────────────────┐
│ streamlit run frontend/app.py  (127.0.0.1)        │  HTTP  │ backend/services/emoca_svc/server.py   │
│  app.py → orchestrator.py ─ emoca_client.py ─┼───────►│ (nghe 127.0.0.1:8765)          │
│                          └─ engine.py        │loopback│  GPU dùng driver Windows       │
│                            (POSTER/MA3D, CPU)│        │                                │
└──────────────────────────────────────────────┘        └────────────────────────────────┘

DỰ PHÒNG (ADR-015) — ảnh đi qua Google + Cloudflare, cần bật xác nhận gửi ảnh
Máy bạn (Windows)                                        Google Colab (GPU)
┌──────────────────────────────────────────────┐        ┌────────────────────────────────┐
│ streamlit run frontend/app.py  (127.0.0.1)        │  HTTPS │ backend/services/emoca_svc/server.py   │
│  app.py → orchestrator.py ─ emoca_client.py ─┼───────►│ (chỉ nghe 127.0.0.1:8765)      │
│                          └─ engine.py        │ tunnel │  ▲ cloudflared quick tunnel    │
│                            (POSTER/MA3D, CPU)│ +token │  │ https://….trycloudflare.com │
└──────────────────────────────────────────────┘        └────────────────────────────────┘
```

Mọi lệnh dưới đây chạy trong **PowerShell**, đứng ở **thư mục gốc dự án** (thư mục chứa `CLAUDE.md`). Đường dẫn của dự án có dấu tiếng Việt và khoảng trắng — **luôn đặt trong ngoặc kép**.

```powershell
cd "C:\đường\dẫn\tới\BTL"      # thay bằng đường dẫn thật trên máy bạn
```

---

## 1. Yêu cầu và cài đặt

| Thứ cần | Giá trị | Nguồn |
|---|---|---|
| Python | **≥ 3.10** (đã kiểm với 3.13.7). Streamlit 1.64 không chạy trên 3.8 | `environments/web/requirements.txt:2-3` |
| Môi trường ảo | đặt **ngoài** thư mục dự án (dự án đang dùng `C:\Users\<bạn>\venvs\ma3d-web`) | SESSION_LOG Session 06 |
| Dung lượng | chỉ mock: **333 MB** sau 1.2(a), cài mất ~104 s (đo 2026-09-25, mạng máy dự án); thêm torch CPU: lớn hơn nhiều (chưa đo) | `documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md` (L8) |

### 1.1 Tạo venv

```powershell
python -m venv "$HOME\venvs\ma3d-web"
$PY = "$HOME\venvs\ma3d-web\Scripts\python.exe"    # gọi thẳng python.exe của venv, khỏi cần Activate.ps1
& $PY --version                                      # phải ≥ 3.10
```

> Không dùng `Activate.ps1` cũng được — nhiều máy Windows chặn script PowerShell (ExecutionPolicy). Mọi lệnh bên dưới dùng `& $PY ...`. Nếu mở cửa sổ PowerShell mới thì đặt lại biến `$PY`.
>
> **Đã có thư mục `$HOME\venvs\ma3d-web`** (máy dùng chung, hoặc chạy lại lần hai): `python -m venv` không báo gì và các lệnh `pip` sẽ cài chồng lên venv cũ. Khi đó bỏ qua 1.1 (chỉ đặt `$PY`), hoặc chọn tên thư mục khác (`documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md`, L9).

### 1.2 Cài thư viện — chọn MỘT trong hai

**(a) Chỉ chế độ mock** (đủ để xem giao diện; không cần torch, không cần checkpoint):

```powershell
& $PY -m pip install -r environments/web/requirements.txt    # streamlit, pillow, pyyaml — không có torch
& $PY -m pip install jsonschema==4.26.0          # chỉ cần để chạy test
```

**(b) Thêm POSTER thật trên CPU** (`backend/web/engine.py` chế độ `real`):

```powershell
& $PY -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
& $PY -m pip install -r environments/web/requirements-real.txt   # gồm requirements.txt + einops, matplotlib, numpy, jsonschema
```

Nguồn: `environments/web/requirements-real.txt` (cài torch CPU trước để pip không kéo bản CUDA từ PyPI). Chế độ thật còn cần **checkpoint** trong `backend/model/ma3d/checkpoints/` (gitignored, 10 file, ~2.47 GB, tải từ các thư mục Google Drive ở `backend/model/ma3d/README.md:67,98`; danh sách tên + kích thước ở `backend/configs/models.yaml`). Không có checkpoint ⇒ mô hình báo `model_unavailable`. Ước lượng RAM: ~1 GB cho hai mô hình (`backend/configs/services.yaml`, mục `engine`).

### 1.3 Lỗi font chữ khi in ra (Windows)

Test và script in tiếng Việt. Trên máy dự án, khi đầu ra bị chuyển hướng (pipe, file, một số terminal) Python dùng bảng mã `cp1258` và **dừng với `UnicodeEncodeError`** (đã gặp khi viết tài liệu này). Đặt trước khi chạy:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

---

## 2. Chạy web ở chế độ mock

```powershell
& $PY -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8599 --server.headless true --browser.gatherUsageStats false
```

Rồi mở **http://127.0.0.1:8599** trong trình duyệt. (Bỏ `--server.port` thì Streamlit dùng cổng mặc định 8501.)

**Vì sao BẮT BUỘC `--server.address 127.0.0.1`:** mặc định Streamlit nghe **mọi** giao diện mạng — nhật ký in dòng "External URL", tức là máy khác cùng mạng Wi-Fi/LAN mở được trang và tải ảnh khuôn mặt lên máy bạn. Chỉ nghe `127.0.0.1` thì chỉ chính máy bạn vào được (ADR-014 mục 5; SESSION_LOG Session 07). Dùng địa chỉ `127.0.0.1`, **không** dùng `localhost`: trên máy dự án `localhost` phân giải sang IPv6 và script bị treo (Session 07).

**Vì sao có `--server.headless true`:** trên máy chưa từng chạy Streamlit (chưa có `~\.streamlit\credentials.toml`), lần đầu Streamlit in "Welcome to Streamlit! … Email:" rồi **chờ nhập**, server chưa nghe cổng. Chạy nền thì treo hẳn (đo 2026-09-25: 60 s không có `/_stcore/health`); thêm cờ này thì có health sau 1.4 s. Nếu lỡ chạy lệnh không có cờ trong cửa sổ PowerShell và thấy lời hỏi `Email:`, bấm **Enter** để bỏ qua (`documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md`, L1). Cờ này cũng khiến Streamlit không tự mở trình duyệt; bạn tự mở địa chỉ ở trên.

**Vì sao có `--browser.gatherUsageStats false`:** không có cờ này thì nhật ký in "Collecting usage statistics" và trình duyệt gửi số liệu sử dụng (không phải ảnh) cho Streamlit; tắt cho khớp nguyên tắc riêng tư của dự án (`documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md`, L4).

**Cờ tuỳ chọn:** `--server.fileWatcherType poll` — thêm khi bạn đang **sửa mã** và muốn server tự nạp lại (Session 07).

**Khi nào PHẢI tắt (Ctrl+C) rồi chạy lại server:**
- sau khi sửa module phụ của `backend/web/` (`orchestrator.py`, `engine.py`, `emoca_client.py`, `mock_data.py`, `theme.py`) — server đang chạy có thể giữ bản cũ trong bộ nhớ (đã gặp: `ImportError: check_emoca`, Session 08). Sửa `app.py` thì Streamlit tự hỏi "Rerun";
- sau khi đổi biến môi trường (`EMOCA_*`, `ENGINE_MODE`) hoặc cài thêm thư viện;
- **không** cần khởi động lại khi thêm/sửa ảnh trong `frontend/assets/theme/` (cache tự làm mới theo thời gian sửa file, `frontend/app.py`, hàm `_signature`).

**Kỳ vọng ở chế độ mock:** tải một ảnh JPEG/PNG/WebP bất kỳ (≥ 32 px cạnh ngắn) → có landmark, ảnh dựng, xác suất 7 nhãn của hai mô hình, và nhãn **MOCK** trên đầu trang. Mọi con số là **dữ liệu giả** (xem mục 7).

**Lấy ảnh ở đâu:** repo **không kèm ảnh khuôn mặt mẫu**. Ở chế độ mock, ảnh nào ≥ 32 px cũng được, kể cả ảnh không có mặt người (kết quả là dữ liệu giả), ví dụ một ảnh chụp màn hình trong `documents/docs/qa/UI_LOADING_NARROW_2026-09-25/`. Không commit ảnh mặt người thật vào repo; muốn có ảnh mặt mẫu thì cần ADR về nguồn và giấy phép (`documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md`, L7).

---

## 2b. Chạy nhanh bằng `run_demo.bat` / `stop_demo.bat` (Windows, EMOCA thật)

Hai tệp ở gốc dự án, bấm đúp là chạy. Không cần quyền quản trị. Cần sẵn: venv `%USERPROFILE%\venvs\ma3d-web` (mục 1), WSL2 `Ubuntu-22.04` với user `viet`, EMOCA đã dựng xong (mục 4-wsl), `curl.exe` (có sẵn trên Windows 10/11).

- **`run_demo.bat`**, theo thứ tự:
  1. Kiểm venv, `curl.exe`, WSL. Thiếu thứ nào thì in hướng dẫn và dừng.
  2. EMOCA: nếu `http://127.0.0.1:8765/v1/health` chưa trả 200 thì bật `~/emoca_wsl/bin/run_emoca_server.sh` trong một cửa sổ thu nhỏ **"EMOCA server (WSL)"**. Cửa sổ này giữ phiên WSL; đóng nó là tắt EMOCA. Log ở `~/emoca_wsl/out/run_server_bat.log`. Script chờ tối đa 180 s; quá hạn thì báo và web chạy ở chế độ **giả lập**.
  3. Web: nếu cổng 8599 chưa chạy thì bật Streamlit trong cửa sổ thu nhỏ **"MA3D web"**, với cùng các cờ ở mục 2. `EMOCA_MODE=real`, `EMOCA_URL=http://127.0.0.1:8765`, `ENGINE_MODE=real` **chỉ** được đặt khi EMOCA đã lên (`backend/web/orchestrator.py`, `resolve_emoca_settings` / `resolve_engine_mode`).
  4. Chờ `/_stcore/health` rồi mở `http://127.0.0.1:8599`. Đặt `DEMO_NO_BROWSER=1` nếu không muốn tự mở trình duyệt.
  - Chạy lại khi đã chạy thì **không bật trùng**: đo 2026-09-26, khi cả hai đã chạy thì script xong trong 0,5 s và không khởi động gì thêm. Nhưng web đang chạy sẽ **giữ chế độ** của lần bật trước; muốn đổi thì chạy `stop_demo.bat` trước.
- **`stop_demo.bat`**: tắt theo **cổng đang nghe**, không tắt theo mẫu tên tiến trình.
  - Web 8599: dừng tiến trình nghe cổng, và tiến trình cha nếu đó là launcher `python.exe` của venv chạy streamlit.
  - EMOCA 8765 trong WSL: lấy PID bằng `ss -ltnp`, gửi SIGTERM, chờ tối đa 10 s rồi mới SIGKILL. `run_emoca_server.sh` tự kết thúc theo.
  - Cuối cùng in trạng thái hai cổng (`trong` = đã tắt).
- Thời gian khởi động từ đầu (stop → run): **chưa đo** — xem báo cáo của lần thử đầu.
- Hai tệp chỉ dùng ký tự ASCII và dòng CRLF, vì `cmd` đọc `.bat` theo code page. Đường dẫn dự án lấy từ `%~dp0` nên chạy được ở thư mục có dấu tiếng Việt và khoảng trắng.

## 3. Cài đặt trên giao diện (bảng "Cài đặt" ở đầu trang)

Mô tả theo `frontend/app.py` (`render_settings`, `render_engine_panel`, `render_emoca_panel`).

Bảng **Cài đặt** (biểu tượng bánh răng) nằm ngay dưới tiêu đề trang, **mặc định đóng** — bấm vào để mở/đóng. Khi mở chia 3 cột: **Giao diện** | **Hiển thị** + **Mô hình POSTER / MA3D-Net** | **Kết nối EMOCA** + **Khác**. Trang không còn thanh bên (sidebar) — bản cũ đặt Cài đặt ở thanh bên, đã bỏ vì thu gọn thanh bên rồi thì không mở lại được (`frontend/app.py:300`).

| Nhóm | Điều khiển | Giá trị | Ghi lên địa chỉ trang? |
|---|---|---|---|
| Giao diện | **Chủ đề màu** | 24 chủ đề, mỗi chủ đề 4 màu lấy từ một ảnh trong `frontend/assets/theme/`; mặc định `cold`. Ảnh hỏng/trùng màu bị loại, lý do nằm trong mục "Ghi chú nguồn màu" | `t` |
| | **Chế độ** | Sáng / Tối (độc lập với chế độ tối của Windows/trình duyệt) | `m` |
| | **Độ tương phản** | Chuẩn / Cao (màu chữ, nút, viền tự chỉnh để đạt WCAG; Cao chặt hơn) | `c` |
| | **Cỡ chữ** | Nhỏ / Vừa / Lớn | `f` |
| | **Giảm chuyển động** | bật/tắt hiệu ứng chuyển mượt | `rm` |
| Hiển thị | Hiện landmark trên ảnh · Hiện khung mặt · Hiện ảnh trung gian & attention | bật/tắt | `lm`, `bb`, `at` |
| Mô hình POSTER / MA3D-Net | **Suy luận** | Giả lập / Thật (CPU) — mặc định lấy từ `ENGINE_MODE` hoặc `backend/configs/services.yaml` | không |
| Kết nối EMOCA | **Nguồn EMOCA** | Giả lập / Máy chủ EMOCA — xem mục 4 | không (có token) |
| Khác | **Đặt lại cài đặt** | về mặc định (chủ đề `cold`, Sáng, Chuẩn, Vừa) | — |

- Cài đặt giao diện được ghi lên địa chỉ trang (`?t=cold&m=dark&c=high…`) nên tải lại/chia sẻ link vẫn giữ nguyên; giá trị lạ tự về mặc định (ADR-014 mục 2).
- Địa chỉ EMOCA, token, xác nhận gửi ảnh và chế độ suy luận **chỉ** nằm trong bộ nhớ phiên, không lên địa chỉ trang.
- Nếu trình duyệt bật "ép tối mọi trang" (Cốc Cốc/Chrome), trang vẫn đọc được nhưng màu chủ đề bị trình duyệt đổi; muốn thấy đúng màu thì tắt tính năng đó trong trình duyệt (ADR-014, đã đo ở Session 07).
- Bộ chọn "Suy luận" mới được thêm vào `frontend/app.py` trong ngày viết tài liệu; nếu giao diện khác mô tả, đọc lại `render_engine_panel` **(chưa kiểm bằng mắt)**.

---

## 4-wsl. EMOCA trên WSL2 (đường chính) **(kiểm một phần — WSL2 + GPU đã đo (4-wsl.1b); dựng env và tài sản đã hoàn tất; M7 real-model smoke qua web đã chạy 2026-10-01; service đã dừng sau kiểm tra. Chưa chạy toàn bộ script smoke riêng bên dưới.)**

Từ 2026-09-25, EMOCA chạy **trên chính máy bạn** trong WSL2 (ADR-018); Colab ở mục 4 dưới đây chỉ còn là **dự phòng**. `backend/services/emoca_svc` nghe `127.0.0.1:8765`, web gọi `http://127.0.0.1:8765`. Ảnh **không rời máy**, và vì là địa chỉ loopback nên không cần hộp xác nhận gửi ảnh (mục 4d).

### 4-wsl.1 Điều kiện (đo 2026-09-25 trên máy dự án)

| Hạng mục | Giá trị đo | Cách đo |
|---|---|---|
| Hệ điều hành | Windows 11 Home, build 26200 | `Win32_OperatingSystem.BuildNumber` |
| Dung lượng trống | C 109.2 GB · D 6.9 GB · E 52.7 GB | `Get-PSDrive` |
| Ảo hoá trong firmware | bật | `Win32_Processor.VirtualizationFirmwareEnabled` |
| GPU | NVIDIA GeForce RTX 3050 4GB Laptop, driver 555.97, 4096 MiB | `nvidia-smi` |
| RAM | 15.7 GB | `Win32_ComputerSystem` |
| WSL | **chưa cài** lúc đo (nay đã cài — xem 4-wsl.1b) | `wsl --status` |

WSL2 **không cần chia ổ**: Ubuntu nằm trong một tệp đĩa ảo `.vhdx` trên ổ sẵn có.

### 4-wsl.1b Đo trên máy này sau khi cài WSL2 (2026-09-25)

Nguồn: đo 2026-09-25 (~11:20–11:40) bằng lệnh, điều phối `btl-bb`, sau khi người dùng cài WSL2 + Ubuntu-22.04 và khởi động lại máy. Cột "Trạng thái" phân biệt điều **đã đo** với điều **chưa đo**.

| Hạng mục | Kết quả | Cách đo | Trạng thái |
|---|---|---|---|
| Phiên bản WSL | `2.7.14.0`, kernel `6.18.33.2` | `wsl --version` | đã đo |
| Ảo hoá | `wsl --status` không còn báo "virtualization is not enabled"; RebootPending = False | `wsl --status` | đã đo |
| Distro | Ubuntu 22.04.5, `python3` 3.10.12, tài khoản Ubuntu `viet` | trong Ubuntu | đã đo |
| Cầu nối GPU | `/dev/dxg` có; thư viện NVIDIA ở `/usr/lib/wsl/lib`; **không** có gói NVIDIA/CUDA cài trong Ubuntu (đúng khuyến nghị) | trong Ubuntu | đã đo |
| `nvidia-smi` **không đối số** | in phần đầu rồi **`Segmentation fault`** (trong `libnvidia-ml.so.1`; kernel báo `dxgkio_query_adapter_info: Ioctl failed: -22`) | `nvidia-smi` | đã đo |
| `nvidia-smi -L` | chạy tốt, thấy RTX 3050 | `nvidia-smi -L` | đã đo |
| `nvidia-smi --query-gpu=...` | thoát mã 0 với `name,memory.total,driver_version`; `...memory.used`; `memory.used,memory.total` | `nvidia-smi --query-gpu=...` | đã đo |
| CUDA driver API (gọi trực tiếp) | `cuInit` = 0, tạo context OK; RTX 3050 4GB Laptop, compute capability 8.6; tổng 4095 MiB, **trống 3292 MiB** (≈ 803 MiB đang do Windows dùng) | gọi CUDA driver API trực tiếp trong Ubuntu | đã đo (lúc **rảnh**) |
| `memory.used` của `nvidia-smi` | báo 0 MiB (chế độ WDDM, cả Windows lẫn WSL) ⇒ **không đáng tin** để đo VRAM | `nvidia-smi --query-gpu=memory.used` | đã đo |
| VRAM thực tế cho EMOCA | ≈ 3,2 GB (suy ra từ số trống ở trên) | — | **chưa đo dưới tải** |
| Đĩa | trong Ubuntu `df ~` báo ~955 GB trống — đó là **đĩa ảo**, không phải giới hạn thật; giới hạn thật là ổ C, còn ~108 GiB | `df ~`, `df /mnt/c` | đã đo |
| RAM | WSL2 thấy 7771 MB (mặc định 50%); Windows còn ~4,4 GB trống lúc đo; **chưa có** `.wslconfig` lúc đo (sau đó đã tạo, xem 4-wsl.4 sự cố 2) | trong Ubuntu / Windows | đã đo |
| Gói Ubuntu | lúc đo chưa có `git-lfs`, `unzip`, `build-essential` (điều phối cài bằng `sudo`) | trong Ubuntu | đã đo |
| localhost WSL2 → Windows | — | — | **chưa đo** |

Lỗi segfault của `nvidia-smi` thuộc nhóm lỗi đã có nhiều báo cáo: [microsoft/WSL#11510](https://github.com/microsoft/WSL/issues/11510), [#11277](https://github.com/microsoft/WSL/issues/11277), [#11870](https://github.com/microsoft/WSL/issues/11870). **Nguyên nhân trên máy này CHƯA xác định** (giả thuyết chưa kiểm: driver Windows 555.97 ngày 2024-05-29 cũ so với kernel WSL mới). Không cần cập nhật driver lúc này — CUDA vẫn tạo context được.

**Khuyến nghị an toàn:**
- **Đừng chạy `nvidia-smi` không đối số** trong Ubuntu (segfault). Kiểm GPU bằng `nvidia-smi -L`, hoặc `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`.
- **Đừng tin `memory.used`** của `nvidia-smi` (luôn 0 MiB ở chế độ WDDM). Số VRAM đáng tin là số CUDA/torch của server báo qua `GET /v1/health`. Smoke test và `run_emoca_server.sh` lấy mẫu endpoint này và ghi ra `~/emoca_wsl/out/smoke_vram_health.json` / `server_vram_health.json` (mục 4-wsl.3). Các script **không còn** dùng `nvidia-smi memory.used`.
- Dung lượng: xem `df /mnt/c`, không xem `df ~`.

#### cuDNN (đo 2026-09-25, điều phối `btl-bb`, trong env `work38` sau lần dựng 3)

| Phép thử | Kết quả | Trạng thái |
|---|---|---|
| `torch.cuda.get_arch_list()` | có `sm_86` | đã đo |
| matmul, phép tính nguyên bản trên GPU | chạy | đã đo |
| `conv2d` **không** dùng cuDNN | chạy | đã đo |
| `conv2d` **có** cuDNN (bản cuDNN 8302), **chưa** đặt `LD_LIBRARY_PATH` | lỗi `CUDNN_STATUS_NOT_INITIALIZED` / `CUDNN_STATUS_INTERNAL_ERROR`; có lần tiến trình tự huỷ: mã thoát 134 (SIGABRT), `dmesg` báo `fatal signal 6` | đã đo |
| Probe `cudnn_probe` (15:32–15:40) | log `Error: libcuda.so: cannot open shared object file: No such file or directory`, rồi SIGABRT (mã 134) | đã đo |
| `conv2d` **có** cuDNN **sau khi** đặt `LD_LIBRARY_PATH` | chạy; `0.73 ms/lần` | đã đo |
| Cùng phép thử, **tắt** cuDNN | `1.12 ms/lần` | đã đo |
| `pytorch3d` `knn_points` trên cuda | chạy | đã đo |
| Script tự thêm `LD_LIBRARY_PATH` | có: `apply_wsl_ldpath` gọi trong `envrun` (`backend/services/emoca_svc/wsl/common.sh:63-86`, biến `EMOCA_WSL_LIB_DIR` ở `:17`) | **đã có trong mã (đọc lại 2026-09-25); chưa kiểm dưới EMOCA thật** |

Hai số ms ở trên đo trên chuỗi conv7x7 → BN → ReLU → conv3x3, đầu vào 1×3×224×224. Đó là phép thử nhỏ, không phải tốc độ EMOCA.

- **Nguyên nhân — ĐÃ XÁC ĐỊNH** (đo 2026-09-25 15:32–15:40, điều phối `btl-bb`). cuDNN 8.3.2 nạp thư viện bằng tên **không có đuôi phiên bản** `libcuda.so`. Tệp này có ở `/usr/lib/wsl/lib`, nhưng ld cache chỉ biết `libcuda.so.1`, nên cuDNN không nạp được `libcuda.so` rồi báo lỗi.
- **Giả thuyết cũ về driver 555.97 là SAI.** **Không cần** cập nhật driver NVIDIA để sửa lỗi cuDNN.
- **Cách sửa (đã đo là chạy được):**
  ```bash
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
  ```
  Script đã tự làm việc này (`common.sh:63-86`, hàm `envrun`); phần này **chưa kiểm dưới EMOCA thật**.
- **Lỗi `nvidia-smi` không đối số là lỗi KHÁC**, vẫn chưa rõ nguyên nhân (NVML / ioctl của dxg trả về -22).
- **Phương án dự phòng `EMOCA_CUDNN=0`** (tắt cuDNN) vẫn giữ lại, nhưng **hiện không cần**.

### 4-wsl.2 Cài WSL2 (người dùng tự làm)

1. Mở **PowerShell bằng quyền Administrator** (chuột phải → *Run as administrator*), chạy:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
   Khởi động lại máy khi được hỏi. Lần mở Ubuntu đầu tiên sẽ hỏi **tên người dùng và mật khẩu Ubuntu** — tự đặt và ghi nhớ.
2. Trong cửa sổ Ubuntu chạy:
   ```bash
   nvidia-smi -L
   ```
   Phải thấy **RTX 3050**. **KHÔNG cài driver NVIDIA bên trong WSL** — WSL dùng driver của Windows. Trên máy dự án, `nvidia-smi` không đối số bị `Segmentation fault` — dùng `-L` (mục 4-wsl.1b).
3. *(Tuỳ chọn, nên làm)* Giới hạn RAM cho WSL2 để không chiếm hết RAM máy (mặc định WSL2 dùng tới 50% RAM): tạo tệp `.wslconfig` trong thư mục người dùng Windows (`C:\Users\<bạn>\.wslconfig`) với nội dung:
   ```ini
   [wsl2]
   memory=8GB
   swap=8GB
   ```
   rồi chạy `wsl --shutdown` để áp dụng.
4. *(Tuỳ chọn)* Muốn giữ ổ C thì chuyển distro sang ổ E bằng `wsl --export` rồi `wsl --import` (E còn 52.7 GB).

### 4-wsl.3 Chạy script dựng môi trường và dịch vụ

> **Trạng thái (cập nhật 2026-09-25 tối):**
> - `setup_emoca_wsl.sh` **đã chạy thật** tới cổng giấy phép (xong phần không cần giấy phép lúc 12:38:30, mục 4-wsl.5). `smoke_test_emoca.sh` và `run_emoca_server.sh` **CHƯA chạy thật**. Tiêu chí nghiệm thu: ROADMAP M0-06.
> - Bản sửa theo `documents/docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md` **đã hoán đổi** (`common.sh`, `setup_emoca_wsl.sh`; không còn tệp `*.next`). Mọi số dòng `setup_emoca_wsl.sh:N` / `common.sh:N` ở mục này là của **bản hiện tại** (đọc lại 2026-09-25). Phần sửa (hồi phục sau sập, `refresh_bin`, probe CUDA…) mới **kiểm tĩnh**, chưa chạy thật.
> - Kiểm tĩnh: `python another/tests/test_wsl_scripts.py` → ALL PASS, 237 dòng `ok` (đo 2026-09-25, sau khi thêm test prefetch).

Mọi lệnh dưới đây gõ **trong cửa sổ Ubuntu (WSL2)**, không phải PowerShell.

**1. Dựng môi trường** — `setup_emoca_wsl.sh`

- **Lần đầu, KHÔNG đặt cờ giấy phép.** Script dựng xong env rồi **dừng trước khi tải tài sản**, thoát mã **10** (`setup_emoca_wsl.sh:8`, `:488-497`):
  ```bash
  bash "$(wslpath 'C:\Users\admin\Desktop\File All\Năm Học\Kì 1 năm 3\Các VĐ hiện đại của CNTT\BTL')/backend/services/emoca_svc/wsl/setup_emoca_wsl.sh"
  ```
- **Sau khi CHÍNH BẠN đã đăng ký và đồng ý điều khoản** ở flame.is.tue.mpg.de và emoca.is.tue.mpg.de, chạy lại với cờ. Cờ là lời xác nhận của bạn — không ai đặt thay bạn, script cũng không tự đồng ý (`:15`). Nên chạy **bước 1b** (tải trước tài sản) trước, vì bước tải của upstream rất chậm:
  ```bash
  ACCEPT_FLAME_EMOCA_LICENSES=1 bash ~/emoca_wsl/bin/setup_emoca_wsl.sh
  ```
- **Chạy lại được:** chặng nào đã xong thì tự bỏ qua (`common.sh:101-113`; đặt `EMOCA_WSL_REDO=1` để chạy lại mọi chặng). Từ lần 2 gõ ngắn `bash ~/emoca_wsl/bin/...`, vì script tự chép các script vào `~/emoca_wsl/bin` (`setup_emoca_wsl.sh:212-240`, `refresh_bin`). Sau khi WSL sập thì chạy bằng đường dẫn dự án (mục 4-wsl.4).
- Cần **mật khẩu sudo** để cài gói hệ thống qua apt; không có terminal để hỏi mật khẩu thì script in sẵn lệnh cho bạn tự chạy rồi thoát **12** (`:125-134`).
- Mã thoát: 0 = xong · 10 = chưa đồng ý giấy phép · 11 = thiếu texture · 12 = cần bạn tự chạy lệnh `sudo` (in sẵn) · mã khác = lỗi, xem log (`:26`).
- Thời gian từng chặng ghi ở `~/emoca_wsl/out/timings.tsv`. Log đầy đủ ở `setup_*.log`, kèm `env_work38.yml`, `pip_freeze_work38.txt`, `pip_check_work38.txt`, `du.txt`, `download_assets.log`, `cudnn_probe.txt` (`:25`).
- **Nếu tài sản đã đủ** (sau bước 1b), script bỏ qua chặng tải của upstream: kiểm bằng `assets_problems` (`common.sh:153`; `setup_emoca_wsl.sh:501-524`). Texture `FLAME_albedo_from_BFM.npz` là **tuỳ chọn** (2026-09-25): người dùng quyết không xin giấy phép BFM; mô hình mặc định có `use_texture=false` (`cfg.yaml:133,292`, theo điều phối) và script mặc định `EMOCA_REQUIRE_TEXTURE=auto` (`common.sh:18`, `:35`) ⇒ không bắt buộc texture; mã 11 chỉ khi texture bắt buộc mà thiếu (`EMOCA_REQUIRE_TEXTURE=1`, hoặc `auto` khi mô hình dùng texture hay không đọc được `cfg.yaml`; `common.sh:202-208`, `setup_emoca_wsl.sh:546-574`).
- **`mediapipe` được ghim 0.10.11** (`setup_emoca_wsl.sh:423-433`): bản 1.0.1 mà pip tự chọn hỏng khi import trên Python 3.8, bản 0.8.10 của upstream không còn trên PyPI.

**1b. Tải trước tài sản** — `prefetch_assets.sh` *(thêm 2026-09-25; đang chạy lần đầu, **chưa có số đo cuối**)*

Máy chủ tải của MPI chậm theo từng kết nối, nên script tải mỗi tệp bằng nhiều kết nối song song (`curl -r`, mỗi đoạn một kết nối), tải tiếp được khi đứt, kiểm `Content-Length` + `unzip -tq`, rồi giải nén đúng chỗ như `download_assets.sh` của upstream (`backend/services/emoca_svc/wsl/prefetch_assets.sh:1-31`). Không cần `sudo`, không cần `aria2c`; cần `curl` và `unzip` (`:57-58`).

```bash
ACCEPT_FLAME_EMOCA_LICENSES=1 bash "$(wslpath 'C:\Users\admin\Desktop\File All\Năm Học\Kì 1 năm 3\Các VĐ hiện đại của CNTT\BTL')/backend/services/emoca_svc/wsl/prefetch_assets.sh"
```

- **Cờ giấy phép** như bước 1: chỉ đặt khi CHÍNH BẠN đã đăng ký và đồng ý điều khoản FLAME/EMOCA; thiếu cờ ⇒ dừng, mã **10** (`:38-45`).
- **Tải gì** (đọc từ mã upstream `e0be0db`, ghi ở `:19-29`): mô hình `EMOCA/models/$MODEL_NAME.zip` (mặc định `EMOCA_v2_lr_mse_20`, `common.sh:28`) → `assets/EMOCA/models`; `FLAME.zip` → `assets`; `data/EMOCA_test_example_data.zip` → `data` (ảnh mẫu cho smoke test); `FaceRecognition.zip` → `assets` **chỉ khi cần** (xem `PREFETCH_FACEREC`).
- **Không tải:** `assets/DECA.zip` — chỉ nạp khi `resume_training` (`gdl/models/DECA.py:3009`), mà `load_deca` đặt `resume_training = False` (`gdl_apps/EMOCA/utils/load.py:47`) ⇒ suy luận ảnh không cần. Các mô hình EMOCA khác (`DECA.zip`, `EMOCA.zip`, `EMOCA_v2_mp.zip`, `EMOCA_v2_lr_cos_1.5.zip`) cũng không dùng (`prefetch_assets.sh:26-29`).
- **Biến** (`:11-17`, `:47-52`):

  | Biến | Mặc định | Ý nghĩa |
  |---|---|---|
  | `PREFETCH_CONN` | 12 | số kết nối (số đoạn) mỗi tệp; tải tiếp giữ cách chia cũ (ghi ở `<tệp>.parts/meta`) |
  | `PREFETCH_ROUNDS` | 6 | số vòng tải lại các đoạn hỏng/đứt |
  | `PREFETCH_FACEREC` | `auto` | `FaceRecognition.zip`: `auto` = tải khi `detail.model.idw` trong `cfg.yaml` của mô hình > 0 hoặc không đọc được (khi đó DECA nạp `resnet50_ft_weight.pkl`, `DECA.py:2774-2806`); `0` = bỏ qua; `1` = luôn tải (`:215-231`) |
  | `EMOCA_ASSET_CACHE` | `~/emoca_wsl/cache` | nơi giữ zip (zip **không** nằm trong `~/emoca/assets`) |
  | `EMOCA_ASSET_BASE_URL` | `https://download.is.tue.mpg.de/emoca/assets` | địa chỉ tải |
  | `PREFETCH_PROGRESS_SEC` | 60 | giây giữa hai dòng báo tiến độ |
- **Mã thoát:** 0 = xong, `assets_problems` rỗng ⇒ bước 1 sẽ bỏ qua chặng tải của upstream (`:242`) · 10 = chưa đặt cờ giấy phép (`:44`) · 1 = lỗi: tệp tải thiếu byte / ghép sai / `unzip -tq` hỏng, hoặc `assets_problems` chưa rỗng (`:237-241`), hoặc biến sai / thiếu `curl`/`unzip` (`die`, `common.sh:35`).
- **Chạy lại được:** tệp đã có đủ byte và `unzip -tq` OK thì không tải lại (`:91-95`); đoạn dở được tải tiếp; tệp hỏng được đổi tên `*.stale.<epoch>`, **không xoá** (`:96`, `:104`).
- **Số đo:** `~/emoca_wsl/out/prefetch_timings.tsv` (giây, byte, KB/s của từng tệp), thêm một dòng mỗi tệp vào `timings.tsv`; log `~/emoca_wsl/out/prefetch_<ngày_giờ>.log` (`:30-31`, `:160-163`). Ghi chú trong script (đo 2026-09-25): 4.5–15 KB/s mỗi kết nối, 12 kết nối ≈ 270 KB/s tổng (`:3-4`). **Thời gian và dung lượng tải cuối: chưa đo.**

**2. Kiểm** — `smoke_test_emoca.sh`

```bash
bash ~/emoca_wsl/bin/smoke_test_emoca.sh
```
- **Phần A (M0-04):** mỗi ảnh mẫu công khai phải có 5 PNG + 5 NPY dài 100/50/50/6/128 (`smoke_test_emoca.sh:3-4`, `:113-125`).
- **Phần B:** chạy `server.py` **tạm** ở cổng **8766** (`:15`, `:26`). Kiểm schema, 68 landmark ở cả `crop224` lẫn `original_px` (`:268-270`), không nạp lại mô hình, ảnh xám ⇒ `no_face` 422 (`:282-283`). Nếu cổng 8765 đang có server chính thì smoke test dừng, để hai bản EMOCA không tranh VRAM 4 GB (`:76-78`).
- **Đo VRAM:** trong lúc chạy phần B, script lấy mẫu `GET /v1/health` mỗi `SMOKE_SAMPLE_SEC` giây (mặc định 1) bằng `python3` của Ubuntu, không torch nên không tạo CUDA context riêng. Kết quả ghi ra `~/emoca_wsl/out/smoke_vram_health.json` (`:7-10`, `:141`, `:161-202`), gồm:
  - `min_gpu_free_mb`, `gpu_total_mb`;
  - `max_vram_reserved_peak_mb`, `max_vram_reserved_mb`, `max_vram_peak_mb`;
  - `vram_after_load_mb`.

  Trường nào server chưa báo thì ghi "không có", không bịa (`:9`). **Không còn** `smoke_vram_nvidia_smi.csv` và không dùng `nvidia-smi memory.used`. `smoke_measurements.json` vẫn được ghi (`:310`).
- **Phần C — thử chuyển tiếp localhost WSL2 → Windows** (M3, không cần GPU). Script mở `python3 -m http.server` tạm ở `127.0.0.1:8799` trong WSL (đổi bằng `SMOKE_FWD_PORT`), rồi gọi từ phía Windows bằng `curl.exe` qua interop (`:11-12`, `:28`, `:34-62`).
  - Kết quả nối thêm vào `~/emoca_wsl/out/smoke_forward.txt`: `DAT` / `KHONG_DAT` / `CHUA_THU_TU_DONG` (`:60-61`). Không gọi được `curl.exe` thì script in lệnh để bạn thử tay trong 60 s.
  - Chạy riêng phần này (nên làm trước, vì không cần dựng xong EMOCA; mã thoát 0 = đạt, 1 = không đạt, 2 = không thử tự động được, `:65-70`):
    ```bash
    bash ~/emoca_wsl/bin/smoke_test_emoca.sh --forward-only
    ```
  - Bỏ phần này bằng `SMOKE_SKIP_FORWARD=1`.

**3. Chạy server** — `run_emoca_server.sh`

```bash
bash ~/emoca_wsl/bin/run_emoca_server.sh        # Ctrl+C để dừng
```
- Nghe `127.0.0.1:8765` (`run_emoca_server.sh:2`, `:94-95`, `common.sh:21`). Log ở `~/emoca_wsl/out/server.log`, xoay vòng khi > 50 MB (`:11`, `:34`).
- `EMOCA_TOKEN` **tuỳ chọn**: không đặt thì không xác thực, chấp nhận được vì chỉ nghe loopback (`:8-9`, `:30-31`).
- **Đo VRAM:** trong suốt lúc server chạy, script lấy mẫu `GET /v1/health` mỗi `VRAM_SAMPLE_SEC` giây (mặc định 5) và ghi vào `~/emoca_wsl/out/server_vram_health.json` (`:12-14`, `:22`, `:37-77`). Các trường giống `smoke_vram_health.json`: `min_gpu_free_mb`, `max_vram_reserved_peak_mb`, `max_vram_peak_mb`…. Không dùng `nvidia-smi memory.used`.
- **Nối web trên Windows:** Cài đặt → Kết nối EMOCA → **Máy chủ EMOCA**, địa chỉ `http://127.0.0.1:8765`. Đây là loopback nên **không cần** ô xác nhận gửi ảnh (mục 4d, ADR-015 quyết định 5). Hoặc đặt biến môi trường `EMOCA_MODE=real` và `EMOCA_URL=http://127.0.0.1:8765` (`backend/web/orchestrator.py:12-13`). *Từ 2026-09-26:* `backend/configs/services.yaml` đã đặt sẵn `emoca_svc.url: http://127.0.0.1:8765` (`:16`) nhưng `mode` **vẫn là `mock`** (`:20`) — web vẫn mặc định giả lập; chọn **Máy chủ EMOCA** trên giao diện hoặc đặt `EMOCA_MODE=real` để dùng EMOCA thật (khi đó không cần `EMOCA_URL`).

**Điểm mới của bản sửa (đã hoán đổi 2026-09-25; chỉ kiểm tĩnh bằng đọc mã):**
- **Mã thoát 12:** thiếu gói hệ thống mà `sudo` cần mật khẩu, trong khi script không có terminal để hỏi. Khi đó script **in sẵn lệnh cho bạn tự chạy** (`sudo dpkg --configure -a`, `sudo apt-get update && sudo apt-get install -y …`) rồi thoát 12 (`setup_emoca_wsl.sh:26`, `:125-134`). Có terminal thì script tự gọi `sudo` như cũ.
- **`EMOCA_HOST_DRIVE`** (mặc định `/mnt/c`): ổ Windows chứa đĩa ảo `.vhdx` của distro. Script đọc dung lượng thật của ổ này bằng `df` thay vì chỉ tin `$HOME`. Ngưỡng cảnh báo 30 GB vẫn là số ước lượng. Đã chuyển distro sang ổ khác thì đặt `EMOCA_HOST_DRIVE=/mnt/e` (`common.sh:14`, `:30`; `setup_emoca_wsl.sh:21`, `:57-73`).
- **`~/emoca_wsl/out/windows_memory.txt`:** kiểm hạn mức bộ nhớ Windows qua `powershell.exe` (chỉ đọc) ở đầu setup và trước chặng mamba. Script **chỉ cảnh báo, không dừng**, khi commit còn trống < 3 GB hoặc không có pagefile (`setup_emoca_wsl.sh:81-116`, `:383`). Lý do: sự cố 2 ở mục 4-wsl.4.
- **`~/emoca_wsl/out/pip_check_work38.txt`:** kết quả `pip check` của env. Nếu có xung đột thì ghi lại và in 10 dòng đầu, nhưng không dừng (`setup_emoca_wsl.sh:25`, `:465-469`).
- **Tự hồi phục sau khi WSL sập:** xem mục 4-wsl.4, Khuyến nghị 3.

**Chưa kiểm:**
- Chuyển tiếp localhost từ WSL2 sang Windows. Nay có lệnh thử `smoke_test_emoca.sh --forward-only` (phần C ở trên), nhưng **chưa chạy**.
- VRAM 4 GB có đủ không (lúc rảnh CUDA báo trống 3292 MiB — mục 4-wsl.1b; dưới tải chưa đo).
- Dung lượng thật của **tài sản** (env đã đo: mục 4-wsl.5). Script chỉ cảnh báo khi còn < 30 GB, và 30 GB là con số ước lượng (`setup_emoca_wsl.sh:57`, `:72`; đọc cả ổ `EMOCA_HOST_DRIVE`).
- ~~mamba/CUDA trên Ubuntu 22.04~~ — **đã chạy được** (env `work38` dựng xong, torch thấy RTX 3050; mục 4-wsl.5, 2026-09-25).

**Chỗ khác notebook Colab là cố ý:** xem chú thích trong script. Patch áp ngay sau clone; `requirements38` sửa trên bản sao ngoài repo (số dòng cũ `:159-171`, `:208-211` là của bản trước khi hoán đổi — xem chú thích M7 trong `setup_emoca_wsl.sh`). Smoke test không dùng `--save_mesh` (`smoke_test_emoca.sh:42`).

### 4-wsl.4 Khi WSL sập giữa lúc dựng

Nguồn: đo 2026-09-25, điều phối `btl-bb`. Trên máy dự án đã có **HAI sự cố** trong ngày:
- **Sự cố 1:** dịch vụ WSL (`wslservice.exe`) sập lúc 11:33:57.
- **Sự cố 2:** Windows hết hạn mức bộ nhớ (commit limit) lúc ~11:50.

Hai sự cố nhiều khả năng có **nguyên nhân chung là thiếu bộ nhớ**. Tuy vậy, **CHƯA chứng minh** được sự cố 1 là do bộ nhớ.

#### Sự cố 1 — `wslservice.exe` sập (11:33:57)

**Chuyện đã xảy ra.** Lần chạy `setup_emoca_wsl.sh` đầu tiên (bắt đầu 11:33:29) hỏng vì **dịch vụ WSL của Windows sập**:

| Thời điểm | Sự kiện |
|---|---|
| 11:33:57 | Application log **id 1000**: `wslservice.exe` 2.7.14.0, module `ntdll.dll` 10.0.26100.9444, mã ngoại lệ `0xc00000fd` (`STATUS_STACK_OVERFLOW`), fault offset `0x165497`, Report Id `c03ed90b-9808-4183-a6b4-03598bf2d53c` |
| 11:34:05, 11:34:08 | Application log **id 1001** |
| 11:34:09–10 | Máy ảo dựng lại; `dmesg`: `EXT4-fs (sdd): 6 orphan inodes deleted; recovery complete` |

- **Hệ quả:** các tệp vừa ghi ngay trước đó (Miniconda đang giải nén) còn **0 byte**.
- **Không do phiên Claude nào gây ra:** ba phiên phụ xác nhận không gọi `wsl`, và không có lệnh `--shutdown`.
- **Nguyên nhân: CHƯA xác định** (có thể liên quan sự cố 2, chưa chứng minh). Có báo cáo trên cùng hệ điều hành / bản WSL: [microsoft/WSL#13714](https://github.com/microsoft/WSL/issues/13714), [#41694](https://github.com/microsoft/WSL/issues/41694), [#13263](https://github.com/microsoft/WSL/issues/13263). Đây là cùng họ lỗi, **chưa chắc cùng nguyên nhân**.
- Người dùng chạy `wsl --update` lúc ~11:40: phiên bản **vẫn 2.7.14.0** (kernel 6.18.33.2-2).
- **Cách xử lý đã dùng (thành công):** xoá thư mục Miniconda dở (toàn tệp 0 byte, do lần chạy hỏng tạo ra) rồi chạy lại script. Lần chạy lại đã qua được chặng Miniconda.

#### Sự cố 2 — hết hạn mức bộ nhớ (~11:50)

**Chuyện đã xảy ra.** Lần dựng thứ hai chết lúc ~11:50 ở chặng mamba (`Solving environment`). Các điểm đã đo:
- `wsl.exe` báo `Wsl/Service/CreateInstance/E_UNEXPECTED` kèm `System.OutOfMemoryException`. **Không** có sự kiện `wslservice.exe` sập mới.
- Windows **không có pagefile**: `Win32_ComputerSystem.AutomaticManagedPagefile = False`, không có `Win32_PageFileUsage`.
- Hạn mức commit **15,7 GB**, chỉ còn trống **0,71 GB**, dù RAM vật lý còn trống 4 GB.
- Các tiến trình dùng nhiều RAM: trình duyệt ~3,5 GB (21 tiến trình), VS Code 2,1 GB, Claude 1,3 GB.
- Sau khi đóng bớt trình duyệt, commit trống lên **5,57 GB**.

**Hư hại do lần tắt đột ngột (đã kiểm):**

| Phần | Kết quả |
|---|---|
| Repo EMOCA chính | `git fsck` sạch |
| Submodule `emonet`, `SwinTransformer`, `face-parsing` | sạch |
| Nhánh submodule lồng `spectre/av_hubert/fairseq/megatron` | tệp chỉ mục git **0 byte** và vài object loose **0 byte**. Nhánh này chỉ dùng cho video, không cần cho ảnh |
| `conda-meta` | 0/155 tệp json hỏng |
| `bin` của Miniconda | không có tệp 0 byte |

**Đã làm:**
- Tạo `C:/Users/admin/.wslconfig` với `memory=5GB`, `swap=4GB` (mẫu ở mục 4-wsl.2 bước 3, nay dùng giá trị này).
- Chạy lại lần 3 từ chặng mamba. Các chặng miniconda, conda_channels, clone_emoca, pull_submodules đã xong từ trước.

**Khuyến nghị cho người dùng (chỉ người dùng tự làm được):** **bật pagefile**. Vào Cài đặt → System → About → Advanced system settings → Performance *Settings* → Advanced → Virtual memory → chọn *Automatically manage*. Cần **khởi động lại máy** để có hiệu lực.

#### Cách phát hiện
- Trên Windows (PowerShell), thay `<giờ>` bằng giờ bắt đầu chạy script:
  ```powershell
  Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=<giờ>} | Where-Object { $_.Id -in 1000,1001 -and $_.Message -match 'wsl' }
  ```
- Trong Ubuntu:
  - `uptime -s` ra giờ **muộn hơn** giờ bạn bắt đầu chạy script ⇒ máy ảo đã khởi động lại giữa chừng.
  - Có tệp 0 byte: `find ~/miniconda3 -size 0 | head`.
- Kiểm hạn mức bộ nhớ trên Windows (PowerShell), **trước khi dựng** và khi nghi hết bộ nhớ:
  ```powershell
  (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory       # commit còn trống, đơn vị KB
  (Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile # False = pagefile không tự quản lý
  ```
  Commit còn trống thấp (lúc sự cố 2 là 0,71 GB) ⇒ đóng bớt ứng dụng hoặc bật pagefile trước khi chạy lại.

#### Khuyến nghị

1. Sau khi WSL sập hoặc hết bộ nhớ giữa lúc dựng, **kiểm tệp 0 byte trước** rồi mới chạy lại.
2. Chạy lại bằng **đường dẫn DỰ ÁN** (lệnh `bash "$(wslpath '...BTL')/backend/services/emoca_svc/wsl/setup_emoca_wsl.sh"` ở mục 4-wsl.3 bước 1). **Không** dùng `~/emoca_wsl/bin/...`, vì bản sao đó cũng có thể hỏng.
3. **Tự hồi phục — bản hiện tại ĐÃ có** (hoán đổi 2026-09-25). Phần này **chỉ kiểm tĩnh**, chưa chạy thật. Xem `documents/docs/qa/REVIEW_WSL_SCRIPTS_2026-09-25.md`. Cách script hồi phục:
   - Vẫn phải chạy lại bằng **đường dẫn DỰ ÁN**. Bản sao trong `~/emoca_wsl/bin` có thể thành 0 byte, và `bash` một tệp rỗng thì thoát 0 mà không làm gì (`setup_emoca_wsl.sh:10-12`).
   - Script tự kiểm từng chặng bằng **chạy thật** (conda/mamba/python/git). Phần hỏng được **dời sang `*.broken.<epoch>`, không xoá** rồi làm lại. Áp dụng cho:
     - Miniconda: `:244-258`;
     - repo EMOCA: `:296-303`, giữ lại `assets/` và `data/`;
     - env `work38`: quanh `:366` (số dòng đọc lại 2026-09-25).
   - Riêng **Miniconda** chỉ bị dời khi có tệp đánh dấu `~/miniconda3/.emoca_wsl_owned` (script tạo tệp này sau khi tự cài, `:270`). Không có tệp đánh dấu thì script **dừng**, in hướng dẫn tự `mv`. Nếu chính script đã cài Miniconda (từ trước khi có tệp đánh dấu) thì chạy `touch ~/miniconda3/.emoca_wsl_owned` rồi chạy lại (`:255-257`).
4. Khuyến nghị 1 vẫn áp dụng: kiểm tệp 0 byte trước khi chạy lại.
5. Trước khi dựng: có `.wslconfig` giới hạn RAM (máy dự án: `memory=5GB`, `swap=4GB`); đóng bớt trình duyệt/ứng dụng nặng; tốt nhất bật pagefile (xem sự cố 2).

### 4-wsl.5 Thời gian dựng thật

Nguồn: đo 2026-09-25, điều phối `btl-bb`. Đây là **số đo thời gian dựng thật đầu tiên**: lần Colab 2025 không có số theo từng chặng. Mạng lúc đo khoảng 10 MB/s (tải installer Miniconda 198 MB mất 18 s). Số liệu lấy từ hai lần chạy:
- **Lần 2** bắt đầu **11:38:10** (log `~/emoca_wsl/out/setup_20260925_113810.log`). Lần này đo được 4 chặng đầu, rồi dừng ở chặng mamba ~11:50 vì hết hạn mức bộ nhớ (sự cố 2, mục 4-wsl.4).
- **Lần 3** bắt đầu **11:53**, chạy tiếp từ chặng mamba.

| Chặng | Lần chạy | Thời gian | Trạng thái |
|---|---|---|---|
| `miniconda` | 2 | 33 s | đã đo |
| `conda_channels` | 2 | 8 s | đã đo |
| `clone_emoca` + 2 patch | 2 | 9 s | đã đo |
| `pull_submodules` | 2 | 572 s | đã đo |
| `mamba` | 3 | 58 s | đã đo (lần 2 chết ở chặng này, sự cố 2) |
| `create_env` | 3 | 22 s | đã đo |
| `mamba env update` | 3 | 769 s (12,8 phút, xong 12:09:30) | đã đo |
| `pip_requirements38` | 3 | 1574 s (12:09:35 → 12:35:49) | đã đo |
| `pip_onnx_mediapipe_mmcv` | 3 | 133 s | đã đo |
| `pip_install_e` | 3 | 4 s | đã đo |
| `pip_opencv_chumpy_tensorboard` | 3 | 24 s (xong 12:38:30) | đã đo |
| tải tài sản (`download_assets`) và các chặng sau | — | — | **chưa đo** (lần 3 dừng ở cổng giấy phép, mã 10). Từ 2026-09-25 tối tài sản đang được tải bằng `prefetch_assets.sh` (bước 1b); **số đo cuối: chưa đo** |

Lần 3 **xong toàn bộ phần không cần giấy phép** rồi dừng ở cổng giấy phép (mã thoát 10) như thiết kế.
- **Tổng thời gian:** lần 3 tính từ 11:53 mất ≈ **45 phút**; cộng các chặng đã đo của lần 2 và lần 3 được ≈ **63 phút** (CPU/mạng, mạng ~10 MB/s).
- **Bước tải tài sản:** lúc chạy lần 3 còn chờ giấy phép FLAME/EMOCA (ROADMAP, Backlog hoãn (l)). *Cập nhật 2026-09-25 tối (theo điều phối `btl-e6`):* người dùng nhắn trong phiên điều phối khoảng 21:35, nguyên văn "tôi đã đăng ký và đồng ý điều khoản / đọc được hết license"; bằng chứng máy: `~/emoca_wsl/out/setup_20260925_213604.log` (lần chạy đầu với cờ). Tài sản đang được tải bằng `prefetch_assets.sh`. Thời gian/dung lượng: **chưa đo**.

**Dung lượng sau lần 3 (đã đo):**

| Phần | Dung lượng |
|---|---|
| env `work38` | 8.6 GB |
| cache gói (`pkgs`) | 5.6 GB |
| `~/emoca` | 801 MB |
| ext4 đã dùng (tổng) | 18 GB |

Tổng này **chưa gồm tài sản** FLAME/EMOCA vì chưa tải.

**Kiểm env sau khi dựng (đã đo):** các gói sau import được, và `torch.cuda.is_available()` = True:

| Gói | Phiên bản |
|---|---|
| torch | 1.11.0 (CUDA 11.5, thấy RTX 3050) |
| pytorch3d | 0.6.2 |
| face_alignment | 1.3.6 |
| numpy | 1.24.4 |
| cv2 | 4.5.5 |

**cuDNN:** lỗi lúc đầu là do thiếu `LD_LIBRARY_PATH=/usr/lib/wsl/lib`, đã xác định nguyên nhân và cách sửa (xem 4-wsl.1b, mục cuDNN).

Tài nguyên trong lúc dựng (đã đo): RAM trống của Windows dao động **1.8–4.2 GB**; `vmmemWSL` chiếm **2.6–3.0 GB**.

---

## 4. EMOCA thật trên Google Colab — **dự phòng** từ 2026-09-25 (ADR-018) **(chưa kiểm — toàn bộ đường Colab chưa ai chạy thật)**

ADR-015 ghi rõ: dựng env, nạp EMOCA, GPU, tunnel, độ trễ, VRAM, hệ toạ độ landmark đều **chưa kiểm chứng**. Phần đã đo chỉ là logic chạy trên máy thường với backend giả. Khả năng cao còn lỗi khi chạy thật (SESSION_LOG Session 08, nợ HIGH).

### 4a. Đóng gói mã trên máy bạn

```powershell
& $PY backend/services/emoca_svc/pack_for_colab.py
```

Ra file `another/notebooks/emoca_svc_bundle.zip` (không commit). Đổi nơi ghi bằng `--out "<đường dẫn>.zip"` (`pack_for_colab.py:103`). Gói **chỉ có mã** dự án (server, runner, patch, contract, config) — không có FLAME/checkpoint. Mã hiện hành dùng `contract_version 0.3.0` (ADR-019; cập nhật 2026-10-03).

### 4b. Chạy notebook `another/notebooks/emoca_colab.ipynb` trên Colab

1. **Giấy phép — bạn phải TỰ làm, không có đường tắt:** đăng ký tài khoản và đồng ý điều khoản ở **FLAME** (flame.is.tue.mpg.de) và **EMOCA** (emoca.is.tue.mpg.de). Cả hai chỉ cho phép dùng **phi thương mại/nghiên cứu**. Chỉ sau đó mới đổi cờ `ACCEPT_FLAME_EMOCA_LICENSES = True` ở ô **1. Thông số**. Mục 4 của notebook dừng (`assert`) nếu cờ còn `False`. Đừng đổi cờ thay người khác và đừng tìm cách bỏ qua bước này.
2. Mở notebook trên Colab → **Runtime → Change runtime type → GPU**.
3. Chạy lần lượt các mục **0 → 11**:

| Mục | Làm gì | Ghi chú |
|---|---|---|
| 0 | Khởi tạo, in phiên bản Python, `nvidia-smi`; **dừng ngay (`assert`) nếu runtime không có GPU** | không có GPU ⇒ quay lại bước 2 |
| 1 | Thông số: cờ giấy phép, `PORT = 8765`, sinh `TOKEN` ngẫu nhiên | token mới mỗi phiên |
| 2 | Tải `emoca_svc_bundle.zip` lên (ô Upload) | bundle cũ/khác commit sẽ bị cảnh báo |
| 3 | Dựng env `work38` (Python 3.8, torch 1.11/cu115, PyTorch3D 0.6.2); ô cuối mục là **kiểm nhanh môi trường** (dừng ngay nếu thiếu GPU/torch/PyTorch3D, trước khi tải tài sản) | **lâu nhất** — lần chạy cũ 2025-11-20 không lưu thời gian từng chặng, nhưng dấu giờ trong đầu ra của `emoca_setup_v2_work38_ok.ipynb` cho cận dưới: **ít nhất ~18 phút** từ tải Miniconda (ô 2, 03:44:38) đến hết tải tài sản (ô 29, 04:02:43); bảng thời gian thật in ở mục 9 |
| 4 | Tải tài sản có giấy phép (FLAME, EMOCA, texture) | cần cờ giấy phép = `True` |
| 5 | Áp patch `0001`, `0002` | không sửa EMOCA tại chỗ |
| 6 | Smoke test EMOCA trên ảnh mẫu công khai (M0-04) | |
| 7 | Chạy `server.py` (chỉ nghe `127.0.0.1`, đòi token) | |
| 8 | Kiểm bằng ảnh thật qua HTTP (M1-03), hệ toạ độ landmark (M0-05) | số liệu để dán lại cho nhóm |
| 9 | Mở Cloudflare quick tunnel, **in Địa chỉ + Token** | địa chỉ `https://….trycloudflare.com` **đổi mỗi phiên** |
| 10 | Giữ phiên (vòng lặp, in mỗi 5 phút) | ô này chạy mãi — bấm **■** để dừng trước khi chạy mục 11; server + tunnel vẫn chạy nền |
| 11 | Tải `emoca_colab_results.zip` (không chứa token) | gửi cho nhóm để ghi vào `documents/docs/qa/M0/` |

**Không chia sẻ notebook sau khi đã chạy** — đầu ra các ô chứa token (notebook, ô 0).

#### 4b-2. Checklist chạy M0 từng bước (viết 2026-09-25 sau khi rà notebook; chưa có bước nào được chạy thật)

**Trước khi mở Colab (trên máy bạn):**
- Gói mã: `another/notebooks/emoca_svc_bundle.zip` (đã đóng 2026-09-24 23:25; mã `backend/services/emoca_svc/` không đổi sau đó). Sửa bất kỳ file nào trong `backend/services/emoca_svc/` thì chạy lại `pack_for_colab.py` (mục 4a). Notebook tự kiểm sha256 từng file trong gói.
- **Tải lại notebook đã sửa** (`another/notebooks/emoca_colab.ipynb`, có ô kiểm GPU, ô kiểm nhanh môi trường, lưu ảnh M0-04, cắt mesh khỏi gói kết quả). Nếu Colab còn bản cũ đang mở: File → Upload notebook và dùng bản mới; bản cũ vẫn chạy được nhưng thiếu các ô kiểm sớm.
- Runtime → Change runtime type → **T4 GPU**. Giữ tab mở suốt phiên (Colab ngắt phiên nhàn rỗi và mất hết dữ liệu; **chưa có bộ nhớ đệm Drive**, mất phiên = dựng lại từ mục 3).

| Mục | Bạn làm | Dấu hiệu ĐẠT (in ra ở ô) | Nếu lỗi |
|---|---|---|---|
| 0 | Chạy | `GPU OK: GPU 0: … T4 …` | `AssertionError: Runtime chưa có GPU` ⇒ đổi runtime rồi chạy lại từ ô 0 |
| 1 | Đổi `ACCEPT_FLAME_EMOCA_LICENSES = True` (lời xác nhận của chính bạn), chạy | `Đã đặt thông số. Cờ giấy phép: True` | — |
| 2 | Chạy, bấm *Choose files*, chọn `emoca_svc_bundle.zip` | `Bundle OK: 11 tệp \| contract_version 0.3.0` | `sha256 không khớp` ⇒ đóng gói lại (4a) và tải lên lại (cập nhật 2026-10-03, ADR-019) |
| 3 | Chạy **lần lượt từng ô**, chờ ô xong mới chạy ô kế | các dòng `[miniconda]`, `[clone + ghim commit]`, `[mamba env update (nặng nhất)]`, `[pip + pip install -e .]`; rồi ô **kiểm nhanh môi trường** in `torch 1.11.0 \| cuda 11.5 \| gpu … T4` và `pytorch3d 0.6.2` | Ô kiểm nhanh `AssertionError` ⇒ gửi nguyên phần stderr nó in. Ô mamba lỗi ⇒ chụp 30 dòng cuối + chạy `!mamba --version` và gửi (không tự ghim phiên bản) |
| 4 | Chạy | `Tìm thấy N ảnh mẫu trong … Dùng EX_DIR = …`; ô texture qua `assert` | Texture: `Tải texture thất bại` ⇒ làm theo hướng dẫn trong thông báo (tải file về máy, `files.upload()`, `!mv`). `wget` lỗi ⇒ chạy lại ô (ghi đè an toàn), lặp lại thì gửi log |
| 5 | Chạy | `[ap    ] 0001-…`, `[ap    ] 0002-…`, `Xong (e0be0db)`; `git status` có `M gdl/utils/DecaUtils.py`, `M gdl/utils/FaceDetector.py`; in `du -sh` của env và assets (ghi lại: cần để quyết định bộ nhớ đệm Drive, gói miễn phí 15 GB) | `[LOI] … không áp được` ⇒ gửi thông báo |
| **6** | Chạy | `M0-04: N/N ảnh đủ 5 PNG + 5 NPY đúng độ dài` + dải ảnh. **NHÌN dải ảnh**: `geometry_*` phải ra khuôn mặt 3D đúng người trong `inputs` (tiêu chí M0-04 (4)) | `please check the test path` ⇒ chạy lại ô tìm ảnh mẫu (mục 4). Thiếu PNG/NPY ⇒ gửi phần cuối `/content/out/m0_04_log.txt` |
| 7 | Chạy (chờ tới 15 phút) | JSON health có `"loaded": true`, `device` là GPU; `[server nạp mô hình + sẵn sàng]` | `server.py đã thoát` ⇒ ô tự in 3000 ký tự cuối `server.log`; gửi |
| **8** | Chạy 2 ô | `PASS: mọi phản hồi đúng emoca_analyze.schema.json…`, `M1-03 AC2 ĐẠT`, `M1-03 AC5 ĐẠT`, dòng VRAM, `Miền landmark…`; rồi ảnh chồng + `NME toàn bộ`. **NHÌN ảnh chồng**: chấm đỏ phải rơi đúng mắt/mũi/miệng/hàm trên ảnh gốc (M0-05) | NME > 0.3 hoặc chấm đỏ lệch một chiều ⇒ nghi sai hệ toạ độ: **chụp ảnh gửi** — đây đúng là điều M0-05 cần bắt |
| 9 | *Nên làm luôn trong phiên này* (mỗi phiên mới phải dựng env lại từ đầu): chạy ô tunnel, ô POST thử, ô in địa chỉ, ô 9c (token hiện một lần, sao chép rồi Enter) | `Tunnel hoạt động; token sai bị từ chối 401`, `PASS: phản hồi qua tunnel…`; in **Địa chỉ** và bảng thời gian từng chặng (`TỔNG`) | Tunnel không lên ⇒ chờ rồi chạy lại ô; gửi `/content/tunnel.log` |
| — | **Thử web ngay lúc này** (giữ phiên Colab): web ⚙ Cài đặt → Kết nối EMOCA → Máy chủ EMOCA → dán địa chỉ + token → *Kiểm tra kết nối* → bật xác nhận gửi ảnh → tải **ảnh của bạn hoặc ảnh công khai** → Phân tích | Ảnh landmark/3DMM hiện là **dữ liệu thật**, chip `emoca_roundtrip` có số ms | Ghi lại mã lỗi hiện trên giao diện (mục 6.1) |
| 10 | Khi xong: bấm ■ ở ô giữ phiên | — | — |
| 11 | Chạy | `N tệp .obj…`, `Gói kết quả: X MB`, trình duyệt tải `emoca_colab_results.zip` | Gói quá nặng ⇒ báo kích thước |
| 12 | Chạy (tắt server + tunnel) | `Đã tắt: server không còn nghe…` | — |

**Gửi lại cho nhóm:** (1) `emoca_colab_results.zip` — có `m0_04_strip_*.png` (dải ảnh dựng), `m0_05_overlay.png`, `analyze_response_sample.json` (fixture vàng), `measurements.json` (thời gian từng chặng, độ trễ, VRAM), `env_work38.yml`, `pip_freeze_work38.txt`; (2) bảng thời gian mục 9; (3) mọi lỗi kèm **số mục** + ~30 dòng cuối đầu ra. Rồi `GENERAL`/`DOCS` tick M0-02..05, M1-01 (AC2), M1-03 **chỉ khi tiêu chí đã đo được**.

**Đầu ra không phải lỗi** (có cả ở lần chạy thành công 2025-11-20): `Error downloading object: …/Resnet50_Final.pth` cùng `git-lfs filter-process failed` (submodule `spectre`), và `Terms of Service not found for … conda-forge`.

### 4c. Nối web với Colab — chọn MỘT cách

**Cách 1 — bảng Cài đặt trên giao diện (khuyên dùng):** mở **Cài đặt** ở đầu trang → cột phải, *Kết nối EMOCA* → **Nguồn EMOCA = Máy chủ EMOCA** → dán **Địa chỉ EMOCA** và **Token** do mục 9 in ra → bấm **Kiểm tra kết nối** (chỉ gọi `GET /v1/health`, không gửi ảnh). Thành công hiện "emoca_svc chạy chế độ real …" kèm độ trễ và GPU.

**Cách 2 — biến môi trường** (đặt trong cùng cửa sổ PowerShell, rồi mới chạy server):

```powershell
$env:EMOCA_MODE  = "real"
$env:EMOCA_URL   = "https://xxxx.trycloudflare.com"   # địa chỉ do notebook in ra
$env:EMOCA_TOKEN = "<token do notebook in ra>"
& $PY -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8599 --server.headless true --browser.gatherUsageStats false
```

**Thứ tự ưu tiên** (`backend/web/orchestrator.py`, `resolve_emoca_settings`): ô trên giao diện > hằng `_EMOCA_MODE` trong mã (mặc định `None`) > biến môi trường `EMOCA_MODE`/`EMOCA_URL`/`EMOCA_TOKEN` > `backend/configs/services.yaml` (`emoca_svc`) > `mock`. Hệ quả thực tế:
- Biến môi trường chỉ làm **giá trị ban đầu** của bảng Cài đặt khi mở phiên; đổi trên giao diện là thắng.
- Ô Token trên giao diện luôn bắt đầu **trống**; để trống thì dùng `EMOCA_TOKEN`. Gõ token vào ô thì ô thắng.
- **Không** ghi token vào `backend/configs/services.yaml` (chú thích đầu file cấm).
- Đổi token không tự phân tích lại — bấm **Phân tích lại**.

**Thử đường "máy chủ EMOCA" mà không cần Colab (chưa kiểm bằng tay; `another/tests/test_emoca_client_remote.py` phủ luồng này):** chạy bản mock của dịch vụ trên máy bạn rồi trỏ web vào nó.

```powershell
& $PY backend/services/emoca_svc/mock_server.py --port 8765     # cửa sổ 1; in "[emoca_svc mock] http://127.0.0.1:8765 … KHÔNG phải EMOCA thật"
```

Ở web (cửa sổ 2): Nguồn EMOCA = Máy chủ EMOCA, Địa chỉ `http://127.0.0.1:8765`, Token để trống. Vì là loopback nên không cần xác nhận gửi ảnh; kết quả vẫn là **dữ liệu giả**.

### 4d. Hộp xác nhận gửi ảnh — vì sao có

Bất biến hạng 0 của dự án (`CLAUDE.md §1.1`, ADR-009): ảnh khuôn mặt không được gửi ra dịch vụ bên thứ ba. Chạy EMOCA trên Colab **vi phạm** điều này — ảnh đi qua Google (Colab) và Cloudflare (tunnel). ADR-015 chấp nhận ngoại lệ có kiểm soát (nợ `[DEBT-HIGH]`, phạm vi còn chờ người dùng xác nhận):

- Khi địa chỉ EMOCA **không phải** loopback (kể cả máy trong LAN), mục *Kết nối EMOCA* trong bảng Cài đặt hiện ô **"Tôi đồng ý gửi ảnh tới máy chủ này"**, luôn bắt đầu ở trạng thái **tắt**.
- Chưa bật ⇒ `orchestrator.analyze` trả lỗi `consent_required` và **không gửi ảnh**.
- Đổi nguồn hoặc đổi địa chỉ ⇒ xác nhận bị **thu hồi** (không thể đồng ý cho máy A rồi ngầm gửi sang máy B).
- Chỉ bật khi đó là ảnh của chính bạn hoặc ảnh công khai. Máy chủ EMOCA của dự án không ghi ảnh ra đĩa, nhưng Google/Cloudflare nằm ngoài tầm kiểm soát.

---

## 5. Chạy test

### 5.0 Cách nhanh nhất: MỘT lệnh chạy hết

```powershell
$env:MA3D_VENV_PYTHON = $PY       # CHỈ cần khi venv KHÔNG nằm ở $HOME\venvs\ma3d-web (xem dưới)
& $PY another/tests/run_all.py             # nhóm NHẸ (50–86 giây): tự chọn python thường / venv web cho từng test, in bảng PASS/FAIL/SKIP
& $PY another/tests/run_all.py --list      # chỉ xem kế hoạch, không chạy
& $PY another/tests/run_all.py --heavy     # thêm test nạp torch (test_engine_poster_real.py, test_engine_registry_extend.py; ~1.5 GB RAM) — chỉ khi máy còn RAM
```
`run_all.py` tự đặt `PYTHONIOENCODING=utf-8` cho các test con. **Gọi bằng `& $PY`, không bằng `python` trần:** mục 1.2 chỉ cài thư viện vào venv, nên trên máy mà python hệ thống chưa cài gì, các test chạy bằng python gọi `run_all.py` sẽ FAIL `ModuleNotFoundError` (`documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md`, L2; suy từ mã, chưa chạy trên máy sạch). **Đặt `MA3D_VENV_PYTHON` chỉ cần khi venv KHÔNG nằm ở `$HOME\venvs\ma3d-web`:** venv web mặc định là `$HOME\venvs\ma3d-web\Scripts\python.exe` theo thư mục người dùng hiện tại (`another/tests/run_all.py:55-57`, sửa 2026-09-25 theo L3); nếu đặt thì biến môi trường thắng (`another/tests/run_all.py:89`). Venv ở chỗ khác mà quên đặt biến ⇒ `test_app_ui.py` và nhóm nặng SKIP "không thấy python venv web". *Kết quả đo 2026-09-25 (venv mock mới, `documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md` bước 7b): 12 PASS, 2 SKIP (nặng), 0 FAIL / 14 test, 85.7 giây (một lần khác 50.5 giây); test lâu nhất là `test_wsl_scripts.py` (11–47 s).* Các lệnh riêng lẻ ở bảng dưới dành cho khi cần chạy từng test.

### 5.1 Từng test

Đặt `$env:PYTHONIOENCODING = "utf-8"` trước (mục 1.3) khi chạy riêng lẻ. Kỳ vọng chung: dòng cuối **`ALL PASS`**, mã thoát 0.

| Lệnh | Cần gì | Kỳ vọng / ghi chú |
|---|---|---|
| `& $PY another/tests/test_configs.py` | pyyaml | ALL PASS. Checkpoint chưa tải ⇒ các kiểm liên quan **SKIP có lý do** (thoát 0); `--require-artifacts` ⇒ FAIL như trước; file có mà sai số byte/sha256 thì luôn FAIL. `--sha256` băm lại ~2.5 GB. |
| `& $PY another/tests/check_emoca_stats.py <file.npz>` | numpy | Kiểm file thống kê chuẩn hoá 3DMM (10 khoá, số chiều, NaN, std>0); `--selftest` tự kiểm công cụ (PASS 8 ca). File thật `backend/model/ma3d/checkpoints/rafdb_emoca_stats.npz` → PASS (đo 2026-09-25). |
| `& $PY another/tests/test_orchestrator.py` | pillow, jsonschema | ALL PASS (23 dòng đạt, đo 2026-09-25 qua `run_all`, `documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md`); chạy được ở cả python thường lẫn venv web |
| `& $PY another/tests/test_ai_mock.py` | pillow, jsonschema | ALL PASS (18 ca theo Session 08) |
| `& $PY another/tests/test_theme.py` | pillow | ALL PASS (10 ca) |
| `& $PY another/tests/test_emoca_svc_mock.py` | pillow, jsonschema | ALL PASS (33 ca, đo 2026-09-25 qua `run_all`); dùng cổng tạm |
| `& $PY another/tests/test_emoca_svc_real.py` | pillow, jsonschema | ALL PASS; backend **giả**, không GPU. *Đã chạy khi viết: ALL PASS.* |
| `& $PY another/tests/test_emoca_client_remote.py` | pillow, jsonschema | ALL PASS (51 ca, đo 2026-09-25 qua `run_all`); HTTP thật trên cổng tạm cục bộ |
| `& $PY another/tests/test_colab_bundle.py` | pillow, pyyaml | ALL PASS. Kiểm patch áp lên EMOCA nên cần `backend/model/emoca/` (chưa kiểm khi thiếu thư mục này). *Đã chạy khi viết: ALL PASS.* |
| `& $PY another/tests/check_ma3d_checkpoints.py` | checkpoint | PASS/FAIL theo kỳ vọng trong `backend/configs/models.yaml`; đọc ~2.5 GB, 1–3 phút, **không cần torch**. `run_all.py` xếp vào nhóm "nhẹ" = không nạp torch/ít RAM, nhưng đọc đĩa nặng |
| `& $PY another/tests/test_app_ui.py` | **streamlit** (venv web) | ALL PASS (15 ca, đo 2026-09-25); thiếu streamlit ⇒ in SKIP, thoát 0 |
| `& $PY -m pytest another/tests/test_face_preprocessor.py -q` | pytest, numpy, pillow (venv web) | Unit test offline với detector giả; không cần mạng, Ultralytics hay checkpoint. Chạy từ thư mục gốc repo; xem M5-01 bên dưới. Chưa có kết quả chạy được ghi nhận. |
| `& $PY another/tests/test_engine_poster_real.py` | **torch + checkpoint** (cài 1.2b) | ALL PASS đã đo (venv web, torch 2.14.0+cpu, Session 09); thiếu torch/checkpoint/RAM ⇒ SKIP, thoát 0. Nặng RAM: kiểm RAM trống trước (nạp cả hai mô hình ~1 GB RSS). |
| `& $PY another/tests/test_engine_registry_extend.py` | **torch + checkpoint** (cài 1.2b); nhóm **nặng** | M2-06: thêm 3 mục registry tạm và chạy qua `engine.predict` mà không đổi mã (ALL PASS, 12 ok, đo 2026-09-25). Tạm thay `engine._MODELS_CONFIG` bằng tệp tạm rồi khôi phục |
| `& $PY another/tests/test_wsl_scripts.py` | **Git Bash** (không cần WSL) | Kiểm tĩnh script `backend/services/emoca_svc/wsl/`; ALL PASS (237 ok, đo 2026-09-25). Thiếu Git Bash ⇒ SKIP phần chạy bash. Test chạy lâu nhất nhóm nhẹ (11–47 s) |
| `& $PY another/tests/audit_contrast_browser.py` | `pip install playwright`, trình duyệt Chromium có sẵn, **server web đang chạy ở cổng 8599** | thoát 1 nếu có chữ không đạt WCAG. `--all` (mọi chủ đề, ~15 phút), `--emoca` (thêm trạng thái Máy chủ EMOCA) |

Các con số "(n ca)" lấy từ SESSION_LOG Session 08–09 và `documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md` (L6), có thể lệch khi test thay đổi; số đo mới nhất luôn là bảng của `& $PY another/tests/run_all.py`.

### 5.2 YOLO26-face — preflight tự động và panel detector phụ

Môi trường đã kiểm chứng là venv web `C:\Users\admin\venvs\ma3d-web` (Python 3.13.5). Cài đúng bộ real/test dependency từ thư mục gốc sau khi cài Torch CPU theo mục 1.2b:

```powershell
& $PY -m pip install -r environments/web/requirements-real.txt
```

`environments/web/requirements-real.txt` ghim `ultralytics==8.4.166` và `pytest==9.1.1`. Checkpoint đặt tại `backend/model/yolo26_face/yolo26_face.pt`; model path, confidence threshold 0.5, crop scale 1.3 và imgsz 640 lấy từ `backend/configs/face_detection.yaml`. Crop scale 1.3 mở rộng chiều rộng và chiều cao quanh tâm bbox đã clamp; `bbox_xyxy` vẫn là bbox detector còn `crop_bbox_xyxy` xác định chính xác vùng crop trong ảnh nguồn (cập nhật 2026-10-03, ADR-024/027/028).

Chạy unit tests offline (fake detector, không tải backend/model/mạng):

```powershell
& $PY -m pytest another/tests/test_face_preprocessor.py -q
```

Trong ứng dụng, preflight YOLO chạy tự động cho **mọi ảnh upload** trước khi chọn luồng một mặt/nhiều mặt (ADR-024/025; cập nhật 2026-10-03, ADR-024/025). Thứ tự hiện tại (ADR-027/028; cập nhật 2026-10-03, ADR-027/028):

1. Chạy YOLO trên ảnh gốc.
2. Nếu YOLO không thấy mặt, làm nét ảnh bằng RealESRGAN rồi chạy YOLO lại.
3. Box được dùng để tạo crop từ **ảnh gốc**; ảnh RealESRGAN chỉ dùng cho lượt dò thứ hai. Một mặt đi theo baseline ảnh upload; nhiều mặt dùng các crop gốc cho phân tích tuần tự, tối đa 30 mặt.
4. Nếu detector lỗi (khác với kết quả hợp lệ 0 mặt), giao diện hiện cảnh báo rồi chạy baseline trên ảnh gốc; không dừng phân tích. Nếu cả hai lượt YOLO hợp lệ đều không thấy mặt, cũng chạy baseline trên ảnh gốc.

Panel **Thử YOLO26-face (tùy chọn)** / **Chạy detector phụ** vẫn là phần xem ảnh trung gian riêng; nó không điều khiển preflight tự động. Ảnh, annotated/crop và kết quả chỉ được giữ trong `st.session_state` của ảnh hiện tại, rồi bị xóa khi ảnh đổi hoặc bị gỡ. Checkpoint weights local được người dùng chấp nhận cho coursework; upstream provenance chưa xác minh độc lập (cập nhật 2026-10-03, ADR-021; Session 41).

Lưu ý: số đo smoke/A/B bên dưới được thực hiện với cấu hình crop cũ sát bbox (scale 1.0), nên không phải kết quả đã xác nhận cho scale 1.3. Smoke cũ trên `database/image/archive/DATASET/test/2/test_2303_aligned.jpg` ghi bbox `(3, 2, 95, 100)`, confidence `0.807876`, crop RGB `92×98`; output chỉ ở RAM. UI smoke bằng AppTest cũng xác nhận shadow stage giữ nguyên kết quả pipeline chính. Kết quả và SHA-256 hiện tại xem `documents/docs/qa/M5_01B_MODEL_SOURCE.md`; file local đã được người dùng chấp thuận cho đồ án, nhưng revision upstream chưa gắn độc lập.

Kết quả smoke/A/B trên là **bằng chứng lịch sử của M5** với cấu hình và scale cũ; không mô tả trạng thái tích hợp hiện tại. A/B trên 21 ảnh ghi nhận YOLO phát hiện 16/21 ở conf 0.5 và mỗi model FER đổi top-1 trên một ảnh trong 16 cặp downstream (`documents/docs/qa/M5_01_AB_2026-09-30.md`). Preflight/crop hiện đã được đưa vào luồng theo ADR-024/025/027/028; cap và routing M7 được kiểm lại trên detector thật, gồm 0/1/2/30/36 mặt (`documents/docs/qa/M7_04_ACCEPTANCE_2026-10-03.md`; cập nhật 2026-10-03).

### 5.3 M7 — Multi-Object Analysis

Khi có ảnh, app chạy YOLO trên ảnh gốc trước suy luận cảm xúc. Chỉ khi lượt gốc hợp lệ không thấy mặt mới chạy RealESRGAN → YOLO lần nữa. Giữ tối đa **30** box lớn nhất, sau đó đánh ID theo thứ tự trên xuống rồi trái sang phải. Crop luôn lấy từ ảnh gốc; ảnh/crop chỉ ở RAM/session (ADR-024/025/027/028; cập nhật 2026-10-03).

- Một mặt: chạy luồng dự đoán thường trên ảnh upload với `max_faces=1`; không hiển thị thao tác Multi-Object Analysis.
- Từ hai mặt trở lên: không gọi dự đoán single-person hoặc giao diện kết quả trực quan/geometry; app tự chạy crop (cắt từ ảnh gốc theo box YOLO) → EMOCA/POSTER/MA3D-Net cho toàn bộ danh sách theo thứ tự, một lần cho mỗi ảnh/cấu hình. Mỗi crop có nhãn dự đoán nhỏ bên dưới (ADR-025/027; cập nhật 2026-10-03).
- Mỗi card có thể mở **Full details**; **View Details** chuyển tới trang riêng có ảnh tham chiếu gốc, crop, bbox/confidence và dự đoán/lỗi. **Phân tích lại** làm mới kết quả nếu cần thử lại.
- **Lịch sử ADR-024:** detector lỗi từng chặn dự đoán; ADR-027 đã thay quy tắc này. Hiện tại YOLO lỗi thì app cảnh báo và chạy baseline ảnh gốc; nếu hai lượt YOLO hợp lệ không thấy mặt thì cũng chạy baseline ảnh gốc (cập nhật 2026-10-03, ADR-027/028).
- Radar chart phân phối xác suất đã được bỏ khỏi giao diện single-person.

Phạm vi cap-30, định tuyến, tự chạy, nhãn dự đoán, grid và route chi tiết đã nghiệm thu (Session 40/41). Báo cáo [`M7_REPORT_2026-10-01.md`](qa/M7_REPORT_2026-10-01.md) là bằng chứng lịch sử cho yêu cầu cap-20; kết quả nghiệm thu phạm vi hiện tại ở [`M7_04_ACCEPTANCE_2026-10-03.md`](qa/M7_04_ACCEPTANCE_2026-10-03.md) (cập nhật 2026-10-03, ADR-024/025).

**Cập nhật nghiệm thu:** M7 cap-30, định tuyến tự động, grid và trang View Details đã được kiểm bằng test và người dùng kiểm bằng mắt (Session 40/41, `documents/docs/qa/M7_04_ACCEPTANCE_2026-10-03.md`; cập nhật 2026-10-03, ADR-024/025). Báo cáo M7 cũ nêu giới hạn cap-20 chỉ là lịch sử.

### 5.4 Sửa roll trong baseline và khu so sánh

Baseline tự chạy 3DDFA-V2 trên CPU để ước lượng roll từ crop YOLO. Nếu `abs(roll_deg)` đạt `roll_threshold_deg`, app xoay ảnh 2D ngược chiều roll rồi gửi ảnh đã xoay qua EMOCA và POSTER/MA3D-Net. Với nhiều mặt, app sửa từng crop. Nếu không tìm thấy mặt hoặc phép đo roll lỗi, app giữ ảnh gốc và tiếp tục. Không hiệu chỉnh yaw/pitch và không warp 3D. Hành vi này được bật theo ADR-029; khu **Experiment: Sửa nghiêng (3DDFA-V2)** ở cuối trang vẫn cho xem đối chứng trước/sau.

Ngưỡng nằm trong [`backend/configs/frontalization.yaml`](../../backend/configs/frontalization.yaml): `roll_threshold_deg: 15`, `enabled_in_baseline: true`. Kaggle RAF-DB sweep chọn 15° vì ngưỡng này có mean Acc(all) cao nhất trong các ngưỡng giữ mức giảm ở ảnh 0° dưới 0,2 điểm phần trăm cho từng mô hình. Benchmark dùng ảnh đã căn chỉnh rồi xoay nhân tạo; nó chưa xác nhận hiệu quả trên ảnh tự nhiên (ADR-028/029, Session 42/44).

### 5.5 Notebook benchmark trên Kaggle

Hai notebook phục vụ benchmark nghiên cứu, không phải bước cần để chạy web. Mở notebook tương ứng trong Kaggle, gắn private dataset RAF-DB/artifacts mà notebook yêu cầu, bật Internet và accelerator được ghi trong notebook/Session 38–39, chạy smoke trước rồi chạy full. Tải kết quả về thư mục đầu ra nêu dưới đây; notebook có thể tiếp tục từ shard khi lượt chạy bị ngắt (cập nhật 2026-10-03, Session 38–41).

| Notebook | Mục đích | Kết quả |
|---|---|---|
| `another/kaggle/notebook_frontalize/run_rafdb_frontalize.ipynb` | Benchmark nghiên cứu 3DDFA-V2 trên RAF-DB ở ảnh gốc/xoay và các góc roll; đồng thời đo YOLO trên ảnh suy giảm tổng hợp, có/không RealESRGAN. Nhánh YOLO dùng cảnh ghép từ nền tổng hợp nên không đại diện ảnh tự nhiên. | Kết quả đã lưu trong `database/results/M8_KAGGLE/`; metrics gồm coverage, Acc, macro F1/F1 theo lớp, chia theo yaw/pitch, McNemar và recall/IoU/false positives/thời gian theo ghi chú notebook. |
| `another/kaggle/notebook_roll/run_rafdb_roll_sweep.ipynb` | Quét `roll_threshold_deg` trên các góc 0/15/30/45/60/90°; kết quả chọn 15° cho cả POSTER và MA3D-Net. | Raw shards, `threshold_sweep.csv` và `summary.json` ở `database/results/M8_ROLL/roll_sweep/`; báo cáo tại [`M8_KAGGLE_RESULTS_2026-10-03.md`](qa/M8_KAGGLE_RESULTS_2026-10-03.md). |

Lượt Kaggle đã hoàn tất. Gói tải đã được giải nén vào `database/results/M8_ROLL/roll_sweep/`; file zip sau đó được dọn theo Session 44. M8-03 đã đóng và ngưỡng 15° được bật trong baseline theo ADR-029. Benchmark dùng RAF-DB đã căn chỉnh rồi xoay tổng hợp, nên không chứng minh hiệu năng trên ảnh tự nhiên.

---

## 6. Xử lý sự cố

### 6.1 Mã lỗi hiện trên giao diện

Lỗi hiện dạng **`[mã] thông báo`**. `status: partial` (hộp vàng) = có một phần kết quả; `error` (hộp đỏ) = không có kết quả. Nguồn: `frontend/app.py` (`ERROR_HINTS`), `backend/web/orchestrator.py`, `backend/web/emoca_client.py`.

| Mã | Nghĩa | Việc cần làm |
|---|---|---|
| `no_face` | Không tìm thấy mặt. Ở mock: cạnh ngắn ảnh < 32 px; ở EMOCA thật: FAN không phát hiện mặt | Ảnh chính diện, đủ sáng, không che mặt |
| `image_invalid` | Không đọc được ảnh, hoặc không phải JPEG/PNG/WebP | Đổi tệp khác |
| `image_too_large` | > 10 MiB, hoặc cạnh dài > 4096 px (phía máy chủ EMOCA) | Thu nhỏ ảnh (`backend/configs/services.yaml`, `limits`) |
| `service_unavailable` | Không nối được EMOCA (phiên Colab tắt, tunnel đổi, token sai — HTTP 401 cũng dùng mã này), hoặc bộ điều phối lỗi import | Bấm **Kiểm tra kết nối**; xem 6.2; nếu thông báo nhắc `backend/web/orchestrator.py` thì xem terminal chạy Streamlit rồi khởi động lại |
| `timeout` | EMOCA không trả lời trong 60 s (`emoca_svc.timeout_s`) | Thử lại (lần gọi đầu sau khi Colab nghỉ có thể chậm); ảnh nhỏ hơn |
| `model_unavailable` | EMOCA: chưa có địa chỉ, hoặc địa chỉ không bắt đầu bằng `http://`/`https://`. Engine thật: thiếu torch/checkpoint, mô hình không có trong `backend/configs/models.yaml` | Dán địa chỉ; hoặc cài mục 1.2b + tải checkpoint; hoặc chuyển "Suy luận" về Giả lập |
| `codes_missing` | MA3D-Net thật không suy luận vì: (1) EMOCA đang **giả lập** — thông báo chứa "giả lập" (`backend/web/engine.py:63`, ADR-017); hoặc (2) mã 3DMM thiếu/sai dạng; hoặc (3) thiếu file thống kê `*_emoca_stats.npz` (RAF-DB đã có từ 2026-09-25) | (1) là **hành vi đúng**: nối EMOCA thật (Cài đặt → Kết nối EMOCA, mục 4) để có kết quả MA3D-Net. POSTER không bị ảnh hưởng |
| `contract_mismatch` | Dữ liệu EMOCA sai contract, không phải JSON, hoặc `contract_version` khác `0.3.0` | Đóng gói lại (4a), tải bundle mới lên Colab, chạy lại từ mục 2; nếu vẫn lỗi báo `GENERAL`/`03_BACKEND` (cập nhật 2026-10-03, ADR-019) |
| `consent_required` | EMOCA trỏ ra ngoài máy mà chưa bật xác nhận; **ảnh chưa được gửi** | Bật "Tôi đồng ý gửi ảnh…" (mục 4d) hoặc chuyển về Giả lập |
| `internal_error` | Lỗi không lường trước trong bộ điều phối (thông báo có tên exception) | Chép thông báo + log terminal, báo nhóm; bấm Phân tích lại |

### 6.2 Tunnel và phiên Colab

Thông báo lấy từ `backend/web/emoca_client.py` (`_http_error_hint`).

| Hiện tượng | Nguyên nhân thường gặp | Việc cần làm |
|---|---|---|
| HTTP **401** "token sai hoặc thiếu" | Token dán sai/thiếu, hoặc token của phiên Colab cũ | Dán lại token mục 9 in ra; ô Token thắng `EMOCA_TOKEN` |
| HTTP **403** "bị chặn" | Tunnel/proxy từ chối yêu cầu | (chưa kiểm) Thử lại; xem `tunnel.log` trên Colab |
| HTTP **404** | Sai địa chỉ | Chỉ dán gốc `https://….trycloudflare.com`, không thêm `/v1/...` |
| HTTP **502** "tunnel không nối được tới dịch vụ" | `server.py` chưa chạy hoặc đã chết | Chạy lại mục 7 trên Colab, xem `/content/server.log` |
| HTTP **530** "tunnel đã tắt" | Phiên Colab đã ngắt / tunnel chết | Chạy lại notebook; **địa chỉ và token mới**, dán lại |
| "Không nối được tới emoca_svc … Địa chỉ tunnel có đổi không?" | Mạng, DNS của tunnel mới chưa kịp, hoặc URL cũ | Đợi vài giây rồi Kiểm tra kết nối; kiểm URL |
| Colab ngắt giữa chừng | Giới hạn thời gian/GPU của Colab | Web trả `partial`, trang không hỏng. Chạy lại notebook (env phải dựng lại mỗi phiên) |
| Địa chỉ đổi | Quick tunnel đổi URL **mỗi phiên** | Luôn lấy địa chỉ mới ở đầu ra mục 9 |
| "emoca_svc chạy chế độ mock (KHÔNG phải EMOCA thật)" | Đang trỏ tới `mock_server.py` | Đúng nếu bạn cố ý thử (4c); nếu không, kiểm lại địa chỉ |

---

## 7. Cái gì là thật, cái gì là giả

| Thành phần | Giả lập (mặc định) | Thật | Trạng thái kiểm chứng |
|---|---|---|---|
| **EMOCA** (landmark, mã 3DMM/FLAME, 4 ảnh dựng) | `backend/web/emoca_client.py` mode `mock`: landmark theo khuôn mẫu, ảnh dựng là ô màu cố định | `backend/services/emoca_svc/server.py` trong **WSL2 trên máy bạn** (đường chính, ADR-018); Colab qua tunnel là dự phòng (ADR-015) | **M7 real-model smoke PASS** trên một mặt, hai crop và đường không mặt; chi tiết/giới hạn ở [`M7_REPORT_2026-10-01.md`](qa/M7_REPORT_2026-10-01.md). Toàn bộ script `smoke_test_emoca.sh` chưa chạy. |
| **POSTER** (xác suất 7 nhãn, attention, `feature_maps`) | `backend/web/engine.py` mode `mock`: số giả | `backend/web/engine.py` mode `real`, CPU trong tiến trình web | Đo chỉ-đọc 1 ảnh: nạp strict 1369 khoá, tổng xác suất 1.0000. Độ trễ không ổn định: 877 ms lúc rảnh – ~5.4 s lúc thiếu RAM (SESSION_LOG Session 09); đo đầu-cuối qua giao diện ở `documents/docs/qa/E2E_ENGINE_REAL_UI.md`: lần đầu (nguội) ≈ 8.6 s, các lần ấm ≈ 0.9–1.2 s, chạy đơn lẻ nên có nhiễu; **chưa** đo p50/p95 (M4-01). **Chưa** đo độ chính xác so với `eval.py` gốc (M2-07) |
| **MA3D-Net** | mock: số giả | `backend/web/engine.py` mode `real` + file thống kê `rafdb_emoca_stats.npz` (có từ 2026-09-25, `backend/configs/models.yaml:69-72`). **Chỉ suy luận khi EMOCA thật**; EMOCA giả lập ⇒ từ chối với `codes_missing` (thông báo chứa "giả lập"), POSTER vẫn chạy bình thường (ADR-017) | **M7 real-model smoke PASS:** hai crop đầu vào nhận dự đoán MA3D từ codes EMOCA thật; dữ liệu thử được tạo trong RAM, không phải đánh giá độ chính xác. Chi tiết ở [`M7_REPORT_2026-10-01.md`](qa/M7_REPORT_2026-10-01.md). |
| **Landmark / 3DMM trên giao diện** | toạ độ `original_px` từ khuôn mẫu giả | `crop224 = lm·112 + 112`, rồi nghịch đảo phép crop về `original_px` (ADR-015 mục 8) | Hệ toạ độ đọc từ mã, **chưa đo** (M0-05) |
| **Giao diện / tương phản** | — | thật | Đo trong Cốc Cốc thật: 96 tổ hợp chủ đề × chế độ, 0 không đạt (Session 07); sau sửa CSS Session 08 chỉ đo lại 12 tổ hợp. Chạy lại đủ 96 tổ hợp (Backlog (c)): **hoãn** 2026-09-25 vì thiếu RAM. Màn hình hẹp 375 px và spinner: `documents/docs/qa/UI_LOADING_NARROW_2026-09-25.md` |
| **Mẫu kết quả EMOCA thật dự phòng** (M4-04 AC2) | — | — | Có fixture tập con thật tại `another/tests/fixtures/emoca_analyze_real_sample.min.json`; đã bỏ ảnh dựng/crop có ràng buộc quyền sử dụng, nên không xác nhận schema đầy đủ có các renders bắt buộc. Đây không phải toàn bộ phản hồi EMOCA. `backend/contracts/web_analyze.example.*.json` vẫn chỉ là fixture contract (cập nhật 2026-10-03, ROADMAP M4-04; ADR-019) |

Giao diện luôn nói rõ nguồn: cảnh báo `mock_ai: …` cho phần giả, `engine_real: …` và `emoca_real: … upstream <commit>` cho phần thật; khi chỉ một phần là thật, giao diện ghi "một phần là dữ liệu MOCK" (ADR-015 mục 6). **Một ảnh chỉ là minh hoạ định tính**, không phải bằng chứng "khớp bài báo" (`CLAUDE.md §1.8`).

---

## 8. Riêng tư và giấy phép

- **Ảnh khuôn mặt** là dữ liệu cá nhân. Web xử lý ảnh trong bộ nhớ, không ghi đĩa, không log nội dung ảnh (`backend/web/orchestrator.py`, đầu file). Ở chế độ mock và khi EMOCA ở loopback, ảnh không rời máy.
- **Colab = ngoại lệ:** ảnh đi qua Google + Cloudflare, chỉ sau khi bạn bật xác nhận (4d). Chỉ dùng ảnh của chính bạn hoặc ảnh công khai. Phạm vi ngoại lệ này **chưa** được người dùng xác nhận bằng văn bản và **chưa** ghi vào `CLAUDE.md §1.1` (SESSION_LOG Session 08, mục đề xuất).
- **FLAME và EMOCA:** giấy phép **phi thương mại**, mỗi người tự đăng ký. Texture `FLAME_albedo_from_BFM.npz` có ràng buộc giấy phép BFM, notebook đang lấy từ một id Google Drive cá nhân — tự kiểm tra quyền dùng (`documents/docs/modules/EMOCA.md §7.4`).
- **Không commit / không chia sẻ:** tài sản FLAME, texture, checkpoint EMOCA (kể cả zip trong `~/emoca_wsl/cache` do `prefetch_assets.sh` tải), checkpoint MA3D/POSTER (`backend/model/ma3d/checkpoints/`), `another/notebooks/emoca_svc_bundle.zip`, `emoca_colab_results.zip` chưa rà soát, notebook đã chạy (chứa token), token. `pack_for_colab.py` chặn tài sản có giấy phép khi đóng gói.
- Không push lên bất kỳ remote nào khi nhóm chưa đồng ý; không bao giờ push lên `august0362/template_claude` (`CLAUDE.md §1.1`).

---

## Điểm còn mâu thuẫn (chưa sửa — báo `GENERAL`/người giao việc)

1. ~~`CLAUDE.md §2` ghi `backend/web/engine.py` "`real` chưa làm" và thiếu `another/tests/test_engine_poster_real.py`~~ — **đã được phiên điều phối sửa** trong lúc viết tài liệu này (bản đồ nay ghi `mock` và `real` trên CPU).
2. **Đã giải quyết (02_CONFIG_METRICS):** tách thành `environments/web/requirements.txt` (chỉ mock) và `environments/web/requirements-real.txt` (torch CPU + jsonschema); mục 1.2 đã sửa theo. Mô tả cũ: ~~**`environments/web/requirements.txt`** nói phần torch "chế độ mock không cần", nhưng các dòng `torch`, `torchvision`, `einops`, `matplotlib`, `numpy` đang có hiệu lực ⇒ `pip install -r environments/web/requirements.txt` cài cả torch. Đồng thời `jsonschema` (cần cho test) chỉ nằm trong chú thích. Runbook này né bằng lệnh cài tường minh (mục 1.2).~~
3. **Không sửa được, đã giảm mức (2026-09-25):** SESSION_LOG Session 08, Exact Next Step 1 ghi "chạy từng mục 0 → 9", còn notebook có mục 0 → 11 (mục 10 là vòng giữ phiên phải dừng tay trước mục 11). SESSION_LOG chỉ thêm, không sửa block cũ ⇒ đọc theo notebook (0 → 11). Từ ADR-018 notebook Colab chỉ còn là dự phòng.
4. **Đã giải quyết (02_CONFIG_METRICS, Session 09):** `another/tests/test_configs.py` nay SKIP có lý do khi thiếu checkpoint (thoát 0), `--require-artifacts` mới FAIL; file có mà sai số byte/sha256 vẫn FAIL. Mô tả cũ: ~~đòi đủ checkpoint trên đĩa ⇒ thành viên mới chưa tải ~2.47 GB sẽ thấy FAIL dù mã đúng~~.
5. **Giảm nhẹ, chưa xử lý gốc (kiểm lại 2026-09-25: 1/12 tệp `another/tests/test_*.py` tự ép UTF-8):** test in tiếng Việt vẫn dừng với `UnicodeEncodeError` (cp1258) nếu chạy riêng lẻ khi đầu ra bị chuyển hướng mà chưa đặt `PYTHONIOENCODING=utf-8`; `another/tests/run_all.py` tự đặt biến này nên chạy qua bộ chạy chung thì không gặp. Sửa gốc (ép UTF-8 ở đầu mỗi test) chưa làm.
6. **ROADMAP M4-04:** AC1 đã đạt qua chạy thử RUNBOOK ngày 2026-09-25; AC2 được tick với fixture tập con EMOCA thật tại `another/tests/fixtures/emoca_analyze_real_sample.min.json`. Fixture bỏ ảnh/render có ràng buộc quyền sử dụng và không thay thế response đầy đủ (cập nhật 2026-10-03, ROADMAP M4-04; ADR-019).
7. **Đã giải quyết (2026-09-25), ghi để đối chiếu:** ROADMAP đã tick M2-06, M3-01, M3-02, M3-05 (người dùng duyệt; bằng chứng dưới từng task). Stats RAF-DB cho MA3D-Net đã có (`backend/configs/models.yaml`, M2-01 ghi chú); MA3D-Net chỉ chạy khi mã 3DMM do EMOCA thật sinh ra (ADR-017). EMOCA đường chính là WSL2 (ADR-018) — các chỗ trong tài liệu này còn nói "Colab" là nói về đường dự phòng.
