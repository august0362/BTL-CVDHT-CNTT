# RUNBOOK_DRYRUN_2026-09-25 — Chạy thử `documents/docs/RUNBOOK_DEMO.md` như người mới (M4-04 AC1)

> Vai: QA (`GENERAL`), giao bởi phiên điều phối btl-e6. Ngày 2026-09-25, Windows 11, PowerShell 7, Python hệ thống 3.13.7.
> Phạm vi: RUNBOOK mục 1.1, 1.2(a), 1.3, 2 (web **mock**) và 5.0 (chạy test). **Không** thử 1.2(b) (torch), 4/4-wsl (EMOCA), 3.
> Cách làm: gõ đúng nguyên văn lệnh trong RUNBOOK, chỉ đổi 2 chỗ theo yêu cầu người giao việc (ghi rõ ở từng bước):
> venv đặt trong thư mục scratch của phiên (không đụng `$HOME\venvs\ma3d-web`), web chạy cổng **8600** (8599 đang dùng).
> Không sửa RUNBOOK hay mã. Mọi thời gian là đo thật (Measure-Command / Stopwatch); mạng và cache pip của máy này ảnh hưởng số đo.

## Kết luận

**ĐẠT CÓ ĐIỀU KIỆN.** Người mới dựng được venv, chạy được web mock và thấy đúng "Kỳ vọng ở chế độ mock"; bộ test chạy
được với venv mới (12 PASS, 2 SKIP nặng, 0 FAIL). Nhưng làm **đúng nguyên văn** thì vấp 2 chỗ chặn/đỏ oan (L1, L2) và
1 chỗ làm test âm thầm SKIP trên máy khác (L3). Sửa 3 dòng tài liệu (+1 dòng mã, đề xuất) là đủ.

## Từng bước

| # | Bước RUNBOOK | Lệnh đã chạy | Thời gian | Kết quả |
|---|---|---|---|---|
| 1 | 1.1 tạo venv (`RUNBOOK_DEMO.md:48-50`) | `python -m venv "<scratch>\rb-venv"` (đổi đường dẫn theo yêu cầu) ; `& $PY --version` | 11.8 s | ĐẠT — Python 3.13.7 |
| 2 | 1.2(a) `pip install -r environments/web/requirements.txt` (`:60`) | nguyên văn | 101.5 s | ĐẠT — mã 0; chỉ có thông báo nâng pip (vô hại) |
| 3 | 1.2(a) `pip install jsonschema==4.26.0` (`:61`) | nguyên văn | 2.1 s | ĐẠT |
| 4 | Dung lượng venv sau 1.2(a) | tổng byte mọi file trong venv | — | **333 MB** (streamlit kéo theo numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1) |
| 5 | 2. chạy web (`:86`) | nguyên văn, chỉ đổi `--server.port 8600` | **treo** (60 s không có health) | **KHÔNG ĐẠT** — xem L1: Streamlit lần đầu hỏi `Email:` và chờ bàn phím |
| 5b | 2. chạy lại với `--server.headless true` (không có trong RUNBOOK) | thêm cờ | 1.4 s tới `/_stcore/health` = `ok` | ĐẠT — chỉ nghe `127.0.0.1:8600` (đã kiểm bằng Get-NetTCPConnection). Log in `Collecting usage statistics…` (L4) |
| 6 | 2. "Kỳ vọng ở chế độ mock" (`:100`) | Playwright (venv ma3d-web, Chromium 1440×900): mở trang, tải ảnh PNG trong repo | mở trang 3.1 s; tải ảnh → có kết quả 3.4 s | ĐẠT — nhãn "Dữ liệu MOCK" ở đầu trang + cảnh báo "DỮ LIỆU MOCK — chưa chạy mô hình thật."; khu landmark/ảnh dựng; 2 thẻ top-1 (POSTER (RAF-DB), MA3D-Net (RAF-DB)); thanh xác suất; bảng Mô hình có cột Checkpoint; ghi chú "minh hoạ định tính"; không có hộp lỗi. Ảnh dùng: `documents/docs/qa/UI_LOADING_NARROW_2026-09-25/mock_result_1440.png` (238 KB) vì repo không có ảnh khuôn mặt mẫu (L7) |
| 7a | 5.0 `python another/tests/run_all.py` (`:529`), đặt `MA3D_VENV_PYTHON` = venv mới như `:533` dặn | nguyên văn | 50.5 s | 11 PASS, 2 SKIP (nặng), **1 FAIL** `test_wsl_scripts.py` — `SyntaxError` ở `another/tests/test_wsl_scripts.py:701`. **Không phải lỗi RUNBOOK:** file bị phiên khác sửa lúc 22:42:57 giữa chừng; bản hiện tại `py_compile` sạch và ở 7b đã PASS. Máy này python hệ thống có sẵn mọi thư viện nên không lộ L2 |
| 7b | 5.0 nhưng gọi bằng python của venv mới: `& $PY another/tests/run_all.py` | đổi `python` → `& $PY` | 85.7 s | **12 PASS, 2 SKIP (nặng), 0 FAIL / 14** — chứng minh venv mock (1.2a) đủ cho mọi test nhẹ, kể cả `test_app_ui.py` (15 ok) |
| 8 | Dọn | dừng PID 3420 + 10092 (web 8600), xoá venv tạm | 2.6 s | Xong; 8600 không còn nghe; web 8599 vẫn `ok`; không tạo `~\.streamlit\credentials.toml` |

## Lỗi tài liệu (kèm đề xuất sửa — CHƯA sửa)

| # | Mức | Chỗ | Vấn đề (đã đo) | Đề xuất |
|---|---|---|---|---|
| L1 | **Chặn** | `RUNBOOK_DEMO.md:86` (và `:496`) | Máy chưa từng chạy Streamlit (không có `~\.streamlit\credentials.toml`) ⇒ Streamlit in "Welcome to Streamlit! … Email:" và **chờ nhập**, server chưa nghe cổng. Ở cửa sổ PowerShell tương tác người mới sẽ thấy lời hỏi nhưng RUNBOOK không nói phải làm gì; chạy nền thì treo hẳn (đo: 60 s không có health) | Thêm `--server.headless true` vào lệnh, **hoặc** một dòng: "Lần đầu Streamlit hỏi Email — bấm Enter để bỏ qua". Khuyên thêm cờ (không ghi file vào thư mục người dùng) |
| L2 | **Đỏ oan** trên máy sạch | `RUNBOOK_DEMO.md:529-531` | Lệnh là `python another/tests/run_all.py` (python **hệ thống**), còn mục 1.2 chỉ cài thư viện vào venv (`& $PY -m pip …`). Nhóm "system" của `run_all.py` cần pillow, jsonschema, pyyaml, numpy ⇒ trên máy mà python hệ thống chưa cài gì, các test đó FAIL `ModuleNotFoundError`. Máy dự án không lộ vì python hệ thống đã có đủ (7a); gọi bằng venv thì 0 FAIL (7b) | Đổi thành `& $PY another/tests/run_all.py` (cả 3 dòng), khớp quy ước "mọi lệnh dùng `& $PY`" ở `:53` |
| L3 | **SKIP âm thầm** trên máy khác | `RUNBOOK_DEMO.md:533`; mã `another/tests/run_all.py:55` | Venv web mặc định ghi cứng `C:\Users\admin\venvs\ma3d-web\…`. Người mới theo đúng 1.1 có venv ở `C:\Users\<tên họ>\venvs\ma3d-web` ⇒ `test_app_ui.py` (và nhóm nặng) SKIP "không thấy python venv web" nếu quên đặt `MA3D_VENV_PYTHON` | Tài liệu: ghi `$env:MA3D_VENV_PYTHON = $PY` ngay trước lệnh run_all. Mã (đề xuất cho người sở hữu `another/tests/`): `DEFAULT_VENV = str(Path.home() / "venvs" / "ma3d-web" / "Scripts" / "python.exe")` |
| L4 | Riêng tư (nhẹ) | `RUNBOOK_DEMO.md:86`, `:91` | Không có `--browser.gatherUsageStats false` ⇒ log in "Collecting usage statistics" — trình duyệt gửi số liệu dùng (không phải ảnh) cho Streamlit. Lệnh do điều phối dùng cho 8599 có cờ này, RUNBOOK thì không | Thêm `--browser.gatherUsageStats false` vào lệnh và một câu giải thích cạnh đoạn "Vì sao BẮT BUỘC `--server.address`" |
| L5 | Số cũ | `RUNBOOK_DEMO.md:529`, `:533` | "~40 giây", "11 PASS, 1 SKIP … 38.5 giây" | Nay 14 test: 12 PASS + 2 SKIP nặng, 50–86 s (đo ở 7a/7b; lâu nhất `test_wsl_scripts.py` 11–47 s) |
| L6 | Số cũ / thiếu dòng | bảng `RUNBOOK_DEMO.md:541-553` | Số ca lệch bảng run_all hôm nay: `test_orchestrator` 22→23, `test_emoca_svc_mock` 27→33, `test_emoca_client_remote` 47→51, `test_app_ui` 12→15 (`test_ai_mock` 18 vẫn khớp). Thiếu dòng cho `test_wsl_scripts.py` (cần Git Bash, không cần WSL; thiếu Git Bash ⇒ SKIP phần chạy bash) và `test_engine_registry_extend.py` (nặng) | Thêm 2 dòng; cân nhắc bỏ số ca cứng, chỉ giữ câu `:555` "số mới nhất là bảng của run_all" |
| L7 | Mơ hồ | `RUNBOOK_DEMO.md:100` | "tải một ảnh JPEG/PNG/WebP bất kỳ" — repo không có ảnh khuôn mặt mẫu; người mới không biết lấy ảnh nào (ảnh mặt người thật = dữ liệu cá nhân) | Ghi rõ: ở mock ảnh bất kỳ ≥ 32 px đều được (kể cả ảnh không có mặt); ảnh chụp màn hình trong `documents/docs/qa/…/*.png` dùng được. Nếu muốn ảnh mặt mẫu thì cần ADR về nguồn/giấy phép, không commit ảnh người thật |
| L8 | Số cũ | bảng `RUNBOOK_DEMO.md:43` | "vài trăm MB (venv 418 MB lúc còn plotly)" | Đo nay: **333 MB** sau 1.2(a), cài mất ~104 s (mạng máy dự án) |
| L9 | Nhỏ | `RUNBOOK_DEMO.md:48` | Nếu `$HOME\venvs\ma3d-web` đã tồn tại (máy dùng chung / chạy lại), `python -m venv` không báo gì và cài chồng vào venv cũ | Một câu: "đã có thư mục này thì bỏ qua 1.1 hoặc chọn tên khác" |

## Không đo / ngoài phạm vi

- 1.2(b) torch CPU, chế độ "Thật (CPU)", `--heavy` với venv mới (venv mock không có torch ⇒ 2 test nặng chắc chắn SKIP).
- Ô `Email:` khi chạy tay ở cửa sổ tương tác (chỉ tái hiện ở chế độ nền); không bấm Enter để khỏi tạo `credentials.toml` trong thư mục người dùng.
- Mục 3 (bảng Cài đặt), 4/4-wsl (EMOCA), `audit_contrast_browser.py`.
- Python hệ thống "sạch" thật (máy này đã cài đủ); L2 suy ra từ việc nhóm "system" của `run_all.py` chạy bằng python gọi nó và các import của test, **chưa** chạy trên máy sạch.

---

## Lần 2 — chạy lại theo RUNBOOK đã sửa L1–L9 (cùng ngày, sau bản sửa của DOCS)

> Cùng cách làm như lần 1: venv mới `<scratch>\rb-venv2` (chưa tồn tại trước), web cổng **8600**, ảnh
> `documents/docs/qa/UI_LOADING_NARROW_2026-09-25/mock_result_1440.png`, test nhóm **nhẹ** (không `--heavy` — RAM máy đang thấp).
> Lệnh gõ nguyên văn RUNBOOK bản mới; chỉ đổi đường dẫn venv và cổng. Vì venv không nằm ở `$HOME\venvs\ma3d-web`, đặt
> `$env:MA3D_VENV_PYTHON = $PY` đúng như RUNBOOK `:561` dặn. Máy vẫn chưa có `~\.streamlit\credentials.toml` (điều kiện L1 còn nguyên).

**Kết luận lần 2: ĐẠT.** Mọi bước chạy đúng nguyên văn, không cần tự thêm cờ hay đổi lệnh; L1–L9 đã được xử lý trong
tài liệu (L3 cả trong mã, `another/tests/run_all.py:55-57`). Còn 2 lỗi tài liệu nhỏ, không chặn (R1, R2).

| # | Bước RUNBOOK (bản mới) | Thời gian | Kết quả |
|---|---|---|---|
| 1 | 1.1 `python -m venv` + `& $PY --version` (`:48-50`) | 11.5 s | ĐẠT — Python 3.13.7 |
| 2 | 1.2(a) `pip install -r environments/web/requirements.txt` (`:62`) | 94.4 s | ĐẠT — mã 0 |
| 3 | 1.2(a) `pip install jsonschema==4.26.0` (`:63`) | 2.1 s | ĐẠT — venv **333 MB**, khớp `:43` |
| 4 | 2. chạy web, lệnh `:88` nguyên văn (cổng 8600) | **1.1 s** tới health `ok` | ĐẠT — không còn lời hỏi `Email:` (L1), không còn "Collecting usage statistics" (L4); chỉ nghe `127.0.0.1:8600` |
| 5 | 2. "Kỳ vọng ở chế độ mock" (`:106`) + "Lấy ảnh ở đâu" (`:108`) — Playwright 1440×900 | mở trang 2.2 s; tải ảnh → kết quả 3.4 s | ĐẠT — nhãn "Dữ liệu MOCK" + cảnh báo "DỮ LIỆU MOCK — chưa chạy mô hình thật."; landmark/ảnh dựng; 2 thẻ top-1 (POSTER (RAF-DB), MA3D-Net (RAF-DB)); thanh xác suất; cột Checkpoint; "minh hoạ định tính"; không hộp lỗi. Ảnh gợi ý ở `:108` dùng được |
| 6 | 5.0 `$env:MA3D_VENV_PYTHON = $PY` ; `& $PY another/tests/run_all.py` (`:561-562`) | 83.1 s | ĐẠT — **12 PASS, 2 SKIP (nặng), 0 FAIL / 14**; `test_app_ui.py` chạy bằng venv mới (15 ok); nằm trong khoảng "50–86 giây" của `:562` |
| 7 | Dọn | — | Dừng PID 316 + 8456 (web 8600), xoá venv tạm; 8600 không còn nghe; 8599 vẫn `ok`; không tạo `credentials.toml` |

### Lỗi tài liệu còn lại (đề xuất — CHƯA sửa)

| # | Mức | Chỗ | Vấn đề | Đề xuất |
|---|---|---|---|---|
| R1 | Hiển thị | `RUNBOOK_DEMO.md:566` | Markdown vỡ: `**Đặt \`MA3D_VENV_PYTHON**\`MA3D_VENV_PYTHON\` chỉ cần khi …:**` — dấu đậm/code lồng sai, tên biến lặp hai lần (có thể DOCS đang sửa dở câu này) | `**Đặt \`MA3D_VENV_PYTHON\` chỉ cần khi venv KHÔNG nằm ở \`$HOME\venvs\ma3d-web\`:**` |
| R2 | Số cũ | `RUNBOOK_DEMO.md:234`, `:587` | "215 dòng `ok`" cho `test_wsl_scripts.py`; hôm nay đo **237** ok (lần 1 bước 7b và lần 2 bước 6) — test đã được phiên khác mở rộng | Đổi thành 237, hoặc bỏ số cứng, dẫn về bảng `run_all` như `:566`/câu cuối mục 5.1 |

Không đo lần 2: như lần 1 (1.2b/torch, `--heavy`, mục 3–4, python hệ thống sạch thật).
