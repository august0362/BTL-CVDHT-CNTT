# Đo giao diện: trạng thái "đang tải" (M3-02) và màn hình hẹp 375 px (Backlog d) — 2026-09-25

- **Vai trò:** QA giao diện, giao việc từ phiên điều phối `btl-e6`. Không sửa `backend/web/*`, `another/tests/*`, ROADMAP; không tick.
- **Đối tượng:** web thật `http://127.0.0.1:8599`, sau khi `btl-56` khởi động lại bằng mã mới (PID 23076 nghe cổng, điều phối xác nhận; DOCS kiểm `Get-NetTCPConnection` → 23076, `/_stcore/health` → `ok`). Đo lúc 21:43–21:50.
- **Công cụ:** playwright 1.63.0 (cài vào venv `ma3d-web`, không sửa `requirements*.txt`), Chrome có sẵn (`C:\Program Files\Google\Chrome\Application\chrome.exe`) headless; không tải Chromium. Script tạm trong scratch của phiên (không thuộc repo): mốc thời gian bằng `MutationObserver` (`performance.now()`, ms tính từ lúc gọi `set_input_files`), đo tràn bằng `getBoundingClientRect`/`scrollWidth`.
- **Ảnh thử:** `backend/model/emoca/gdl_apps/EMOCA/emoca.png` (ảnh minh hoạ có sẵn trong upstream EMOCA, 911×247, 334 KB) — không dùng ảnh cá nhân của người dùng. EMOCA ở chế độ **Giả lập** trong mọi lần đo ⇒ nội dung ảnh không ảnh hưởng kết quả.
- **Chủ đề:** mặc định (Cold · Sáng · tương phản Chuẩn · cỡ chữ Vừa). Mỗi kịch bản đo **một lần**.

## 1. Spinner "Đang phân tích" (`frontend/app.py:802`)

| Chế độ POSTER/MA3D-Net | Spinner xuất hiện (DOM) | Playwright thấy `visible` | Spinner biến mất (DOM) | Ảnh |
|---|---|---|---|---|
| Giả lập (mock) | **không xuất hiện** (0 lần) | không (chờ 15 s → Timeout) | — | `mock_result_1440.png` (kết quả, 1440 px) |
| Thật (CPU), lần đầu nạp mô hình | **651.6 ms** sau khi tải ảnh | 901.9 ms | **6507.3 ms** | `real_spinner_1440.png` (đang quay), `real_result_1440.png` (sau khi xong) |

- **Vì sao mock không có spinner:** Streamlit 1.64.0 chỉ hiện nội dung `st.spinner` khi khối chạy lâu hơn `DELAY_SECS = 0.5` giây (`streamlit/elements/spinner.py:36`, trong venv `ma3d-web`; bộ đếm `threading.Timer(DELAY_SECS, …)` ở `:132`). Phân tích mock xong nhanh hơn ngưỡng này (chip thời gian ở một lần mock: `total 23 ms`) ⇒ spinner không bao giờ hiện. Đây là hành vi của Streamlit, **không phải lỗi** ở `frontend/app.py`.
- **Chế độ Thật:** mốc xuất hiện 651.6 ms khớp ngưỡng 0.5 s cộng thời gian rerun và truyền về trình duyệt. Spinner hiện liên tục ~5.9 s (651.6 → 6507.3 ms); chữ đầy đủ: "Đang phân tích (EMOCA → POSTER / MA3D-Net)… Lần đầu nạp mô hình thật, có thể vài giây." (xem `real_spinner_1440.png`).
- **RSS tiến trình web (PID 23076, `WorkingSet64`):** 99 MB (21:43:29, trước đo) → 100 MB (21:44:04, sau đo mock) → **860 MB** (21:44:33, sau đo Thật — torch + mô hình nằm lại trong tiến trình tới lần khởi động lại sau; điều phối đã đồng ý).
- **Lỗi JavaScript (`pageerror`):** 0 ở cả hai lần.
- **Giới hạn:** chỉ đo lần nạp đầu (nguội). Ở lần Thật ấm (suy luận ~0.5–0.7 s theo `documents/docs/qa/E2E_ENGINE_REAL_UI.md`), spinner có thể hiện chập chờn quanh ngưỡng 0.5 s — **chưa đo**.

## 2. Màn hình hẹp 375 px

Viewport 375 px; chụp màn hình đầu trang ở 812 px chiều cao (×2 mật độ điểm ảnh), chụp toàn bộ kết quả bằng viewport 375×6000 (Streamlit cuộn trong `section[data-testid=stMain]` nên `full_page` của playwright không chụp quá màn đầu).

| Bảng Cài đặt | Trạng thái trang | Tràn ngang | Ảnh |
|---|---|---|---|
| đóng | trống (chưa tải ảnh) | không (`stMain` 375/375) | `mock_375_settings_closed_empty.png` |
| mở | trống | không | `mock_375_settings_open_empty.png` |
| đóng | có kết quả (mock) | **CÓ** — `stMain` scrollWidth **568** / clientWidth 375 | `375_settings_closed_result_full.png` |
| mở | có kết quả (mock) | **CÓ** — cùng nguyên nhân | `375_settings_open_result_full.png` |

- **Lỗi hiển thị thấy được (MED):** bảng mô hình `table.mx-table` trong thẻ "Ghi chú trung thực" rộng **536 px** (7 cột: Mô hình, id, Dataset, Checkpoint, Số lớp, Bật, Kết quả — `frontend/app.py:694`), trong khi ô chứa chỉ rộng 311 px. Mọi cha từ `stMarkdownContainer` tới `stMainBlockContainer` đều `overflow-x: visible`; phần tử cuộn đầu tiên là `section[data-testid=stMain]` (`overflow-x: auto`) ⇒ **cả vùng nội dung chính cuộn ngang 193 px**, không riêng cái bảng. Khi chưa cuộn, cột Checkpoint bị cắt ở mép phải (chỉ thấy "rafdb_POSTE…") và các cột Số lớp/Bật/Kết quả nằm ngoài màn (xem nửa dưới `375_settings_closed_result_full.png`). Hướng sửa (việc của `04_FRONTEND`, chưa làm): bọc bảng trong khối `overflow-x: auto` riêng, hoặc xếp dọc ở màn hẹp.
- Cột Checkpoint là mới (có sau lần khởi động lại, `frontend/app.py:694`); bản web đo lúc chạy thử trước đó chưa có cột này.
- **Không thấy chữ chồng lên nhau** ở phần còn lại (đã xem bằng mắt: đầu trang + Cài đặt mở 1 cột ở màn đầu; 2/3 dưới của kết quả khi Cài đặt đóng — ảnh dựng, hai cột dự đoán xếp dọc, radar, ghi chú. Chưa xem từng phần của bản Cài đặt mở dài 5660 px). Phép đo chữ bị cắt (phần tử lá có `scrollWidth > clientWidth`): 0 ở cả 4 tổ hợp.
- **Sai sót đo của chính phiên này, đã sửa:** lần chạy thử đầu chỉ so `document.documentElement.scrollWidth` với viewport và bỏ qua phần tử nằm trong cha có `overflow` ⇒ báo nhầm "không tràn ngang" (đã gửi điều phối). Kết luận đúng là bảng trên.
- `mock_report.json`, `real_report.json` là đầu ra thô của script; các tên ảnh `*_375_settings_*_result.png` và `real_375_*` nhắc trong JSON đã được xoá (ảnh chỉ có màn đầu, hoặc trùng byte với bản mock vì phiên trình duyệt mới luôn ở chế độ Giả lập) — thay bằng hai tệp `375_settings_*_result_full.png`.

## 3. Kết luận cho ROADMAP (không tick)

- **M3-02 "trạng thái đang tải":** ĐÃ ĐO ĐƯỢC ở chế độ Thật (spinner 651.6 → 6507.3 ms, có ảnh). Ở chế độ mock spinner không hiện do ngưỡng 0.5 s của Streamlit — giao diện không có trạng thái đang tải *nhìn thấy được* ở mock, nhưng cũng không cần vì kết quả về trong < 0.5 s. Quyết định tick thuộc `GENERAL`.
- **Backlog (d):** đã đo. Có một lỗi tràn ngang MED do bảng mô hình 7 cột khi có kết quả; phần còn lại không tràn, không chồng chữ.

## 4. Cập nhật (2026-09-25, sau khi `btl-56` sửa)

- **Lỗi tràn ngang 375 px ở mục 2: ĐÃ SỬA** bởi `btl-56`. Bảng mô hình nay nằm trong `div.mx-table-wrap` (`frontend/app.py:697`), với CSS `max-width:100%; overflow-x:auto` (`frontend/theme.css:177-179`). Mã sửa lúc 21:47; web cổng 8599 đã khởi động lại (PID nghe cổng lúc đo lại: 2112).
- **Đo lại** (cùng ảnh thử, viewport 375×6000, Cài đặt đóng, EMOCA/engine Giả lập): `section[data-testid=stMain]` scrollWidth **375** / clientWidth 375 (trước khi sửa: **568**) ⇒ trang không còn cuộn ngang. Bảng rộng 699 px, cuộn trong khối bọc (clientWidth 311 / scrollWidth 699). Bảng rộng hơn trước (536 → 699 px) vì ô bảng nay `white-space:nowrap` (`frontend/theme.css:179`).
- Chưa đo lại tổ hợp Cài đặt mở. Chưa chụp ảnh mới sau khi sửa, nên các ảnh `375_settings_*_result_full.png` trong thư mục vẫn là ảnh **trước khi sửa**.
