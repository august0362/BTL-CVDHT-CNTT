# E2E — engine `real` qua giao diện web thật

- **Ngày đo:** 2026-09-24 · **Người đo:** phiên 01_AI (theo yêu cầu điều phối btl-f9) · **Phục vụ:** M4-01
- **Loại số đo:** mỗi kịch bản đo **MỘT lần**. Máy đang chạy nhiều phiên khác nên số có nhiễu. **Không** suy ra p50/p95 từ các số này.
- **Không sửa mã:** chỉ đo. `backend/web/`, `another/tests/`, `backend/configs/` giữ nguyên.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| Server | `ma3d-web\Scripts\python.exe -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8600 --server.headless true` (server riêng, không đụng 8599) |
| Tiến trình được đo | PID 17136 (tiến trình con của launcher 5168 do venv tạo; là tiến trình giữ cổng 8600) |
| Biến môi trường | `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=2`, `PYTHONIOENCODING=utf-8` |
| torch | 2.14.0+cpu, CUDA=False |
| Trình duyệt | Cốc Cốc 152.0.7977.124, headless, qua Playwright; khung nhìn 1440×1500 |
| URL | `http://127.0.0.1:8600/?m=light&c=normal&t=cold` (và `m=dark`) |
| Ảnh thử | `test_face.png` 373×420 px, 134 KB (một khuôn mặt) |
| EMOCA | **mock** (mặc định) — chỉ POSTER/MA3D-Net chạy thật |
| RAM trống trước khi đo | 4.60 GB vật lý, 2.74 GB commit |

**Cách đo thời gian (phía trình duyệt):**
- Bắt đầu tính giờ ngay trước thao tác (tải ảnh / bấm "Thật (CPU)" / bấm "Phân tích lại").
- Dừng khi Streamlit hết trạng thái chạy (không còn `stStatusWidget`/`stSpinner`) **và** mục "So sánh biểu cảm" có mặt.
- Ở mọi bước, script đều thấy trạng thái chạy trước khi dừng, nên không có bước nào bị tính là "xong" trước khi chạy.
- Thời gian phía server lấy từ chip `timing_ms` trên trang.

**Cách đo RAM:**
- WS = WorkingSetSize của PID 17136, đọc bằng `K32GetProcessMemoryInfo`, tương đương `WorkingSet64`.
- "Commit" = PagefileUsage (bộ nhớ riêng đã cam kết) của cùng tiến trình.

## 2. Thời gian và RAM

| # | Kịch bản | Trình duyệt (s) | Chip server: emoca / ma3d / total (ms) | WS trước → sau (MB) | Đỉnh WS (MB) | Commit sau (MB) |
|---|---|---|---|---|---|---|
| a | Mock: tải ảnh (light) | 1.02 | 13 / 0 / 25 | 85 → 93 | 94 | 77 |
| b | **Thật — lần đầu, NGUỘI** (bấm "Thật (CPU)", tự phân tích lại) | **8.60** | 9 / 641 / **7432** | 93 → **1080** | **1373** | 1304 |
| c1 | Thật — ẤM, "Phân tích lại" lần 1 | 1.21 | 9 / 712 / 744 | 1080 → 1083 | 1373 | 1324 |
| c2 | Thật — ẤM, lần 2 | 1.02 | 8 / 602 / 633 | 1083 → 1082 | 1373 | 1340 |
| c3 | Thật — ẤM, lần 3 | 0.92 | 8 / 558 / 591 | 1082 → 1085 | 1373 | 1360 |
| d-a | Mock: tải ảnh (dark, phiên trình duyệt mới) | 0.94 | 9 / 0 / 22 | 1086 → 1087 | 1373 | 1363 |
| d-b | Thật (dark) — **ĐÃ ẤM** (mô hình đã nạp trong tiến trình từ bước b) | 1.22 | 8 / 666 / 695 | 1087 → 1086 | 1373 | 1379 |

- WS lúc server vừa khởi động, chưa có trình duyệt: 75 MB.
- Log server ở bước b ghi: `poster_rafdb: sẵn sàng sau 1.7 s`, `ma3d_rafdb: sẵn sàng sau 1.3 s`, cả hai đều strict (1369 khoá, không thiếu/thừa).
- Phần còn lại của 7.4 s ở bước b gồm import torch + mã MA3D và băm sha256 (không đo tách riêng).
- Hai thanh chụp thêm (khung nhìn cao, sau bước d) cho thời gian suy luận POSTER là 764 ms (light) và 836 ms (dark). Bước này chỉ để chụp ảnh, không tính vào bảng.

**Nhận xét:**
1. **Lần đầu nguội ≈ 8.6 s. Các lần ấm ≈ 0.9–1.2 s** ở phía trình duyệt, trong đó phần suy luận POSTER trên CPU chiếm khoảng 0.56–0.71 s.
2. **Chế độ Thật tốn ≈ 1.0 GB WS**; đỉnh WS 1.37 GB, commit ≈ 1.3 GB. Lý do: `backend/configs/models.yaml` → `default_comparison` gồm **cả** POSTER và MA3D-Net, nên giao diện yêu cầu cả hai. Nạp lười (việc trước) **không** giảm RAM ở luồng mặc định này. MA3D-Net được nạp (~240 MB theo đo ở `another/tests/test_engine_poster_real.py`) nhưng rồi trả `codes_missing` vì thiếu `*_emoca_stats.npz`.
3. **Chip `ma3d` ở lần nguội (641 ms) không chứa thời gian nạp.** Chỉ `total` (7432 ms) phản ánh thời gian nạp. Đọc chip `ma3d` riêng sẽ hiểu sai lần nguội.
4. Commit tăng dần qua các lần chạy ấm: 1304 → 1324 → 1340 → 1360 → 1363 → 1379 MB, khoảng +15–20 MB mỗi lần phân tích. Có thể là cache của Streamlit/bộ cấp phát, cũng có thể là rò rỉ. **Mới có 5 điểm đo, chưa kết luận được.**

## 3. Giao diện ở chế độ Thật (chép đúng những gì hiển thị)

**Chip nguồn** (cạnh ảnh xem trước): `test_face.png` · `134 KB` · `EMOCA: giả lập` · `POSTER/MA3D-Net: thật (CPU)`. Nhãn đầu trang: `Dữ liệu MOCK` và `Cold · Sáng` / `Cold · Tối`.

**Cột POSTER (RAF-DB):**
- Dòng phụ: `model_id poster_rafdb · dataset RAF-DB`.
- Top-1: **😄 Vui vẻ — "Top-1 · độ tin cậy 87.6%"**.
- Thanh xác suất theo thứ tự hiển thị:

| Nhãn | Xác suất |
|---|---|
| Ngạc nhiên | 1.8% |
| Sợ hãi | 1.8% |
| Ghê tởm | 2.0% |
| **Vui vẻ ★** | **87.6%** |
| Buồn | 1.8% |
| Tức giận | 1.7% |
| Trung tính | 3.3% |

- Tiếp theo là `Thời gian suy luận: 558 ms` (lần c3) và mục thu gọn "Ảnh trung gian & attention (7×7)".
- Ảnh vào là crop **giả lập** vì EMOCA đang mock, nên con số này chỉ mang tính minh hoạ.

**Cột MA3D-Net:**
- Tiêu đề thẻ: **`ma3d_rafdb`** (id thô, không phải tên hiển thị). Dòng phụ: `model_id ma3d_rafdb · dataset ?`.
- Hộp lỗi đỏ: "Mô hình lỗi **[codes_missing]** Chưa có file thống kê chuẩn hoá 3DMM (*_emoca_stats.npz, `stats` trong backend/configs/models.yaml); không suy luận với mã chưa chuẩn hoá hay điền 0."

**Banner lỗi đầu trang** (vàng, phía trên "Hình học khuôn mặt"):
- "[codes_missing] Chưa có file thống kê chuẩn hoá 3DMM (…) không suy luận với mã chưa chuẩn hoá hay điền 0."
- Dòng thứ hai: "**Thiếu mã 3DMM từ EMOCA nên MA3D-Net không chạy được.**"

**Radar Chart:** chỉ có chuỗi POSTER (RAF-DB), vì MA3D lỗi nên không có dữ liệu.

**Ghi chú trung thực:**
- Banner vàng: "**MỘT PHẦN LÀ DỮ LIỆU MOCK:** landmark, mã 3DMM và ảnh dựng là dữ liệu giả (EMOCA đang giả lập); POSTER chạy thật trên CPU nhưng nhận ảnh cắt giả lập (chưa căn chỉnh khuôn mặt) nên kết quả chỉ mang tính minh hoạ."
- Cảnh báo:
  - `engine_real: poster_rafdb chạy thật trên CPU trong tiến trình web`
  - `mock_ai: emoca_client (landmark, 3DMM, ảnh dựng) chạy chế độ mock — dữ liệu giả, không phải suy luận thật`
  - `Mặt #0: mock: dữ liệu giả sinh trong backend/web/, không phải suy luận thật`
- Bảng mô hình: **chỉ một dòng** — POSTER (RAF-DB) · poster_rafdb · RAF-DB · 7 · có.
- Chip thời gian: `emoca 8 ms` · `ma3d 558 ms` · `total 591 ms` · `contract_version 0.2.0` (lần c3).

**Thanh bên:** mục "Mô hình POSTER / MA3D-Net" có "Suy luận: Giả lập | **Thật (CPU)**". Chú thích ghi "~1 GB RAM cho hai mô hình". Con số này khớp số đo ở §2 (≈ 1.0 GB WS).

## 4. Ảnh chụp

Thư mục: `C:\Users\admin\AppData\Local\Temp\claude\C--Users-admin-Desktop-File-All-N-m-H-c-K--1-n-m-3-C-c-V--hi-n---i-c-a-CNTT-BTL\8a263c74-d592-40a1-aa27-fe3d934a6ea3\scratchpad\e2e\shots\`. Đây là thư mục tạm, không commit, có thể bị dọn.

| File | Nội dung |
|---|---|
| `light_real_full.png`, `dark_real_full.png` | Toàn trang ở chế độ Thật (khung nhìn 1440×4200) |
| `light_real_crop.png`, `dark_real_crop.png` | Cắt vùng "So sánh biểu cảm" + "Ghi chú trung thực" |
| `light_a_mock.png`, `light_b_real.png`, `light_c_real_warm.png`, `dark_a_mock.png`, `dark_b_real.png` | Từng bước; chỉ 1500 px đầu, vì Streamlit cuộn trong khung trong nên `full_page` không chụp hết |
| `*.txt` | `inner_text` của trang ở từng bước |
| `../e2e_log.json` | Số đo thô |
| `../st8600.log` | Log server |

Script: `e2e_real.py` (đo), `shoot_tall.py` (chụp), cùng thư mục `e2e\`.

## 5. Lỗi và điểm lệch trên giao diện (chưa sửa — `frontend/app.py` thuộc phiên khác)

| # | Mức | Mô tả | Bằng chứng |
|---|---|---|---|
| U1 | Trung bình | **Thông điệp sai nguyên nhân:** banner đầu trang ghi "Thiếu mã 3DMM từ EMOCA nên MA3D-Net không chạy được", trong khi mã 3DMM **có** (mock). Nguyên nhân thật là thiếu file thống kê `*_emoca_stats.npz`, đúng như dòng trên của chính banner. Dòng này có vẻ là gợi ý chung cho mọi `codes_missing`. | `light_real_full.png`, `dark_b_real.png` (y≈690–810) |
| U2 | Thấp | **Thẻ MA3D-Net mất tên hiển thị/dataset khi lỗi:** tiêu đề là `ma3d_rafdb`, `dataset ?`. Mục lỗi từ engine không kèm `display_name`/`dataset` (đúng schema lỗi), nên giao diện cần tra registry để hiển thị. | `*_real_crop.png` (cột phải) |
| U3 | Thấp | **Bảng "Mô hình" ở Ghi chú trung thực chỉ có POSTER**, thiếu dòng MA3D-Net (enabled) dù mô hình này đã được yêu cầu và đã nạp. Chưa kiểm `models_info` được sinh ở đâu. | `*_real_crop.png` (cuối) |
| U4 | Thấp | Chip `ma3d` ở lần nguội không gồm thời gian nạp (xem §2, nhận xét 3). Nên có chip riêng cho thời gian nạp, hoặc chú thích. | e2e_log.json bước b |
| U5 | Chưa đo | Chế độ tối: nút đang chọn của các thanh chọn ("Tối", "Thật (CPU)", "Chuẩn", "Vừa") nhìn bằng mắt có chữ xanh nhạt trên nền xanh, có thể tương phản thấp. **Chưa đo** (theo yêu cầu, không chạy `audit_contrast_browser.py`). | `dark_b_real.png` thanh bên |

**Những gì không thấy:**
- Không có exception trên trang (`stException` = 0 ở mọi bước).
- Không có `pageerror` JS.
- Log Streamlit không có dòng lỗi/traceback.
- Không thấy chữ chồng hoặc tràn trong các ảnh đã xem.

## 6. Điều chưa chắc

- Mỗi kịch bản chỉ đo một lần, nhiều phiên khác chạy song song: sai số không xác định.
- "Nguội" ở đây là nguội **trong tiến trình** (torch và mô hình chưa nạp). Cache đĩa của hệ điều hành đã ấm, vì trước đó `another/tests/test_engine_poster_real.py` vừa đọc cùng checkpoint. Lần khởi động đầu tiên sau khi bật máy có thể chậm hơn.
- Bước d-b (dark) **không phải** lần nguội: mô hình đã nạp sẵn trong tiến trình (cache mức module dùng chung giữa các phiên trình duyệt). Muốn đo nguội ở chế độ tối phải khởi động lại server.
- Thời gian phía trình duyệt gồm cả độ trễ poll 50 ms và thời gian vẽ lại của Streamlit.
- WS phụ thuộc cách Windows cắt bớt working set khi thiếu RAM. Commit (PagefileUsage) ổn định hơn để so sánh.
- Xu hướng commit tăng dần (§2, nhận xét 4) cần đo dài hơn (vài chục lần phân tích) trước khi kết luận là rò rỉ.

## 7. Kiểm tra rò bộ nhớ (engine trực tiếp, 60 lần)

- **Ngày đo:** 2026-09-24 · **Người đo:** phiên 01_AI · đo một lần, không qua trình duyệt.
- **Script tạm** (không nằm trong repo): `scratchpad\e2e\leak_check.py`, chạy bằng python venv `ma3d-web` với `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=2`.
- **Cách chạy:** gọi `engine.predict(ảnh, codes, models=["poster_rafdb"], attention=True, feature_maps=True, mode="real")` 60 lần liên tiếp.
  - Ảnh: `test_face.png` 373×420.
  - Ghi số sau lần 1 và sau mỗi 5 lần; gọi `gc.collect()` ngay trước mỗi lần ghi.
  - WS = WorkingSetSize, Commit = PagefileUsage, đọc bằng `K32GetProcessMemoryInfo` của chính tiến trình.

| Mốc | Lần | WS (MB) | Commit (MB) |
|---|---|---|---|
| trước khi nạp | 0 | 33.2 | 21.9 |
| sau khi nạp + lần 1 (5.88 s) | 1 | 798.6 | 938.9 |
| | 5 | 800.7 | 940.5 |
| | 10 | 730.6 | 848.4 |
| | 15 | 731.3 | 848.5 |
| | 20 | 731.0 | 848.3 |
| | 25 | 728.1 | 845.4 |
| | 30 | 731.1 | 848.4 |
| | 35 | 730.9 | 848.2 |
| | 40 | 712.7 | 828.2 |
| | 45 | 712.3 | 827.8 |
| | 50 | 719.4 | 835.2 |
| | 55 | 719.1 | 835.2 |
| | 60 | 719.2 | 835.2 |

- **Độ dốc từ lần 10 đến lần 60:** WS −11.4 MB (−0.23 MB/lần), Commit −13.2 MB (−0.26 MB/lần). Không tăng.
- **Thời gian suy luận** (`latency_ms`, lần 2–60): min 512, trung vị 527, max 568 ms.

**Kết luận: không thấy rò trong 60 lần** ở `engine.predict` POSTER real có attention + feature_maps.
- Commit đi ngang hoặc giảm nhẹ sau ~10 lần đầu, thấp hơn nhiều so với ngưỡng nghi ngờ +2 MB/lần.
- Forward hook được `remove()` trong `finally` (`backend/web/engine.py`, `_real_prediction`), và `captured` là biến cục bộ của mỗi lần gọi. Số đo không cho thấy chúng giữ tensor lại.
- **Vì vậy không sửa mã.**

**Phạm vi và điều chưa chắc:**
- Mức tăng ~15–20 MB mỗi lần ở §2 (qua giao diện) **không tái hiện được ở tầng engine**. Nguồn có thể nằm ở tầng Streamlit/orchestrator, ví dụ `session_state`, ảnh/bytes của file tải lên, bộ nhớ đệm của phần tử giao diện, hoặc nhiều phiên trình duyệt trong cùng tiến trình. **Chưa đo** tầng đó.
- Chưa đo đường MA3D-Net có `x_3d` (cần `*_emoca_stats.npz` thật) và chưa đo `orchestrator.analyze`.
- Mới đo 60 lần một lần chạy. Rò rất nhỏ (< 0.2 MB/lần) có thể bị nhiễu che mất.
