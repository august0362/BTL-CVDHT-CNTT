# E2E — giao diện web với EMOCA THẬT (WSL2) + POSTER/MA3D-Net thật — 2026-09-25

- **Người đo:** phiên 01_AI (+03_BACKEND), giao việc từ điều phối `btl-e6`. Phục vụ **M3-04**, **M2-05**; bằng chứng thêm cho M0-06 AC(4).
- **Không sửa mã:** `backend/web/`, `backend/configs/`, `backend/contracts/` giữ nguyên. Cấu hình bằng biến môi trường (cơ chế có sẵn, `backend/web/orchestrator.py:9-13`).
- **Mỗi kịch bản đo MỘT lần.** Máy Windows khi đo chỉ còn **0,71 GB RAM trống** (các phiên khác đang chạy) ⇒ thời gian có nhiễu, **không** suy ra p50/p95.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| EMOCA | `backend/services/emoca_svc/wsl/run_emoca_server.sh` trong WSL2 (Ubuntu-22.04), `127.0.0.1:8765`, `EMOCA_v2_lr_mse_20` (detail), cuDNN bật, RTX 3050 4 GB. Nạp mô hình 21,45 s. Log `~/emoca_wsl/out/run_server_e2eweb_20260925_234154.log` |
| Web | venv `ma3d-web`: `python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8601 --server.headless true` (web thứ hai; cổng 8599 không đụng) |
| Biến môi trường | `EMOCA_MODE=real`, `EMOCA_URL=http://127.0.0.1:8765`, `ENGINE_MODE=real`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=2` |
| `backend/configs/services.yaml` | **không đổi** (`emoca_svc.mode: mock` bị biến môi trường ghi đè) |
| Trình duyệt | Chrome có sẵn, headless, Playwright (venv), khung nhìn 1440×1500. URL `http://127.0.0.1:8601/?m=light&c=normal&t=cold` |
| Ảnh thử | `EMOCA_test_example_data/images/affectnet_test_examples/0000_0000_00.png` (224×224, 65 KB) — ảnh **công khai** trong bộ dữ liệu mẫu của EMOCA. Không chép ảnh gốc vào repo; chỉ lưu ảnh chụp giao diện |

## 2. Kết quả

| # | Tiêu chí | KQ | Bằng chứng |
|---|---|---|---|
| 1 | EMOCA thật, không có nhãn "giả lập" | **ĐẠT** | Chip nguồn: `EMOCA: cục bộ` · `POSTER/MA3D-Net: thật (CPU)`. Không dòng nào chứa "giả lập"/"mock" trên trang. Ghi chú trung thực: `emoca_real: EMOCA_v2_lr_mse_20 (detail), upstream e0be0db+patches` và `engine_real: ma3d_rafdb, poster_rafdb chạy thật trên CPU trong tiến trình web` |
| 2 | POSTER + MA3D-Net cùng `ok`; MA3D-Net không còn `codes_missing` | **ĐẠT** | Không có "Mô hình lỗi", không có `codes_missing`. POSTER: Buồn 73,1 %; MA3D-Net: Buồn 64,8 %. Hai phân phối **khác nhau** (vd. Ngạc nhiên 5,6 % vs 17,3 %) ⇒ nhánh 3DMM có tác động. Suy luận: POSTER 745 ms, MA3D-Net 614 ms (lần đầu) |
| 3 | Landmark + ảnh dựng thật; có feature_maps | **ĐẠT** (xem bằng mắt) | "Ảnh vào + 68 landmark (original_px)" — điểm nằm trên mắt/mũi/miệng, có khung mặt. "3DMM coarse (EMOCA)" và "3DMM detail (EMOCA)" là ảnh dựng thật (dáng môi chu khớp ảnh vào), không phải placeholder. Feature map 7×7: POSTER = Nhánh ảnh, Nhánh landmark, Attention; MA3D-Net = thêm **Tác động của 3DMM** (`fusion_delta`) |
| 4 | Thời gian lần đầu/lần sau; timeout đủ cho lần đầu | **ĐẠT, biên hẹp** | Bảng mục 3 |

Server EMOCA sau 3 yêu cầu từ web (`/v1/health`):
- `requests_served` 3, `cudnn_enabled` true, `cudnn_fallbacks` 0.
- `vram_peak_mb` 1677, `vram_reserved_peak_mb` 1826.
- `landmark_norm_range_seen` [-0,700; 0,802].
- Không có lỗi JavaScript (`pageerror` = 0).

## 3. Thời gian (một lần mỗi dòng)

| Lượt | Trình duyệt (s) | Chip server: emoca / emoca_roundtrip / engine_total / total (ms) |
|---|---|---|
| Tải ảnh — lần đầu (web nguội + EMOCA vừa nạp) | **46,4** | 26 594 / **26 919** / 15 627 / 42 583 |
| "Phân tích lại" 1 | 14,4 | 12 194 / 12 273 / 1 259 / 13 542 |
| "Phân tích lại" 2 | 2,5 | 436 / 463 / 1 233 / 1 708 |

**Timeout:**
- Web chờ EMOCA **60 s**: `backend/configs/services.yaml:16` (`timeout_s: 60`) → `backend/web/orchestrator.py:153`. Mặc định dự phòng: `backend/web/orchestrator.py:52` (`_DEFAULT_EMOCA_TIMEOUT_S = 60.0`), `backend/web/emoca_client.py:165` (`_DEFAULT_TIMEOUT_S = 60.0`), dùng ở `backend/web/emoca_client.py:217`.
- Lần đầu khứ hồi EMOCA 26,9 s ⇒ **còn biên ~33 s**, đủ. Biên không lớn nếu máy chậm hơn.

**Nhận xét:**
- **EMOCA chậm 2 lần liên tiếp, lần 3 mới ấm.** Hai lần gọi đầu từ web chậm (26,6 s rồi 12,2 s), lần 3 còn 0,44 s. Khi chạy `another/tests/e2e_wsl_emoca.py` trên server trước (23:39), chỉ lần 1 chậm (11,3 s). Nguyên nhân **chưa đo**. Giả thuyết: cuDNN tìm thuật toán/khởi động CUDA cho kích thước tensor mới ở bước detect (FAN).
- **`engine_total` lần đầu 15,6 s** gồm nạp POSTER + MA3D-Net. Số này cao hơn 7,4 s ở `documents/docs/qa/E2E_ENGINE_REAL_UI.md`, có thể do RAM trống thấp (0,71 GB) ⇒ phân trang. Không đo tách.

## 4. Tệp

Thư mục `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25/`:
- `real_emoca_top_1440.png`: đầu trang — chip nguồn, landmark, ảnh dựng coarse/detail, đầu phần so sánh.
- `real_emoca_first_1440.png`: toàn trang (khung nhìn 1440×5200) — xác suất, feature map đã mở, radar, ghi chú trung thực.
- `report.json`: đầu ra thô của script (thời gian, chip, trích văn bản trang). **Lưu ý đọc JSON:** các trường `toggle_landmark`, `toggle_att`, `feature_map_imgs` = 0 và `warnings_lines` rỗng là do **selector của script sai**, KHÔNG phải trang thiếu. Landmark và feature map có hiện theo mặc định, kiểm bằng hai ảnh chụp; dòng cảnh báo có tiền tố "Cảnh báo:" nên script không bắt được.

Script đo nằm trong scratch của phiên, không đưa vào repo. Sau khi đo, web 8601 và server EMOCA đã tắt (cổng 8601, 8765 trống).

## 5. Chưa đo

- p50/p95 (mỗi lượt đo một lần).
- Nguyên nhân hai lần EMOCA chậm đầu tiên.
- Ảnh nhiều mặt / ảnh lớn: VRAM trống thấp nhất đo trong smoke là 116 MB.
- Hành vi khi server EMOCA tắt giữa lúc phân tích (M4-05 "Kèm").
