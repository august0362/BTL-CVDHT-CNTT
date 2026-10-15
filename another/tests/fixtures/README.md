# tests/fixtures

## `emoca_analyze_real_sample.min.json` — fixture vàng RÚT GỌN của EMOCA thật

- **Là gì:** tập con của **một** phản hồi `POST /v1/analyze` thật của `services/emoca_svc` (EMOCA **thật**, không phải mock). Dùng cho M1-01 AC (2) và M0-04 AC (5).
- **Nguồn:**
  - Mô hình `EMOCA_v2_lr_mse_20` (mode `detail`), upstream `e0be0db+patches`, contract `0.2.0` lúc đo.
  - 2026-09-26 (ADR-019): trường `contract_version` trong fixture đổi thành `0.3.0` — **số đo lúc 0.2.0, nội dung không
    đổi** (0.3.0 chỉ thêm trường tuỳ chọn cho `web_analyze`; hình dạng `emoca_analyze` giữ nguyên).
  - Chạy trong WSL2 trên RTX 3050 4GB, ngày 2026-09-25, trong lần smoke test `smoke_20260925_233239`.
  - Phản hồi gốc: `~/emoca_wsl/out/analyze_sample_20260925_233239.json` (chỉ có trên máy chạy, không đưa vào repo). sha256 của tệp gốc ghi trong khối `_fixture` của fixture.
- **Ảnh đầu vào:** một ảnh 224×224 trong `EMOCA_test_example_data/images/affectnet_test_examples`, bộ ảnh mẫu công khai đi kèm EMOCA. **Không kèm ảnh.**
- **Đã bỏ** `faces[].renders` (4 ảnh dựng) và `faces[].crop_image`, vì:
  - ảnh dựng sinh từ mô hình FLAME có giấy phép;
  - ảnh gốc AffectNet không được phát lại (`CLAUDE.md §1.1`, ADR-009).
- **Giữ lại:** `contract_version`, `service`, `model`, `image`, `timing_ms`, và trong mỗi mặt: `face_index`, `bbox_original_px`, `crop_transform`, `landmarks` (cả `crop224` và `original_px`), `codes`, `warnings`.
- **Hệ quả:** contract bắt buộc `renders`, nên fixture **KHÔNG qua schema đầy đủ** `contracts/emoca_analyze.schema.json`.
  - `tests/test_fixture_emoca_real.py` kiểm bằng **schema con**: suy ra từ chính contract, chỉ bỏ `renders` khỏi `required` của face; không sửa contract.
  - Test cũng kiểm thêm:
    - `contract_version` 0.3.0 (xem ghi chú ADR-019 ở trên);
    - codes dài 100/50/50/6/128;
    - 68 landmark ở cả hai hệ;
    - `crop_transform · original_px ≈ crop224` sai lệch ≤ 1 px.
  - Test chạy trong nhóm nhẹ của `tests/run_all.py`.
- **Không dùng để:** đánh giá độ chính xác, hay thay dữ liệu cho giao diện (không có ảnh). Đây chỉ là **một** ảnh, minh hoạ định tính (`CLAUDE.md §1.8`).
- **Muốn có fixture validate được schema đầy đủ:** cần ADR cho `renders` thành tuỳ chọn (tăng `contract_version`), hoặc cần ảnh dựng được phép phát lại. Phương án (C) trong trao đổi điều phối ngày 2026-09-26, chưa làm.
