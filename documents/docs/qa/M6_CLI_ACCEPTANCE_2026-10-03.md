# Nghiệm thu M6-01 / M6-03 — 2026-10-03

## Môi trường và lệnh

Windows, Python 3.13.5 từ `C:/Users/admin/venvs/ma3d-backend/web/Scripts/python.exe`,
PyTorch 2.14.0+cpu, Spandrel 0.4.1, Ultralytics 8.4.166, Pillow 12.3.0,
NumPy 2.5.3. `PYTHONPATH=.` và `PYTHONIOENCODING=utf-8`. CUDA không khả dụng;
`--device auto` chọn CPU.

- Inline Python smoke: mỗi ảnh chạy trong process riêng; đo khởi tạo
  `SpandrelRestorer`, `restore()` và Windows `PeakWorkingSetSize`.
- `python -m services.image_preprocessing.cli database/image --output out_m6_cli --device auto --confidence 0.5 --padding 0.15 --max-side 4096`
- `python another/tests/run_all.py --only test_m6_pipeline.py` — **1 PASS**, 0,9 s.
- `python another/tests/run_all.py --heavy --only test_m6_real_integration.py` — **1 PASS**,
  28,4 s; kiểm tra CLI thật đạt. So sánh paired 21 ảnh được SKIP vì bản sao này
  không có RAF-DB archive.

## M6-01 — smoke x4v3 thật

SHA-256 tính lại từ checkpoint là
`8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`;
trùng giá trị trong roadmap và tài liệu M6. Kích thước checkpoint: 4.885.111
byte. Spandrel load thành công, scale **4×**, device **CPU**.

| Ảnh | Kích thước | Load | Inference | Peak working set |
|---|---:|---:|---:|---:|
| `afraid.jpg` | 606×404 | 6,403 s | 4,803 s | 494,15 MiB |
| `angry.jpg` | 525×350 | 4,675 s | 2,336 s | 489,36 MiB |
| `base64-…-crop-…webp` | 1200×630 | 4,217 s | 10,709 s | 1085,16 MiB |
| `cong_dong_la_gi_5_c0a7fabc23.jpg` | 800×586 | 6,006 s | 7,052 s | 787,09 MiB |

Peak working set là đỉnh tiến trình riêng cho từng hàng sau load và inference.
Load được đo trong process mới cho từng ảnh; thời gian có thể dao động theo tải
CPU.

**Verdict M6-01: PASS.** Checkpoint khớp hash đã ghi; x4v3 load và inference
thành công trên đủ 4 ảnh, có số đo thiết bị, scale, thời gian và bộ nhớ.

## M6-03 — CLI đầu-cuối

| Chỉ số | Kết quả |
|---|---:|
| Ảnh đầu vào / thành công / lỗi | 4 / 4 / 0 |
| Enhanced image / metadata JSON | 4 / 4 |
| Mặt phát hiện / crop được lưu | 21 / 21 |
| Tổng thời gian CLI | 61,172 s |
| Tổng thời gian xử lý metadata / trung bình mỗi ảnh | 52,206 s / 13,052 s |
| Tổng x4v3 / YOLO theo stage | 49,051 s / 2,187 s |

| Ảnh | Mặt/crop | Thời gian xử lý |
|---|---:|---:|
| `afraid.jpg` | 1 | 6,579 ms |
| `angry.jpg` | 1 | 5,545 ms |
| `base64-…-crop-…webp` | 13 | 21,403 ms |
| `cong_dong_la_gi_5_c0a7fabc23.jpg` | 6 | 18,680 ms |

Kiểm tra đủ 4 enhanced image, 4 metadata, cấu hình detector `confidence=0.5`,
`padding=0.15`, `max_side=4096`, CPU, stage timings và hash đúng của cả hai
checkpoint. Mỗi bbox/crop nằm trong ảnh; cả 21 crop so từng pixel đều bằng vùng
ảnh gốc tại `crop_bbox_xyxy`. Output RGB, giữ kích thước nguồn (không upscale),
không có khóa CodeFormer trong metadata. Mỗi ảnh mẫu đều có mặt; hành vi no-face
được kiểm bằng test pipeline giả hiện có, chưa có ảnh no-face thật trong bộ này.

**Verdict M6-03: PASS cho CLI, batch, crop và metadata.** A/B chất lượng ảnh và
benchmark paired 21 ảnh: **NOT MEASURABLE** trong bản sao này (thiếu RAF-DB cho
benchmark; chưa có review chất lượng A/B). Chưa dùng kết quả làm đầu vào nhận
dạng/huấn luyện trước khi review A/B.

## Quyết định CLI và việc còn mở

CLI tiếp tục luôn chạy x4v3: chức năng offline của nó là xuất ảnh đã tăng cường
cùng crop và metadata cho từng đầu vào, nên enhanced image là artifact bắt buộc.
YOLO chạy trên ảnh đã tăng cường; crop vẫn lấy pixel từ ảnh gốc. Luồng app theo
ADR-028 riêng: YOLO thử ảnh gốc trước, chỉ dùng x4v3→YOLO khi không thấy mặt.

- Benchmark paired 21 ảnh không đo được vì không có `database/image/archive/` trong
  bản sao này.
- Chưa có đánh giá A/B chất lượng rộng trên ảnh cho tác vụ nhận dạng/huấn luyện.
- Kết quả chỉ đo CPU; GPU chưa đo trong venv CPU-only.
- Đã xóa `out_m6_cli/` sau khi kiểm tra; xác nhận thư mục không còn.
