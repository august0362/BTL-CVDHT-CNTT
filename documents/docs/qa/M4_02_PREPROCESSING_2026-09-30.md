# M4-02 — Đo ảnh hưởng tiền xử lý trên RAF-DB

Ngày đo: 2026-09-30  
Verdict: **ĐẠT AC đo paired và ghi tác động lên xác suất**

## Phạm vi và phương pháp

- Lấy mẫu phân tầng ngẫu nhiên, seed `20260930`: 6 ảnh cho mỗi lớp trong RAF-DB test (`database/image/archive/DATASET/test`, nhãn `test_labels.csv`), tổng 42 ảnh. Đủ cả 7 lớp, không lưu ảnh hay mã 3DMM vào báo cáo.
- Chạy dịch vụ EMOCA thật trong WSL cục bộ (model `EMOCA_v2_lr_mse_20`, detail, contract `0.3.0`). Mỗi ảnh được phân tích một lần; cùng mã 3DMM thật được tái sử dụng cho mọi biến thể và cả hai model để cô lập ảnh hưởng của ảnh đầu vào.
- So ba biến thể trên cùng ảnh: (A) crop224 trả về bởi EMOCA; (B) ảnh RAF-DB căn chỉnh sẵn 100×100; (C) `model_input` 100×100 được tạo theo tiền xử lý backend/web/ADR-020 từ crop224 và landmark EMOCA. Engine áp dụng đúng đường inference hiện hành cho POSTER và MA3D-Net RAF-DB.
- Không có lỗi EMOCA hay mẫu bị loại: 42/42 cặp hợp lệ. Inference chạy trên CPU; tổng thời gian phép đo khoảng 173 giây.

## Kết quả

`top-1 flip` là số ảnh mà nhãn dự đoán thay đổi giữa hai đầu vào. `Δp` là trị tuyệt đối chênh lệch xác suất theo từng lớp; các thống kê gộp trên 42 ảnh × 7 lớp.

| Model | So sánh | Flip top-1 | Accuracy A → B | Mean Δp | P95 Δp | Max Δp |
|---|---|---:|---:|---:|---:|---:|
| POSTER | crop224 → RAF aligned 100 | 13/42 (30.95%) | 64.29% → 92.86% | 0.0747 | 0.6036 | 0.8657 |
| POSTER | crop224 → web `model_input` 100 | 15/42 (35.71%) | 64.29% → 85.71% | 0.0828 | 0.6264 | 0.8663 |
| POSTER | RAF aligned 100 → web `model_input` 100 | 3/42 (7.14%) | 92.86% → 85.71% | 0.0165 | 0.0336 | 0.6179 |
| MA3D-Net | crop224 → RAF aligned 100 | 11/42 (26.19%) | 64.29% → 85.71% | 0.0867 | 0.4088 | 0.8192 |
| MA3D-Net | crop224 → web `model_input` 100 | 13/42 (30.95%) | 64.29% → 88.10% | 0.0881 | 0.4047 | 0.8172 |
| MA3D-Net | RAF aligned 100 → web `model_input` 100 | 4/42 (9.52%) | 85.71% → 88.10% | 0.0163 | 0.0577 | 0.4806 |

## Kết luận và giới hạn

1. Thay crop224 EMOCA bằng ảnh aligned theo miền RAF làm đổi top-1 ở 26–31% mẫu; chênh lệch xác suất trung bình theo lớp khoảng 0.075–0.087. Tiền xử lý ảnh có ảnh hưởng lớn đến dự đoán ở tập paired này.
2. Đầu vào web `model_input` hiện tại gần ảnh RAF aligned hơn crop224: so với ảnh aligned, top-1 flip là 7.14% cho POSTER và 9.52% cho MA3D-Net. Tuy vậy hai đầu vào không tương đương hoàn toàn, và không thể suy rộng rằng căn chỉnh hiện tại luôn cải thiện dự đoán.
3. Accuracy là trên tập con cân bằng 42 ảnh, chỉ nhằm đo độ nhạy với tiền xử lý; không phải kết quả benchmark RAF-DB test đầy đủ hay tái lập con số trong bài báo. Không thay đổi code production hoặc ngưỡng dự đoán trong phép đo này.

## Tái lập

Seed `20260930`; lấy 6 ảnh/lớp từ archive; EMOCA thật qua endpoint loopback; engine `web.engine.predict`, mode `real`, `codes_real=True`, hai model `poster_rafdb` và `ma3d_rafdb`. Checkpoint được engine nạp strict. So sánh paired dùng chung kết quả EMOCA cho từng ảnh.
