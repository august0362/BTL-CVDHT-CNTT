# QA — Kết quả Kaggle M8: chỉnh thẳng 3DDFA-V2 và RealESRGAN→YOLO trên ảnh xấu — 2026-10-03

**Cập nhật 2026-10-04:** §1–2 ghi lại kết quả và kết luận tại thời điểm benchmark ban đầu; §3 bổ sung sweep roll-only hoàn tất và ngưỡng được chọn. Trạng thái baseline hiện hành theo ADR-029.

- **Nguồn:** `another/kaggle/notebook_frontalize/run_rafdb_frontalize.ipynb`, chạy trên Kaggle với 2 GPU T4.
- **Dữ liệu thô:** `database/results/M8_KAGGLE/`: `metrics.csv`, `summary.json`, `predictions.csv`/`scenes.csv`, `shard_*.jsonl`.
- **Mã Dataset Kaggle:** bản chụp ngày 30/09, nên `analyze()` cũ là single-face mặc định.
- **Không lỗi chạy:** 0 lỗi worker. Mọi "lỗi" trong bảng là EMOCA/FAN `no_face`, được tính là dự đoán sai trong Acc(all).

## 1. Thí nghiệm 1 — 3DDFA-V2 frontalize trên RAF-DB test (3068 ảnh × 3 góc xoay roll)

Nhánh `raw`: ảnh (đã xoay) → EMOCA → mô hình. Nhánh `frontal`: ảnh (đã xoay) → 3DDFA-V2 → EMOCA → mô hình. Không có RealESRGAN/YOLO/CodeFormer.

| Góc | Mô hình | Coverage raw → frontal | **Acc(all)** raw → frontal | Macro F1 raw → frontal | Sửa đúng / làm sai | McNemar p |
|---|---|---|---|---|---|---|
| 0° | POSTER | 99,3 → 99,0 % | 88,0 → **86,2 %** (−1,8) | 83,0 → 79,6 | 68 / 122 | 1,1e-4 |
| 0° | MA3D-Net | 99,3 → 99,0 % | 88,6 → **86,8 %** (−1,8) | 81,6 → 79,3 | 59 / 113 | 4,7e-5 |
| 45° | POSTER | 94,7 → 97,1 % | 79,2 → **81,4 %** (+2,2) | 74,0 → 75,5 | 245 / 178 | 1,3e-3 |
| 45° | MA3D-Net | 94,7 → 97,1 % | 80,6 → **81,8 %** (+1,2) | 74,7 → 74,7 | 218 / 179 | 0,056 |
| 90° | POSTER | 83,0 → 95,6 % | 24,8 → **79,7 %** (+54,9) | 11,2 → 74,3 | 1764 / 79 | < 1e-300 |
| 90° | MA3D-Net | 83,0 → 95,6 % | 18,0 → **80,4 %** (+62,4) | 13,1 → 74,6 | 1985 / 71 | < 1e-300 |

**Góc tự nhiên (0°), chia theo |yaw| do 3DDFA ước lượng:**

| Nhóm \|yaw\| | n | POSTER Acc(all) raw → frontal | MA3D Acc(all) raw → frontal |
|---|---|---|---|
| < 15° | 2719 | 89,0 → 87,9 | 89,7 → 88,6 |
| 15–30° | 328 | 81,1 → **75,6** | 79,9 → **75,3** |
| ≥ 30° | 21 | 61,9 → **38,1** | 71,4 → **38,1** |

Chia theo |pitch| cũng ra cùng chiều: nhóm ≥ 30° (n = 31) giảm từ 71 % xuống khoảng 48–52 %.

**Kiểm tra bộ ước lượng pose:**
- Roll ước lượng có trung vị 43,1° với ảnh xoay 45° và 92,5° với ảnh xoay 90°, khớp với góc xoay thật.
- Trên ảnh RAF-DB 0°, phần lớn ảnh có |roll| < 5°. Nhóm 3DDFA ước |roll| 10–20° (n = 25) nhiều khả năng là ước sai, vì ảnh RAF-DB đã căn thẳng; nhóm này mất 32 điểm khi chỉnh thẳng.

**Thời gian (trung vị, Kaggle T4):** `analyze` (EMOCA + POSTER + MA3D) khoảng 0,55 s; frontalize khoảng 0,29 s.

**Trần tham chiếu:** sau khi chỉnh ảnh xoay 90°, độ chính xác đạt khoảng 80 %, vẫn thấp hơn ảnh gốc đứng thẳng (88 %) khoảng 8 điểm. Đây là phần mất đi do bước warp 3D.

### Kết luận thí nghiệm 1
1. **Lợi ích chỉ đến từ việc sửa roll (xoay trong mặt phẳng ảnh).** Mặt bị xoay 90° được cứu từ khoảng 20 % lên khoảng 80 %. Phần lớn lợi ích nằm ở coverage, vì FAN của EMOCA tìm lại được mặt.
2. **Với mặt quay ngang (yaw) hoặc ngửa/cúi (pitch) tự nhiên, chỉnh thẳng 3D làm hại**, và càng hại ở nhóm góc lớn: −5 điểm ở yaw 15–30°, −24 đến −33 điểm ở yaw ≥ 30° (n nhỏ). Ảnh warp tạo vết và làm méo các chi tiết biểu cảm. Kết quả này bác bỏ giả thuyết "mặt nghiêng → chỉnh 3D thẳng → dự đoán tốt hơn" trên RAF-DB.
3. **"Luôn chạy chỉnh thẳng" không nên dùng.** Ở 0° nó làm giảm 1,8 điểm, và ý nghĩa thống kê rõ.
4. **Đề xuất ban đầu, đã được đo ở §3:** dùng 3DDFA chỉ để ước lượng roll rồi xoay ảnh 2D ngược lại khi vượt ngưỡng, không warp 3D. Kaggle roll-only sweep trên các góc 0/15/30/45/60/90° chọn ngưỡng 15° cho cả hai mô hình.
5. **M8-02 (Hassner)** sau đó bị huỷ theo ADR-028.

## 2. Thí nghiệm 2 — RealESRGAN có giúp YOLO bắt mặt trên ảnh xấu không

- **Dữ liệu:** 1000 ảnh RAF-DB (lớp fear lấy đủ 74), dán vào nền tổng hợp 640×480, cỡ mặt 24/48/96 px, 6 kiểu làm xấu, tổng 18.000 cảnh, 0 lỗi.
- **Cách tính:** recall với IoU ≥ 0,4, ngưỡng conf 0,5 của app.

| Kiểu ảnh | YOLO trên ảnh gốc | RealESRGAN → YOLO | **YOLO ảnh gốc, 0 mặt thì RealESRGAN** (mô phỏng) | % cảnh phải chạy RealESRGAN | McNemar (gốc vs ESRGAN) |
|---|---|---|---|---|---|
| clean | 56,1 % | 51,5 % | 56,7 % | 43,6 % | p = 2,6e-29 (ESRGAN kém hơn) |
| blur σ2 | 34,3 % | 28,4 % | 36,3 % | 65,4 % | p = 1,9e-26 (kém hơn) |
| down ×4 | 31,7 % | 24,9 % | 33,5 % | 68,1 % | p = 1,4e-32 (kém hơn) |
| noise σ20 | 62,1 % | 52,0 % | 69,1 % | 37,3 % | p = 1,2e-29 (kém hơn) |
| combo | 35,3 % | 34,9 % | 41,6 % | 64,4 % | p = 0,58 |
| **JPEG q10** | 20,7 % | **37,7 %** | **40,0 %** | 79,3 % | p = 8,5e-101 (**tốt hơn**) |
| **Tổng** | 40,1 % | 38,2 % | **46,2 %** | 59,7 % | — |

- **JPEG q10 theo cỡ mặt:** 48 px 23,0 → 34,9 % (rule fallback 40,2 %); 96 px 39,0 → 78,1 % (fallback 79,5 %).
- **Số khung nhầm** rất thấp ở mọi nhánh (≤ 0,012 mỗi cảnh). IoU của các lần bắt đúng 0,83–0,90.
- **Thời gian:** RealESRGAN khoảng 443 ms mỗi ảnh 640×480 trên T4; YOLO khoảng 12 ms.

### Kết luận thí nghiệm 2
1. **Luôn chạy RealESRGAN trước YOLO làm hại** ở hầu hết các kiểu ảnh (−4,6 đến −10,1 điểm recall). Chỉ có lợi với ảnh **nén JPEG nặng**.
2. **Rule tốt nhất trong mô phỏng:** chạy YOLO trên **ảnh gốc** trước; chỉ khi **không thấy mặt nào** mới chạy RealESRGAN → YOLO. Rule này không bao giờ kém YOLO ảnh gốc (vì chỉ đổi kết quả khi YOLO ra 0 mặt) và thắng ở mọi kiểu ảnh: tổng 46,2 % so với 40,1 %.
3. **Tại thời điểm benchmark, code sau M6-04 chạy RealESRGAN trước.** Kết quả này dẫn tới ADR-028: YOLO chạy trên ảnh gốc trước, chỉ dùng RealESRGAN khi không thấy mặt.
4. **Giới hạn:** mặt RAF-DB đã căn sẵn được dán lên nền tổng hợp, không phải ảnh chụp tự nhiên. Ngay cả ảnh sạch 96 px, YOLO cũng chỉ bắt 89 %, cho thấy có lệch miền. Mặt 24 px gần như không bao giờ bắt được ở mọi nhánh. Với ảnh thật, YOLO trên ảnh gốc thường đã thấy mặt, nên RealESRGAN hiếm khi phải chạy và tổng thời gian giảm.

## 3. Thí nghiệm 3 — Kaggle roll-only threshold sweep (ADR-028)

- **Nguồn:** `another/kaggle/notebook_roll/run_rafdb_roll_sweep.ipynb`; dữ liệu đã giải nén tại `database/results/M8_ROLL/roll_sweep/` từ gói tải ngày 2026-10-03 (SHA-256 `6D6BBDFA3F624605F6E3E130B04C68E2DA48A89C3C9F3F84A9815FB3675BAB0F`). Gói zip được dọn sau khi giải nén (Session 44).
- **Độ đầy đủ:** hai shard có 9.204 dòng mỗi shard và đều báo tiến độ 100%; tổng 18.408 cặp ảnh–góc duy nhất. Mỗi góc 0/15/30/45/60/90° có 3.068 ảnh. `summary.json` ghi nhận cùng số dòng; lỗi dự đoán được tính là sai trong Acc(all).
- **Chọn ngưỡng:** so raw với policy chỉ xoay 2D khi `|estimated roll| ≥ T`; trong các ngưỡng thỏa mức giảm Acc(all) ở ảnh 0° không quá 0,2 điểm phần trăm, bộ tổng hợp chọn ngưỡng có mean Acc(all) cao nhất. Cả hai mô hình được đề xuất `T=15`. Theo ADR-029, giá trị này hiện được bật trong baseline tại `backend/configs/frontalization.yaml`.

| Mô hình | Mean coverage (6 góc) | Mean Acc(all) | Mean macro F1 | Acc(all) tại 0°: raw → T=15 | Giảm tại 0° |
|---|---:|---:|---:|---:|---:|
| POSTER | 97,441 % | 86,386 % | 81,934 % | 87,973 → 87,907 % | 0,065 điểm % |
| MA3D-Net | 97,441 % | 86,897 % | 80,833 % | 88,559 → 88,462 % | 0,098 điểm % |

| Góc xoay | n | POSTER Acc(all), raw → T=15 | MA3D-Net Acc(all), raw → T=15 |
|---:|---:|---:|---:|
| 0° | 3.068 | 87,973 → 87,907 % | 88,559 → 88,462 % |
| 15° | 3.068 | 86,734 → 86,473 % | 87,321 → 87,027 % |
| 30° | 3.068 | 84,681 → 86,538 % | 85,202 → 87,484 % |
| 45° | 3.068 | 79,172 → 86,180 % | 80,574 → 86,701 % |
| 60° | 3.068 | 58,246 → 85,724 % | 61,213 → 86,050 % |
| 90° | 3.068 | 24,772 → 85,495 % | 18,057 → 85,658 % |

Mean là trung bình đều trên sáu góc, mỗi góc có cùng 3.068 ảnh. T=15 đạt giới hạn suy giảm tại 0°; góc 15° giảm nhẹ so với raw, trong khi cải thiện ở 30–90° làm tăng mean tổng thể. Ban đầu giá trị được dùng cho nhánh thí nghiệm; ADR-029 sau đó đưa sửa roll 2D 15° vào baseline.

**Giới hạn:** ảnh RAF-DB test đã căn chỉnh rồi được xoay nhân tạo theo góc; sweep chọn ngưỡng trong miền benchmark này, không chứng minh hiệu quả trên ảnh tự nhiên hoặc ảnh người dùng.
