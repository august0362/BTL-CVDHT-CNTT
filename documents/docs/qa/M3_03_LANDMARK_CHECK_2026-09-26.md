# M3-03 — So landmark EMOCA với FAN 2D (5 điểm chuẩn) — 2026-09-26

- **Người đo:** phiên 01_AI, giao việc từ điều phối `btl-e6`. **Ngưỡng do GENERAL đặt:** mọi điểm ≤ **5 % IOD** (khoảng cách hai tâm mắt, tính trên FAN).
- **Kết luận: KHÔNG ĐẠT** theo ngưỡng này — 12/25 điểm đạt, sai lệch lớn nhất 8,3–14,5 % IOD. Chi tiết và cách hiểu ở mục 3.
- Không sửa mã. Ảnh không vào repo. Số thô: `~/emoca_wsl/out/m3_03_landmarks.json` (WSL). Script tạm `~/emoca_wsl/m3_03.py` (ngoài dự án).

## 1. Cách đo

| Mục | Giá trị |
|---|---|
| EMOCA | server thật trong WSL2 (`run_emoca_server.sh`, `EMOCA_v2_lr_mse_20` detail, RTX 3050). Lấy `faces[0].landmarks.original_px` từ `POST /v1/analyze`. Đây là landmark **EMOCA dự đoán** (FLAME chiếu lên ảnh, `predicted_landmarks`), đổi về px ảnh gốc qua `crop_transform` |
| FAN 2D | `face_alignment` 1.3.6 trong env `work38`, `LandmarksType.TWO_D`, **CPU**, gọi độc lập trên cùng ảnh; chọn mặt gần tâm landmark EMOCA nhất (ảnh nào cũng chỉ có 1 mặt) |
| Ảnh | 5 ảnh công khai đầu tiên của `EMOCA_test_example_data/images/affectnet_test_examples/` (`0000_0000_00` … `0000_0004_00`), 224×224 px |
| Điểm (chỉ số 0-based, chuẩn 68 điểm) | tâm mắt 36–41 (mắt bên trái ảnh), tâm mắt 42–47, đỉnh mũi 30, khoé miệng 48, khoé miệng 54 |
| Sai lệch | khoảng cách Euclid (px) giữa điểm EMOCA và FAN; % IOD = px / IOD<sub>FAN</sub> × 100 |

## 2. Kết quả (px / % IOD; **đậm** = vượt 5 %)

| Ảnh | IOD (px) | Mắt 36–41 | Mắt 42–47 | Mũi 30 | Miệng 48 | Miệng 54 | Max % | TB 68 điểm % |
|---|---|---|---|---|---|---|---|---|
| 0000_0000_00 | 75,06 | 2,06 / 2,75 | 5,53 / **7,37** | 4,08 / **5,44** | 6,25 / **8,33** | 4,29 / **5,71** | 8,33 | 6,89 |
| 0000_0001_00 | 69,46 | 1,02 / 1,47 | 1,14 / 1,65 | 6,12 / **8,81** | 5,19 / **7,47** | 3,43 / 4,94 | 8,81 | 6,45 |
| 0000_0002_00 | 73,14 | 7,76 / **10,61** | 5,32 / **7,27** | 3,21 / 4,39 | 3,79 / **5,19** | 2,05 / 2,80 | 10,61 | 8,39 |
| 0000_0003_00 | 68,67 | 0,92 / 1,33 | 3,44 / **5,01** | 9,93 / **14,46** | 4,08 / **5,95** | 2,81 / 4,09 | 14,46 | 8,14 |
| 0000_0004_00 | 75,83 | 2,94 / 3,88 | 0,43 / 0,57 | 6,38 / **8,42** | 3,63 / 4,79 | 2,27 / 3,00 | 8,42 | 5,16 |
| **Đạt ≤ 5 %** | | 4/5 | 2/5 | 1/5 | 1/5 | 4/5 | **12/25** | |

Độ lệch có dấu trung bình trên 5 ảnh (EMOCA − FAN, px; y hướng xuống):

| Điểm | dx | dy |
|---|---|---|
| Mắt 36–41 | −0,53 | +2,40 |
| Mắt 42–47 | +0,31 | +1,99 |
| Mũi 30 | −0,64 | **−5,74** (5/5 ảnh đều âm) |
| Miệng 48 | **+2,78** | +2,08 |
| Miệng 54 | **−2,24** | +0,11 |

## 3. Cách hiểu (để GENERAL quyết)

1. **Không phải lỗi phép đổi toạ độ.** Nếu `crop_transform` hay hệ quy chiếu sai, mọi điểm sẽ lệch cùng hướng hoặc cùng tỉ lệ. Ở đây độ lệch **tuỳ từng điểm và nhất quán theo điểm**:
   - mũi của EMOCA luôn cao hơn FAN khoảng 5,7 px;
   - hai khoé miệng của EMOCA luôn nằm vào trong (48 lệch sang phải, 54 lệch sang trái);
   - hai tâm mắt gần như trùng theo x.
   
   Hai hệ `crop224` ↔ `original_px` của EMOCA khớp nhau tới 0,001 px (`another/tests/e2e_wsl_emoca.py`, 2026-09-25).
2. **Nhiều khả năng là khác định nghĩa điểm.** Landmark EMOCA là 68 điểm **gắn trên lưới FLAME** (landmark embedding) được mô hình 3D khớp rồi chiếu lên ảnh. FAN là bộ **dò 2D** huấn luyện trực tiếp trên nhãn 68 điểm. Đỉnh mũi và khoé miệng là nơi hai định nghĩa hay khác nhau. Điều này **chưa được kiểm** bằng tài liệu FLAME/EMOCA — chỉ là giả thuyết khớp với dấu độ lệch.
3. **Ngưỡng 5 % IOD** hợp với việc so hai bộ dò 2D. Với landmark từ mô hình dựng 3D, sai lệch trung bình 68 điểm đo được ở đây là 5,2–8,4 % IOD. Các lựa chọn:
   - (a) giữ 5 % ⇒ M3-03 **không đạt**;
   - (b) đặt ngưỡng riêng cho landmark dựng 3D, vd. từng điểm ≤ 15 % và TB 68 điểm ≤ 10 % IOD ⇒ **đạt** với 5 ảnh này (max 14,46 %, TB max 8,39 %);
   - (c) nếu mục tiêu của M3-03 là "overlay đúng chỗ trên ảnh gốc", so thêm với landmark FAN mà chính server dùng để cắt ảnh (`detect`, chưa trả qua contract).
   
   Nhìn bằng mắt, overlay 68 điểm trên ảnh `0000_0000_00` nằm đúng mắt/mũi/miệng (`documents/docs/qa/E2E_REAL_EMOCA_2026-09-25/real_emoca_top_1440.png`).

## 4. Chưa đo

- Ảnh lớn / nghiêng mạnh / nhiều mặt (5 ảnh đều 224×224, một mặt, gần chính diện).
- So với nhãn tay (ground truth) — không có trong bộ ảnh mẫu.
