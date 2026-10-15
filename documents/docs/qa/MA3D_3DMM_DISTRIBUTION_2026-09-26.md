# Chẩn đoán MA3D-Net: phân phối 3DMM và ảnh hưởng của nhánh 3D — 2026-09-26

- **Người đo:** phiên 01_AI, giao việc từ điều phối `btl-e6`. **Chỉ đọc/đo — không sửa mã.**
- **Câu hỏi:** người dùng thấy MA3D-Net dự đoán kém hơn POSTER trên web. Có phải do mã 3DMM trên web lệch với lúc train không?
- **Kết luận ngắn:**
  1. Có lệch **nhẹ** ở `pose` (ảnh AffectNet nghiêng đầu nhiều hơn RAF-DB đã căn chỉnh). Các nhóm còn lại nằm trong miền train.
  2. **Lệch 3DMM KHÔNG giải thích được khác biệt.** Ở checkpoint `raf_MA3D.pth`, nhánh 3D gần như không đổi dự đoán: thay x_3d thật bằng 0 ⇒ |Δp| ≤ 0,001; kể cả x_3d cực đoan (z = ±20) ⇒ |Δp| ≤ 0,004. Khác biệt POSTER ↔ MA3D-Net trên web đến từ **phần trọng số còn lại** của hai checkpoint (các khối FiLM + backbone được train riêng), không đến từ mã 3DMM.
  3. Lệch **lớn hơn** nằm ở **ảnh vào**: lúc train, cả hai mô hình nhận ảnh RAF-DB đã căn chỉnh (`<mẫu>.jpg`); trên web, cả hai nhận `crop_image` của EMOCA (khung FAN × 1,25, lỏng hơn). Cắt chặt lại làm đổi top-1 ở 1/5 ảnh, **cho cả hai mô hình**.
- 5 ảnh không có nhãn đúng ⇒ **không** đo được "mô hình nào đúng hơn". Đây là chẩn đoán cơ chế, không phải đánh giá độ chính xác (M2-07).

## 1. Cách đo

| Mục | Giá trị |
|---|---|
| Mã EMOCA | server thật đang chạy `127.0.0.1:8765` (của điều phối, không tắt), `EMOCA_v2_lr_mse_20` detail; `POST /v1/analyze` lấy `faces[0].codes` + `crop_image` |
| Ảnh | 5 ảnh công khai `EMOCA_test_example_data/images/affectnet_test_examples/0000_0000_00 … 0004_00` (224×224). Ảnh chỉ nằm trong scratch |
| Chuẩn hoá | `backend/model/ma3d/checkpoints/rafdb_emoca_stats.npz`, z = (x − mean)/std theo từng chiều — đúng như `backend/web/engine.py:329-338` và `backend/model/ma3d/Read_dataset/RAF_DB.py:76-77` |
| Mô hình | `backend/web/engine.py` `predict(mode="real", codes_real=True)`, venv `ma3d-web`, CPU |
| Số thô | `ma3d_diag.json` trong scratch của phiên (không vào repo) |

## 2. Phân phối z của mã EMOCA thật (5 ảnh gộp)

| Nhóm (số chiều) | TB \|z\| | % \|z\| > 2 | % \|z\| > 3 | max \|z\| | TB z (có dấu) | độ lệch chuẩn z |
|---|---|---|---|---|---|---|
| exp (50) | 1,12 | 16,4 | 5,6 | 4,90 | −0,15 | 1,46 |
| detail (128) | 1,12 | 15,8 | 3,6 | 4,03 | −0,02 | 1,40 |
| **pose (6)** | **1,62** | 13,3 | **10,0** | **9,92** | **−0,67** | **2,65** |
| shape (100) | 0,96 | 10,8 | 1,2 | 4,02 | +0,03 | 1,20 |
| tex (50) | 0,90 | 6,8 | 1,2 | 3,93 | −0,03 | 1,13 |

Tham chiếu: nếu đúng phân phối train (chuẩn) thì TB |z| ≈ 0,80, % > 2 ≈ 4,6, % > 3 ≈ 0,3.
- exp/detail/shape/tex rộng hơn một chút (độ lệch chuẩn z 1,1–1,5), không lệch tâm. Điều này bình thường với 5 ảnh AffectNet ngoài miền RAF-DB.
- **pose bất thường:** max |z| 9,92 (ảnh `0001`), TB có dấu −0,67. `pose_std` lúc train rất nhỏ (0,017–0,192), vì RAF-DB aligned đã xoay thẳng mặt. Ảnh AffectNet nghiêng thật nên ra ngoài miền.
- Ảnh `0002` có exp 20 % chiều > 3 — biểu cảm mạnh hơn trung bình RAF-DB.

## 3. Ablation — nhánh 3D đẩy dự đoán thế nào

| Ảnh | POSTER (top-1, p) | MA3D x_3d thật | MA3D x_3d = 0 (mã = mean) | \|Δp\| lớn nhất |
|---|---|---|---|---|
| 0000 | buồn 0,731 | buồn 0,648 (ngạc nhiên 0,173) | buồn 0,649 (ngạc nhiên 0,172) | 0,001 |
| 0001 | ngạc nhiên 0,742 | ngạc nhiên 0,719 | ngạc nhiên 0,719 | < 0,001 |
| 0002 | buồn 0,874 | buồn 0,864 | buồn 0,864 | 0,001 |
| 0003 | buồn 0,889 | buồn 0,888 | buồn 0,888 | 0 |
| 0004 | buồn 0,889 | buồn 0,888 | buồn 0,888 | 0 |

Độ nhạy với x_3d cực đoan (ảnh 0000, dấu ngẫu nhiên, seed 0):

| x_3d | \|Δp\| lớn nhất so với x_3d = 0 |
|---|---|
| z = ±1 mọi nhóm | 0,0006 |
| z = ±5 | 0,0024 |
| z = ±20 | 0,0039 |
| z = ±20 chỉ exp / detail / pose / shape / tex | 0,0034 / 0,0010 / 0,0017 / 0,0022 / 0,0033 |

- Δp tăng theo z ⇒ nhánh 3D **có được nối** (engine truyền đúng x_3d). Nhưng ảnh hưởng **rất nhỏ**.
- `ThreeDMMFusion` (`backend/model/ma3d/models/ThreeDMM_Adaptive.py:116-147`, 4,89 triệu tham số) không có tham số cổng. Nó sinh γ/β (FiLM) từ x_3d cho 5 khối `ThreeDMMAdaptiveBlock`. Các khối này có trọng số riêng ⇒ biến đổi đặc trưng ảnh gần như **không phụ thuộc** x_3d ở checkpoint này.
- Khớp với log train: RAF-DB val — MA3D-Net 90,67 % vs POSTER 90,44 % (`backend/model/ma3d/log/raf_MA3D.txt`, `raf_POSTER.txt`), chênh 0,23 điểm.

## 4. Đường train vs đường web

| Mục | Lúc train (đọc mã/README) | Web hiện tại | Khớp? |
|---|---|---|---|
| Ảnh vào mô hình | ảnh RAF-DB **đã căn chỉnh** `<mẫu>/<mẫu>.jpg` (`RAF_DB.py:49,85-86`; README: `test_0002_aligned.jpg`), val_transform Resize 224 | `crop_image` 224 của EMOCA (FAN bbox × 1,25, ADR-006) | **Không** — khung/căn chỉnh khác |
| Mô hình EMOCA trích 3DMM | **không ghi** trong repo (README chỉ nói "EMOCA"). Notebook `emoca_setup_v2_work38_ok.ipynb` chỉ có một lệnh chạy: video Chèo với `EMOCA_v2_lr_mse_20` (ô 42) | `EMOCA_v2_lr_mse_20` | chưa xác minh — cần hỏi nhóm train |
| Chế độ | detail (có `detail.npy` 128 chiều) | detail | khớp |
| Ảnh đưa vào EMOCA | nhiều khả năng là chính ảnh RAF-DB aligned (npy nằm cùng thư mục ảnh) — EMOCA tự dò FAN + crop × 1,25 trên ảnh đó. **Chưa xác minh** | ảnh người dùng tải lên | khác nguồn ảnh (RAF aligned 100×100 vs ảnh tuỳ ý) |
| Chuẩn hoá | `rafdb_emoca_stats.npz`, (x − mean)/std | như trên | khớp (người dùng đã xác nhận đúng file stats, 2026-09-25) |
| Thứ tự ghép x_3d | `prepare_batch` (`backend/model/ma3d/engine.py:27`): exp, detail, pose, shape, tex | `_X3D_ORDER` như vậy | khớp |

**Thử crop chặt** — cắt lõi 1/1,25 của `crop_image` rồi resize 224, gần khung ảnh aligned hơn (chỉ để thăm dò):

| Ảnh | POSTER crop web → chặt | MA3D crop web → chặt |
|---|---|---|
| 0000 | buồn 0,731 → **vui 0,834** | buồn 0,648 → **vui 0,638** |
| 0001 | ngạc nhiên 0,742 → 0,849 | ngạc nhiên 0,719 → 0,653 |
| 0002–0004 | không đổi top-1 | không đổi top-1 |

⇒ Khung ảnh ảnh hưởng tới **cả hai** mô hình, nhiều hơn hẳn mã 3DMM. Ảnh 0000 (môi chu) đổi nhãn. Không có nhãn đúng để nói khung nào đúng hơn.

## 5. Đề xuất (để GENERAL quyết — không tự sửa)

1. **Không** kỳ vọng MA3D-Net khác POSTER nhờ 3DMM với checkpoint này: nhánh 3D gần như bất hoạt (mục 3). Giao diện nên nói rõ điều này, tránh người dùng hiểu MA3D-Net "dùng 3D để đoán tốt hơn".
2. Kiểm lệch **ảnh vào** (ADR-006: dùng chung `crop_image` EMOCA cho mọi mô hình). Phương án: căn chỉnh kiểu RAF-DB (5 điểm, đồng dạng, 100×100 → 224) từ landmark có sẵn, rồi đo trên tập có nhãn (M2-07).
3. Hỏi nhóm train: mô hình EMOCA nào đã dùng để trích 3DMM RAF-DB; EMOCA chạy trên ảnh aligned 100×100 hay ảnh gốc.
4. Đánh giá thật cần RAF-DB test có nhãn (M2-07). 5 ảnh AffectNet không nhãn chỉ là minh hoạ định tính.

## 6. Bổ sung (ADR-020, 2026-09-26): chuỗi ảnh lúc train và 5 biến thể ảnh vào

### 6.1 Chuỗi lúc train — đọc mã (chỉ đọc)

| Bằng chứng | Nội dung |
|---|---|
| EMOCA ghi đầu ra | `gdl_apps/EMOCA/demos/test_emoca_on_images.py:75-85`: thư mục `<output>/<model>/<tên ảnh>/`, tên = stem của ảnh vào (`gdl/datasets/ImageTestDataset.py:74,158`). Mã: `shape/exp/tex/pose/detail.npy` (`io.py:62-74`). Crop 224 chỉ có khi `save_images` bật, dưới dạng `inputs.png` (`io.py:55`) |
| VKIST, Chèo | loader đọc **`inputs.png`** — tức crop 224 của EMOCA (`Read_dataset/VKIST_3087.py:40`, `DatasetCheo.py:47`) ⇒ khớp lời tác giả ("lấy ảnh 224×224 của EMOCA đẩy vào MA3D") |
| **RAF-DB** (checkpoint web đang dùng) | loader đọc **`<mẫu>/<mẫu>.jpg`** (`Read_dataset/RAF_DB.py:49,85-86`); README: `test_0002_aligned/test_0002_aligned.jpg` + `*.npy`. EMOCA **không** sinh ra tệp `.jpg` trùng tên thư mục ⇒ tệp này do nhóm tự chép vào. Tên trùng ảnh RAF-DB aligned ⇒ **nhiều khả năng là ảnh aligned 100×100 gốc**, nhưng cũng có thể là `inputs.png` đổi tên. Từ mã **không phân định được** |
| CAER-S, FERPlus, CheoFaMo | lấy ảnh đầu tiên trong thư mục (`Caers.py:49`, `FER_Plus.py:47`, `Cheo_FaMo.py:39`) |
| Mã 3DMM của RAF-DB | nằm cùng thư mục tên `<mẫu>_aligned` ⇒ EMOCA đã chạy **trên ảnh aligned 100×100** (tự dò FAN + crop × 1,25 + phóng 224) |
| `_lmk.npy` | loader bỏ qua `*_lmk.npy` / `inputs_lmk.npy` — EMOCA gốc không ghi tệp này ⇒ nhóm đã sửa io của EMOCA để lưu landmark. Khớp mục `io.py` bỏ trống trong notebook (M0-03) |

⇒ Với checkpoint RAF-DB, chuỗi train là **B/C** (aligned 100 → resize 224) nếu tệp `.jpg` là ảnh aligned gốc, hoặc **D** (crop EMOCA của ảnh aligned 100) nếu là `inputs.png`. Cần tác giả xác nhận **tệp `.jpg` trong thư mục RAF-DB là tệp nào**.

### 6.2 Đo 5 biến thể trên 7 ảnh (5 ảnh mẫu công khai + 2 ảnh người dùng; ảnh không lưu, chỉ ghi số)

Mã EMOCA: server thật 8765. Engine: `predict(mode="real")`, CPU. Căn chỉnh: `backend/web/face_align.py` với mẫu **TẠM** `rafdb_aligned_100` (ArcFace thu về 100×100, `backend/configs/align.yaml`, `status: provisional`).
- **A** — web hiện tại: crop EMOCA 224.
- **B** — aligned 100 (mẫu tạm), mã EMOCA từ ảnh gốc.
- **C** — aligned 100 + mã EMOCA chạy trên chính ảnh aligned 100.
- **D** — crop EMOCA **của ảnh aligned 100** (lời tác giả áp cho RAF).
- **E** — crop EMOCA hạ 100 rồi phóng 224 (chỉ mô phỏng độ phân giải thấp).

Mỗi ô: POSTER / MA3D-Net, top-1 %.

| Ảnh | A (web) | B | C | D | E |
|---|---|---|---|---|---|
| mẫu 0000 (chu môi) | buồn 73,1 / buồn 64,8 | vui 88,5 / vui 83,2 | = B | vui 87,2 / vui 64,0 | buồn 59,1 / buồn 52,2 |
| mẫu 0001 | ngạc nhiên 74,2 / 71,9 | ngạc nhiên 40,9 / 64,6 | = B | ngạc nhiên 88,5 / 82,3 | ngạc nhiên 77,4 / 74,8 |
| mẫu 0002 | buồn 87,4 / 86,4 | buồn 80,4 / 59,7 | = B | buồn 88,9 / 89,0 | buồn 86,3 / 86,7 |
| mẫu 0003 | buồn 88,9 / 88,8 | buồn 70,1 / 65,9 | = B | buồn 88,9 / 88,7 | buồn 89,0 / 88,8 |
| mẫu 0004 | buồn 88,9 / 88,8 | buồn 81,4 / 59,4 | = B | buồn 88,6 / 88,3 | buồn 88,9 / 88,6 |
| **người dùng A** (đàn ông cười rõ) | vui 42,4 / **buồn 45,3** | **vui 88,0 / vui 80,0** | = B | vui 61,0 / buồn 59,7 | vui 61,7 / buồn 33,0 |
| **người dùng B** (bé gái cau mày) | **giận 84,3 / giận 50,7** | buồn 43,5 / trung tính 39,2 | = B | buồn 49,8 / buồn 54,4 | giận 81,2 / ghê tởm 44,1 |

- **C trùng B tới từng số** ⇒ đổi mã 3DMM (ảnh gốc ↔ aligned) không đổi gì. Khớp mục 3: nhánh 3D gần bất hoạt.
- **Khớp trực giác** — chỉ 2 ảnh người dùng có nhãn hiển nhiên (cười ⇒ vui; cau mày ⇒ giận). Tính trên 4 dự đoán (2 ảnh × 2 mô hình):

| Biến thể | Số dự đoán khớp |
|---|---|
| A (web) | 3/4 — chỉ MA3D-Net ở ảnh A sai |
| B/C | 2/4 — sửa được ảnh A, làm hỏng ảnh B |
| D | 1/4 |
| E | 2/4 — hoặc 2,5 nếu tính "ghê tởm" gần "giận" |

- Ảnh "người dùng A" ở biến thể A lệch với số điều phối đo qua web (POSTER buồn 49,1 / MA3D buồn 56,5). Ở đây ảnh đã được xoay EXIF rồi mã hoá lại PNG trước khi gửi EMOCA, nên crop có thể khác một chút — **chưa kiểm**.
- **Kết luận:** với 7 ảnh (2 có nhãn hiển nhiên) **không biến thể nào tốt hơn rõ ràng**. Căn chỉnh bằng mẫu TẠM sửa được ảnh cười nhưng làm hỏng ảnh cau mày. Đúng như quyết định giữ căn chỉnh là **tuỳ chọn, mặc định tắt**. Cần (1) tác giả xác nhận tệp `.jpg` của RAF-DB, (2) mẫu 5 điểm **đo** từ ảnh RAF-DB aligned thật (`estimate_template`), (3) đánh giá trên RAF-DB test có nhãn (M2-07) trước khi bật.
