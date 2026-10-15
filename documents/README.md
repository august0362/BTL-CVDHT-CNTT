# MA3D-Net — Nhận diện biểu cảm khuôn mặt bền vững bằng điều biến thích ứng dẫn hướng bởi 3DMM

> **MA3D-Net**: *Multi-modal Adaptive 3D Morphable Model Modulation Network for Robust Facial Expression Recognition*.
> Tài liệu này tổng hợp ngữ cảnh của dự án từ hai file trong thư mục: bài báo (`MA3D_VC_revised_1.pdf`) và slide thuyết trình (`MA3D-Net-Presentation.pptx`).

> **M6 image preprocessing plan:** scope and proposed execution are documented in [`documents/docs/M6_IMAGE_PREPROCESSING.md`](docs/M6_IMAGE_PREPROCESSING.md). No pipeline code has been added.

---

## 1. Tóm tắt nhanh

| Mục | Nội dung |
|---|---|
| **Bài toán** | Nhận diện biểu cảm khuôn mặt (Facial Expression Recognition – FER) từ ảnh tĩnh |
| **Vấn đề** | FER 2D nhạy cảm với tư thế, ánh sáng, trang điểm, che khuất; đặc trưng biểu cảm bị lẫn với danh tính (identity bias) → khái quát hoá kém giữa các đối tượng |
| **Ý tưởng cốt lõi** | Dùng tham số 3D Morphable Model (3DMM/FLAME, trích bằng EMOCA) như một **tín hiệu điều biến** (scale + shift kiểu FiLM) để tinh chỉnh đặc trưng ảnh 2D ở nhiều tầng, thay vì chỉ ghép (concatenate) |
| **Nền tảng** | Mở rộng kiến trúc **POSTER** (cross-attention giữa đặc trưng ảnh và landmark) bằng một nhánh 3DMM |
| **3 đóng góp** | (1) Kiến trúc MA3D-Net đa phương thức 2D + 3DMM; (2) Cơ chế điều biến chéo phương thức (3DMM Modulation); (3) Hàm mất mát **MACE** (Margin-Aware Cross-Entropy) |
| **Kết quả (headline)** | RAF-DB **90.67%**, FERPlus **90.31%**, CAER-S **91.76%**, CheoFaMo **83.92%** – đều cao hơn POSTER |
| **Dữ liệu mới** | Đánh giá thêm trên **CheoFaMo** (biểu cảm trong nghệ thuật Chèo Việt Nam: trang điểm sân khấu, ánh sáng phức tạp) |
| **Hướng tiếp theo** | Mở rộng sang FER động (video), mô hình nhẹ, tổng quát hoá miền |

---

## 2. Nội dung thư mục

Thư mục `BTL` (theo tên thư mục: môn *Các vấn đề hiện đại của CNTT*, kì 1 năm 3).

| File | Loại | Vai trò |
|---|---|---|
| `MA3D_VC_revised_1.pdf` | Bản thảo bài báo (bản "revised") | Nguồn đầy đủ: phương pháp, thực nghiệm, ablation, hạn chế, tài liệu tham khảo |
| `MA3D-Net-Presentation.pptx` | 23 slide thuyết trình | Bản rút gọn: động lực → phương pháp → kết quả → kết luận |

### Cấu trúc slide

| Slide | Nội dung |
|---|---|
| 1–2 | Tiêu đề, tác giả, đơn vị; mục lục |
| 3–5 | **Motivation**: hạn chế của FER 2D · vai trò của 3DMM · câu hỏi nghiên cứu |
| 6 | **Contributions** |
| 7 | **Related Work** (FER, multi-modal FER, 3DMM) |
| 8 | Tổng quan phương pháp (sơ đồ kiến trúc) |
| 9 | Trích xuất đặc trưng đa phương thức |
| 10–12 | 3DMM Modulation Module |
| 13 | Multi-scale Cross-interaction Module |
| 14 | MACE loss |
| 15–16 | Bộ dữ liệu; thống kê CheoFaMo |
| 17–18 | So sánh với SOTA (RAF-DB, FERPlus, CAER-S, CheoFaMo) |
| 19–20 | Trực quan hoá; các trường hợp thất bại |
| 21 | Ablation |
| 22–23 | Kết luận & hướng phát triển; cảm ơn |

---

## 3. Động lực và câu hỏi nghiên cứu

- **Biến thiên môi trường**: tư thế, ánh sáng, trang điểm, che khuất làm thay đổi mạnh diện mạo → biểu diễn biểu cảm 2D thiếu ổn định.
- **Thiếu nhận thức hình học**: đặc trưng 2D thuần không mô hình hoá tường minh hình học khuôn mặt và thường trộn lẫn thông tin danh tính với biểu cảm.
- **3DMM là lời giải hợp lý**: tách *hình dạng danh tính* khỏi *biến dạng biểu cảm*, cho biểu diễn gọn, dễ diễn giải và bền hơn giữa các đối tượng. Các bộ ước lượng 3D từ ảnh đơn (như EMOCA) đã đủ tin cậy để đưa vào FER.
- **Khó khăn còn lại**: tích hợp thông tin 3D (thấp chiều, hình học) với đặc trưng thị giác (cao chiều, ngữ nghĩa) — ghép trực tiếp gây *representation mismatch*.

> **Câu hỏi nghiên cứu (slide 5):** làm thế nào tận dụng tiên nghiệm hình học 3D dày đặc để điều biến thích ứng đặc trưng ngoại hình 2D, nhằm nhận diện biểu cảm bất biến với danh tính và bền với tư thế?

## 4. Định vị so với công trình liên quan

- **FER 2D (deep learning)**: SCN, RAN (loại bỏ vùng nhiễu), DMUE, FDRL (bất định, tách đặc trưng), TSIVENet (hai luồng), TransFER và **POSTER** (transformer, cross-attention với landmark), WeiCL và các hướng self-/semi-supervised. Hạn chế chung: phụ thuộc biểu diễn 2D, landmark chỉ cho hình học **thưa**.
- **Multi-modal FER**: POSTER (landmark), ORSANet (phân đoạn ngữ nghĩa, chống che khuất). Các chiến lược fusion gồm early / late / **intermediate** (attention, cross-modal) — intermediate cho kết quả tốt nhất nhưng thường tương tác ngầm, dựa vào các modality 2D.
- **3DMM**: FLAME (mô hình tham số thấp chiều: shape, expression, pose) và EMOCA (ước lượng shape, expression, pose, detail, texture từ một ảnh RGB). Các nghiên cứu trước thường fusion 3D–2D theo kiểu đơn giản, chưa mô hình hoá tương tác.

**Điểm khác biệt của MA3D-Net**: tiêm tín hiệu điều kiện 3DMM **theo từng lớp, tăng dần (progressive layer-wise)** để hiệu chỉnh đặc trưng ngoại hình, kết hợp liền mạch với cross-interaction dựa trên landmark; không dùng ghép đặc trưng trực tiếp.

---

## 5. Phương pháp

### 5.1 Tổng quan pipeline

Bốn thành phần: (1) Multi-modal Feature Extraction → (2) 3DMM Modulation Module → (3) Multi-scale Cross-interaction Module → (4) Classification head.

```
                     ┌─► IR50 (huấn luyện) ───────────► X_face ∈ R^{P×D} ─┐
                     │                                                     ▼
Ảnh khuôn mặt I ─────┼─► EMOCA/FLAME (đóng băng) ─► X_3dmm ∈ R^334 ─► MLP ─► {α_i, β_i}
                     │                                                     │  (điều biến)
                     │                       3DMM Modulation: N = 5 Adaptive Block (FAM + residual bottleneck + SE)
                     │                                                     │
                     └─► MobileFaceNet (đóng băng) ─► X_lmk ∈ R^{P×D} ─────┤
                                                                           ▼
             Multi-scale Cross-interaction (3 tỉ lệ, Pyramid / Cross-Fusion Block, cross-attention hai chiều)
                                                                           │  gộp classification token các tỉ lệ → z_cls
                                                                           ▼
                                                   MLP head → xác suất lớp ⇄ MACE loss
```

### 5.2 Trích xuất đặc trưng đa phương thức

| Nhánh | Mô hình | Đầu ra | Trạng thái |
|---|---|---|---|
| Ảnh (ngoại hình, texture) | **IR50** | `X_face ∈ R^{P×D}` (chuỗi token không gian) | Huấn luyện |
| Landmark (hình học thưa, cấu trúc) | **MobileFaceNet** (ảnh resize 112×112) | `X_lmk ∈ R^{P×D}` | Đóng băng |
| 3DMM (hình học dày, tách danh tính) | **EMOCA** → tham số FLAME | `X_3dmm ∈ R^334` | Đóng băng |

- 334 tham số 3DMM = shape 100 + expression 50 + pose 6 + detail 128 + texture 50; **giữ nguyên cả 5 nhóm** và nối thành một vector.
- Mỗi chiều được chuẩn hoá độc lập theo mean/std của tập train, dùng lại cho val/test.

### 5.3 3DMM Modulation Module

Hình học được biến thành *tín hiệu điều biến* để hiệu chỉnh đặc trưng trung gian, thay vì ghép trực tiếp.

```
z = MLP(X_3dmm)
[α_1, β_1, …, α_N, β_N] = MLP(z)              # α_i, β_i ∈ R^C; chia sẻ theo không gian, broadcast
X̂_i     = σ(IN(X_i)) ⊙ (1 + α_i) + β_i        # FAM (Feature-wise Adaptive Modulation, kiểu FiLM)
X_{i+1} = X_i + Block(X̂_i)                    # kết nối residual
```

- `IN` = Instance Normalization, `σ` = LeakyReLU, `⊙` = nhân theo phần tử.
- **FAM** đặt ngay sau instance norm và trước phép tích chập.
- **3DMM Adaptive Block** (residual bottleneck): Conv 1×1 (giảm kênh) → Conv 3×3 → Conv 1×1 (khôi phục kênh); có IN, LeakyReLU, Dropout và **Squeeze-and-Excitation (SE)**.
- Xếp chồng **N = 5** block → điều biến hình học *lũy tiến* từ cấu trúc thô đến chi tiết mịn. Kết nối residual giữ lại thông tin ngoại hình trong khi tiêm dần tinh chỉnh hình học.
- Một embedding chung `z` sinh đồng thời tham số cho mọi block (thay vì sinh riêng từng lớp).

### 5.4 Multi-scale Cross-interaction Module

Lấy cảm hứng từ POSTER.

- Đặc trưng mặt (đã điều biến) và landmark được hạ mẫu phân cấp thành **3 tỉ lệ** (đầy đủ / trung bình / thấp) và xử lý qua các **Pyramid Block** xếp chồng.
- Ở mỗi tỉ lệ dùng **cross-attention hai chiều** (landmark→mặt và mặt→landmark):

```
Q_lmk = X_lmk·W_Q,  K_face = X_enh·W_K,  V_face = X_enh·W_V
F_{lmk←face} = Softmax(Q_lmk · K_faceᵀ / √d) · V_face          # đối xứng cho face←lmk
X_lmk_out = X_lmk + MLP(F_{lmk←face})
X_enh_out = X_enh + MLP(F_{face←lmk})
y_pred    = MLP(z_cls)          # z_cls: gộp classification token của các tỉ lệ
```

### 5.5 Margin-Aware Cross-Entropy (MACE)

Các biểu cảm giống nhau (Fear/Surprise, Anger/Disgust) khó phân biệt; CE chuẩn coi mọi mẫu như nhau → underfit mẫu mơ hồ, overfit mẫu dễ.

```
m = (p_y − p_max) / (p_y + p_max + ε),   p_max = max_{j≠y} p_j,   ε = 1e-8
w = 1 − m²
L_MACE = (1 + w) · (−log p_y)
```

| Biên `m` | Ý nghĩa | Hành vi MACE |
|---|---|---|
| `m ≈ 0` | Mẫu khó, mơ hồ | `w` lớn → tập trung học mạnh hơn |
| `m > 0` (lớn) | Dự đoán tự tin | `w` nhỏ → giảm ảnh hưởng mẫu dễ |
| `m < 0` | Dự đoán sai | Không tăng thêm trọng số, tránh nhấn mạnh nhãn có thể sai |

Hệ số `1 + w` đảm bảo mọi mẫu đóng góp *không thấp hơn* CE chuẩn. So với focal loss, MACE xét trực tiếp biên giữa lớp đúng và lớp cạnh tranh mạnh nhất.

---

## 6. Thực nghiệm

### 6.1 Bộ dữ liệu

| Dataset | Train | Test | Số lớp | Ghi chú |
|---|---:|---:|---:|---|
| **RAF-DB** | 12,271 | 3,068 | 7 | ~30k ảnh in-the-wild; dùng tập biểu cảm cơ bản (neutral, happiness, surprise, sadness, anger, disgust, fear) |
| **FERPlus** | 28,709 | 3,589 | 8 | Bản cải tiến FER2013, ảnh 48×48, nhãn tổng hợp từ nhiều người gán, thêm lớp *contempt* (có thêm 3,589 ảnh val) |
| **CAER-S** | 44,996 | 20,987 | 7 | 65,983 ảnh tĩnh của CAER, phục vụ FER theo ngữ cảnh |
| **CheoFaMo** | 4,675 | 1,169 | 6 | Dữ liệu mới về Chèo; xem bên dưới |

Chỉ số đánh giá: độ chính xác tổng thể (accuracy); riêng CheoFaMo bổ sung **Macro-F1** và **Balanced Accuracy** do mất cân bằng lớp.

### 6.2 CheoFaMo

- Bộ dữ liệu *ảnh chuỗi phân đoạn* cho phân tích tâm trạng trong **Chèo** (nghệ thuật sân khấu truyền thống Việt Nam): 37,198 frame gom thành **5,844 segment**, mỗi segment là một chuỗi biểu cảm theo mô hình onset–apex–offset (O-A-O).
- 6 lớp tâm trạng sân khấu (không có neutral), do chuyên gia lĩnh vực gán nhãn. Khó vì trang điểm sân khấu, ánh sáng mạnh, biểu cảm cường điệu.
- Chia **80/20 stratified theo lớp**. **Rất mất cân bằng**:

| Lớp | Tổng | Train | Test |
|---|---:|---:|---:|
| Happiness | 3,323 | 2,658 | 665 |
| Sadness | 1,562 | 1,249 | 313 |
| Fear | 332 | 266 | 66 |
| Love | 271 | 217 | 54 |
| Anger | 320 | 256 | 64 |
| Hatred | 36 | 29 | 7 |
| **Tổng** | **5,844** | **4,675** | **1,169** |

Lý do mất cân bằng (theo bài báo): Chèo thiên về nội dung đạo đức, nhân văn, lạc quan nên biểu cảm thù ghét/thù địch hiếm.

### 6.3 Thiết lập huấn luyện

| Hạng mục | Giá trị |
|---|---|
| Optimizer / scheduler | AdamW; ExponentialLR, γ = 0.98 |
| Weight decay | 1e-4 |
| Số epoch | 100 (MA3D-Net và POSTER cùng thiết lập để so sánh công bằng) |
| Hàm mục tiêu | CE + label-smoothing CE (hệ số 0.2, trọng số gấp đôi CE); **warm-up 10 epoch**, sau đó chuyển hoàn toàn sang **MACE** |
| Batch size / LR | 8 / ≈ 1e-5 (quy tắc căn bậc hai từ cấu hình gốc của POSTER: batch 100, LR 4e-5) |
| Phần cứng | NVIDIA GTX 1660 (6 GB) — bị hạn chế bộ nhớ (POSTER gốc dùng RTX 3090 24 GB) |
| Số block điều biến | N = 5 |

Warm-up là cần thiết vì đầu quá trình huấn luyện dự đoán chưa ổn định nên biên `m` chưa phản ánh đúng độ khó của mẫu.

### 6.4 Kết quả

**a) Con số công bố (headline)** — MA3D-Net so với POSTER:

| Dataset | POSTER | **MA3D-Net** |
|---|---:|---:|
| RAF-DB | 90.44 | **90.67** |
| FERPlus | 89.82 | **90.31** |
| CAER-S | 91.42 | **91.76** |
| CheoFaMo | 83.11 | **83.92** |

**b) Kiểm định độ ổn định** — chạy lại độc lập cả hai mô hình (5 lần; CheoFaMo được nêu là 4 lần), trung bình ± độ lệch chuẩn, t-test hai phía ghép cặp:

| Dataset | POSTER (tái lập) | MA3D-Net (tái lập) | Chênh lệch TB | t | p |
|---|---:|---:|---:|---:|---:|
| RAF-DB | 90.29 ± 0.10 | 90.51 ± 0.16 | +0.216 | 3.4256 | 0.0266 |
| FERPlus | 89.69 ± 0.46 | 90.10 ± 0.30 | +0.408 | 3.1797 | 0.0335 |
| CAER-S | 90.27 ± 0.66 | 91.10 ± 0.37 | +0.828 | 5.8528 | 0.0043 |
| CheoFaMo | 82.89 ± 0.07 | 83.49 ± 0.35 | +0.60 | 4.4016 | 0.0117 |

Cả bốn đều có ý nghĩa thống kê ở mức 0.05. Khoảng tin cậy 95%: RAF-DB [90.31, 90.71] vs [90.17, 90.41]; FERPlus [89.72, 90.47] vs [89.12, 90.25]; CAER-S [90.63, 91.56] vs [89.46, 91.08]; CheoFaMo [83.13, 84.02] vs [82.79, 83.08] (MA3D-Net vs POSTER).

**c) CheoFaMo — chỉ số bổ sung** (do mất cân bằng lớp):

| Mô hình | Accuracy | Macro-F1 | Balanced Acc. |
|---|---:|---:|---:|
| POSTER (tái lập) | 82.89 ± 0.07 | 64.81 ± 1.37 | 63.45 ± 2.15 |
| MA3D-Net (tái lập) | 83.49 ± 0.35 | 66.18 ± 1.93 | 64.97 ± 2.68 |

Mức cải thiện ở Macro-F1 (+1.37) và Balanced Acc. (+1.52) lớn hơn ở Accuracy (+0.60) → không chỉ do lớp đa số. Phân tích theo lớp: lớp thiểu số có recall thấp (khoảng 39–47% cho lớp Fear, 43–71% cho Hatred, ~50% cho Love), trong khi Happiness (~92–93%) và Sadness (~80–82%) cao → accuracy tổng thể cao hơn nhiều so với Macro-F1. *(Bài báo chỉ dùng chỉ số lớp 0–5; tên lớp ở đây là suy luận, xem mục 10, điểm 6.)* Sai số tập trung quanh lớp Happiness (vd. seed 42: 45 mẫu Sadness → Happiness và 45 mẫu Happiness → Sadness).

**d) So sánh SOTA (rút gọn từ bảng trong bài; các số của phương pháp khác lấy từ công bố gốc, không tái lập)**

| Phương pháp | Venue | RAF-DB | FERPlus |
|---|---|---:|---:|
| SCN | CVPR 2020 | 87.03 | 89.39 |
| RAN | TIP 2020 | 86.90 | 89.16 |
| DMUE | CVPR 2021 | 89.42 | – |
| FDRL | CVPR 2021 | 89.47 | – |
| EAC | ECCV 2022 | 90.35 | 89.64 |
| POSTER | ICCV 2023 | 90.44 | 89.82 |
| NLFER | The Visual Computer 2026 | 89.21 | 89.06 |
| **MA3D-Net** | – | **90.67** | **90.31** |

| Phương pháp | Venue | CAER-S |
|---|---|---:|
| DNS | ICML 2018 | 75.19 |
| CAER-Net-S | ICCV 2019 | 73.51 |
| GRERN | IEEE Access 2020 | 81.31 |
| EfficientFace | AAAI 2021 | 85.87 |
| MA-Net | TIP 2021 | 88.42 |
| GLAMOR-Net | NCA 2021 | 89.88 |
| POSTER | ICCV 2023 | 91.42 |
| **MA3D-Net** | – | **91.76** |

### 6.5 Ablation (độ chính xác, %)

**Thành phần chính** (RAF-DB / CheoFaMo):

| Cấu hình | RAF-DB | CheoFaMo | Δ RAF-DB | Δ CheoFaMo |
|---|---:|---:|---:|---:|
| **Full MA3D-Net** | **90.67** | **83.92** | – | – |
| w/o 3DMM Features | 90.44 | 83.11 | −0.23 | −0.81 |
| w/o 3DMM Modulation (thay bằng concat) | 90.48 | 83.54 | −0.19 | −0.38 |
| w/o Multi-scale Design | 90.48 | 83.75 | −0.19 | −0.17 |
| w/o Cross-Interaction | 90.54 | 83.77 | −0.13 | −0.15 |
| w/o MACE Loss | 90.59 | 83.23 | −0.08 | −0.69 |

Bỏ 3DMM ảnh hưởng nhiều nhất, đặc biệt ở CheoFaMo (điều kiện khắc nghiệt hơn).

**Ablation bổ sung** (RAF-DB / CheoFaMo):

| Yếu tố | Cấu hình → kết quả |
|---|---|
| Nhóm tham số 3DMM | Exp only 89.20 / 82.89 · Shape+Exp 89.56 / 82.81 · Pose+Exp 89.69 / 82.81 · **All 90.67 / 83.92** |
| Cách tích hợp 3DMM | Direct concat 89.36 / 82.04 · **FiLM-style 90.67 / 83.92** (+1.31 / +1.88) |
| Số Adaptive Block | 1: 89.62 / 82.89 · 2: 89.26 / 82.38 · 3: 89.49 / 82.29 · 4: 89.92 / 82.81 · **5: 90.67 / 83.92** |
| SE module | w/o SE 89.16 / 83.23 · **Full 90.67 / 83.92** |
| MACE warm-up (epoch) | 0: 89.39 / 82.55 · 5: 89.46 / 82.29 · **10: 90.67 / 83.92** · 15: 88.93 / 82.81 |
| Label smoothing | 0.0: 89.33 / 82.04 · 0.1: 89.36 / 82.72 · **0.2: 90.67 / 83.92** |

Nhận xét chính: dùng **toàn bộ** tham số 3DMM tốt nhất (shape/pose không gây hại, hiệu quả phụ thuộc dữ liệu); điều biến lũy tiến 5 block tốt nhất và không đơn điệu theo số block; warm-up 10 epoch là điểm tối ưu.

### 6.6 Kết quả định tính

- **Attention map**: POSTER có attention phân tán; MA3D-Net tập trung hơn vào vùng phân biệt (mắt mở lớn ở *Fear*, vùng miệng và má ở *Happiness*).
- **t-SNE** (RAF-DB): cụm của MA3D-Net gọn hơn (intra-class compactness) và tách biệt hơn (inter-class separability), nhất là với các biểu cảm dễ nhầm.

### 6.7 Hạn chế

- **Phụ thuộc chất lượng tái dựng 3D**: với ảnh mờ, méo hoặc che khuất nặng, EMOCA ước lượng kém → tiên nghiệm hình học suy giảm → điều biến kém hiệu quả, attention kém chính xác.
- **Văn hoá và chủ quan trong gán nhãn**: biểu cảm không ánh xạ phổ quát sang trạng thái nội tâm; biểu cảm Chèo mang tính cách điệu, sân khấu.
- **Mất cân bằng lớp** ở CheoFaMo (ví dụ lớp Hatred chỉ 36 mẫu).
- **Nguy cơ rò rỉ dữ liệu khi chia tập**: vì CheoFaMo là chuỗi ảnh phân đoạn, các đoạn kề nhau của cùng một buổi diễn/đối tượng nằm ở cả train và test có thể làm ước lượng lạc quan. Bài báo khuyến nghị chia theo nhóm (subject/performance) khi metadata cho phép.
- **Quyền riêng tư và đồng thuận** khi triển khai FER trong ứng dụng thực tế.

---

## 7. Kết luận và hướng phát triển

**Kết luận**: MA3D-Net tích hợp đặc trưng ngoại hình 2D với tiên nghiệm hình học 3DMM qua cơ chế điều biến thích ứng, kết hợp MACE loss, đạt kết quả tốt nhất trên bốn bộ dữ liệu (RAF-DB, FERPlus, CAER-S, CheoFaMo) trong các so sánh của bài báo.

**Hướng phát triển**:
1. Mở rộng sang **FER động (DFER)** trên video.
2. Mô hình hoá thời gian bằng RNN hoặc Transformer; cơ chế temporal attention cho khung hình/chuyển động quan trọng.
3. Ràng buộc **nhất quán thời gian** của tiên nghiệm 3D giữa các frame.
4. Mô hình **nhẹ** cho triển khai thời gian thực; kỹ thuật **domain generalization**.

---

## 8. Tác giả và tài nguyên

- **Tác giả**: Quoc Viet Hoang, Minh Kien Nguyen, Tien Phuc Nguyen, Thanh Ha Le, Quang Trung Dinh, Thi Chau Ma (tác giả liên hệ: chaumt@vnu.edu.vn).
- **Đơn vị**: HMI Laboratory, University of Engineering and Technology (UET), Vietnam National University, Hanoi (VNU-UET); Hanoi Academy of Theatre and Cinema (Quang Trung Dinh).
- **Tài trợ**: đề tài QG.25.03 — *"The research of reconstructing facial expressions and movements on 3D face model in acting using, serving the purpose of teaching in traditional arts training institutions"* (ĐHQGHN).
- **Mã nguồn** (theo bài báo: gồm code, checkpoint, cấu hình, script train/eval): https://github.com/HoangViet252006/MA3D-Net *(chưa kiểm tra nội dung repo)*
- **Dữ liệu CheoFaMo**: Mendeley Data, V2 (2026), DOI [10.17632/2h6hxwkbwn.2](https://doi.org/10.17632/2h6hxwkbwn.2)

## 9. Thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| **FER** | Facial Expression Recognition — nhận diện biểu cảm khuôn mặt |
| **3DMM** | 3D Morphable Model — mô hình khuôn mặt 3D tham số hoá |
| **FLAME** | 3DMM thấp chiều: shape (danh tính), expression, pose |
| **EMOCA** | Mạng ước lượng tham số FLAME (thêm detail, texture) từ một ảnh RGB |
| **IR50 / MobileFaceNet** | Backbone ảnh / backbone landmark |
| **POSTER** | Pyramid cross-fusion transformer cho FER; mô hình nền và đối thủ chính |
| **FiLM / FAM** | Điều biến đặc trưng theo kênh bằng scale + shift; FAM là biến thể dùng trong bài |
| **SE** | Squeeze-and-Excitation — hiệu chỉnh trọng số theo kênh |
| **MACE** | Margin-Aware Cross-Entropy — CE có trọng số theo biên giữa lớp đúng và lớp cạnh tranh |
| **DFER** | Dynamic FER — nhận diện biểu cảm trên video |
| **Chèo** | Nghệ thuật sân khấu truyền thống Việt Nam |
| **O-A-O** | Onset–Apex–Offset — mô hình ba pha của một biểu cảm |

---

## 10. Lưu ý khi dùng tài liệu này

Một số điểm chưa nhất quán hoặc cần làm rõ khi trích dẫn hay thuyết trình:

1. **Tiêu đề khác nhau**: slide ghi *"Multi-modal Adaptive 3D Morphable Model Modulation Network for Robust Facial Expression Recognition"*, còn PDF ghi *"MA3D-Net: 3DMM-Guided Adaptive Modulation for Robust Facial Expression Recognition"*.
2. **Số headline so với số tái lập**: các con số nổi bật (90.67, 90.31, 91.76, 83.92) cao hơn trung bình nhiều lần chạy (90.51, 90.10, 91.10, 83.49). Slide chỉ dùng con số headline; kiểm định thống kê, Macro-F1, Balanced Acc., phân tích theo lớp và các bảng ablation bổ sung chỉ có trong PDF.
3. **POSTER = 83.11 trên CheoFaMo** được bài báo gọi là "originally reported", nhưng bộ dữ liệu CheoFaMo (Mendeley Data V2, 2026) ra đời sau POSTER (2023), nên bài báo gốc của POSTER không thể có số này — *suy luận*: đây là kết quả do nhóm tác giả tự chạy. Con số này cũng **trùng khít** với dòng "w/o 3DMM Features" (90.44 / 83.11, giống POSTER trên RAF-DB) — nên chuẩn bị giải thích mối quan hệ giữa hai kết quả này.
4. **Số lần chạy không đồng nhất**: văn bản nói 5 lần chạy cho cả bốn bộ dữ liệu nhưng CheoFaMo được nêu là 4 lần, và Bảng 6 (precision/recall theo lớp) chỉ liệt kê 3 seed (42, 3407, 9999).
5. **"Direct concatenation" có hai kết quả khác nhau**: Bảng 7 (w/o 3DMM Modulation) là 90.48 / 83.54, Bảng 9 là 89.36 / 82.04. Bài báo giải thích là "controlled setting" khác, nhưng nên nắm rõ khác biệt thiết lập.
6. **Ánh xạ chỉ số lớp CheoFaMo** ở Bảng 6 không được nêu rõ; theo kích thước tập test (Anger 64, Fear 66, Happiness 665, Hatred 7, Love 54, Sadness 313) và các giá trị recall thì thứ tự khớp với thứ tự bảng chữ cái (0 = Anger … 5 = Sadness) — đây là suy luận, không phải phát biểu của bài báo.
7. **Nguồn tổng hợp**: nội dung lấy từ văn bản PDF và toàn bộ slide. Công thức chép lại từ văn bản trích xuất (một số ký hiệu Hy Lạp bị mất khi trích, đã khôi phục theo ngữ cảnh) — cần đối chiếu PDF gốc trước khi trích dẫn chính thức.
