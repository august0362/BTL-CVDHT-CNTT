# M0-05 (3) — Vẽ landmark EMOCA thật lên ảnh gốc — 2026-09-25

- **Vai trò:** QA/DOCS, giao việc từ phiên điều phối `btl-e6`. Không tick.
- **Đầu vào (đọc CHỈ ĐỌC từ WSL, không bật GPU/server):** `~/emoca_wsl/out/analyze_sample_20260925_233239.json` (phản hồi `POST /v1/analyze` thật: `EMOCA_v2_lr_mse_20`, `e0be0db+patches`, contract 0.2.0) và `~/emoca_wsl/out/sample_image_20260925_233239.png` (ảnh đã gửi, 224×224, lấy từ bộ ảnh mẫu `EMOCA_test_example_data/images/affectnet_test_examples`).
- **Cách làm:** script tạm trong scratch của phiên (Pillow, python Windows, không thuộc repo). Script vẽ 68 điểm `landmarks.original_px` lên ảnh gốc kèm khung `bbox_original_px`, và 68 điểm `landmarks.crop224` lên `crop_image` (base64 trong phản hồi). Màu theo nhóm iBUG-68, đánh số từ 0: hàm 0–16, lông mày 17–26, mũi 27–35, hai mắt 36–41 / 42–47, miệng 48–67. Ảnh phóng ×3.
- **Ảnh chồng — CHỈ lưu trong WSL, KHÔNG vào repo** (ảnh AffectNet không được phát lại):
  - `~/emoca_wsl/out/m0_05/m0_05_overlay_original_px.png` — ảnh gốc + 68 điểm `original_px` + khung.
  - `~/emoca_wsl/out/m0_05/m0_05_overlay_crop224.png` — ảnh crop 224 + 68 điểm `crop224`.
  - `~/emoca_wsl/out/m0_05/m0_05_report.json` — mọi số dưới đây.
  - Bản sao tạm trên Windows (ảnh, JSON) đã xoá sau khi chép vào WSL.

## Kết quả (một ảnh — minh hoạ định tính, `CLAUDE.md §1.8`)

| Hạng mục | Giá trị |
|---|---|
| Ảnh gốc / crop | 224×224 / 224×224 |
| Số điểm | 68 `original_px`, 68 `crop224` |
| `crop_transform.matrix_3x3` (original_px → crop224) | gần như phép đồng dạng: tỉ lệ 1.0045, dịch y −1.0045 (ảnh mẫu vốn đã được cắt sẵn quanh mặt) |
| Sai lệch hai hệ: `matrix_3x3 · original_px` so với `crop224` | lớn nhất **0.001 px**, trung bình 0.0005 px |
| `bbox_original_px` | [26, 35, 196, 189] |
| Điểm nằm trong khung | 63/68. 5 điểm ngoài khung đều thuộc đường hàm/cằm, ở phía dưới khung; khung phát hiện mặt thường không bao hết cằm |
| Dải toạ độ điểm (x; y) | x 33.4–201.0; y 35.2–199.9 |

**Vị trí trung bình từng nhóm (`original_px`, x; y — y tăng xuống dưới):**

| Nhóm | Trung bình |
|---|---|
| Lông mày (17–26) | (115.3; 53.3) |
| Mắt bên trái ảnh (36–41) | (71.2; 67.5) |
| Mắt bên phải ảnh (42–47) | (145.4; 86.9) |
| Mũi (27–35) | (98.0; 103.7); đáy mũi (31–35): (94.9; 116.9) |
| Miệng (48–67) | (89.0; 139.6) |
| Hàm (0–16) | (98.3; 146.0) |
| Khoảng cách hai tâm mắt | 76.6 px |

**Kiểm thứ tự hình học (6/6 đúng):**
- lông mày ở trên mắt;
- mắt ở trên đáy mũi;
- đáy mũi ở trên miệng;
- mắt trái ảnh ở bên trái mắt phải ảnh;
- mũi nằm giữa hai mắt theo trục x;
- miệng nằm giữa hai mắt theo trục x.

Hai mắt lệch nhau 19 px theo trục y vì mặt trong ảnh hơi nghiêng.

**Nhận xét bằng mắt** (người viết xem ảnh chồng `original_px`):
- Điểm xanh lá nằm đúng trên hai mắt.
- Điểm xanh dương chạy dọc sống mũi xuống cánh mũi.
- Điểm hồng bao quanh môi.
- Đường hàm ôm theo mặt tới cằm.
- Lông mày: các điểm nằm trên cung mày. Riêng bên trái ảnh, vài điểm hơi cao hơn cung mày.

Đây là đánh giá của người viết. **Người dùng chưa xem.**

## Giới hạn

- Chỉ **một** ảnh, và ảnh đã được cắt sẵn quanh mặt, nên `crop_transform` gần như đồng nhất. Phép thử này **không** kiểm được nhánh biến đổi lớn (ảnh gốc to, mặt nhỏ, lệch tâm). Nhánh đó đã được kiểm trên backend giả (`another/tests/test_emoca_svc_real.py`, khép kín sai số 0.0003 px); với EMOCA thật, `another/tests/e2e_wsl_emoca.py` kiểm sai lệch ≤ 1 px trên ảnh do người dùng chọn (theo điều phối: 3/3 PASS).
- Không có độ chính xác landmark so với nhãn chuẩn: không có nhãn thật của ảnh này.
- Kết luận cho M0-05 (3) là việc của `GENERAL` và người dùng. Theo AC, ảnh minh chứng phải nằm trong `documents/docs/qa/M0/`, nhưng ảnh này **không được đưa vào repo** vì giấy phép AffectNet. Muốn có ảnh trong repo thì cần một ảnh được phép phát lại (vd ảnh của chính người dùng, có đồng ý) — cần quyết.
