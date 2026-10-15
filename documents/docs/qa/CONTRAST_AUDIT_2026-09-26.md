# CONTRAST_AUDIT_2026-09-26 — Đo tương phản WCAG AA trong trình duyệt (backlog c)

> Vai: QA (`GENERAL`), giao bởi phiên điều phối btl-e6. **Chỉ báo cáo — không sửa theme.**
> Công cụ: `another/tests/audit_contrast_browser.py` (Chromium headless qua Playwright, venv `ma3d-web`), đo trên web đang chạy
> `http://127.0.0.1:8599` (chế độ mặc định: POSTER/MA3D-Net **giả lập**, EMOCA giả lập — **không** dùng `--engine-real`).
> Mỗi chủ đề 4 tổ hợp: `m` ∈ {sáng, tối} × `c` ∈ {chuẩn, cao}; khung 1440×3300, mở bảng Cài đặt, tải ảnh gradient 320×320
> sinh tại chỗ, mở mọi expander, rồi đo **mọi phần tử có chữ** (pha nền bán trong suốt). Ngưỡng AA: ≥ 4.5 (chữ thường),
> ≥ 3 (chữ lớn). Chạy tiền cảnh từng lượt ~6 chủ đề; kiểm RAM trống trước mỗi lượt, < 1.5 GB thì dừng.

## Kết luận

**DỪNG GIỮA CHỪNG vì RAM thấp — mới đo 12/24 chủ đề.** Trong phần đã đo: **47/47 tổ hợp đo đầy đủ đạt AA, 0 chữ không
đạt, không có cặp màu nào trượt.** 1 tổ hợp đo **không đầy đủ** (không tính là đạt, xem dưới). 12 chủ đề còn lại **chưa đo**.

| Lượt | Chủ đề | RAM trống trước lượt | Thời gian | Kết quả |
|---|---|---|---|---|
| 1 | cold, dark-cold, dark-winter, fall, summer, winter | 2.06 GB | 223.5 s | 24/24 tổ hợp đạt, 0 chữ không đạt (mã 0) |
| 2 | hunt-be5b50, hunt-f5cbcb, hunt-ff9d9d, hunt-ffb4a2, hunt-ab886d, hunt-bf9264 | 6.25 GB | 229.6 s | 23/24 đạt đầy đủ, 0 chữ không đạt (mã 0); 1 tổ hợp đo thiếu |
| 3 | hunt-d8ae7e, hunt-da8359, hunt-ffd6ba, hunt-ffdab3, hunt-674636, hunt-bbdc12 | **1.48 GB** | — | **Không chạy** (< 1.5 GB) |
| 4 | hunt-3368a0, hunt-8ca9ff, hunt-9564dd, hunt-bd5579, hunt-febbcc, hunt-946d6d | — | — | **Chưa tới lượt** |

Danh sách 24 chủ đề lấy từ `frontend/theme.py` `load_palettes(frontend/assets/theme)` (1 ảnh bị loại: `Color Hunt Palette e3f2fd90caf92196f30d47a1.png` trùng màu với `ColdColor.png`).

## Chi tiết: tỉ lệ nhỏ nhất mỗi tổ hợp (tất cả ≥ ngưỡng)

Mỗi tổ hợp đo 154 kiểu chữ (155 ở `hunt-be5b50` sáng/cao), trừ dòng đánh dấu ⚠.

| Chủ đề | sáng · chuẩn | sáng · cao | tối · chuẩn | tối · cao |
|---|---|---|---|---|
| cold | 4.65 | 7.02 | 5.88 | 7.02 |
| dark-cold | 4.62 | 7.02 | 5.26 | 7.03 |
| dark-winter | 4.61 | 7.03 | 4.66 | 7.05 |
| fall | 4.64 | 7.03 | 5.72 | 7.65 |
| summer | 4.64 | 7.03 | 5.75 | 7.38 |
| winter | 4.64 | 7.03 | 5.76 | 8.03 |
| hunt-be5b50 | 4.63 | 7.02 | 4.69 | 7.11 |
| hunt-f5cbcb | 4.61 | 7.07 | 5.86 | ⚠ 8.64 (chỉ 61 kiểu chữ) |
| hunt-ff9d9d | 4.61 | 7.07 | 5.86 | 7.19 |
| hunt-ffb4a2 | 4.64 | 7.03 | 5.74 | 7.20 |
| hunt-ab886d | 4.61 | 7.02 | 5.63 | 7.18 |
| hunt-bf9264 | 4.63 | 7.02 | 5.70 | 7.06 |

Cặp màu trượt ngưỡng: **không có** (script in tối đa 6 cặp tệ nhất cho mỗi tổ hợp có lỗi; không tổ hợp nào có).

## Nhận xét

1. **⚠ `hunt-f5cbcb` · tối · cao đo không đầy đủ:** chỉ 61 kiểu chữ so với ~154 ở mọi tổ hợp khác ⇒ nhiều khả năng trang chưa
   hiện xong khu kết quả khi đo (script chờ cố định 5 s sau khi tải ảnh; lượt 2 chạy ngay sau lượt 1). 61 phần tử đo được
   đều đạt (nhỏ nhất 8.64) nhưng **không** kết luận tổ hợp này đạt. Cần đo lại riêng: `--themes hunt-f5cbcb` (4 tổ hợp, ~40 s).
2. **Biên sát ngưỡng ở chế độ sáng · chuẩn:** mọi chủ đề đã đo có tỉ lệ nhỏ nhất 4.61–4.65, chỉ hơn ngưỡng 4.5 khoảng 2–3 %;
   `dark-winter` tối · chuẩn 4.66, `hunt-be5b50` tối · chuẩn 4.69 cũng sát. Đạt, nhưng một thay đổi CSS nhỏ (vd làm nhạt chữ phụ)
   có thể làm trượt — nên giữ bài đo này như một bước kiểm khi sửa `frontend/theme.css`/`frontend/theme.py`. Chế độ "cao" luôn ≥ 7.02.
3. Phạm vi: chỉ trạng thái mặc định (giả lập) ở 1440 px. **Không** đo `--emoca` (bảng Máy chủ EMOCA), `--engine-real`,
   màn hẹp 375 px, chế độ "ép tối" của trình duyệt.

## Việc còn lại

- Chạy lượt 3 và 4 khi RAM trống ≥ 1.5 GB (lệnh ở trên, mỗi lượt ~225 s), và đo lại `hunt-f5cbcb`.
- Sau đó cập nhật bảng này (số tổ hợp đạt/không trên đủ 96).

Log thô: đầu ra của từng lượt nằm trong scratch của phiên QA (không commit); các số trong báo cáo chép nguyên từ đó.
