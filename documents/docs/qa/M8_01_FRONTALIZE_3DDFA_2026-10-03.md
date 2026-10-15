# QA — M8-01 Frontalize 3DDFA-V2 (ADR-026) — 2026-10-03

**Verdict:** PASS có điều kiện cho phần triển khai và độ ổn định. Chưa đánh giá độ chính xác vì chưa có tập ảnh nghiêng có nhãn. Mọi con số về biểu cảm dưới đây là **minh hoạ định tính** (bất biến §1.8).

## 1. Phạm vi và cách làm

- Codex viết code trong sandbox, ở bản làm việc `C:\Users\admin\btl-wt\M8-01`. Sandbox của Codex không ghi được dưới Desktop, cách xử lý giống ENV-1 của dự án IT Studio. GENERAL review rồi mới gộp về dự án.
- Tệp đã gộp: `backend/services/frontalization/{__init__,tdfa_v2_wrapper}.py`, `backend/web/frontalize_experiment.py`, `backend/web/orchestrator.py` (thêm `run_frontalization_experiment`), `frontend/app.py` (khu thí nghiệm cuối trang), `another/tests/test_frontalization.py`, `another/tests/smoke_frontalization.py`, `another/tests/run_all.py` (đưa test mới vào nhóm nặng), `another/tests/test_app_ui.py` (thêm 1 test UI).
- **Không gộp:** các sửa đổi Codex làm riêng cho bản sao, gồm cho `test_theme`, `test_colab_bundle`, `test_wsl_scripts`, `check_ma3d_checkpoints` SKIP khi thiếu tệp. Lý do: gộp về sẽ làm yếu test của dự án đầy đủ.

## 2. Lỗi GENERAL phát hiện và đã sửa khi review

| # | Lỗi | Ảnh hưởng | Sửa |
|---|---|---|---|
| 1 | `frontend/app.py` dùng `re.search` nhưng không `import re` | Bấm nút thí nghiệm luôn báo lỗi (`NameError` bị bắt) | thêm `import re` |
| 2 | Mặt chính diện bị co theo bề ngang của mặt nghiêng (`factor=min(...)`) | Mặt càng nghiêng thì ảnh thẳng càng nhỏ | giữ tỉ lệ camera, chỉ căn tâm |
| 3 | Lật trục y lệch 1 px so với `similar_transform` của upstream | Lệch nhẹ | `y = 120 − y` như upstream |
| 4 | Pháp tuyến và z-buffer bị ngược dấu (đo trên crop thật: 85–92 % tam giác hướng +z, z lớn nhất ở đỉnh mũi) | `occluded` 0,95–0,98 trên mặt thẳng, ảnh lốm đốm | `nz > 0`, z-buffer theo max |
| 5 | z-buffer ô 4 px, dung sai 1,5 % quá chặt | Ảnh lốm đốm, `occluded` 0,65–0,85 | ô ~3 % bề rộng mặt, dung sai 20 % độ sâu, tính theo đa số tam giác kề, thêm 1 lượt làm mượt trên lưới → `occluded` 0,08–0,34 |
| 6 | `smoke_frontalization.py` nuốt lỗi (`except: continue`) | Vi phạm §6 | in lỗi ra |
| 7 | Hàm `run()` dùng chung của `test_app_ui.py` (có từ Session 34/35, không phải do M8): `Image.open` trên dữ liệu rác, và `import orchestrator` làm vỡ test "không nạp được orchestrator" | `test_app_ui` FAIL từ trước | không đọc được ảnh → 0 mặt; chỉ import orchestrator khi có upload |

## 3. Kiểm chứng tự động (dự án đầy đủ, venv `ma3d-web`, Python 3.13)

- `python another/tests/run_all.py`: 17 PASS, 5 SKIP (nhóm nặng), 1 FAIL. FAIL là `test_app_ui.py` do lỗi #7. Sau khi sửa, `PYTHONPATH=. python another/tests/test_app_ui.py` cho **ALL PASS**, gồm test mới `frontalization_experiment_section_is_disabled_in_mock_mode`.
- `python another/tests/run_all.py --heavy --only test_frontalization.py`: **PASS**. Test nhận `no_face`/`fit_failed` và test điều phối giả (chạy tuần tự từng mặt, lỗi một mặt không làm mất mặt khác, không chạy lại RealESRGAN/YOLO, không ghi tệp) đều đạt.
- **Hạn chế:** test dùng ảnh BFM tự render bị SKIP. Mạng ước lượng yaw −51° cho ảnh render chính diện, tức ảnh tổng hợp không giống ảnh thật, và ngưỡng test không bị nới. Vì vậy AC "mặt thẳng |yaw| < 10°" và "|yaw| > 30°" được kiểm trên ảnh thật (mục 4), không phải bằng test tổng hợp.

## 4. Smoke với model thật (`another/tests/smoke_frontalization.py database/image`, 29 crop YOLO, CPU)

- **Mặt gần thẳng** (angry.jpg, afraid.jpg): yaw −3…−4°, sau khi chỉnh vẫn |yaw| < 5°.
- **Ước lượng lại pose trên ảnh đã chỉnh thẳng** (trước → sau):

  | Trường hợp | Trước | Sau |
  |---|---|---|
  | yaw / pitch | 28° / 36° | 2–6° / 3–9° |
  | roll | 104° | 1–5° |
  | roll | −89° | −5° |
  | pitch | 46° | 17–19° (còn méo vùng cằm) |

- **Chưa có ảnh nào |yaw| > 30°.** Yaw lớn nhất trong dữ liệu là 29°. AC "|yaw| > 30°" vẫn mở.
- **Thời gian chỉnh thẳng:** khoảng 150–230 ms/crop sau khi đã nạp model; lần nạp đầu khoảng 15 s.
- **Kiểm bằng mắt** trên 8 crop có pose lớn nhất (ảnh ghép tạm trong scratchpad, đã xoá):
  - Mặt được dựng thẳng; nụ cười, nếp mắt và nếp miệng vẫn còn.
  - Vùng cần lấp đối xứng là các mảng liền, không còn lốm đốm.
  - Giới hạn: vật che trước mặt (tay, áo) bị kéo vào vùng mặt; mặt cúi mạnh (pitch 46°) bị méo cằm và miệng.

## 5. Đầu-cuối: EMOCA thật (WSL2, RTX 3050 4GB) + POSTER/MA3D-Net thật (CPU)

Luồng chạy: `prepare_multiface_views` → `run_frontalization_experiment`, 3 ảnh, tổng 16 mặt, **0 lỗi**.

| Ảnh | Số mặt | Tổng thời gian | Chỉnh thẳng | CodeFormer (trước + sau) | analyze (trước + sau) |
|---|---|---|---|---|---|
| afraid.jpg | 1 | 23,7 s | 0,16 s | 14,1 s | 9,4 s |
| ảnh webp nhiều mặt | 14 | 345 s (~24,6 s/mặt) | 2,8 s | 212,9 s | 129,2 s |

- **Tài nguyên:**
  - VRAM EMOCA: peak 1677 MB, reserved 1900 MB, trong ngân sách ≤ 2 GB. Bước chỉnh thẳng không dùng GPU.
  - RAM tiến trình đo (cả bộ RealESRGAN, YOLO, CodeFormer, 3DDFA, engine): **peak working set 1928 MB**. Con số này **vượt** ngân sách 1,5 GB của tiến trình web (ngân sách đặt ra trước M6/M7, khi chưa có bộ tiền xử lý). Cần đo riêng phần 3DDFA tăng thêm. Theo kích thước tệp thì ước khoảng ≤ 100 MB, nhưng chưa đo.
- **Biểu cảm trên mặt gần thẳng** (kiểm tra chỉnh thẳng có làm mất biểu cảm không):
  - angry.jpg: POSTER anger 89,0 → 88,8 %; MA3D anger 89,7 → 88,9 %.
  - afraid.jpg: POSTER surprise 88,8 → 88,8 %; MA3D 91,2 → 90,6 %.
  - Thay đổi dưới 1 điểm %.
- **Ảnh nhiều mặt, chỉ minh hoạ:**
  - MA3D tăng tự tin hoặc đổi top-1 ở các mặt nghiêng: mặt 3 sadness 41 % → happiness 88 % (POSTER happiness 88 % ở cả hai nhánh); mặt 5 (yaw 28°, pitch 36°) neutral 45 → 85 %; mặt 4 neutral 54 → 87 %; mặt 10 neutral 50 → 72 %.
  - Có mặt bị giảm: mặt 13 POSTER happiness 85 → 61 %, MA3D neutral → sadness; mặt 11 MA3D disgust 88 → 64 %; mặt 7 MA3D neutral 87 → 63 %.
  - **Không kết luận "tốt hơn"** khi chưa có nhãn.

## 6. Riêng tư

Ảnh và crop chỉ nằm trong RAM và `session_state`. Test điều phối chặn `Image.save` ra đường dẫn tệp. EMOCA chạy cục bộ. Ảnh ghép kiểm tra bằng mắt chỉ nằm trong scratchpad tạm và đã xoá ngay.

## 7. Còn mở

1. Thiếu ảnh |yaw| > 30° có nhãn (cặp nghiêng/thẳng như ví dụ 27 % → 89 % Happy) để đo đúng mục tiêu của M8.
2. RAM 1928 MB peak khi chạy cả bộ tiền xử lý: cần đo riêng phần tăng thêm của 3DDFA và xem lại ngân sách RAM sau M6/M7.
3. Test ảnh tổng hợp bị SKIP (mạng không nhận ra ảnh render).
4. Mỗi mặt mất khoảng 25 s; CodeFormer chiếm khoảng 60 %.
5. UI thật trong Streamlit (bấm nút với EMOCA thật) chưa chụp màn hình.

## 8. Bổ sung — ảnh `database/image/1.jpg` của người dùng (mặt nằm nghiêng, đang cười)

Chạy cùng ngày, EMOCA thật + POSTER/MA3D-Net thật, 1 mặt (`RealESRGAN + YOLO`), 0 lỗi.

- **Pose:** trước khi chỉnh, yaw/pitch/roll = 16,2° / 3,6° / **93,5°**. Ước lượng lại trên ảnh đã chỉnh: 22,3° / −0,5° / −3,2°. Roll được sửa; yaw không giảm (16 → 22°). `occluded` 0,18.
- **Kiểm bằng mắt** (ảnh ghép tạm, đã xoá): mặt được dựng đứng, nụ cười và nếp mắt còn nguyên. Mép trái mặt bị ảnh nền lấp vào, vì crop YOLO chạm mép ảnh gốc.
- **Thời gian:** frontalize 1,2 s (gồm lần nạp đầu của phần lưới), CodeFormer 10,6 s + 7,4 s, analyze 13,2 s (lần đầu nạp engine) + 4,0 s.
- **Xác suất** (trước → sau chỉnh thẳng):

  | Mô hình | happiness | neutral |
  |---|---|---|
  | POSTER | 75,0 → **88,7 %** (+13,8) | — |
  | MA3D-Net | 59,8 → **88,6 %** (+28,9) | 12,5 → 2,1 % |

  Top-1 giữ nguyên là happiness.
- **Tài nguyên:** RAM peak 1948 MB (cả bộ tiền xử lý); EMOCA VRAM peak 1677 MB.

Đây là một ảnh, chỉ là minh hoạ định tính. Kết quả cùng hướng với quan sát ban đầu của người dùng (nghiêng 27 % → thẳng 89 % Happy). Ảnh này là trường hợp **roll** lớn, chưa phải |yaw| > 30°, nên AC |yaw| > 30° của M8-01 vẫn mở.
