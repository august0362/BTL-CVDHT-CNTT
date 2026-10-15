# M7: Multi-Object Analysis

> **Cập nhật 2026-10-03 (ADR-027):** CodeFormer đã bị **xoá khỏi toàn dự án** (mã, cấu hình, test, `backend/model/`, `backend/third_party/`, gói benchmark). RealESRGAN chỉ còn dùng để làm nét ảnh **đưa vào YOLO**; crop đưa vào EMOCA/POSTER/MA3D cắt từ **ảnh gốc** theo box YOLO. Mọi mô tả CodeFormer bên dưới là lịch sử, không còn đúng với code hiện tại.

## Behavior

- Run local YOLO detection on the **original** image before emotion inference; only if it finds zero faces, sharpen with RealESRGAN and run YOLO again (ADR-028). Crops always come from the original pixels (ADR-027). Keep at most 30 largest boxes, then assign IDs from top to bottom and left to right.
- Exactly one detected person follows the normal single-person prediction path on the uploaded image with `max_faces=1`; multi-object analysis is not shown.
- More than one detected person skips the normal single-person prediction and all single-person visual/geometry/comparison output. Show a responsive crop-card grid and automatically run EMOCA/MA3D on every original-pixel crop serially once per image/configuration. Each card shows a compact top prediction under its crop.
- YOLO detection results are cached per uploaded image and reused by automatic Multi-Object Analysis; do not repeat the detector call.
- Each card can expand full details and has a View Details button that opens the dedicated `page=face-details&face_id=…` view. Details include the original-image location reference, face crop, bbox/confidence, and result/error.
- No Run Multi-Object Analysis button is shown; changing the image/settings or using **Phân tích lại** allows a fresh automatic run. A failed run is not retried on every rerender.
- If detection fails (YOLO error or exception), show a warning and run the normal single-person workflow on the original upload (ADR-027; supersedes ADR-024). If detection completes with zero faces, use the existing normal workflow.
- Remove the probability radar chart from the single-person comparison UI. Multi-person inputs must not show the normal visual result, geometry or single-face comparison sections.
- User images and intermediate crops stay in memory for the active Streamlit session; nothing is saved to disk.

## Limits and contract

`backend/configs/services.yaml` sets `limits.max_faces: 30`. The web orchestrator, EMOCA client/service, real runner and YOLO preprocessor clamp requested counts to 30. The public `web_analyze` response schema is unchanged; routing and multi-object UI state are internal to the app.

## Scope

This is a face-handling and UI change. It does not claim that restoration improves emotion accuracy. The normal workflow stays on the original image for a single detected face; multi-person runs return per-face MA3D outputs without also running a single-person comparison.

## UI behavior and acceptance checklist

| Detected faces | Expected behavior |
|---|---|
| 0 | Keep the normal upload workflow and its no-face/fallback handling. Do not show multi-object controls. |
| 1 | Run the standard single-person prediction. Do not show multi-object controls. |
| 2–30 | Skip single-person inference and its visual/geometry/comparison sections. Automatically run multi-object inference once, then show up to 30 face cards in a responsive grid with a small top emotion/probability label under each crop. |
| More than 30 | Keep the 30 largest detections and assign stable spatial face IDs. |
| Detector error / exception | Show a warning and run the normal single-person workflow on the original upload (ADR-027). |

For each multi-face card, confirm **Full details** expands inline and **View Details** opens the dedicated page. Confirm the page shows original-image location, crop, detector metadata, and full prediction/error. Confirm a normal rerender does not start a second batch; **Phân tích lại** explicitly resets the run so it can retry.

## Acceptance status

The initial cap-20 M7 requirements passed a real local smoke on 2026-10-01; evidence is in [`documents/docs/qa/M7_REPORT_2026-10-01.md`](qa/M7_REPORT_2026-10-01.md). The expanded cap-30, routing, grid, detail page, radar-removal and automatic multi-analysis/card-summary requirements were added in ADR-024/025 and still need acceptance verification. The old smoke report does not claim acceptance of these expanded requirements.
