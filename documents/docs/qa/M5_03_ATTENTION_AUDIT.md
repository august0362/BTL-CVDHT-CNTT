# M5-03 · Attention map — báo cáo audit nghiệm thu

> **Cập nhật phạm vi:** M5-03 đã được đưa ra khỏi milestone YOLO. Attention/overlay hiện hữu thuộc luồng phân tích cảm xúc và được giữ nguyên; báo cáo audit này là tài liệu lịch sử, không còn yêu cầu nghiệm thu riêng để hoàn tất M5.

**Ngày:** 2026-09-30  
**Phạm vi:** audit code, contract/ADR và các test mock/UI đã chỉ định. Không chạy suy luận model thật. Không tick backlog.

## Kết luận

**Chưa nghiệm thu toàn bộ AC.** Có triển khai attention map cho POSTER/MA3D-Net và overlay UI, cùng test nguồn; nhưng lượt chạy audit này không thực thi được test UI do môi trường Python hiện hành thiếu Streamlit. Test real-engine chứa suy luận POSTER/MA3D thật nên được chủ động không chạy. Vì vậy phần code/test hiện có không đồng nghĩa AC đã được nghiệm thu bằng kết quả chạy trong phiên này.

## Kết quả theo AC

| AC / nội dung đối chiếu | Kết quả | Bằng chứng và giới hạn |
|---|---|---|
| Phạm vi và định nghĩa attention; mô hình hỗ trợ; mock phân biệt rõ | **PASS (code/contract)** | ADR-013 và contract 0.2.0 định nghĩa attention là `attns[-1]["s"]["img_attn"]`, 7×7 trên `crop224`, chuẩn hoá 0..1, dùng được cho cả POSTER và MA3D-Net. `backend/web/engine.py` có đường trích xuất thật và mock; UI gắn nhãn dữ liệu giả, mô tả rõ attention mock không phải mức đóng góp dự đoán. Bản đồ `fusion_delta` chỉ MA3D, không phải attention. Test real-engine có assert POSTER trả attention 49 giá trị 7×7 trong [0,1], nhưng không chạy ở lượt audit này. |
| Ảnh nền đúng không gian ảnh model nhận (ADR-019/020) | **PASS (code/test hiện hữu); NOT VERIFIED (chạy trong phiên)** | ADR-019 cho crop EMOCA; ADR-020 yêu cầu phủ ảnh `model_input` khi có. `frontend/app.py::visual_tiles` ưu tiên `face.model_input.image`, fallback `crop_image`; chú thích tile ghi nguồn ảnh. Test UI `test_visual_block_missing_attention_and_model_input` mô phỏng model_input và kiểm tra nhãn nguồn; `test_attention_overlay_helpers` kiểm kích thước đầu ra cả với ảnh nền 100×100. Hai test chưa chạy vì thiếu Streamlit. |
| Nóng/lạnh, phóng lưới 7×7, shape sai/attention thiếu | **PASS (code/test hiện hữu); NOT VERIFIED (chạy trong phiên)** | `attention_overlay` kiểm `space == crop224`, số giá trị bằng rows×cols, resize bicubic lên 224×224, tô Turbo và blend. Test helper kiểm điểm nóng góc trên trái đỏ hơn điểm lạnh cuối lưới xanh hơn; thiếu attention và 10 giá trị trả thông báo/None. Đây là kiểm hình dạng tổng quát; hàm hiện không ép riêng grid phải đúng [7,7] (chấp nhận grid bất kỳ nếu số phần tử khớp), dù hợp đồng của mô hình là 7×7. Tests chưa chạy trong phiên. |
| Mock/UI liên quan | **NOT VERIFIED** | Đã thử `python -m pytest another/tests/test_app_ui.py -q`: không chạy, Python 3.11 thiếu `pytest`. Đã thử `python another/tests/test_app_ui.py`: dừng ở import vì thiếu `streamlit`; thông báo SKIP thứ cấp gây `UnicodeEncodeError` do terminal encoding, không phải lỗi assert của test. Không tìm thấy test UI nào thực thi thành công. |
| Attention model thật trên POSTER/MA3D-Net | **NOT VERIFIED** | `another/tests/test_engine_poster_real.py` chạy suy luận thật CPU; không chạy theo giới hạn audit “không inference thật nặng”. Test có kiểm attention POSTER, còn phần MA3D trong đoạn liên quan chủ yếu assert `fusion_delta`, không assert rõ shape/range attention MA3D. |

## Lệnh đã chạy

- `python -m pytest another/tests/test_app_ui.py -q` — **không chạy**: `No module named pytest`.
- `python another/tests/test_app_ui.py` — **không chạy**: `No module named streamlit`; phần in SKIP tiếp theo lỗi encoding terminal.
- Đọc/đối chiếu `ROADMAP_BACKLOG.md` phần M5-03, ADR-013/019/020, `backend/web/engine.py`, `backend/web/orchestrator.py`, `frontend/app.py`, `another/tests/test_app_ui.py` và đoạn attention trong `another/tests/test_engine_poster_real.py`.
- Không chạy `another/tests/test_engine_poster_real.py` vì file gọi inference model thật trên CPU.

## Còn thiếu để nghiệm thu toàn AC

1. Chạy các test mock/UI liên quan bằng đúng môi trường dự án có Streamlit (và pytest nếu dùng pytest), ghi kết quả PASS/FAIL.
2. Nếu AC yêu cầu xác minh dữ liệu attention model thật, cần chạy test real-engine với checkpoint/artifact và điều kiện inference được cho phép; lượt audit này không cung cấp bằng chứng đó.
3. Có thể cân nhắc assert grid đúng `[7, 7]` trong validator/test UI nếu AC muốn lỗi rõ với mọi shape sai kích thước, thay vì chỉ từ chối shape không khớp số values. Đây là khoảng cách giữa contract attention cố định 7×7 và validator UI tổng quát; báo cáo không sửa code.

**Trạng thái tại thời điểm audit:** M5-03 chưa tick theo phạm vi roadmap lúc audit. Quyết định phạm vi hiện hành ở đầu tài liệu đã đưa M5-03 khỏi milestone YOLO; kết luận audit này chỉ còn giá trị lịch sử.
