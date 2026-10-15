# M5-04a — Thiết kế hiển thị lưới 3D EMOCA trong trình duyệt

> **Cập nhật phạm vi:** Mesh 3D tiếp tục được hoãn ngoài v1 theo ADR-018 và không thuộc milestone YOLO M5 hiện hành. Tài liệu thiết kế này được giữ để tham khảo; không có yêu cầu mở spike, contract, renderer hay nghiệm thu M5-04.

Ngày: 2026-09-30  
Trạng thái: đề xuất thiết kế, chưa triển khai  
Phạm vi: lưới một mặt EMOCA trong luồng phân tích Streamlit hiện có.

## Tóm tắt đề xuất

Trả **coarse mesh** của EMOCA dưới dạng các mảng số JSON: đỉnh `float32` XYZ và tam giác `uint16`/`uint32` (chỉ số zero-based), cùng metadata hệ tọa độ và màu bề mặt trung tính. Không gửi texture, UV hay normals ở đợt đầu; Three.js tự tính normals. Giữ một lưới đầy đủ ở mức coarse, không decimate mặc định để tránh méo hình học. Giới hạn kích thước sau serialize ở 2 MiB/mặt và thời gian lấy lưới tối đa 10 giây; lưới vượt giới hạn hoặc lỗi thì bỏ riêng phần mesh, phân tích vẫn trả kết quả bình thường kèm mã lỗi/warning có cấu trúc.

Thêm trường tùy chọn `mesh` cho từng face của `web_analyze`, dự kiến tăng contract chung từ `0.3.0` lên `0.4.0` (minor, trường tùy chọn). Renderer là Three.js trong iframe cô lập do `components.html`, không tải thư viện từ CDN, chỉ nhận dữ liệu mesh của đúng `run_key` ảnh hiện tại. Dùng `st.session_state` hiện có để cache theo digest/config; xóa mesh cũ khi upload rỗng, đổi ảnh, hoặc không còn face tương ứng.

## Căn cứ hiện trạng

- Roadmap M5-04 chỉ yêu cầu lưới xem được trong trình duyệt, giao `GENERAL` chốt ADR gửi đỉnh/mặt, `01_AI` lấy dữ liệu và `04_FRONTEND` dùng Three.js. Ghi chú khởi động M5 nêu rõ hiện chưa có contract/renderer mesh (`documents/docs/context/ROADMAP_BACKLOG.md`, mục M5-04 và ghi chú khởi động M5).
- ADR-012 giữ `emoca_svc` là HTTP JSON riêng và gộp engine + Dashboard vào một tiến trình Streamlit. ADR-018 chọn WSL2 cục bộ làm đường chính, Colab chỉ dự phòng. Do đó nên trả mesh theo contract EMOCA hiện có, không tạo service renderer riêng.
- Contract hiện hành là `0.3.0`; ADR-019 đã thêm trường tùy chọn vào face và tăng minor version, yêu cầu đóng gói lại service và đồng bộ phía client do kiểm tra phiên bản tuyệt đối (`documents/docs/context/CONTRACTS_ADR.md`, ADR-019).
- `backend/services/emoca_svc/emoca_runner.py` xây face từ kết quả infer, hiện đưa ra `codes`, `landmarks`, `renders`, transform và timing. `EmocaBackend` có thể lấy kết quả từ tensor infer trước khi giải phóng; nên rút gọn mesh ở đây, không đọc artifact trên đĩa.
- Upstream `backend/model/emoca/gdl_apps/EMOCA/utils/io.py::save_obj` lấy `opdict['verts'][i]`, topology từ `emoca.deca.render.faces[0]`, UV texture/UV faces và normal map để ghi coarse OBJ; sau đó `util.upsample_mesh(...)` tạo detailed vertices/colors/faces và ghi detailed OBJ. Demo upstream gọi hàm này khi `--save_mesh`. Đây là đường ghi file và tạo cả hai mesh, không phù hợp cho request web vốn chỉ cần dữ liệu trong RAM.
- `backend/web/orchestrator.py` chép các trường EMOCA cần thiết vào `faces[]`; `frontend/app.py` giữ kết quả trong `st.session_state['result']`, khóa cache theo `(digest, emoca_signature, engine_mode)`, và xóa kết quả khi uploader rỗng. Ảnh đầu vào không được ghi xuống đĩa.

## Nguồn mesh và biểu diễn hình học

### Coarse hay detailed

| Lựa chọn | Ưu điểm | Chi phí / rủi ro |
|---|---|---|
| **Coarse (khuyến nghị)** | Sát topology mặt FLAME/DECA chuẩn, đủ cho xem xoay và kiểm tra hình khối; ít đỉnh/mặt, serialize nhanh; không cần dense template/displacement upsampling | Không thể hiện chi tiết vi mô của da/độ nhám; shading khác ảnh render EMOCA |
| Detailed | Có displacement detail và dense colors từ `upsample_mesh`; nhìn gần có nhiều chi tiết hơn | Mesh dày hơn đáng kể, tăng JSON/payload, RAM và tải GPU; màu gắn theo vertex có thể nặng; chỉ phục vụ trực quan, không tăng giá trị suy luận |

Chọn coarse cho spike và bản đầu. Không đưa `detailcode` 128 chiều thay cho geometry: mã latent không phải mesh render được. Không suy topology từ ảnh render.

### Đỉnh, mặt, chỉ số

- `vertices`: mảng phẳng `[x0,y0,z0, x1,y1,z1, ...]`, số hữu hạn, đơn vị chuẩn hóa EMOCA, dtype logic `float32`.
- `faces`: mảng phẳng `[i0,i1,i2, ...]`, tam giác, index zero-based vào `vertices`; dtype logic `uint16` nếu `vertex_count <= 65535`, nếu không `uint32`.
- Đỉnh và mặt phải cùng một thứ tự từ `opdict['verts']` và `emoca.deca.render.faces`; không tự đảo winding trong server. Ghi `front_face: "ccw"` sau khi xác minh bằng spike; nếu upstream winding là ngược, khai báo chính xác hoặc đảo một lần có kiểm chứng, không để frontend đoán.
- Metadata bắt buộc: `coordinate_space`, `units`, `vertex_count`, `face_count`, `index_type`, `topology`, `front_face`, `bounds`.
- Hệ tọa độ đầu vào phải được xác nhận trực tiếp từ tensor và ảnh `geometry_coarse`/camera của upstream. Dự kiến là hệ local 3D EMOCA/FLAME, không phải `original_px`, `crop224` hay tọa độ màn hình. Không gắn nhãn `right_handed` hoặc đổi trục khi chưa kiểm tra mã renderer. Three.js adapter có thể đổi trục một lần để khớp camera web; phép đổi đó phải được ghi trong metadata/renderer và không sửa dữ liệu contract.

### Texture và normals

Không truyền texture, UV, normal map hoặc normals ở v0.4.0. `save_obj` có chúng, nhưng payload ảnh texture/base-color và tangent/normal data làm phình request, khiến JSON phức tạp và tăng bộ nhớ; nhu cầu hiện tại chỉ là quan sát hình khối. Renderer tạo `BufferGeometry`, đặt màu đơn sắc dễ đọc, gọi `computeVertexNormals()`, bật ánh sáng hai phía hoặc kiểm winding để tránh mặt tối do culling. Có thể bổ sung `vertex_colors` hoặc texture như tính năng mới nếu spike cho thấy màu/chi tiết thật sự cần thiết.

Không decimate khi chưa có đo chất lượng. Decimation giúp payload nhưng có thể làm thay đổi đường viền môi, mũi, mắt; mesh coarse đã là mức giảm độ phức tạp hợp lý so với detailed. Nếu kích thước coarse vượt ngân sách, ưu tiên nén nhị phân/đổi transport ở ADR tiếp theo; chỉ decimate có sai số được định lượng (ví dụ Hausdorff RMS ≤ 0.5% đường kính hộp bao và không đổi topology vùng mắt/mũi/miệng) sau khi lưu bản full làm đối chứng.

## Serialize và ngân sách tài nguyên

**Khuyến nghị v0.4.0:** JSON UTF-8 theo cấu trúc đã nêu, không base64 cho số liệu. `float32` làm tròn 6 chữ số thập phân trước encode; integer index ở dạng JSON number. Ưu điểm: dễ kiểm tra schema và debugger; trade-off: JSON dùng nhiều byte hơn binary. Không serialize `numpy.ndarray` thô hoặc pickle.

Ngân sách ban đầu cho mỗi mặt:

- `vertices` ≤ 100,000 điểm; `faces` ≤ 200,000 tam giác; giới hạn cứng theo kích thước thực tế sẽ được chốt trong spike.
- JSON mesh ≤ **2 MiB** UTF-8 sau serialize; phía browser giữ không quá một mesh active (mục tiêu bộ nhớ mesh + buffer ≤ 16 MiB/mặt, chưa tính WebGL driver).
- Tạo/serialize trên service ≤ **2 giây**; tổng EMOCA request giữ trong timeout hiện tại 60 giây, nhưng vượt 10 giây riêng cho trích/serialize mesh thì trả phân tích không mesh.
- Chỉ `max_faces` hiện hành được trả; mỗi mặt có budget riêng. Tổng response nên ≤ 8 MiB với cấu hình hiện tại tối đa 4 mặt. Nếu nhiều mặt vượt tổng ngân sách, giữ mesh mặt theo diện tích giảm dần và báo `mesh_payload_limited` cho phần bị bỏ.
- Đo RSS service trước/sau và JS heap trong trình duyệt trên một, rồi tối đa số mặt cấu hình; không suy ra dung lượng chỉ từ kích thước JSON.

Đây là ngân sách đề xuất để spike kiểm chứng, không phải số đo đã có. JSON decimal có thể lớn hơn đáng kể so với `float32` nhị phân; nếu coarse không đạt các ngưỡng trên, phương án kế là buffer nhị phân (base64 hoặc endpoint nhị phân có xác thực). Base64 tăng khoảng 33% kích thước và thêm bản sao bộ nhớ; binary endpoint phức tạp hơn vì hiện client/contract dùng JSON.

## Contract và lỗi

Thêm trường **tùy chọn** `faces[].mesh` trong `emoca_analyze` và `web_analyze`:

```json
"mesh": {
  "format": "triangles-f32-json-v1",
  "coordinate_space": "emoca_local",
  "units": "model_unit",
  "topology": "triangles",
  "index_type": "uint16",
  "front_face": "ccw",
  "vertex_count": 3,
  "face_count": 1,
  "bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]},
  "vertices": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
  "indices": [0, 1, 2]
}
```

Ví dụ chỉ minh họa hình dạng, không phải tọa độ mesh thật. Tên `coordinate_space`, đơn vị và winding phải được chốt sau spike. Contract chung `0.3.0` → **`0.4.0`**, vì chỉ thêm field tùy chọn; thêm ADR riêng (M5-04b), cập nhật hai schema, ví dụ valid/invalid và fixture vàng, cập nhật service/client đồng thời. Không sửa contract trong task thiết kế này.

Quy tắc tương thích/lỗi:

- Mesh vắng mặt hợp lệ; client cũ/mới không được thất bại chỉ vì không có mesh.
- Sai `contract_version`, định dạng không hỗ trợ hoặc face lỗi cấu trúc: theo quy tắc hiện tại báo `contract_mismatch`; không cố render mảng không rõ semantics.
- NaN/Infinity, số đỉnh/mặt không khớp, index âm/vượt `vertex_count`, bounding box không hữu hạn, payload vượt giới hạn: bỏ mesh của face đó, ghi warning có mã `mesh_invalid` / `mesh_too_large`; giữ landmark/renders/dự đoán và status phân tích chính nếu chúng thành công.
- EMOCA inference lỗi như hiện tại vẫn áp dụng status `error`/`partial`; lỗi mesh không được biến thành `no_face` hoặc làm hỏng prediction.
- Không trả thông báo chứa tọa độ hoặc dữ liệu ảnh trong log. Warning chỉ chứa mã, số đếm và giới hạn.

## Renderer Three.js và vòng đời Streamlit

- Dùng `streamlit.components.v1.html`/iframe với kích thước responsive; dựng scene trong vùng riêng, canvas `width: 100%`, chiều cao desktop khoảng 420 px và mobile 300 px. ResizeObserver cập nhật camera aspect/renderer; có fallback nếu WebGL không khả dụng.
- Không dùng CDN, script từ xa, analytics hoặc tải asset ngoài. Đóng gói phiên bản Three.js đã được dự án chấp thuận trong asset local; nếu component hiện tại không hỗ trợ asset local an toàn, spike phải xác minh cách bundle trước khi chọn triển khai. Hạn chế iframe chỉ nhận JSON từ parent; validate kích thước/kiểu trước cấp buffer.
- Camera orbit có giới hạn zoom, auto-fit theo bounds, nền tương phản chủ đề và nút reset; không cho phép URL, HTML hoặc shader từ payload.
- Mỗi component nhận `mesh_key = (run_key, face_index, mesh_digest)` làm key ổn định. Trong `frontend/app.py`, kết quả mesh chỉ lấy từ `st.session_state['result']` tương ứng với digest ảnh hiện upload và chữ ký cấu hình hiện tại. Khi upload khác, bấm phân tích lại, bỏ upload, chuyển face hoặc kết quả mới thiếu mesh: bỏ component/key cũ trước khi render; không dùng biến module global/cache_resource cho dữ liệu ảnh.
- Vì Streamlit rerun toàn script, component chỉ được render từ kết quả đã khớp `run_key`; không gọi lại EMOCA chỉ để vẽ. Giữ `mesh` trong cùng tuple kết quả đang cache; không tạo cache dùng chung giữa session người dùng. Với `max_faces > 1`, mỗi face có component/key riêng.

## Riêng tư và lưu trữ

Mesh được suy ra từ ảnh khuôn mặt và vẫn là dữ liệu nhạy cảm có thể nhận dạng. Chỉ xử lý trong RAM: tensor → mảng CPU → JSON; không gọi `save_obj`, không dùng temporary OBJ/NPY/GLB, không lưu vào thư mục output/log. Xóa tham chiếu tensor/mảng trung gian sau khi serialize; phía web chỉ giữ mesh trong session result hiện tại. Khi uploader bị xóa hoặc ảnh đổi, xóa cache result cũ; browser thay component để hủy WebGL buffers (`dispose()` geometry/material/renderer khi unload). Tránh giữ mesh trong `st.cache_data`, `st.cache_resource`, query params, browser localStorage, sessionStorage hoặc telemetry.

WSL2 cục bộ theo ADR-018 là đường chính. Nếu cấu hình Colab dự phòng được dùng, việc truyền mesh ra khỏi máy tuân theo đúng xác nhận gửi ảnh hiện có; không tạo đường truyền riêng hoặc lưu lại mesh trên tunnel/server lâu hơn request.

## Spike trước end-to-end

Thực hiện một spike nhỏ trên **một ảnh fixture được phép sử dụng** và một lần infer EMOCA:

1. Trích `verts` và `render.faces` ngay trong memory trước khi request kết thúc; ghi số đỉnh/mặt, dtype, min/max, winding, kích thước JSON, thời gian copy CPU/serialize và delta RSS. Không tạo artifact OBJ.
2. Dựng một trang HTML/Three.js tối giản hoặc component thử để tải JSON đó, kiểm tra mặt trước/sau, trục và hướng mặt bằng so sánh cạnh lưới với `geometry_coarse` render. Xoay/zoom và resize viewport.
3. Chạy lại cùng request qua rerun giả lập với cùng digest rồi ảnh digest khác: xác nhận cache chỉ dùng đúng mesh; xóa upload phải giải phóng component/buffer. Kiểm tra iframe không gọi mạng ngoài.

Spike đạt khi hệ tọa độ/winding được xác định, coarse mesh có vẻ tương ứng geometry render, JSON ≤2 MiB/mặt, trích+serialize ≤2 giây, RSS tạm tăng ≤32 MiB và browser mở được trong ≤1 giây sau khi nhận payload trên máy mục tiêu. Nếu không đạt, báo số đo và đề xuất binary/giảm lưới trước khi viết contract/schema.

## Tiêu chí nghiệm thu M5-04 end-to-end

1. Với EMOCA thật, một ảnh có mesh coarse xem được trong trình duyệt; camera xoay/zoom/reset và resize đúng, không phụ thuộc CDN.
2. Số đỉnh/mặt, bounds, winding, hệ trục và index được validate; không NaN/Infinity hay index ngoài miền.
3. Contract/schema/fixtures tăng đồng bộ lên 0.4.0; response cũ không mesh vẫn chạy; lỗi mesh chỉ làm ẩn mesh và không làm mất dự đoán/ảnh hiện có.
4. Đo payload và thời gian theo ngân sách; kiểm ít nhất một ảnh và số mặt tối đa được cấu hình. Ghi peak RSS service và số liệu browser.
5. Qua rerun cùng ảnh không gọi lại EMOCA; đổi ảnh, đổi face, phân tích lại và xóa upload không để lộ mesh của ảnh trước.
6. Không phát sinh OBJ/NPY/GLB hoặc ảnh mesh trên đĩa; không có request mạng tới CDN/asset host; component xử lý WebGL unavailable gọn và rõ.

## Tài liệu/mã đã đọc

- `documents/docs/context/ROADMAP_BACKLOG.md` — M5-04, ghi chú trạng thái mở đầu M5; M0-04 về quyết định trước đây tắt `--save_mesh`.
- `documents/docs/context/CONTRACTS_ADR.md` — quy tắc version/schema; ADR-012, ADR-018, ADR-019.
- `backend/services/emoca_svc/server.py`, `backend/services/emoca_svc/emoca_runner.py` — HTTP JSON, runner và `EmocaBackend`.
- `backend/model/emoca/gdl_apps/EMOCA/utils/io.py::save_obj`, `backend/model/emoca/gdl_apps/EMOCA/demos/test_emoca_on_images.py` — cách upstream xuất coarse/detailed OBJ.
- `backend/web/emoca_client.py`, `backend/web/orchestrator.py`, `frontend/app.py` — decode/forward dữ liệu, cache theo digest, Streamlit rerun và session state.
