# Kiểm toán ROADMAP — 2026-10-04

Phạm vi: mọi dòng task M0–M8 và Backlog hoãn trong `documents/docs/context/ROADMAP_BACKLOG.md`. Phân loại theo trạng thái mới nhất có bằng chứng trong bản sao này; các tuyên bố cũ hơn trong cùng file được xem là lịch sử. Đã kiểm tra tồn tại các báo cáo/test được trích dẫn bên dưới. `documents/docs/qa/M0_M1_ACCEPTANCE_MAP_2026-09-25.md` là bản đồ bằng chứng, không phải log thô; các log trong `~/emoca_wsl/out/` được dẫn trong roadmap/session nhưng không có trong bản sao.

## Quy ước phân loại

1. **Tick đủ bằng chứng** — các AC được báo cáo là đạt và có báo cáo QA, session hoặc test tương ứng.
2. **Tick thiếu bằng chứng** — ghi rõ AC nào không được chứng minh hoặc bằng chứng chỉ đạt một phần.
3. **Mở nhưng đã xong** — có bằng chứng hoàn thành; nên tick.
4. **Mở, còn cần** — nêu phần thiếu và chủ thể cần hành động.
5. **Lỗi thời/thay thế** — quyết định/ADR sau này đã bỏ hoặc thay hành vi ban đầu.

## M0 — Chốt quyết định & EMOCA

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M0-01 · Chốt Q1–Q6 và cấu trúc repo | **2 — tick thiếu bằng chứng** | Q1–Q6 và quyết định không git/push, giữ `template_claude/` được ghi ở `documents/docs/context/SESSION_LOG.md` Session 41; phần đổi hướng ADR-003/005 có trong `documents/docs/context/CONTRACTS_ADR.md`. Nhưng AC(1) đòi ADR-001/003/005 được Accepted hoặc sửa: ADR-001 vẫn ghi **Proposed** trong `CONTRACTS_ADR.md` (dù bố cục sau được ADR-010 cập nhật). AC(2)–(3) đã có quyết định cuối. Thiếu chứng cứ chốt rõ AC(1) cho ADR-001; cần chỉnh ADR hoặc sửa AC trước khi coi tick đầy đủ. |
| M0-02 · Env EMOCA tái lập | **2 — tick thiếu bằng chứng** | Có số đo chạy script thoát 0, thời gian, `env_work38.yml` và `pip_freeze_work38.txt` được tóm tắt trong roadmap và `documents/docs/qa/M0_M1_ACCEPTANCE_MAP_2026-09-25.md`; Session 11 có lịch sử dựng. Tuy nhiên AC(1) nói máy Windows + WSL2 **sạch**; báo cáo xác nhận chưa thử máy thứ hai/môi trường sạch độc lập. AC(2)–(3) có số đo/tệp được ghi nhận. Thiếu AC(1) theo đúng tiêu chí tái lập; hoặc thu hẹp tiêu chí thành máy dự án đã dựng. |
| M0-03 · Bộ patch EMOCA | **1 — tick đủ bằng chứng** | AC(1)–(4): `documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md` mục 2 ghi `revert.sh`/`git status`, hash diff trước-sau khi apply, lý do trong các patch và script revert đã chạy; roadmap M0-03 ghi chi tiết tương ứng. Việc không có patch `io.py` được giải thích là không cần cho demo ảnh, và được chấp nhận cùng M0-04. |
| M0-04 · Smoke một ảnh | **1 — tick đủ bằng chứng theo AC đã được người dùng chấp thuận** | AC(1) bỏ `--save_mesh` theo quyết định người dùng vì web không dùng mesh; AC(2)–(3) smoke kiểm đủ 5 PNG/5 NPY và kích thước code; AC(4) người dùng xem ảnh; AC(5) fixture thật dạng tập con. Các quyết định/kết quả được ghi ở roadmap M0-04 và `documents/docs/qa/M0_M1_ACCEPTANCE_MAP_2026-09-25.md`. Giới hạn: fixture không có renders do quyền tài sản, không phải fixture đầy đủ của contract (xem M1-01). |
| M0-05 · Miền landmark và `tform` | **1 — tick đủ bằng chứng theo ngoại lệ đã chấp thuận** | AC(1) miền landmark; AC(2) phép đổi `crop_transform`; AC(3) overlay 68 điểm và kiểm hình học 6/6 có trong roadmap và `documents/docs/qa/M0_05_LANDMARK_OVERLAY_2026-09-25.md`. Ảnh chỉ lưu trong WSL thay vì `documents/docs/qa/M0/` theo chấp thuận của người dùng; `documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md` mục 7 ghi thêm kết quả overlay qua web. |
| M0-06 · EMOCA cục bộ WSL2 | **1 — tick đủ bằng chứng** | AC(1) GPU trong WSL; AC(2) setup + 5 PNG/5 NPY; AC(3) VRAM 1677 MB, reserved 1826 MB, không OOM; AC(4) web qua loopback và ảnh không rời máy. Có trong roadmap M0-06, `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25.md` (UI thật, loopback) và `documents/docs/qa/M0_M1_ACCEPTANCE_MAP_2026-09-25.md` (map từng AC). |

## M1 — Dịch vụ EMOCA

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M1-01 · Contract và fixture vàng | **2 — tick thiếu bằng chứng** | AC(1) schema nhận ví dụ hợp lệ/từ chối ví dụ lỗi, AC(3) contract version có test/bằng chứng ở Session 03 và các schema. AC(2) chỉ có `another/tests/fixtures/emoca_analyze_real_sample.min.json`: đã bỏ `renders`/ảnh, nên không qua schema contract đầy đủ; `another/tests/test_fixture_emoca_real.py` kiểm schema con. Roadmap ghi rõ fixture đầy đủ cần thay đổi quy định `renders`; `documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md` mục 1 cũng xác nhận. Thiếu AC(2) theo đúng yêu cầu fixture vàng đầy đủ; tick hiện tại chỉ hợp lệ nếu ghi rõ AC được thu hẹp thành fixture tập con. |
| M1-02 · `emoca_svc` mock | **1 — tick đủ bằng chứng** | AC trả mẫu đúng schema, không GPU: `another/tests/test_emoca_svc_mock.py` ALL PASS (27 ca theo roadmap); test kiểm schema, lỗi và tọa độ; nội dung được ghi ở `documents/docs/context/SESSION_LOG.md` Session 08 và phần M1-02 của roadmap. |
| M1-03 · `emoca_svc` thật | **1 — tick đủ bằng chứng** | AC(1) schema + landmark hai hệ; (2) không nạp lại; (3) load/latency/VRAM; (4) không ghi ảnh; (5) `no_face`: smoke real được ghi trong roadmap M1-03; `documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md` mục 1, 9, 10 ghi schema, quét không có ảnh mới và `no_face` thật. Test backend thật `another/tests/test_emoca_svc_real.py` bổ sung các ca hợp đồng/luồng giả, không được dùng thay smoke thật. |

## M2 — POSTER / MA3D-Net

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M2-01 · Artifact và bảng nhãn | **1 — tick đủ bằng chứng trong phạm vi đã chốt** | AC(1) hash checkpoint, backbone và stats; AC(2) thứ tự nhãn kiểm bằng hai nguồn cho mô hình `enabled`. `another/tests/test_configs.py --sha256`, `another/tests/check_emoca_stats.py`, `backend/configs/models.yaml` và quyết định thu hẹp phạm vi được ghi ở roadmap M2-01/`documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md` mục 6. Chỉ RAF-DB còn dùng; các bộ khác `enabled: false` theo quyết định người dùng Session 41. |
| M2-02 · Clone MA3D-Net và ghim commit | **1 — tick đủ bằng chứng** | AC commit và không sửa upstream: Session 02 ghi `6959e5a`, working tree sạch; roadmap M2-02. Đường dẫn wrapper làm ở M2-05. |
| M2-03 · Chốt cách dựng POSTER | **1 — tick đủ bằng chứng** | AC lớp mô hình/checkpoint chạy được và nguồn: `another/tests/check_ma3d_checkpoints.py`; `another/tests/test_engine_poster_real.py` strict-load 1369 keys; giao diện real POSTER có trong `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25.md`. Người dùng chốt giữ cách `MA3D` bỏ `mm_fusion`; ghi ở roadmap M2-03 và Session 09. |
| M2-04 · FER Engine mock | **1 — tick đủ bằng chứng** | AC hàm nội bộ trả xác suất đúng schema, chạy không GPU: `another/tests/test_ai_mock.py` ALL PASS (18 ca), gồm tổng xác suất/top1/attention và lỗi `codes_missing`; Session 05 và roadmap M2-04. |
| M2-05 · FER Engine thật | **1 — tick đủ bằng chứng** | AC(1) preprocessing so `val_transform`; (2) thứ tự và chuẩn hóa 3DMM bằng stats thật; (3) `codes_missing`; (4) tổng xác suất; (5) nạp một lần. `another/tests/test_engine_poster_real.py` (60 ca) và `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25.md` mục 2 chứng minh luồng EMOCA thật → POSTER/MA3D thật. `documents/docs/qa/M2_07_EVAL_PARITY_2026-09-30.md` xác nhận đường tính ở service/reference khớp. |
| M2-06 · Registry dùng chung | **1 — tick đủ bằng chứng** | AC thêm model qua config mà không sửa engine: `another/tests/test_engine_registry_extend.py` ALL PASS (12 ca), thêm ba entry tạm và hash `backend/web/engine.py`/`backend/configs/models.yaml` không đổi; roadmap M2-06. Test monkeypatch registry qua biến module, giới hạn này được ghi rõ trong roadmap. |
| M2-07 · Parity với `eval.py` | **1 — tick đủ bằng chứng** | AC ≥200 ảnh có 3DMM và dung sai: `documents/docs/qa/M2_07_EVAL_PARITY_2026-09-30.md` ghi 222 ảnh, 222/222 top-1 trùng, accuracy 83.3333% hai đường, max delta 4.17e-7 < 1e-5. Báo cáo cũng nêu đúng giới hạn: parity triển khai, không tái lập benchmark công bố/toàn tập. |

## M3 — Web

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M3-01 · Orchestrator + mock E2E | **1 — tick đủ bằng chứng** | AC(1) gọi client/engine, schema hợp lệ; (2) trạng thái mock/real; (3) không giả định GPU: `another/tests/test_orchestrator.py` và Session 05/08; test schema, mode/status và orchestrator không import torch. Các bằng chứng được tổng kết ở roadmap M3-01. |
| M3-02 · UI Streamlit mock | **1 — tick đủ bằng chứng** | AC upload/render, loading, lỗi, partial: `another/tests/test_app_ui.py` PASS (roadmap ghi 15 ca); ảnh UI/Playwright và spinner thật 651.6–6507.3 ms cùng layout 375 px được đo trong `documents/docs/qa/UI_LOADING_NARROW_2026-09-25.md`; `another/tests/test_emoca_client_remote.py`/AppTest bao phủ lỗi và partial. |
| M3-03 · Tọa độ/overlay | **1 — tick đủ bằng chứng với tiêu chí được thu hẹp** | AC web vẽ đúng `original_px`: `another/tests/test_fixture_emoca_real.py` sai lệch 0.001 px, `documents/docs/qa/M0_05_LANDMARK_OVERLAY_2026-09-25.md` và `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25.md` xác nhận overlay đúng vùng giải phẫu. QA cũng đo sai lệch landmark EMOCA so với FAN vượt ngưỡng 5% IOD; đây là độ chính xác mô hình, không phải lỗi hệ tọa độ web, được chấp nhận theo lựa chọn người dùng (roadmap M3-03). |
| M3-04 · Nối dịch vụ thật | **1 — tick đủ bằng chứng** | AC đường EMOCA thật→engine thật: `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25.md` mục 2; service lỗi/tắt/timeout trả `partial`, URL cấu hình được: `another/tests/test_emoca_client_remote.py` (HTTP giả lập thật) và roadmap M3-04. Chưa đo tắt server EMOCA thật giữa request, nhưng đây không phải thiếu AC riêng; hành vi đó được ghi là rủi ro chưa thử. |
| M3-05 · So sánh POSTER / MA3D-Net | **1 — tick đủ bằng chứng** | AC top1/xác suất/bất đồng/model-checkpoint/dataset/ghi chú định tính: `another/tests/test_app_ui.py::test_comparison_agree_disagree_and_checkpoint` kiểm hai nhánh dự đoán, metadata/checkpoint; `documents/docs/qa/UI_LOADING_NARROW_2026-09-25.md` có ảnh/trình duyệt và phép đo màn hẹp. Chưa có xác nhận xem bằng mắt của người dùng, nhưng AC không bắt buộc bước đó. |

## M4 — Tích hợp và QA

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M4-01 · Đo tài nguyên | **1 — tick đủ bằng chứng** | AC phần cứng RTX 3050/WSL2, p50/p95, VRAM, ngân sách: `documents/docs/qa/M4_01_RESOURCES_2026-09-26.md` ghi 40 lượt ấm, p50/p95, VRAM/RAM và đề xuất ngân sách; `documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md` mục 8 ghi ngân sách được GENERAL chấp nhận. Việc chép ngân sách vào `PROJECT_CONTEXT.md §6` còn chờ trong ghi chú cũ, nhưng AC của dòng này là **đề xuất** ngân sách, đã có báo cáo; nên đồng bộ tài liệu, không mở lại phép đo. |
| M4-02 · Lệch preprocessing | **1 — tick đủ bằng chứng** | AC so crop backend/web/EMOCA với RAF aligned và tác động xác suất: `documents/docs/qa/M4_02_PREPROCESSING_2026-09-30.md` ghi 42/42 paired, tỷ lệ đổi top-1 cho từng engine, giới hạn tập con. |
| M4-03 · Audit 10 mục | **1 — tick đủ bằng chứng** | AC QA verdict PASS hoặc defect giao owner: `documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md` có 8 PASS, 2 N-A có lý do, 0 FAIL và rủi ro/owner. |
| M4-04 · Runbook và kết quả dự phòng | **1 — tick đủ bằng chứng theo lựa chọn fixture tập con** | AC(1) người mới chạy: `documents/docs/qa/RUNBOOK_DRYRUN_2026-09-25.md` lần 2 đạt. AC(2) dữ liệu mẫu: fixture EMOCA thật `another/tests/fixtures/emoca_analyze_real_sample.min.json` được kiểm bằng `another/tests/test_fixture_emoca_real.py`; được chấp nhận là tập con không kèm ảnh có giấy phép. Ghi chú cũ coi Colab là đường chính đã lỗi thời sau ADR-018; WSL2 hiện là đường chính. |
| M4-05 · Trả nợ riêng tư | **1 — tick đủ bằng chứng theo ADR-018** | AC nhánh (a): EMOCA WSL2, loopback và ảnh ở máy người dùng, chứng minh trong `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25.md` và M0-06. Phần đo ngắt Colab không còn áp dụng khi chọn WSL2; hành vi server local tắt giữa request chưa đo và được ghi ở E2E mục 5, không phủ định nhánh AC đã chọn. |

## M5 — YOLO

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M5-01 · YOLO single-face | **5 — bị thay thế** | ADR-021 chốt utility/shadow, ADR-027/028 đổi luồng YOLO và cấm forward crop xử lý tăng cường vào baseline. `documents/docs/qa/M5_REVIEW_2026-10-03.md` ghi 01d và AC tổng (5) superseded; 01b provenance/license và 01e tài liệu còn hở, nhưng không còn giữ nguyên pipeline M5 như task hoạt động. M5 chỉ nên được đóng là “superseded”, không diễn đạt như mọi AC ban đầu đã đạt. |
| M5-02 · Multi-face | **5 — bị thay thế** | Chức năng được chuyển/mở rộng thành M7: ADR-023 → ADR-024/025 (cap 30, routing, cards/details, auto batch). `documents/docs/qa/M5_REVIEW_2026-10-03.md` và `documents/docs/qa/M7_04_ACCEPTANCE_2026-10-03.md` xác nhận thay thế và nghiệm thu M7. |
| M5-03 · Attention map | **5 — ngoài phạm vi** | Roadmap ghi attention/overlay đã là phần phân tích cảm xúc và không phải điều kiện YOLO; ADR-013 định nghĩa feature maps, ADR-018 giữ mesh/nhánh khác ngoài v1. Không còn việc M5 cần nghiệm thu. |
| M5-04 · Mesh 3D | **5 — hoãn ngoài v1** | ADR-012 nêu mesh tương tác để sau v1; ADR-018 giữ ngoài phạm vi. `documents/docs/qa/M5_STATUS_2026-09-30.md` xác nhận không phải backlog YOLO. |

## M6 — Tiền xử lý ảnh độc lập

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M6-01 · Model stack và môi trường | **1 — tick đủ bằng chứng** | Checkpoint x4v3 load/inference thật, SHA-256, CPU, thời gian/RAM: `documents/docs/qa/M6_CLI_ACCEPTANCE_2026-10-03.md` mục M6-01, PASS. GPU không đo do venv CPU; không phải thiếu AC vì báo cáo yêu cầu ghi thiết bị đã dùng. |
| M6-02 · Pipeline tuần tự | **1 — tick đủ bằng chứng** | Implementation độc lập nằm trong `backend/services/image_preprocessing/`; offline stage/order/RGB/bounds/no-face được test ở `another/tests/test_m6_pipeline.py` (được gọi trong `documents/docs/qa/M6_CLI_ACCEPTANCE_2026-10-03.md`). CLI thật x4v3 chứng minh crop RGB/bounds/limit và output hiện hành. Lưu ý mô tả roadmap M6-02 còn nhắc CodeFormer; cần sửa cho phù hợp ADR-027 và pipeline hiện tại. |
| M6-03 · Batch CLI / output / metadata | **1 — tick đủ bằng chứng** | CLI x4v3 đệ quy 4/4 ảnh, 21 crop đúng pixel, metadata/hash/thời gian; `documents/docs/qa/M6_CLI_ACCEPTANCE_2026-10-03.md` verdict PASS. QA nêu paired benchmark và review A/B chưa đo được trong bản sao/không cần cho app theo ADR-027; giới hạn đó không nằm trong AC kỹ thuật batch còn hiệu lực. |
| M6-04 · Bỏ CodeFormer khỏi app | **1 — tick đủ bằng chứng** | AC grep sạch, tests, luồng thật, RAM: `documents/docs/context/SESSION_LOG.md` Session 38 ghi grep sạch, 19 PASS, các test nặng PASS, 2 mặt 5.66→2.57 s, RAM và loại CodeFormer; `another/tests/test_m6_real_integration.py`, `another/tests/test_m7_real_acceptance.py`, `another/tests/test_frontalization.py` có trong bản sao. ADR-027 là quyết định nền. |

## M7 — Multi-Face

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M7-01 · Cap và crop N mặt | **1 — tick đủ bằng chứng** | Cap 30, thứ tự ID, >30: `documents/docs/qa/M7_04_ACCEPTANCE_2026-10-03.md` có YOLO thật 0/1/2/30/36 → giữ tối đa 30, crop đúng pixel; `another/tests/test_m7_real_detection_cap.py`. |
| M7-02 · Routing single/multi | **1 — tick đủ bằng chứng** | Single chạy baseline; multi không gọi baseline và batch tự động; detector lỗi không đoán số người: `documents/docs/qa/M7_04_ACCEPTANCE_2026-10-03.md` AppTest 10/10; ADR-024/025 và ADR-027 xác định fallback hiện hành. |
| M7-03 · UI nhiều mặt | **1 — tick đủ bằng chứng** | Grid, summary, expand/details, ẩn single views/radar, auto batch: AppTest trong `documents/docs/qa/M7_04_ACCEPTANCE_2026-10-03.md`; người dùng kiểm bằng mắt màn rộng/hẹp và View Details theo cập nhật cuối QA cùng Session 41. |
| M7-04 · Nghiệm thu mở rộng | **1 — tick đủ bằng chứng** | 0/1/2/30/>30, lỗi detector, chạy một lần/retry/navigation/no-write: `documents/docs/qa/M7_04_ACCEPTANCE_2026-10-03.md` (10/10 AppTest + YOLO thật); người dùng kiểm UI thật. Câu cũ trong roadmap nói “đang mở” đã bị phần nghiệm thu 2026-10-03 và Session 41/42 thay thế. |

## M8 — Chỉnh mặt

| Task | Phân loại | Đối chiếu AC và bằng chứng |
|---|---|---|
| M8-01 · Frontalize 3DDFA-V2 dạng warp 3D | **5 — bị ADR-028 thay thế** | Phần wrapper/test/UI ban đầu có QA trong `documents/docs/qa/M8_01_FRONTALIZE_3DDFA_2026-10-03.md`; nhưng ADR-028/Session 40 kết luận warp 3D gây hại với yaw/pitch tự nhiên và không dùng trong app. Luồng “luôn warp 3D” bị thay bằng M8-03: chỉ ước lượng roll rồi xoay 2D. Không nên coi M8-01 là AC còn hiệu lực hoặc task thuần túy pass. |
| M8-03 · Sửa roll 2D, YOLO ảnh gốc trước | **1 — tick đủ bằng chứng** | Sweep hoàn tất 18.408 cặp, ngưỡng T=15, thay cấu hình; `documents/docs/qa/M8_KAGGLE_RESULTS_2026-10-03.md` §3, `database/results/M8_ROLL/roll_sweep/` và `documents/docs/context/SESSION_LOG.md` Session 42. Session 39 ghi test chiều xoay/luồng và ADR-028. ADR-029/M8-04 sau đó bật roll 15° trong baseline (Session 44). |

## Backlog hoãn

| Mục | Phân loại | Đối chiếu và ai cần hành động |
|---|---|---|
| (a) Rà ROADMAP, tick task đã đạt | **3 — mở nhưng đã xong; đề nghị tick** | Audit này phân loại toàn bộ task và liệt kê sửa đổi đề nghị. Báo cáo này là bằng chứng mới cho việc hoàn thành mục (a). |
| (b) Cache Google Drive Colab | **4 — còn việc có điều kiện** | Chỉ cần nếu người dùng chọn chạy Colab dự phòng; hiện WSL2 là đường chính theo ADR-018. Nếu quyết dùng Colab: người dùng khởi chạy phiên đầu/cung cấp số liệu dung lượng; agent sau đó triển khai cache. |
| (c) Chạy lại audit tương phản 96 tổ hợp | **5 — hoãn theo quyết định người dùng** | `documents/docs/qa/CONTRAST_AUDIT_2026-09-26.md` ghi 12/24 chủ đề đã đo, 47/47 tổ hợp đạt và phần còn lại chưa đo. Người dùng đã chọn không chạy tiếp, ưu tiên demo/hiệu ứng về sau (ghi trong chính report và `documents/docs/qa/QA_REPORT_M4_03_2026-09-26.md`). Không còn việc hiện hành trừ khi người dùng mở lại. |
| (d) Đo UI màn hẹp | **1 — tick đủ bằng chứng** | `documents/docs/qa/UI_LOADING_NARROW_2026-09-25.md` đo lại sau sửa: `stMain` 375/375, không cuộn ngang. |
| (e) M2-07 và M4-02 | **1 — tick đủ bằng chứng** | `documents/docs/qa/M2_07_EVAL_PARITY_2026-09-30.md` và `documents/docs/qa/M4_02_PREPROCESSING_2026-09-30.md` lần lượt có parity 222 ảnh và paired preprocessing 42 ảnh. |
| (f) M4-05 riêng tư | **1 — tick đủ bằng chứng** | `documents/docs/qa/E2E_REAL_EMOCA_2026-09-25.md` ghi WSL2/loopback; ADR-018 chọn chạy cục bộ. |
| (g) Người dùng tự sửa PROJECT_CONTEXT và CLAUDE §1.1 | **4 — còn cần, chủ sở hữu theo từng file** | `documents/docs/qa/DOC_CONSISTENCY_SCAN.md` nêu điểm cần sửa và chủ: `PROJECT_CONTEXT.md` thuộc người dùng; `CLAUDE.md §1.1` cũng cần quyết định/chỉnh chính sách riêng tư của người dùng. Session 41 cho phép Claude cập nhật PROJECT_CONTEXT nói chung, nhưng yêu cầu hiện tại giới hạn chỉ sửa file audit; chưa có bằng chứng mục này đã hoàn tất. Người dùng cần quyết định/hoàn tất nội dung chính sách; sau đó agent có thể cập nhật PROJECT_CONTEXT khi được giao trong phạm vi thích hợp. |
| (h) `/handover` ghi block session cuối buổi | **3 — mở nhưng đã xong; đề nghị tick** | `documents/docs/context/SESSION_LOG.md` có Session 42 ngày 2026-10-04, có trạng thái, kết quả và bước tiếp; đáp ứng việc ghi handover cuối phiên. |
| (i) `.gitattributes` khi `git init` | **5 — huỷ theo quyết định người dùng** | Session 41 quyết định không git/push; roadmap mục (i) cũng ghi hủy 2026-10-03. Không tạo `.gitattributes` theo điều kiện đã bị bỏ. |
| (j) M5, M6 | **3 — mở nhưng đã xử lý; đề nghị tick** | M5-01/02 đóng dạng superseded theo `documents/docs/qa/M5_REVIEW_2026-10-03.md`; M6-01/03 nghiệm thu trong `documents/docs/qa/M6_CLI_ACCEPTANCE_2026-10-03.md`, M6-02/04 có tests và session tương ứng. Mục tổng hợp này không còn công việc độc lập. |
| (k) Texture BFM | **1 — tick đủ bằng chứng** | Người dùng quyết không xin giấy phép; texture tùy chọn `use_texture=false` và `EMOCA_REQUIRE_TEXTURE=auto` được ghi trong roadmap và Session 11. |
| (l) Giấy phép FLAME/EMOCA | **1 — tick đủ bằng chứng** | Session 41 ghi người dùng đã có giấy phép và đồng ý điều khoản; điều kiện bật cờ chỉ sau xác nhận đã được đáp ứng. |
| (m) CAER-S / FERPlus / CheoFaMo | **5 — huỷ khỏi phạm vi** | Session 41: người dùng chỉ dùng RAF-DB; roadmap đánh dấu hủy, các entry ngoài RAF-DB giữ `enabled: false`. Không cần tạo nhãn/stats cho bộ dữ liệu này nếu không có quyết định mới. |

## Đếm theo phân loại

Đếm mỗi task M0–M8 và mỗi mục (a)–(m) đúng một lần: **53 mục**.

| Phân loại | Số mục |
|---|---:|
| 1. Tick đủ bằng chứng | 37 |
| 2. Tick thiếu bằng chứng | 3 |
| 3. Mở nhưng thực tế đã xong | 3 |
| 4. Mở và còn cần | 2 |
| 5. Lỗi thời / bị thay thế / bị huỷ | 8 |

## Đề nghị sửa roadmap

- **Sửa tick thiếu:** mở lại M0-01 đến khi ADR-001 được sửa/đánh dấu rõ đã superseded; ghi giới hạn “chưa thử máy sạch độc lập” ở M0-02; M1-01 đổi AC fixture thành tập con được chấp nhận hoặc bỏ tick đến khi có fixture đầy đủ.
- **Đồng bộ trạng thái:** gỡ các câu lịch sử gây mâu thuẫn “M7 đang mở” sau phần nghiệm thu 2026-10-03; cập nhật phần đầu roadmap vốn ghi M6-01/03 mở lại, hiện đã đạt theo QA 2026-10-03; ghi M8-01 là superseded bởi ADR-028 và M8-03 là hạng mục đóng. ADR-029/M8-04 sau đó bật sửa roll trong baseline.
- **Đóng backlog đã hoàn tất:** tick (a), (h), (j); ghi (c), (i), (m) là hoãn/hủy, không phải việc mở; giữ (b) có điều kiện chỉ khi dùng Colab dự phòng.
- **Sửa mô tả lỗi thời M6:** xóa CodeFormer khỏi mô tả pipeline M6-02/03; phân biệt CLI batch M6 với luồng app ADR-027/028.
- **Giữ rõ việc còn cần:** mục (g) chỉ đóng sau khi các chỉnh sửa PROJECT_CONTEXT/CLAUDE được xác nhận; không nhập chung việc này với tình trạng nghiệm thu mô hình.
