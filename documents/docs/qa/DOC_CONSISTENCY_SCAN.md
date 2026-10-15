# DOC_CONSISTENCY_SCAN — rà mâu thuẫn tài liệu (chỉ đọc)

- **Ngày:** 2026-09-24 · **Người rà:** phiên `02_CONFIG_METRICS` đóng vai `DOCS` (DOCS.md §2.6), theo giao việc của phiên điều phối `btl-f9`.
- **Cách làm:** đọc các file trong phạm vi; grep từ khoá lỗi thời; quét đường dẫn bằng script tạm (ngoài repo); chạy `python another/tests/run_all.py --list` và 7 test nhẹ để đối chiếu số ca. **Không sửa file nào ngoài báo cáo này.**
- **Phạm vi:** `CLAUDE.md`, `.claude/agents/*.md` (không `_archive`), `documents/docs/context/{PROJECT_CONTEXT,CONTRACTS_ADR,ROADMAP_BACKLOG,SESSION_LOG}.md` (chỉ block Session 06–09 được xét là trạng thái hiện tại), `documents/docs/RUNBOOK_DEMO.md`, `documents/docs/modules/EMOCA.md`, `backend/configs/*.yaml`, `backend/web/requirements*.txt`, ô markdown của `another/notebooks/emoca_colab.ipynb`, docstring/chú thích của `another/tests/*.py`, `backend/web/*.py`, `backend/services/emoca_svc/*.py`.
- **Tổng:** 4 HIGH · 13 MED · 9 LOW.

Mức: **HIGH** = dẫn người đọc làm sai hoặc tạo rủi ro riêng tư · **MED** = sai sự thật nhưng không gây hại · **LOW** = chính tả, định dạng, tên gọi cũ.
Chủ sở hữu: `PROJECT_CONTEXT.md` và `CLAUDE.md` §1, §3–§7 thuộc **người dùng** (ở đây chỉ đề xuất). Bản đồ `CLAUDE.md` §2, `ROADMAP`, `SESSION_LOG` do `DOCS` ghi. `CONTRACTS_ADR.md` thuộc `GENERAL`. `backend/configs/` thuộc `CONFIG`. File vai trò do `GENERAL` quản, `DOCS` ghi.

---

## HIGH

| # | file:dòng | Câu (rút gọn) | Mâu thuẫn với (bằng chứng) | Đề xuất sửa | Chủ |
|---|---|---|---|---|---|
| H1 | `documents/docs/context/PROJECT_CONTEXT.md:31-47` | Sơ đồ `backend/web/backend` "KHÔNG có torch" → HTTP → `emoca_svc` + `ma3d_svc`; "mỗi module một env, giao tiếp qua HTTP JSON (ADR-002)" | ADR-012 `CONTRACTS_ADR.md:203` (Accepted: POSTER/MA3D + web trong một tiến trình Streamlit); `CLAUDE.md:10` (ghi chú ADR-012); `backend/configs/services.yaml:3` ("không còn ma3d_svc"). File này tự nhận là "thắng trí nhớ" (`:3`), nên agent đọc nó sẽ dựng sai kiến trúc | Vẽ lại sơ đồ: `backend/web/` (app → orchestrator → engine, torch CPU) ⇄ HTTP ⇄ `emoca_svc` (Colab, ADR-015); sửa câu `:47` thành "chỉ EMOCA tách env/tiến trình (ADR-012)". Kéo theo `:105-108` (nguồn dữ liệu `ma3d_svc.predictions`/`.attention` → `engine.predict` trong `web_analyze.faces[].predictions`) | Người dùng |
| H2 | `.claude/agents/03_BACKEND.md:10` | Gửi ảnh ra máy không phải loopback "là **ngoại lệ có kiểm soát** của điều cấm 'gửi ảnh tới dịch vụ bên thứ ba'" | `CLAUDE.md:9` §1.1 vẫn cấm tuyệt đối, chưa có câu ngoại lệ; tiêu đề ADR-015 `CONTRACTS_ADR.md:241` ghi "phạm vi ngoại lệ riêng tư **chờ người dùng xác nhận**"; `ROADMAP_BACKLOG.md:136-137` M4-05 `[DEBT-HIGH]` còn mở; `RUNBOOK_DEMO.md` §8 ghi "chưa được người dùng xác nhận bằng văn bản" | Thêm "(TẠM, chờ người dùng xác nhận phạm vi — M4-05, ADR-015)" vào câu `:10`, để agent BACKEND không coi đây là quyền đã được cấp | GENERAL (file vai trò) |
| H3 | `backend/configs/services.yaml:20` | `engine.mode`: "real cần torch + checkpoint (**environments/web/requirements.txt**)" | Requirements đã tách: `environments/web/requirements.txt` chỉ còn bản mock, torch nằm ở `environments/web/requirements-real.txt` (`CLAUDE.md:48-49`, `RUNBOOK_DEMO.md` §1.2b). Người làm theo chú thích sẽ cài thiếu torch rồi gặp `model_unavailable` | Đổi thành `(environments/web/requirements-real.txt; cài torch CPU trước — xem chú thích đầu file đó)` | CONFIG (lỗi sót của chính phiên này ở việc 3) |
| H4 | `.claude/agents/01_AI.md:61` | "nạp một lần… (theo ADR-012, dùng `st.cache_resource` hoặc tương đương vì chạy chung tiến trình Streamlit)" | ADR-016(1) `CONTRACTS_ADR.md:270`: "singleton mức module, **không dùng Streamlit**"; ADR-012 `:208` và `01_AI.md:23`: `engine.py` không import Streamlit. Agent AI làm theo `:61` sẽ đưa Streamlit vào `engine.py`, phạm ranh giới module (bất biến §1.2) | Sửa thành "nạp một lần bằng singleton mức module trong `backend/web/engine.py` (ADR-016), không dùng `st.cache_resource`". `ROADMAP_BACKLOG.md:78` (M2-05 AC5) cũng nên thêm "(ADR-016: singleton mức module)" | GENERAL; ROADMAP → DOCS |

## MED

| # | file:dòng | Câu (rút gọn) | Mâu thuẫn với (bằng chứng) | Đề xuất sửa | Chủ |
|---|---|---|---|---|---|
| M1 | `CLAUDE.md:116` (§6) | "chạy lệnh kiểm chứng của module (ghi trong `IMPL`; **chưa có `verify` chung**)" | `another/tests/run_all.py` đã có (`CLAUDE.md:53`, `RUNBOOK_DEMO.md:193-199`; `python another/tests/run_all.py --list` → 12 test, đo 2026-09-24) | "…; lệnh chung: `python another/tests/run_all.py` (`--heavy` khi đủ RAM)" | Người dùng |
| M2 | `CLAUDE.md:54` (§2) | `documents/docs/qa/` "(sẽ tạo) báo cáo QA" | Thư mục đã được tạo cùng báo cáo này | "báo cáo QA — `DOC_CONSISTENCY_SCAN.md` (rà tài liệu, 2026-09-24)" | DOCS |
| M3 | `documents/docs/context/CONTRACTS_ADR.md:4` | "bản nháp `contract_version: 0.1.0`, **mọi ADR ở trạng thái `Proposed`**" | Cả ba schema dùng `"const": "0.2.0"` (`backend/contracts/emoca_analyze.schema.json:219`, `ma3d_predict.schema.json:14`, `web_analyze.schema.json:17`); ADR-009, 010, 012, 015 là Accepted (`:179,182,203,241`); các ví dụ `:38,75,103` còn ghi 0.1.0 | "contract hiện hành `0.2.0` (ADR-013); trạng thái từng ADR ghi ở tiêu đề của nó; ví dụ JSON ở Phần B là bản 0.1.0 rút gọn" | GENERAL |
| M4 | `documents/docs/context/CONTRACTS_ADR.md:133` (Phần C) | `services.yaml`: `emoca_svc{url,timeout_s,mode}`, **`ma3d_svc{url,timeout_s,mode}`**; giới hạn 3 khoá | `backend/configs/services.yaml:11-29`: không có `ma3d_svc`, thêm `host`, `port`, mục `engine.mode`, `limits.min_face_side_px`, `limits.max_image_side_px`. `another/tests/test_configs.py` kiểm `"ma3d_svc" not in services` và đòi `host/port/timeout_s/mode`. Tiêu đề Phần C `:127` còn "(`04` điền giá trị, `01` định nghĩa hình dạng)", theo số vai trò cũ | Cập nhật dòng `:133` theo file thật + ADR-012/016; đổi `:127` thành `02_CONFIG_METRICS` / `GENERAL` | GENERAL |
| M5 | `CONTRACTS_ADR.md:190, 216, 228, 268` | ADR-011, ADR-013, ADR-014, ADR-016 còn "*Proposed*" | Cả bốn đã được dùng trong code: `arch: MA3D_no_fusion` (`backend/configs/models.yaml`, ADR-011); schema `0.2.0` + `feature_maps` (ADR-013); `frontend/theme.py`/`theme.css` (ADR-014, `CLAUDE.md:42-43`); `backend/web/engine.py` chế độ `real` (ADR-016, `ROADMAP:80`) | Người dùng/GENERAL chốt Accepted, hoặc ghi rõ "đang dùng thử, chưa chốt" ngay trong tiêu đề | GENERAL + người dùng |
| M6 | `.claude/agents/01_AI.md:44` | ADR-011: POSTER "chỉ chấp nhận khi khoá `state_dict` thiếu/thừa đúng bằng `mm_fusion.*`…, nếu không dùng mã POSTER gốc" | Phần cập nhật ADR-011 `CONTRACTS_ADR.md:196-198`: tiêu chí (1) "**sai theo nghĩa đen**" (POSTER có đủ 1369 khoá kể cả `mm_fusion`); đề xuất tiêu chí mới đã được code dùng (`MA3D_no_fusion`) | Trỏ tới tiêu chí đã đề xuất ở `CONTRACTS_ADR.md:198` ("nạp strict + `mm_fusion` chưa huấn luyện"), ghi "chờ người dùng chốt" | GENERAL |
| M7 | `.claude/agents/01_AI.md:72` | Đầu vào #2: checkpoint `raf_MA.pth`, `raf_ori.pth`, `ir50_o1.pth`, `mobilefacenet_model_best.pth_o1.tar`; "thiếu thì chỉ chạy mock" | Đã có 10 file dưới tên khác: `raf_MA3D.pth`, `rafdb_POSTER.pth`, `ir50.pth`, `mobilefacenet_model_best.pth.tar`… (`ls backend/model/ma3d/checkpoints`, `backend/configs/models.yaml`, `CLAUDE.md:35`). Thứ thật sự còn thiếu chỉ là `*_emoca_stats.npz` | "Checkpoint: theo `backend/configs/models.yaml` (đã có). **Còn thiếu:** `rafdb_emoca_stats.npz` (kiểm bằng `another/tests/check_emoca_stats.py`)" | GENERAL |
| M8 | `documents/docs/context/ROADMAP_BACKLOG.md:70` (M2-03) | "**Còn thiếu:** chạy thật (cần torch, chưa cài…)" | `ROADMAP:80` (M2-05, Session 09): torch 2.14.0+cpu đã cài, `rafdb_POSTER.pth` nạp strict 1369 khoá, `test_engine_poster_real.py` ALL PASS. AC của M2-03 (`:69`) vẫn ghi `raf_ori.pth` (file thật: `rafdb_POSTER.pth`, archive tên `raf_ori`, `models.yaml:38`) | Thêm ghi chú Session 09: đã chạy thật; chỉ còn chờ người dùng chốt tiêu chí (1) của ADR-011 (xem M6) rồi mới tick | DOCS |
| M9 | `ROADMAP_BACKLOG.md:113` (M3-04) | "…**hiện engine vẫn mock**" | Engine `real` đã có và chọn được bằng `engine_mode=`/`ENGINE_MODE`/`engine.mode` (`ROADMAP:85`, `backend/configs/services.yaml:19`) | "engine có chế độ `real` (POSTER chạy được; MA3D-Net trả `codes_missing` vì thiếu stats); còn thiếu EMOCA thật" | DOCS |
| M10 | `ROADMAP_BACKLOG.md:25` (M0-03 AC3); `.claude/agents/01_AI.md:31`; `PROJECT_CONTEXT.md:71` | Bộ patch gồm `np.int`, `LandmarksType`, `FaceVideoDataModule`, `io.py`; đường dẫn `gdl_apps/utils/io.py` | `backend/services/emoca_svc/patches/`: bắt buộc chỉ có `0001` (np.int), `0002` (LandmarksType); FaceVideoDataModule nằm ở `optional/0101`, `0102` (dành cho demo video, `CLAUDE.md:38`); **không có patch `io.py`**. `backend/model/emoca/gdl_apps/utils/io.py` không tồn tại; đường đúng là `gdl_apps/EMOCA/utils/io.py` (`PROJECT_CONTEXT.md:61`, `EMOCA.md:51`) | Ghi rõ cách chia bắt buộc/tuỳ chọn, bỏ hoặc giải thích patch `io.py`, sửa đường dẫn | DOCS (ROADMAP), GENERAL (01_AI), người dùng (PROJECT_CONTEXT) |
| M11 | `documents/docs/RUNBOOK_DEMO.md:267` (§7, hàng POSTER) | "Đo chỉ-đọc 1 ảnh… **~877 ms CPU** (Session 08)" | Nợ kỹ thuật `SESSION_LOG.md:74` (Session 09): độ trễ "không ổn định (877 ms lúc rảnh, ~5.4 s lúc thiếu RAM) ⇒ **không tuyên bố số nào**"; bất biến `CLAUDE.md` §1.8 | Bỏ con số, hoặc ghi "877 ms – 5.4 s, chưa đo p50/p95 (M4-01)" | DOCS |
| M12 | `PROJECT_CONTEXT.md:20`; `ROADMAP_BACKLOG.md:145` (M5-03) | Attention map ở cột "Để sau" / M5 | Đã hiển thị: `ROADMAP:101` ("radar + attention"), ADR-013 (`feature_maps` + `attention`), `RUNBOOK_DEMO.md:267` (POSTER trả attention) | Chuyển attention sang phạm vi v1 (đã có, chưa đo đúng); M5-03 ghi "làm một phần ở ADR-013" | Người dùng; DOCS |
| M13 | `PROJECT_CONTEXT.md:85, 87, 142-143, 145` | §4: POSTER "**chưa xác minh**", checkpoint "không có trong bản clone"; §7 Q2/Q3 hỏi `raf_ori.pth`, `raf_MA.pth`; Q5 mặc định "HTML+JS thuần" | ADR-011 cập nhật `CONTRACTS_ADR.md:195-199` (đã có bằng chứng trên checkpoint); 10 checkpoint đã có (`CLAUDE.md:35`); ADR-012 (Streamlit) đã trả lời Q5 | Cập nhật §4/§7: Q2 → "đã có bằng chứng, chờ chốt tiêu chí"; Q3 → "chỉ còn thiếu `rafdb_emoca_stats.npz`"; Q5 → "đã chốt Streamlit (ADR-012)" | Người dùng |

## LOW

| # | file:dòng | Câu (rút gọn) | Mâu thuẫn với | Đề xuất | Chủ |
|---|---|---|---|---|---|
| L1 | `.claude/agents/GENERAL.md:42`; `01_AI.md:3`; `CONTRACTS_ADR.md:206` | env `work38_cu11` | Notebook và server dùng `work38` (26 lần trong `another/notebooks/emoca_colab.ipynb`, `backend/services/emoca_svc/server.py`, `emoca_runner.py`; `CONTRACTS_ADR.md:254`) | Thống nhất một tên là `work38` | GENERAL |
| L2 | `ROADMAP_BACKLOG.md:104, 108`; `PROJECT_CONTEXT.md:92, 126-135`; `CONTRACTS_ADR.md:192` | Vai trò đánh số cũ: `06`, `05`, cột "Chủ" 01–06, "`05` ghi" | Cơ cấu 6 vai trò mới (`CLAUDE.md:62`) | Đổi sang tên vai trò mới (`GENERAL`, `01_AI`…) | DOCS / người dùng / GENERAL |
| L3 | `.claude/agents/04_FRONTEND.md:19, 39` | "chỉ gọi `backend/web/orchestrator.analyze(...)`" | Cùng file, `:10`: còn gọi `emoca_status`, `check_emoca` | "chỉ gọi hàm của `backend/web/orchestrator.py`" | GENERAL |
| L4 | `CONTRACTS_ADR.md:206` | `emoca_svc`… "cổng riêng, ví dụ 8001" | `backend/configs/services.yaml:13`: `port: 8765` | Ghi 8765 hoặc "cổng trong `services.yaml`" | GENERAL |
| L5 | `another/tests/run_all.py` (nhóm của `check_ma3d_checkpoints.py`) so với `RUNBOOK_DEMO.md:216` | `--list` xếp test này vào nhóm "nhẹ" | RUNBOOK: "đọc ~2.5 GB, 1–3 phút" | Ghi chú: "nhẹ" = không nạp torch/RAM, nhưng đọc đĩa nặng | GENERAL (tests) |
| L6 | `ROADMAP_BACKLOG.md:19-20` (M0-02) | Đích `backend/services/emoca_svc/env/`; AC "máy Windows + WSL2 sạch" | Ghi chú Session 08 `:22` đã thay bằng notebook Colab, nhưng hai dòng gốc chưa gạch; thư mục `env/` không tồn tại | Gạch dòng gốc hoặc thêm "(thay bằng `:22`)" ngay trên dòng AC | DOCS |
| L7 | `ROADMAP_BACKLOG.md:98` (M3-01) | AC(2) mock/real "chờ `GENERAL` quyết" | Đã có `resolve_emoca_settings`/`resolve_engine_mode` (`backend/web/orchestrator.py:9-13`), cảnh báo `engine_real:`/`mock_ai:` (`RUNBOOK_DEMO.md:273`) — **không chắc** như vậy đã đủ AC(2) | GENERAL xem lại; tick nếu đạt | GENERAL/DOCS |
| L8 | `CLAUDE.md:47` | `mock_data.py`: "nhãn/registry tạm thời **cho tới khi có backend/configs/ thật**" | `backend/configs/` đã có, chế độ `real` đọc `backend/configs/models.yaml` (`backend/web/engine.py:7`); `mock_data.py` nay chỉ phục vụ chế độ mock | "dữ liệu giữ chỗ cho chế độ mock (chế độ real đọc `backend/configs/`)" | DOCS |
| L9 | `backend/web/mock_data.py:16-18` (chú thích) | Nhãn RAF-DB "phải do 02_CONFIG_METRICS xác nhận ở M2-01" | `ROADMAP:61`: nhãn RAF-DB đã xác minh bằng 2 nguồn, ghi vào `backend/configs/models.yaml` | Trỏ tới `backend/configs/models.yaml`; tên hằng `RAFDB_LABELS_UNVERIFIED` giữ cho mock | 01_AI |

---

## Quét đường dẫn tự động (mục 2)

Script tạm `scan_paths.py`, đặt trong thư mục scratchpad của phiên, **không** đưa vào repo. Nó lấy mọi token dạng đường dẫn (có `/` hoặc đuôi `.py/.md/.yaml/.json/.ipynb/.npz/.pth/...`) trong backtick của `.md`/ô notebook, và trong docstring/chú thích của `.py`/`.yaml`/`.txt`. Mỗi token được kiểm lần lượt so với: gốc dự án, thư mục của file chứa nó, `backend/model/emoca`, `backend/model/ma3d`, `backend/services/emoca_svc`, `web`, `tests`, `contracts`, `notebooks`. Tên trần (không có `/`) thì tìm ở bất kỳ đâu trong repo. Với đường dẫn upstream dưới `backend/model/`, chỉ kiểm phần đầu.

- **Thô:** 346 token không tìm thấy, gồm cả SESSION_LOG. Phần lớn là nhiễu: cặp chữ có dấu `/` ("mean/std", "PASS/FAIL"), URL/endpoint (`/v1/analyze`), tên repo GitHub, mảnh tiếng Việt.
- **Sau khi lọc tay, ngoài SESSION_LOG** (SESSION_LOG là lịch sử, không tính):

| Nhóm | Token (file:dòng) | Kết luận |
|---|---|---|
| **Sai thật** | `gdl_apps/utils/io.py` (`01_AI.md:31`, `PROJECT_CONTEXT.md:71`) | Sai đường dẫn ⇒ M10 |
| **Sai thật** | `backend/services/emoca_svc/env/` (`ROADMAP_BACKLOG.md:19`) | Kế hoạch cũ, đã thay ⇒ L6 |
| Tên checkpoint cũ | `raf_ori.pth` ×12, `raf_MA.pth` ×5, `ir50_o1.pth` ×4, `mobilefacenet_model_best.pth_o1.tar` ×3 (`01_AI.md:44,72`, `PROJECT_CONTEXT.md:85,88,142,143`, `CONTRACTS_ADR.md:190-192`, `ROADMAP:68-69`) | Tên theo log/mã gốc; file thật mang tên khác (`models.yaml` có ánh xạ + `_BACKBONE_REDIRECT` trong `backend/web/engine.py:49`). Riêng các chỗ mô tả **trạng thái hiện tại** ⇒ M7, M8, M13 |
| Chưa sinh (đúng kỳ vọng) | `documents/docs/qa/M0/`, `documents/docs/qa/M2/`, `documents/docs/qa/<milestone>/`, `rafdb_emoca_stats.npz`, `*_emoca_stats.npz`, `analyze_response_sample.json`, `emoca_colab_results.zip`, `another/notebooks/emoca_svc_bundle.zip`, `env_work38.yml`, `pip_freeze_work38.txt`, `measurements.json` | Sản phẩm của notebook/M0/M2, chưa chạy; không phải lỗi |
| Tài sản upstream tải sau | `assets/EMOCA/models`, `assets/FLAME/geometry/`, `FLAME_albedo_from_BFM.npz`, `backend/model/settings` (`EMOCA.md:142,197,245`) | Tải bởi `download_assets.sh`; không phải lỗi |
| Nhắc có chủ ý | `backend/web/backend`, `backend/web/frontend`, `backend/services/ma3d_svc/` (`CLAUDE.md:41,62`, `CONTRACTS_ADR.md:204`) | Đều nằm trong câu "đã bỏ/thay thế" ⇒ đúng. `test.py` (`PROJECT_CONTEXT.md:94`) nói rõ là file không tồn tại ⇒ đúng |
| Phương án bị loại | `backend/model/poster/` (`CONTRACTS_ADR.md:192`), `.streamlit/config.toml` (`CONTRACTS_ADR.md:238`) | Nằm trong "phương án (B)" / "Loại bỏ" ⇒ đúng |
| Mẫu/giữ chỗ | `backend/contracts/web_analyze.example.*` (bị cắt ở `*`), `file.npz`, `cfg.yaml`, `another/tests/test_` | Minh hoạ ⇒ bỏ qua |

## Đối chiếu số ca test (mục 3)

`python another/tests/run_all.py --list` → 12 test (11 nhẹ + 1 nặng), khớp `RUNBOOK_DEMO.md:199` ("11 PASS, 1 SKIP"). Đã chạy lại ngày 2026-09-24 trên python hệ thống (đếm dòng `ok`), tất cả ALL PASS, exit 0:

| Test | Đo lại | Tài liệu nói | Khớp |
|---|---|---|---|
| `test_ai_mock.py` | 18 | 18 (`RUNBOOK:210`) | ✔ |
| `test_orchestrator.py` | 21 | 21 (`RUNBOOK:209`); 19 ở `SESSION_LOG.md:95` là số của Session 08 (lịch sử) | ✔ |
| `test_emoca_svc_mock.py` | 27 | 27 | ✔ |
| `test_emoca_svc_real.py` | 52 | 52 (`SESSION_LOG.md:95`) | ✔ |
| `test_emoca_client_remote.py` | 47 | 47 | ✔ |
| `test_theme.py` | 10 | 10 | ✔ |
| `test_colab_bundle.py` | 52 | 52 | ✔ |

**Không chạy** (máy thiếu RAM hoặc cần venv): `test_app_ui.py` (tài liệu nói 11 ca), `test_engine_poster_real.py`, `check_ma3d_checkpoints.py` (đọc ~2.5 GB), `test_configs.py --sha256`. `test_configs.py` mặc định và `check_emoca_stats.py --selftest` đã chạy ở việc trước: ALL PASS / PASS 8 ca.

## Thuật ngữ (mục 4)

- **Biến môi trường** `EMOCA_MODE`, `EMOCA_URL`, `EMOCA_TOKEN`, `ENGINE_MODE`, `MA3D_VENV_PYTHON`: tên khớp giữa code (`os.environ.get` trong `backend/web/`, `another/tests/run_all.py`), `backend/configs/services.yaml:5-6`, `RUNBOOK_DEMO.md` và `03_BACKEND.md:10`. Không lệch.
- **Mã lỗi:** enum trong `backend/contracts/emoca_analyze.schema.json` (`image_invalid, image_too_large, no_face, service_unavailable, timeout, model_unavailable, codes_missing, contract_mismatch`) khớp các file vai trò. `consent_required` chỉ có ở mức `web_analyze` (trường `code` là chuỗi tự do), đúng như `CONTRACTS_ADR.md:262`. `03_BACKEND.md:29` và `04_FRONTEND.md:30` liệt `partial` chung hàng với mã lỗi, trong khi `partial` là giá trị của `status`. Việc này rất nhỏ nên không tách thành mục riêng.
- **Tên env:** xem L1. **Tên vai trò:** xem L2. **Khoá config:** xem M4.

## Điều chưa chắc

1. L7 (M3-01 AC2): chưa đọc đủ `frontend/app.py` để khẳng định giao diện đã hiện rõ trạng thái mock/real của từng phần.
2. M10: không tìm thấy ghi chép nào giải thích vì sao patch `io.py` bị bỏ. Có thể nó chỉ cần cho demo video. Cần `01_AI` xác nhận.
3. Chưa rà hết `documents/docs/modules/EMOCA.md` (400 dòng): chỉ quét từ khoá và đường dẫn. Các cụm "Colab/Linux/WSL là lựa chọn thực tế" (`:313`) là phân tích kỹ thuật, không phải quyết định, nên không tính là mâu thuẫn.
4. Mức HIGH của H3 và H4 là đánh giá của người rà (hai câu này dẫn người đọc làm sai), không phải số đo.

---

## Trạng thái xử lý (2026-09-24, cùng phiên rà, theo giao việc của `btl-f9`)

**Nguyên tắc:** các file `.claude/agents/*.md` là chỉ dẫn/cấu hình cho Claude, cùng loại với `CLAUDE.md`. Phiên này **không sửa** chúng khi yêu cầu đến từ một phiên khác. Mọi mục chạm tới các file đó ghi là "chờ người dùng" (người dùng tự sửa, hoặc cho phép một phiên sửa trực tiếp).

| ID | Trạng thái | file:dòng mới / ghi chú |
|---|---|---|
| H1 | chờ người dùng | `PROJECT_CONTEXT.md` — không sửa theo giao việc |
| H2 | chờ người dùng | `.claude/agents/03_BACKEND.md:10` — file vai trò; đề xuất thêm "(TẠM, chờ người dùng xác nhận phạm vi — ADR-015, ROADMAP M4-05; CLAUDE.md §1.1 vẫn cấm)". `01_AI.md`/`04_FRONTEND.md` chưa rà câu tương tự |
| H3 | đã sửa | `backend/configs/services.yaml:20` (chỉ chú thích → `environments/web/requirements-real.txt`) |
| H4 | sửa một phần | ROADMAP M2-05 AC5 `documents/docs/context/ROADMAP_BACKLOG.md:81` thêm "(ADR-016: singleton mức module, nạp lười từng mô hình)". `01_AI.md:61` **chờ người dùng** (file vai trò) |
| M1 | chờ người dùng | `CLAUDE.md` §6 |
| M2 | bỏ qua (giao cho `btl-f9`) | `CLAUDE.md` §2 — phiên điều phối tự cập nhật bản đồ |
| M3 | đã sửa | `documents/docs/context/CONTRACTS_ADR.md:4` (contract hiện hành 0.2.0; trạng thái ADR ghi ở tiêu đề; ví dụ Phần B là bản 0.1.0) |
| M4 | đã sửa | `CONTRACTS_ADR.md:127` (tên vai trò mới), `:133` (khớp `backend/configs/services.yaml` thật; câu "Mục dịch vụ MA3D riêng đã bỏ theo ADR-012") |
| M5 | chờ người dùng | trạng thái Accepted/Proposed của ADR-011/013/014/016 |
| M6 | chờ người dùng | `.claude/agents/01_AI.md:44` — file vai trò |
| M7 | chờ người dùng | `.claude/agents/01_AI.md:72` — file vai trò; đề xuất "checkpoint theo `backend/configs/models.yaml`; thiếu duy nhất `rafdb_emoca_stats.npz` (kiểm bằng `another/tests/check_emoca_stats.py`)" |
| M8 | đã sửa | ghi chú mới ở `ROADMAP_BACKLOG.md:73` (Session 09 đã chạy thật POSTER; chờ người dùng chốt tiêu chí ADR-011) |
| M9 | đã sửa | ghi chú mới ở `ROADMAP_BACKLOG.md:117` |
| M10 | sửa một phần | ghi chú mới ở `ROADMAP_BACKLOG.md:27` (0001/0002 bắt buộc, 0101/0102 tuỳ chọn, không có patch `io.py`). `01_AI.md:31` **chờ người dùng**; `PROJECT_CONTEXT.md:71` chỉ ghi ở báo cáo này |
| M11 | đã sửa | `documents/docs/RUNBOOK_DEMO.md:267` (bỏ "~877 ms" làm số chính; ghi dải 877 ms – ~5.4 s và số đầu-cuối của `documents/docs/qa/E2E_ENGINE_REAL_UI.md`: nguội ≈ 8.6 s, ấm ≈ 0.9–1.2 s; chưa có p50/p95) |
| M12, M13 | chờ người dùng | `PROJECT_CONTEXT.md`; phần ROADMAP M5-03 của M12 không nằm trong danh sách được giao |
| L1 | sửa một phần | `CONTRACTS_ADR.md:206` → `work38`. `GENERAL.md:42`, `01_AI.md:3` **chờ người dùng**. Phát hiện thêm: README upstream `backend/model/emoca/README.md:144,178` dùng `work38_cu11`, đó là nguồn gốc của tên này; notebook dự án dùng `work38` |
| L2 | bỏ qua | quá rộng, để sau (theo `btl-f9`) |
| L3 | chờ người dùng | `.claude/agents/04_FRONTEND.md:19,39` — file vai trò |
| L4 | đã sửa | `CONTRACTS_ADR.md:206` (cổng = `emoca_svc.port`, hiện 8765) |
| L5 | đã sửa | `documents/docs/RUNBOOK_DEMO.md:216` |
| L6 | đã sửa | `ROADMAP_BACKLOG.md:19` (dòng ghi chú ngay trên hai dòng gốc) |
| L7 | bỏ qua | `GENERAL` sẽ xem |
| L8 | bỏ qua (giao cho `btl-f9`) | `CLAUDE.md` §2 |
| L9 | đã sửa | `backend/web/mock_data.py:16-18` (chỉ chú thích; giữ tên hằng) |

**Tổng:** đã sửa 10 (H3, M3, M4, M8, M9, M11, L4, L5, L6, L9) · sửa một phần 3 (H4, M10, L1) · chờ người dùng 9 (H1, H2, M1, M5, M6, M7, M12, M13, L3; thêm phần còn lại của H4/M10/L1) · bỏ qua 4 (M2, L2, L7, L8).

**Kiểm sau khi sửa (2026-09-24, máy này):** `python another/tests/test_configs.py` → ALL PASS (exit 0). `python another/tests/run_all.py` → **11 PASS, 1 SKIP (nhóm nặng), 0 FAIL / 12 test, 38.8 giây**, exit 0. Grep: Phần C của `CONTRACTS_ADR.md` không còn `ma3d_svc` (chỉ còn câu "đã bỏ theo ADR-012"); `work38_cu11` chỉ còn ở `01_AI.md:3`, `GENERAL.md:42` (chờ người dùng) và upstream `backend/model/emoca/README.md`; `st.cache_resource` còn 1 lần ở `01_AI.md` (chờ người dùng).
**Số ca đã đổi so với lúc rà:** trong lần chạy này, `test_orchestrator.py` ra 22 ok (lúc rà là 21) và `test_app_ui.py` ra 12 (RUNBOOK ghi 11). Có thể do các phiên khác vừa thêm test; con số "(n ca)" ở `RUNBOOK_DEMO.md:209,217` cần cập nhật.

## Cập nhật của phiên điều phối (`btl-f9`, cùng ngày)

Phiên xử lý (`btl-c9`) không sửa `.claude/agents/*.md` vì coi là file chỉ dẫn cho Claude và yêu cầu đến từ phiên khác — cách làm thận trọng và hợp lý. Điều phối đã tự kiểm chứng H2, H3, H4 (đúng) rồi **tự áp** các sửa tối thiểu trong file vai trò, theo phán đoán riêng dựa trên bằng chứng của báo cáo và ADR-015/016, không phải vì yêu cầu của phiên khác: **H2** (`03_BACKEND.md:10`, `01_AI.md:10`: ghi rõ đây là biện pháp giảm nhẹ TẠM, còn chờ người dùng xác nhận phạm vi — M4-05; `CLAUDE.md` §1.1 vẫn cấm), **H4** (`01_AI.md:61` → singleton mức module, không dùng `st.cache_resource`), **M6** (`01_AI.md:44` → trỏ tới tiêu chí ADR-011 đã cập nhật), **M7** (`01_AI.md:72` → checkpoint theo `backend/configs/models.yaml`, còn thiếu duy nhất `*_emoca_stats.npz`), **M10** (`01_AI.md:31` → đường dẫn `io.py` và phân loại patch bắt buộc/tuỳ chọn), **L1** (`01_AI.md:3`, `GENERAL.md:42` → `work38`), **L3** (`04_FRONTEND.md:19,39`). Còn **chờ người dùng**: H1, M1, M5, M12, M13 (các file của người dùng; ADR-011/013/014/016 cần người dùng chốt trạng thái). Số ca test trong `RUNBOOK_DEMO.md` đã cập nhật (orchestrator 22, app_ui 12).

## Trạng thái áp dụng 2026-10-04

Đối chiếu báo cáo với `PROJECT_CONTEXT.md`, `CLAUDE.md` (§1.1/§2), toàn bộ tiêu đề/nội dung ADR, configs và code có trong working copy. Chỉ sửa `PROJECT_CONTEXT.md`, `CLAUDE.md` và báo cáo này theo phạm vi được phép. Working copy không chứa `.claude/`, `backend/model/`, notebook hoặc các artifact weights; khi xác minh artifact dùng cấu hình/hash và ghi nhận hiện có trong tài liệu nguồn, không giả định các file đó nằm trong bản copy.

| ID | Quyết định | Căn cứ / xử lý ngày 2026-10-04 |
|---|---|---|
| H1 | APPLIED-ALREADY | `PROJECT_CONTEXT.md` §§1–2 đã mô tả Streamlit `backend/web/` chạy POSTER/MA3D-Net trên CPU và chỉ EMOCA tách qua HTTP; ADR-012/018 xác nhận WSL2 là đường chính, Colab dự phòng. Dữ liệu prediction/attention ở §5 lấy từ engine trong `backend/web/`. |
| H2 | APPLIED-ALREADY | Phần cập nhật điều phối ngày 2026-09-24 trong báo cáo ghi đã thêm caveat TẠM/chờ xác nhận vào hướng dẫn vai trò. `CLAUDE.md` §1.1 vẫn cấm gửi ảnh ra dịch vụ bên thứ ba; ADR-015 chỉ là fallback và nêu rõ phạm vi ngoại lệ còn chờ người dùng. Thư mục `.claude/` không có trong working copy. |
| H3 | APPLIED-ALREADY | Chú thích `backend/configs/services.yaml` hiện trỏ `environments/web/requirements-real.txt`; đối chiếu file requirements cho thấy đây là bộ cài torch CPU. |
| H4 | APPLIED-ALREADY | Báo cáo ghi điều phối đã sửa chỉ dẫn vai trò. `backend/web/engine.py` xác nhận singleton mức module, nạp lười, không dùng `st.cache_resource`; điều này khớp ranh giới module của ADR-016. |
| M1 | APPLY-NOW | Sửa `CLAUDE.md` §6 để nêu `python another/tests/run_all.py`, `--heavy` và `--list`; ghi cập nhật 2026-10-04. Không chạy test theo yêu cầu. |
| M2 | APPLY-NOW | Viết lại `CLAUDE.md` §2 theo cây thật của working copy, bỏ các đường dẫn không tồn tại như `backend/model/`, `.claude/`, `another/notebooks/`, `another/kaggle/`, `frontend/assets/theme/`, `template_claude/` và script chạy demo; ghi rõ các module mới đang có. CodeFormer đã bị loại theo ADR-027. |
| M3 | APPLIED-ALREADY | `CONTRACTS_ADR.md` hiện ghi contract `0.3.0` theo ADR-019/020, nêu từng ADR có trạng thái tại tiêu đề và ví dụ Phần B là bản rút gọn `0.1.0`; schema trong working copy cũng dùng `0.3.0`. |
| M4 | APPLIED-ALREADY | Phần C `CONTRACTS_ADR.md` đã mô tả các khóa thực của `services.yaml`, ghi rõ không còn `ma3d_svc` và dùng tên vai trò mới. |
| M5 | OBSOLETE | Việc một ADR đã có triển khai không tự động biến trạng thái thành Accepted. Các tiêu đề ADR-011/013/014/016 vẫn Proposed; ADR-011 còn điều kiện đánh giá chưa đạt trong chính nội dung. Không có quyết định người dùng 2026-10-03/04 chấp nhận đổi trạng thái. Đã làm rõ ở `PROJECT_CONTEXT.md` rằng cách triển khai POSTER được chốt nhưng trạng thái ADR-011 vẫn Proposed. |
| M6 | APPLIED-ALREADY | Phần cập nhật điều phối trong báo cáo ghi `01_AI.md` đã được sửa để trỏ tới tiêu chí strict-load và `mm_fusion` chưa huấn luyện của ADR-011. `.claude/` không có trong working copy. |
| M7 | OBSOLETE | Đề xuất “còn thiếu stats” đã lỗi thời: `backend/configs/models.yaml` khai báo artifact RAF-DB `rafdb_emoca_stats.npz` với hash; ROADMAP ghi đã có từ 2026-09-25, và `PROJECT_CONTEXT.md` §7 Q3 xác nhận. Demo chỉ dùng RAF-DB theo quyết định Session 41. |
| M8 | APPLIED-ALREADY | ROADMAP đã có ghi chú lịch sử M2-03 theo M8 của báo cáo; M2-05 xác nhận POSTER chạy thật. Tiêu chí ADR-011 còn mở nên chưa đánh dấu hoàn tất. |
| M9 | APPLIED-ALREADY | ROADMAP đã ghi chú engine có chế độ thật; `backend/configs/services.yaml` và `backend/web/orchestrator.py` xác nhận chọn chế độ qua config/biến môi trường. |
| M10 | APPLY-NOW | Sửa `PROJECT_CONTEXT.md` §3: chỉ `0001`/`0002` bắt buộc, `optional/0101`/`0102` dành cho demo video, bỏ patch `io.py` và sửa đường dẫn upstream thành `gdl_apps/EMOCA/utils/io.py`. ROADMAP và chỉ dẫn vai trò đã có ghi nhận cập nhật trong trạng thái xử lý trước đây của báo cáo. |
| M11 | APPLIED-ALREADY | Trạng thái xử lý trước đây ghi RUNBOOK đã thay tuyên bố latency một số bằng dải đo và ghi rõ chưa có p50/p95; không sửa RUNBOOK theo giới hạn file hiện tại. |
| M12 | APPLY-NOW | `PROJECT_CONTEXT.md` §1 nay liệt kê attention/feature maps trong phạm vi hiện tại, dẫn ADR-013; không chỉnh ROADMAP vì ngoài phạm vi file được phép. |
| M13 | APPLIED-ALREADY | `PROJECT_CONTEXT.md` §7 đã ghi Q2 có kết quả kiểm chứng/triển khai, Q3 xác nhận RAF-DB và stats, Q5 chốt Streamlit theo ADR-012. |
| L1 | APPLIED-ALREADY | Trạng thái xử lý trước đây ghi `CONTRACTS_ADR.md` và hướng dẫn vai trò đã thống nhất `work38`; ADR-012 cũng nêu tên env EMOCA `work38`. Thư mục `.claude/`/notebook upstream không có trong working copy. |
| L2 | APPLY-NOW | Đổi các chủ sở hữu số cũ trong bảng rủi ro của `PROJECT_CONTEXT.md` sang `01_AI`, `02_CONFIG_METRICS`, `03_BACKEND`, `04_FRONTEND`; ghi nguồn CLAUDE §4. Những chỗ ở ROADMAP/CONTRACTS_ADR ngoài danh sách file cho phép không sửa. |
| L3 | APPLIED-ALREADY | Phần cập nhật điều phối trong báo cáo ghi `04_FRONTEND.md` đã đổi sang “hàm của `backend/web/orchestrator.py`”. Thư mục `.claude/` không có trong working copy. |
| L4 | APPLIED-ALREADY | ADR-012 trong `CONTRACTS_ADR.md` đã dẫn cổng từ `emoca_svc.port` trong config (8765), không còn ví dụ cổng 8001. |
| L5 | APPLIED-ALREADY | Trạng thái xử lý trước đây ghi RUNBOOK đã làm rõ test checkpoint đọc đĩa nặng dù không nạp torch/RAM. |
| L6 | APPLIED-ALREADY | Trạng thái xử lý trước đây ghi ROADMAP đã đánh dấu kế hoạch `backend/services/emoca_svc/env/` được thay bằng notebook/Colab. Trong working copy, `backend/services/emoca_svc/env/` không tồn tại. |
| L7 | APPLIED-ALREADY | `backend/web/orchestrator.py` có `resolve_engine_mode`; `frontend/app.py` hiển thị cảnh báo theo `engine_real:`/`mock_ai:` và RUNBOOK mô tả trạng thái từng phần. Điều kiện lựa chọn chế độ và thông báo hiện đã có bằng chứng trong code/UI. |
| L8 | APPLIED-ALREADY | `CLAUDE.md` §2 mô tả `mock_data.py` là dữ liệu giữ chỗ cho chế độ mock và real đọc `backend/configs/`; `backend/web/engine.py` đọc `backend/configs/models.yaml`. |
| L9 | APPLIED-ALREADY | `backend/web/mock_data.py` ghi nhãn `RAFDB_LABELS_UNVERIFIED` chỉ dùng cho mock và nói nhãn thật đã xác minh nằm ở `backend/configs/models.yaml`. |

