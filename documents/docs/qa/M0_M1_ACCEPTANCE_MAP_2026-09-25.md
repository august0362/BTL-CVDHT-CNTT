# Bản đồ nghiệm thu M0/M1 — AC → bằng chứng từ script chạy EMOCA thật — 2026-09-25

- **Vai trò:** QA/DOCS, giao việc từ phiên điều phối `btl-e6`, để **tick nhanh sau smoke test**. Bản đồ này **không tick** và **không có số đo** — nó chỉ nói *số nào / tệp nào* sẽ chứng minh AC nào.
- **Nguồn đã đọc (2026-09-25, bản hiện tại):** AC nguyên văn ở `documents/docs/context/ROADMAP_BACKLOG.md` (M0-02..M0-06, M1-01, M1-03, M4-04, M4-05); `backend/services/emoca_svc/wsl/smoke_test_emoca.sh` (339 dòng), `run_emoca_server.sh` (99 dòng), `setup_emoca_wsl.sh`, `common.sh`, `another/tests/e2e_wsl_emoca.py` (220 dòng), `backend/services/emoca_svc/emoca_runner.py` (các trường `/v1/health`), `backend/services/emoca_svc/patches/`.
- **Thư mục kết quả:** `OUT = ~/emoca_wsl/out` (`common.sh:48`). Mô hình mặc định `MODEL_NAME = EMOCA_v2_lr_mse_20` (`common.sh:28`).
- Ký hiệu cột "Đo được": **Đủ** = script sinh đủ bằng chứng · **Một phần** = còn AC con phải đo thêm · **Không** = không script nào đo.

## 0. Lưu ý trước khi chạy (để không mất bằng chứng)

1. **Đầu ra màn hình của `smoke_test_emoca.sh` KHÔNG được ghi ra tệp.** `log()` chỉ in ra màn hình (`common.sh:34`). Các dòng kết luận quan trọng chỉ nằm trên stdout: "M0-04: n/n anh du 5 PNG + 5 NPY…" (`smoke_test_emoca.sh:138-144`), "PASS: dung emoca_analyze.schema.json…" (`:285`), "PASS: khong nap lai mo hinh; anh xam -> no_face 422" (`:311`), "Mien landmark chuan hoa…" (`:315`), "Smoke test DAT" (`:339`). **Khuyến nghị:** chạy `bash ~/emoca_wsl/bin/smoke_test_emoca.sh 2>&1 | tee ~/emoca_wsl/out/smoke_$(date +%Y%m%d_%H%M%S).log` (hoặc chạy nền và giữ tệp output). Mã thoát 0 của smoke = mọi `assert`/`die` đều qua.
2. `another/tests/e2e_wsl_emoca.py` cũng chỉ in ra màn hình (thoát 0 = PASS, `:11`, `:136-140`) — giữ đầu ra lại (chuyển hướng ra tệp trong `documents/docs/qa/` hoặc scratch, **không** kèm ảnh).
3. `smoke_test_emoca.sh` **dừng** nếu cổng 8765 đang có server (`:80-82`) ⇒ chạy smoke **trước**, rồi mới `run_emoca_server.sh`, rồi mới `e2e_wsl_emoca.py`.

## 1. Bảng AC → bằng chứng → ngưỡng

### M0-02 · Env EMOCA thành script tái lập — **Một phần**

AC gốc: *"(1) từ một máy Windows + WSL2 sạch, chạy script tới hết mà không phải sửa tay ngoài danh sách tài sản tải tay đã ghi; (2) ghi thời gian dựng thực tế; (3) `conda list`/`pip freeze` được lưu."* (ghi chú Session 08 cho đường Colab; ADR-018: WSL2 là đường chính, nghiệm thu ở M0-06).

| AC | Bằng chứng | Ngưỡng đạt |
|---|---|---|
| (1) chạy tới hết không sửa tay | `setup_emoca_wsl.sh` (lần có cờ giấy phép) thoát **0**; log `OUT/setup_<ngày_giờ>.log` (lần đầu với cờ: `setup_20260925_213604.log`, theo điều phối) | mã 0, không có `[LOI]`. **Cần đo thêm / cần quyết:** trong ngày đã có can thiệp tay (xoá Miniconda 0 byte sau sự cố 1; tạo `.wslconfig`; bật pagefile) — đó là sửa môi trường máy, không phải sửa script; `GENERAL` quyết có tính là "sửa tay" không. "Máy **sạch**": chưa thử trên máy thứ hai |
| (2) thời gian dựng | `OUT/timings.tsv` (một dòng/chặng, `common.sh:53`); thêm `OUT/prefetch_timings.tsv` cho tải tài sản (`prefetch_assets.sh:30`) | có số cho mọi chặng. Đã có 10/11 chặng không cần giấy phép (thiếu dòng `pull_submodules`, có số 572 s ở SESSION_LOG Session 11); **cần** các dòng tài sản + texture |
| (3) `conda list`/`pip freeze` | `OUT/env_work38.yml`, `OUT/pip_freeze_work38.txt` (`setup_emoca_wsl.sh:474-476`, gọi ở `:485`, `:557`) | hai tệp tồn tại, không rỗng |

### M0-03 · Bộ patch cho EMOCA — **Một phần**

AC: *"(1) `git status` trong `backend/model/emoca/` sạch khi chưa áp patch; (2) áp patch xong demo chạy được; (3) mỗi patch có một dòng lý do (`np.int`, `LandmarksType`, `FaceVideoDataModule`, `io.py`); (4) có script gỡ patch."*

| AC | Bằng chứng | Ngưỡng đạt |
|---|---|---|
| (1) sạch khi chưa áp | **Không script nào đo.** Cần đo thêm: trong WSL `bash <dự án>/backend/services/emoca_svc/patches/revert.sh ~/emoca` rồi `git -C ~/emoca status --porcelain --ignore-submodules=all` | đầu ra rỗng (trừ `assets/`, `data/` không theo dõi); sau đó `apply.sh` lại |
| (2) áp xong demo chạy | Smoke phần A: `test_emoca_on_images.py` chạy trên `~/emoca` đã áp 0001+0002 (`setup_emoca_wsl.sh:324-326`; `smoke_test_emoca.sh:99-119`) | dòng "M0-04 (1)-(3) DAT" (`:144`) |
| (3) lý do từng patch | Tĩnh, đã đọc: header `Ly do:` ở `patches/0001-…patch:4-5` (np.int), `0002-…patch:4-5` (LandmarksType); `optional/0101`, `0102` (FaceVideoDataModule) | mỗi patch có dòng `Ly do:`. **Không có patch `io.py`** — lý do chưa ghi (ROADMAP ghi chú M0-03) ⇒ `01_AI` cần ghi một câu |
| (4) script gỡ | Tĩnh: `patches/revert.sh` có | **cần chạy thật** một lần (cùng bước AC (1)) |

### M0-04 · Smoke test EMOCA trên 1 ảnh — **Một phần**

AC: *"(1) chạy `test_emoca_on_images.py` (hoặc adapter) với `--save_codes True --save_mesh True`; (2) có `inputs.png`, `geometry_coarse.png`, `geometry_detail.png`, `out_im_coarse.png`, `out_im_detail.png`; (3) có `shape/exp/tex/pose/detail.npy` với độ dài 100/50/50/6/128; (4) ảnh dựng nhìn đúng khuôn mặt; (5) lưu làm fixture vàng."*

| AC | Bằng chứng | Ngưỡng đạt |
|---|---|---|
| (1) chạy demo | `OUT/m0_04_log.txt` (tee đầu ra upstream, `smoke_test_emoca.sh:113`); dòng `m0_04_test_emoca_on_images` trong `timings.tsv` | chạy xong, không có "please check the test path" (`:115-117`). **Lệch AC có chủ đích:** smoke **không** truyền `--save_mesh` (`:101`) ⇒ `GENERAL` quyết chấp nhận (mesh không dùng cho web) hay chạy thêm một lần có `--save_mesh True` |
| (2) 5 PNG | `OUT/m0_04/EMOCA_v2_lr_mse_20/<ảnh>/{inputs,geometry_coarse,geometry_detail,out_im_coarse,out_im_detail}.png`; kiểm ở `:125`, `:129` | dòng "M0-04: n/n anh du 5 PNG + 5 NPY dung do dai" với n = số ảnh (tối đa `SMOKE_MAX_IMAGES` = 5) |
| (3) 5 NPY đúng độ dài | cùng thư mục, `{shape,exp,tex,pose,detail}.npy`; kiểm `np.load(...).size` ở `:126`, `:131-135` | cùng dòng trên; smoke thoát 0 |
| (4) nhìn đúng khuôn mặt | các PNG ở (2) | **cần mắt người** (`:144` tự nhắc) — người dùng xem, ghi lại |
| (5) fixture vàng | **Không script nào lưu.** Cần đo thêm: chọn 1 ảnh mẫu công khai, lưu bộ PNG/NPY (hoặc phản hồi `/v1/analyze`, xem M1-01) vào nơi nhóm chốt | tệp fixture có mặt + ghi nguồn ảnh; **không** commit tài sản có giấy phép |

### M0-05 · Miền landmark và `tform` — **Một phần**

AC: *"(1) đo và ghi miền giá trị của `predicted_landmarks` (`DECA.py:968-971`); (2) vá `ImageTestDataset.py` để trả `tform`/bbox; (3) vẽ landmark lên **ảnh gốc** khớp mắt/mũi/miệng (ảnh minh chứng trong `documents/docs/qa/M0/`)."* (ghi chú Session 08: (2) đổi cách — `EmocaBackend` tự có `tform`.)

| AC | Bằng chứng | Ngưỡng đạt |
|---|---|---|
| (1) miền giá trị | `/v1/health` trường `landmark_norm_range_seen` (`emoca_runner.py:307`); smoke in ở `:315` và lưu trong `OUT/smoke_measurements.json` → `health.landmark_norm_range_seen` (`:316-317`) | có giá trị, nằm trong [-1, 1] (kỳ vọng từ mã) |
| (2) có `tform` | `crop_transform.matrix_3x3` trong phản hồi; `e2e_wsl_emoca.py:78-87` kiểm `matrix_3x3 · original_px ≈ crop224` | dòng "ok … crop_transform·original_px ≈ crop224 (sai lệch lớn nhất … px)", ngưỡng ≤ **1.0 px** (`LANDMARK_TOL_PX`, `:46`) |
| (3) vẽ lên ảnh gốc | **Không script WSL nào vẽ.** Cần đo thêm: web (Cài đặt → Máy chủ EMOCA, bật "Hiện landmark trên ảnh") với một ảnh mặt → chụp màn hình vào `documents/docs/qa/M0/`; hoặc notebook mục 8 (Colab) | ảnh chồng khớp mắt/mũi/miệng (mắt người). Không lưu ảnh mặt người thật vào repo nếu không được phép — dùng ảnh mẫu công khai của EMOCA |

### M0-06 · EMOCA chạy cục bộ qua WSL2 — **Đủ** (AC (4) xem ghi chú)

AC: *"(1) `wsl --install` xong và `nvidia-smi` trong Ubuntu thấy RTX 3050; (2) script dựng môi trường chạy hết, smoke test có 5 PNG + 5 NPY đúng độ dài (như M0-04); (3) VRAM đỉnh khi server chạy đo được (`vram_peak_mb` ở `/v1/health`) và không OOM; (4) web nối `http://127.0.0.1:8765`, ảnh không rời máy."*

| AC | Bằng chứng | Ngưỡng đạt |
|---|---|---|
| (1) | **Đã đo** (ROADMAP ghi chú M0-06: `nvidia-smi -L` thấy RTX 3050) | — |
| (2) | `setup_emoca_wsl.sh` thoát 0 (xem M0-02 (1)) + smoke phần A (xem M0-04 (2)(3)) | cả hai đạt |
| (3) VRAM đỉnh | `OUT/smoke_vram_health.json` (`smoke_test_emoca.sh:154`, `:182-217`) và `OUT/server_vram_health.json` (`run_emoca_server.sh:40`, `:78-80`): `max_vram_peak_mb`, `max_vram_reserved_peak_mb`, `min_gpu_free_mb`, `gpu_total_mb`, `samples` | `samples` > 0 và `max_vram_peak_mb` là **số** (không phải "khong co"). Nên ghi cả `max_vram_reserved_peak_mb` và `min_gpu_free_mb` (sát thực tế hơn — REVIEW H1) |
| (3) không OOM | smoke phần B qua (`:330`), `OUT/smoke_server.log`, `OUT/server.log` | không có "out of memory" / "CUDA error" trong hai log (`grep -ci "out of memory"` = 0); `min_gpu_free_mb` > 0 |
| (4) web nối 127.0.0.1:8765 | (a) chuyển tiếp localhost: `OUT/smoke_forward.txt` (`:62-63`) hoặc `smoke_test_emoca.sh --forward-only` thoát 0; (b) `python another/tests/e2e_wsl_emoca.py --image <ảnh> --url http://127.0.0.1:8765` (đi qua `backend/web/emoca_client.py`, `:96`, `:114`) thoát 0; loopback bắt buộc (`:207-209`) | (a) dòng cuối `DAT`; (b) dòng cuối `PASS`, `mode` ≠ `mock` trong phần health |
| (4) ảnh không rời máy | địa chỉ loopback (e2e từ chối địa chỉ khác, `:207`); server chỉ nghe `127.0.0.1` (`run_emoca_server.sh:98-99`) | theo thiết kế; không cần hộp xác nhận |

*Ghi chú (4):* e2e đi qua đúng module web (`emoca_client`) nhưng **không qua giao diện**. Nếu `GENERAL` hiểu "web nối" là giao diện Streamlit thì **cần đo thêm**: Cài đặt → Kết nối EMOCA → Máy chủ EMOCA `http://127.0.0.1:8765` → "Kiểm tra kết nối" OK + một lần phân tích có cảnh báo `emoca_real:` và MA3D-Net `status: ok` với `codes_real=True` (ADR-017).

### M1-01 · Contract v0.1 thành file — **Một phần** (chỉ còn AC (2))

AC: *"(1) schema validate ví dụ hợp lệ, từ chối ví dụ lỗi; (2) có fixture vàng từ M0-04; (3) `contract_version` được ghi ở mọi schema."* — (1), (3) **đã đo** (Session 03).

| AC | Bằng chứng | Ngưỡng đạt |
|---|---|---|
| (2) fixture vàng | **Không script WSL nào lưu phản hồi đầy đủ**: `smoke_measurements.json` chỉ có `rows` (thời gian) + `health` (`smoke_test_emoca.sh:316-317`); e2e không ghi đĩa (`:14-15`). Cần đo thêm: lưu **một** phản hồi `POST /v1/analyze` trên ảnh mẫu công khai của EMOCA (vd `backend/contracts/emoca_analyze.example.real.json`, cần ADR/`contract_version` giữ nguyên), kèm tên ảnh + commit upstream | tệp validate qua `emoca_analyze.schema.json`; codes 100/50/50/6/128; 68 landmark hai hệ. Việc đổi `backend/contracts/` cần theo luật §1.4 (`GENERAL`) |

### M1-03 · `emoca_svc` thật — **Một phần** (thiếu AC (4) trên EMOCA thật)

AC: *"(1) `POST /v1/analyze` trả response hợp lệ schema với landmark ở cả `crop224` và `original_px`; (2) hai lần gọi liên tiếp không nạp lại mô hình; (3) đo thời gian nạp, độ trễ/ảnh, VRAM; (4) không ghi ảnh ra đĩa; (5) không thấy mặt → `no_face`."*

| AC | Bằng chứng | Ngưỡng đạt |
|---|---|---|
| (1) | smoke phần B: 3 ảnh × 3 lần, `jsonschema.validate` + codes + 68 landmark hai hệ (`:270-285`); e2e (`:120-129`) | dòng "PASS: dung emoca_analyze.schema.json…" (`:285`); e2e `PASS` |
| (2) | `load_seconds` trước = sau (`:305-306`); `requests_served` trong health (`emoca_runner.py:305`) | dòng "PASS: khong nap lai mo hinh…" (`:311`) |
| (3) thời gian nạp | `health.load_seconds` (`emoca_runner.py:497`), in ở `:265`, lưu trong `smoke_measurements.json` → `health` | có số |
| (3) độ trễ/ảnh | `smoke_measurements.json` → `rows[]` (`roundtrip_ms`, `timing_ms`) (`:284`, `:316`); `OUT/latency_cudnn.tsv` (p50/max phía server, p50 khứ hồi, trạng thái cuDNN; `:295-302`); e2e in trung vị khứ hồi (`:132-134`) | có p50 với n ≥ 2 lần ấm (smoke: n = 6). Chỉ là số một máy, **không** suy p95 |
| (3) VRAM | như M0-06 (3) | như M0-06 (3) |
| (4) không ghi ảnh | **Chưa đo trên EMOCA thật** (chỉ đo trên backend giả, `another/tests/test_emoca_svc_real.py`). Cần đo thêm: trước/sau smoke phần B chạy `find ~/emoca_wsl/svc ~/emoca /tmp -newer <mốc> -type f` (hoặc `inotifywait`) — không có tệp ảnh mới (ngoài log) | 0 tệp ảnh mới |
| (5) `no_face` | ảnh xám 320×320 → 422 `no_face` (`:307-311`) | dòng PASS ở `:311` |

### M4-04 AC2 · Bộ kết quả dự phòng — **Không**

AC: *"… có bộ kết quả mẫu lưu sẵn khi Colab không sẵn sàng."*

- **Không script nào tạo.** Cần đo thêm (một lần, sau smoke): lưu một phản hồi `/v1/analyze` + các PNG M0-04 của **ảnh mẫu công khai EMOCA** (cùng việc với M1-01 (2), M0-04 (5)) thành một bộ web đọc được khi EMOCA không chạy. Cần `GENERAL` chốt nơi lưu và việc có commit hay không (ảnh dựng từ tài sản có giấy phép FLAME ⇒ **không** commit nếu chưa rõ giấy phép).

### M4-05 · Nợ `[DEBT-HIGH]` riêng tư của ADR-015 — **Đủ** (qua M0-06)

AC: *"một trong hai — (a) chuyển EMOCA về chạy cục bộ (WSL2 …) để ảnh không còn rời máy; hoặc (b) …. Kèm: đo hành vi khi Colab bị ngắt giữa lúc đang xử lý."*

- (a) đạt khi **M0-06 đạt** (đặc biệt AC (4): loopback, ảnh không rời máy).
- Phần "Kèm: đo hành vi khi Colab bị ngắt" gắn với đường Colab; khi dùng (a) thì `GENERAL` quyết còn áp dụng không (Colab chỉ còn dự phòng). Nếu cần giữ: tắt `run_emoca_server.sh` (Ctrl+C) giữa lúc web đang phân tích → web phải trả `partial`, không hỏng trang — **cần đo thêm**.

## 2. Tóm tắt

| Mức | Task |
|---|---|
| **Đủ** (2) | M0-06 (nếu chấp nhận e2e qua `emoca_client` là "web nối"), M4-05 (qua M0-06) |
| **Một phần** (6) | M0-02 (máy sạch / "không sửa tay"), M0-03 (chạy `revert.sh` + `git status`), M0-04 (`--save_mesh`, mắt người, fixture), M0-05 (ảnh chồng), M1-01 (fixture vàng), M1-03 (không ghi đĩa trên EMOCA thật) |
| **Không** (1) | M4-04 AC2 |

Ba việc đo thêm rẻ nhất, gộp được vào **một** lượt sau smoke: (i) `tee` đầu ra smoke ra tệp; (ii) lưu một phản hồi `/v1/analyze` của ảnh mẫu công khai (M0-04 (5), M1-01 (2), M4-04 AC2); (iii) `find -newer` quanh phần B (M1-03 (4)).
