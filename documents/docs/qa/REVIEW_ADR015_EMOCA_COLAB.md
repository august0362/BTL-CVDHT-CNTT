---
id: QA-ADR015-REVIEW-01
type: QA_REPORT
from: GENERAL
to: [AI, BACKEND]
task: M1-03 / M0-04 / M0-05 (đường EMOCA-trên-Colab, ADR-015)
status: READY
created: 2026-09-24
---

# Review độc lập — đường EMOCA trên Colab (ADR-015)

**Loại:** review CHỈ ĐỌC (không chạy EMOCA/Colab/torch). **Verdict:** FAIL có điều kiện — 1 lỗi HIGH chặn notebook ở bước đo (D1); phần gọi API EMOCA trong `emoca_runner.py` khớp upstream ở mọi điểm đã đối chiếu.

**Phạm vi đã đọc:** `backend/services/emoca_svc/{emoca_runner.py, server.py, mock_server.py, pack_for_colab.py, patches/*}`, `backend/web/emoca_client.py`, `backend/web/orchestrator.py` (cổng `consent_required`, `resolve_emoca_settings`), phần EMOCA của `frontend/app.py`, `another/notebooks/emoca_colab.ipynb` (43 ô, đọc từ JSON; "ô N" = chỉ số 0-based trong mảng `cells`), `backend/configs/services.yaml`.
**Đối chiếu upstream** (`backend/model/emoca` @ `e0be0db`, đã xác nhận bằng `git log -1`): `gdl_apps/EMOCA/utils/{io.py, load.py}`, `gdl_apps/EMOCA/demos/{test_emoca_on_images.py, download_assets.sh}`, `gdl/datasets/{ImageTestDataset.py, ImageDatasetHelpers.py, DecaDataModule.py}`, `gdl/utils/{FaceDetector.py, DecaUtils.py, lightning_logging.py}`, `gdl/models/DECA.py`, `setup.py`, `requirements38.txt`; và notebook đã chạy được `emoca_setup_v2_work38_ok.ipynb` (ở gốc dự án).

**Giới hạn:** không có số đo nào trong báo cáo này. Mục "nghi ngờ" cần chạy thật trên Colab mới kết luận được.

## (a) Bảng DEFECT

| ID | Mức | file:dòng | Mô tả | Bằng chứng (đã đọc) | Độ chắc | Đề xuất |
|---|---|---|---|---|---|---|
| D1 | **HIGH** | `another/notebooks/emoca_colab.ipynb` ô 4 (`EX_DIR = "/content/emoca/assets/data/EMOCA_test_example_data/images/affectnet_test_examples"`) | Sai đường dẫn ảnh mẫu. `download_assets.sh` giải nén dữ liệu mẫu vào `/content/emoca/data/…`, không phải `assets/data/…`. Hậu quả: ô 28 gọi `test_emoca_on_images.py --input_folder {EX_DIR}` ⇒ `TestData` in "please check the test path" rồi `exit()` (mã 0, trông như chạy xong); ô 29 `assert report …` hỏng (M0-04); ô 33 `assert imgs` hỏng (M1-03 AC2/3/5); ô 35 (M0-05) không có `first_ok`. **Cả ba phép đo của notebook bị chặn**, dù server (ô 31) vẫn lên. | `download_assets.sh:1-3` (`cd ../../..` → gốc repo, `cd assets`), `:73-78` (`cd ../` → gốc repo, `mkdir data`, `cd data`, `unzip EMOCA_test_example_data.zip`); zip có gốc `EMOCA_test_example_data/` (`emoca_setup_v2_work38_ok.ipynb:7728-7730`, đầu ra unzip); notebook cũ dùng `/content/emoca/data/EMOCA_test_example_data/video/cheo_1.mp4` (`emoca_setup_v2_work38_ok.ipynb:8639, 8646, 8905`); `ImageTestDataset.py:42-49` (`os.path.isdir` sai ⇒ `print(...)`, `exit()`). Lưu ý: chính upstream cũng sai — mặc định của demo là `assets/data/...` (`test_emoca_on_images.py:37`), notebook chép theo. `backend/model/emoca` không có thư mục `data/`/`assets/` được git theo dõi (`git ls-files` = 0). | chắc chắn | Đặt `EX_DIR = "/content/emoca/data/EMOCA_test_example_data/images/affectnet_test_examples"`; thêm `assert os.path.isdir(EX_DIR)` ngay cuối ô 21 để hỏng sớm, có thông báo rõ. (Việc của `01_AI`.) |
| D2 | MED | `another/notebooks/emoca_colab.ipynb` ô 37; `backend/services/emoca_svc/mock_server.py:281, 291` | Notebook **không bao giờ gửi `POST /v1/analyze` qua tunnel** trước khi đưa URL cho người dùng: ô 33 chỉ POST tới `LOCAL` (127.0.0.1), ô 37 qua tunnel chỉ `GET /v1/health` (đúng/sai token). Đường POST có thân ~MB qua Cloudflare là đường duy nhất web dùng mà chưa có phép thử nào. Thêm rủi ro: tầng HTTP chỉ đọc theo `Content-Length` (mặc định `"0"`), không xử lý `Transfer-Encoding: chunked` ⇒ nếu proxy chuyển thân dạng chunked, server đọc thân rỗng và trả 400 `contract_mismatch` "JSON không đúng B1" — lỗi trông như lỗi contract. | `mock_server.py:281` (`int(self.headers.get("Content-Length", "0"))`), `:291` (`self.rfile.read(length)`); không có `Transfer-Encoding`/`chunked` trong `backend/services/emoca_svc/*.py` (grep); ô 33 dùng `call(LOCAL, "/v1/analyze", …)`; ô 37 chỉ `call(URL, "/v1/health", …)`. Hành vi thật của cloudflared với thân POST: **chưa kiểm** (không có nguồn trong repo). | nghi ngờ (cần chạy thật) | Ô 37: sau khi tunnel lên, POST một ảnh mẫu công khai qua `URL`, `jsonschema.validate`, in độ trễ khứ hồi qua tunnel (số đo cho M4). Tầng HTTP: thiếu `Content-Length` với POST ⇒ trả 411 rõ ràng (hoặc đọc chunked) thay vì coi là thân rỗng. |
| D3 | MED | `backend/web/orchestrator.py:65` + (không có chỗ nào đọc `max_image_side_px` trong `backend/web/`); `backend/services/emoca_svc/mock_server.py:176-179` | Web không áp giới hạn cạnh ảnh `limits.max_image_side_px: 4096` trước khi gửi. Ảnh máy ảnh thường (vd 6000×4000, < 10 MiB) được mã hoá lại và **gửi qua Google/Cloudflare**, rồi mới bị server từ chối 413 `image_too_large`. Vừa hỏng trải nghiệm với ảnh phổ biến, vừa để ảnh rời máy vô ích (đi ngược mục đích giảm nhẹ hạng 0 của ADR-015). | `grep max_image_side backend/web/*.py` = không có kết quả; `backend/configs/services.yaml` `limits.max_image_side_px: 4096`; server kiểm ở `mock_server.py:176-179`; client gửi nguyên kích thước (`emoca_client.py:192-199, 257-261`). | chắc chắn (theo mã) | `03_BACKEND`: đọc `max_image_side_px` và từ chối trước khi gửi (hoặc, nếu chọn thu nhỏ, phải ghi phép co vào `crop_transform`/`original_px` theo CLAUDE.md §1.7 — cần ADR nhỏ). |
| D4 | LOW | `backend/services/emoca_svc/mock_server.py:75, 167`; `emoca_runner.py:166-176` | `limits.max_faces` được đọc vào `cfg` nhưng không được áp: `check_request` chỉ kiểm `max_faces ≥ 1`, runner chạy encode+decode cho mọi mặt tới `max_faces` do request đặt. Người có token gửi `max_faces` lớn có thể chiếm GPU (khoá tuần tự chặn mọi request khác). | `mock_server.py:75` (đọc `max_faces`), `:167-168` (chỉ kiểm ≥ 1); `emoca_runner.py:166` (`[: max(1, int(max_faces))]`). | chắc chắn | Kẹp `max_faces = min(max_faces, cfg["max_faces"])` hoặc trả 400 khi vượt. |
| D5 | LOW | `backend/services/emoca_svc/emoca_runner.py:350-351` (trong vòng lặp `:169-171`) | Một mặt PHỤ quá nhỏ (cạnh crop < 16 px) làm cả request thành 422 `no_face`, dù mặt lớn nhất đã xử lý được. Chỉ xảy ra khi `max_faces > 1` (v1 = 1 nên ảnh hưởng thấp). | `crop()` raise `RunnerError("no_face", …)` ở `:350-351`, gọi trong vòng lặp `analyze` `:169-171`, không bắt riêng. | chắc chắn | Bỏ qua mặt đó kèm `warnings`, chỉ raise khi không còn mặt nào. |
| D6 | LOW | `backend/services/emoca_svc/emoca_runner.py:330-334`; notebook ô 16 | Nhãn tái lập `upstream_commit` có hậu tố `+patches` khi cây mã "bẩn" — nhưng notebook ô 16 sửa `requirements38.txt` (tệp được git theo dõi) nên nhãn luôn là `+patches` kể cả khi CHƯA áp bản vá nào; nhãn cũng không nói bản vá nào. | `emoca_runner.py:332-334` (`git status --porcelain --untracked-files=no` khác rỗng ⇒ `+patches`); ô 16 ghi đè `/content/emoca/requirements38.txt`. | chắc chắn | Kiểm từng patch bằng `git apply -R --check` (như `apply.sh`) và ghi danh sách tên patch đã áp, hoặc truyền danh sách từ notebook qua biến môi trường. |
| D7 | LOW | `backend/services/emoca_svc/server.py:46-56`; `backend/web/emoca_client.py:217-230` | Client bỏ cuộc sau `timeout_s` (60 s) nhưng server vẫn xử lý và giữ khoá GPU; người dùng bấm "Phân tích lại" ⇒ request mới xếp hàng sau request cũ. Với Colab chậm có thể dẫn tới chuỗi timeout. | Khoá `self._gpu` bao toàn bộ `runner.analyze` (`server.py:46-48`); client chỉ có timeout phía mình (`emoca_client.py:217, 229-233`). | nghi ngờ (phụ thuộc độ trễ thật — chưa đo) | Đo độ trễ ở ô 33/D2 trước; nếu cần, server trả 503 `retryable` khi khoá đang bận quá N giây thay vì xếp hàng vô hạn. |
| D8 | LOW | `backend/services/emoca_svc/server.py:51-56` | Sau CUDA OOM chỉ trả 503; không `torch.cuda.empty_cache()`, nên các request sau có thể tiếp tục OOM do bộ nhớ cache phân mảnh. FAN có tự `empty_cache` sau phát hiện (`FaceDetector.py:97`) nhưng encode/decode thì không. | `server.py:51-56`; `FaceDetector.py:96-97`. | nghi ngờ (cần chạy thật) | Ở nhánh OOM gọi `empty_cache()` qua một hook của backend (giữ `server.py` không import torch trực tiếp). |
| D9 | LOW | notebook ô 38, ô 40 (không có ô dừng) | (i) Token in thẳng ra đầu ra ô 38, mà Colab lưu đầu ra vào notebook trên Drive (đã cảnh báo ở ô 0 mục 5 — rủi ro đã được chấp nhận). (ii) Không có ô "tắt server + tunnel": ô 40 khi bị ngắt in "server và tunnel vẫn chạy nền", tunnel công khai sống tới khi runtime chết. | ô 38 `print("  Token  :", TOKEN)`; ô 40 `except KeyboardInterrupt: print("… vẫn chạy nền …")`; không ô nào gọi `server.terminate()`/`tunnel.terminate()`. | chắc chắn | Thêm ô 12 "Dừng": `tunnel.terminate(); server.terminate()`. Cân nhắc gợi ý "Edit → Clear all outputs" trước khi lưu/chia sẻ notebook. |
| D10 | LOW | notebook ô 8, 12, 37 | Công cụ dựng không ghim phiên bản: `Miniconda3-latest`, `conda install mamba` (không số phiên bản), `cloudflared …/releases/latest/…`. Bằng chứng "chạy được" là từ 2025-11-20; nay 2026-09. Bước dựng env là bước mong manh nhất; lỗi ở đây sẽ không tái lập được (CLAUDE.md §1.5). | ô 8 (`Miniconda3-latest-Linux-x86_64.sh`), ô 12 (`conda install mamba -n base -c conda-forge -y`), ô 37 (`releases/latest/download/cloudflared-linux-amd64`); ngày trong `emoca_setup_v2_work38_ok.ipynb:7716`. | nghi ngờ (chỉ lộ khi chạy thật) | Ghim bản Miniconda/cloudflared cụ thể; ô 26 đã ghi `env_work38.yml` — thêm `conda --version`, `mamba --version`, `cloudflared --version` vào `/content/out`. |

**Tổng:** HIGH 1 · MED 2 · LOW 7.

## (b) Độ chắc chắn (tóm tắt)

- Chắc chắn: D1, D3, D4, D5, D6, D9.
- Nghi ngờ, cần chạy thật: D2, D7, D8, D10.

## (c) Đã kiểm và thấy ĐÚNG

**Lời gọi upstream (`emoca_runner.EmocaBackend`)**
- `load_model(models_dir, model_name, "detail")` đúng chữ ký và trả `(deca, conf)` (`load.py:159-193`); `sys.exit(0)` khi thiếu checkpoint (`load.py:50-52`) được bắt thành `SystemExit` (`emoca_runner.py:306`). Sau đó `emoca.cuda(); emoca.eval()` giống demo (`test_emoca_on_images.py:60-62`).
- `FAN()` mặc định `device='cuda'` (`FaceDetector.py:77`); `run(image, with_landmarks=True)` nhận RGB uint8 H×W×3, trả `(boxes [left, top, right, bottom], 'kpt68', kpts)`, không thấy mặt ⇒ `([], 'kpt68', [])` (`FaceDetector.py:91-120`). Runner truyền uint8 RGB (`emoca_runner.py:152`).
- `bbox2point(left, right, top, bottom, type=…)` — runner truyền đúng thứ tự (`emoca_runner.py:347-348` ↔ `ImageDatasetHelpers.py:25`).
- Crop giống hệt nhánh một mặt của `TestData`: `size = int(old_size*1.25)`, 3 điểm nguồn, đích `[0,0],[0,223],[223,0]`, `estimate_transform('similarity', src, dst)`, `warp(image/255., tform.inverse, output_shape=(224,224))` (`emoca_runner.py:349-357` ↔ `ImageTestDataset.py:144-155`).
- `encode({"image": x}, training=False)` với `x` [1,3,224,224] float trong [0,1]: `DECA.encode` nhận tensor 4 chiều (`DECA.py:482-485`); giống `io.test` (`io.py:77-83`). `decode` qua `io.decode` trả `(values, visualizations)` và tự bọc `no_grad` (`io.py:86-106`).
- Khoá `codedict`: `shapecode/texcode/expcode/posecode` (`DECA.py:883-886`), `detailcode` chỉ ở `DecaMode.DETAIL` (`DECA.py:889-891`) — runner ép `mode="detail"` và báo `codes_missing` nếu thiếu (`emoca_runner.py:265-267, 204-207`).
- Khoá `visdict`: `geometry_coarse`, `geometry_detail`, `output_images_coarse`, `output_images_detail` (`DECA.py:2617, 2622, 2624, 2635`) — trùng `save_images` (`io.py:56-59`) và `RENDER_SOURCES` (`emoca_runner.py:40-45`). `torch_img_to_np` (CHW→HWC, `io.py:15-16`) + `_fix_image` (×255 nếu max < 30, về uint8, `lightning_logging.py:7-11`) ⇒ PNG RGB.
- `predicted_landmarks` [B,68,2], chiếu trực giao, lật trục y (`DECA.py:968-971`), lưu vào `codedict` (`DECA.py:1186`); runner lấy `[0]` ⇒ (68,2) và kiểm shape (`emoca_runner.py:211-212`).

**Hệ toạ độ (CLAUDE.md §1.7)**
- `crop224 = lm·112 + 112` khớp `tensor_vis_landmarks` (`DecaUtils.py:736`) VÀ chuẩn hoá lúc huấn luyện `kpt / image_size * 2 - 1` (`DecaDataModule.py:395`, và các dòng 532, 664, 801, 915, 1012). Còn lệch nửa pixel về quy ước (crop ánh xạ tới 0..223, chuẩn hoá theo 224) — nằm trong sai số, giống hệt upstream, không phải lỗi.
- `tform.params` của `estimate_transform(src=original, dst=crop)` là phép `original_px → crop224` ⇒ ghi `from: original_px, to: crop224` đúng (`emoca_runner.py:234-235`); `original_px = inv(M)·crop224` đúng chiều (`emoca_runner.py:222`).
- EXIF: web xoay EXIF rồi mã hoá lại PNG (không còn EXIF) trước khi gửi; server xoay EXIF lần nữa (no-op) (`mock_server.py:182`) ⇒ `original_px` của hai phía cùng hệ.

**Python 3.8 (env `work38`)**
- Không có cú pháp/API 3.9+ lúc chạy trong `backend/services/emoca_svc/*.py` (grep `removeprefix`, `functools.cache`, `:=`, `match`, `X | Y`…); `str | None` chỉ ở chú thích kiểu có `from __future__ import annotations` (`mock_server.py:25, 216`). `ThreadingHTTPServer` có từ 3.7.
- `gdl` và `gdl_apps` được `pip install -e .` cài thành gói (`setup.py:18`) nên import được khi cwd là `/content/svc`. `io.py` import `FaceVideoDataModule` ở mức module (`io.py:3`) — hai bản vá tuỳ chọn chỉ sửa logic lúc chạy, không cần cho việc import. `requirements38.txt:15` ghim `face-alignment~=1.3.3` (tên `_2D`), bản vá `0002` hỗ trợ cả hai tên; `numpy>=1.23.1` (`:36`) nên bản vá `0001` là cần thiết.

**Bảo mật / riêng tư**
- Server mặc định nghe `127.0.0.1` (`backend/configs/services.yaml`), từ chối địa chỉ khác loopback khi không có token (`server.py:61-62`); token đọc từ biến môi trường, không qua argv (`server.py:81`; notebook ô 31 `env=dict(os.environ, EMOCA_TOKEN=TOKEN)`).
- Xác thực trước khi đọc thân request và đọc bỏ thân khi từ chối (`mock_server.py:251-260, 273-275`); so sánh hằng thời gian (`:255`); log chỉ có method/đường dẫn/mã (`:236-237`); giới hạn thân trước khi đọc (`:288-290`); timeout socket 120 s (`:225`). Runner không ghi đĩa (`png_b64` dùng `BytesIO`, `emoca_runner.py:76-83`).
- Notebook kiểm token sai qua tunnel ⇒ 401 (ô 37) và quét token trong `/content/out` trước khi đóng gói (ô 42); cờ giấy phép mặc định `False` + `assert` (ô 4, ô 21); `pack_for_colab.py` chặn đuôi tài sản có giấy phép và tệp > 2 MiB (`pack_for_colab.py:47-48, 76-80`).
- Cổng xác nhận: `analyze` chặn TRƯỚC khi gọi AI (`orchestrator.py:353-358`); nhánh `mock_case="partial"` không gọi client (`orchestrator.py:254-257`); chỉ IP loopback/`localhost` là cục bộ, LAN và tên miền đều là "từ xa" (`orchestrator.py:126-137`); giao diện khởi tạo xác nhận = `False` (`app.py:216`) và thu hồi khi đổi chế độ/địa chỉ (`app.py:235-240`). Không tìm thấy đường vòng qua cổng.
- Client không đưa token vào thông báo lỗi; chỉ chuyển tiếp mã lỗi thuộc enum của contract, mã lạ ⇒ `contract_mismatch` (`emoca_client.py:161-162, 266-272`); giới hạn phản hồi 64 MiB (`:164, 240-241`).

**HTTP / contract**
- Mã lỗi: `image_invalid` 400, `image_too_large` 413, `no_face` 422, `contract_mismatch` 400/404/500, lỗi nội bộ/OOM ⇒ `service_unavailable` 503 retryable; 401 ⇒ `service_unavailable` không retry đúng ADR-015 §9. Base64 cho phép thân ≈ 4/3·`max_image_bytes` + 4096, khớp giới hạn phía client (`emoca_client.py:256-259`).
- Runner kiểm hình dạng trước khi trả (`check_body`, NaN/inf, ma trận suy biến) ⇒ `contract_mismatch` thay vì gửi dữ liệu sai (`emoca_runner.py:94-128, 192-195`).

**Notebook trên Colab (theo mã, chưa chạy)**
- cwd: `%cd /content/emoca` (ô 9) trước các bước dùng đường dẫn tương đối (`environment.yml`, `pull_submodules.sh`, `requirements38.txt`, `pip install -e .`); `%cd …/demos` (ô 21) trước `download_assets.sh` (script tự `cd ../../..`).
- Nội suy `{EMOCA_COMMIT}`, `{EX_DIR}`, `{MODEL_NAME}`, `"{sample_path}"`, `{FLAME_ALBEDO_DRIVE_ID}` trong lệnh `!` là cú pháp IPython hợp lệ; `%%writefile` đứng dòng đầu ô 34.
- `source activate work38; …` trong `!` giống nguyên văn notebook đã chạy được; server chạy bằng `bash -c "source /content/miniconda/bin/activate work38 && …"` (ô 31) — đúng cách với `subprocess`.
- Đầu ra M0-04: demo ghi vào `<output_folder>/<model_name>/<tên>/{inputs,geometry_*,out_im_*}.png` và `{shape,exp,tex,pose,detail}.npy` (`test_emoca_on_images.py:53, 74-85`; `io.py:49-74`) ⇒ ô 29 tìm đúng chỗ (khi D1 đã sửa). `--save_codes True` hợp lệ.
- `apply.sh` kiểm commit `e0be0db` và idempotent; `requirements38.txt` bị ô 16 sửa không cản `git apply` vì hai patch không chạm tệp đó.

## Việc tiếp theo đề xuất

1. `01_AI`: sửa D1 (một dòng + một `assert`), thêm POST qua tunnel ở ô 37 (D2), thêm ô dừng (D9).
2. `03_BACKEND`: D3 — áp `max_image_side_px` phía web trước khi gửi.
3. Người dùng chạy notebook sau khi sửa D1; gửi `emoca_colab_results.zip` để đo M0-04/M0-05/M1-03 và kết luận D2, D7, D8, D10.
