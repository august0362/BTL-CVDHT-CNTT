# Module EMOCA — Báo cáo phân tích mã nguồn

> Module: bộ trích xuất **3DMM (FLAME)** đóng băng dùng trong MA3D-Net.
> Nguồn phân tích: repo [`radekd91/emoca`](https://github.com/radekd91/emoca), đã clone tại [`../../backend/model/emoca/`](../../../backend/model/emoca/)) (chưa sửa gì — `git status` sạch).
> Bối cảnh dự án: xem [`../../README.md`](../../README.md)). Báo cáo này chỉ nói về EMOCA.

**Cách đọc độ tin cậy** — mỗi khẳng định được gắn nhãn:

| Nhãn | Nghĩa |
|---|---|
| **[Mã]** | Đọc trực tiếp từ mã/cấu hình trong repo (kèm `đường_dẫn:dòng` tính từ gốc repo `backend/model/emoca/`) |
| **[Chạy]** | Đã chạy lệnh kiểm chứng |
| **[Notebook]** | Lấy từ `emoca_setup_v2_work38_ok.ipynb` của bạn (Colab, đã chạy được) |
| **[Suy luận]** | Suy ra, chưa kiểm chứng trực tiếp |

---

## 1. Thông tin repo

| Mục | Giá trị |
|---|---|
| Repo | https://github.com/radekd91/emoca |
| Nhánh phân tích | `release/EMOCA_v2` (nhánh mặc định; còn `release/EMOCA`, `radekd91-patch-1`) |
| Commit | `e0be0db` — "Update README.md", Radek Daněček, 06/12/2024 |
| Quy mô | 1.076 commit · 163 file `.py` · 196 file `.yaml` · ~87 MB (gồm `.git`, chưa kéo submodule) |
| Bài báo | *EMOCA: Emotion Driven Monocular Face Capture and Animation*, CVPR 2022 — Danecek, Black, Bolkart (MPI-IS) |
| Nền tảng | Xây trên **DECA** (Feng et al., SIGGRAPH 2021); bản v2 lấy ý tưởng lip-reading loss từ **SPECTRE** |
| Framework | PyTorch, PyTorch Lightning 1.4.9, Hydra/OmegaConf (cấu hình) |
| Trạng thái | README ghi rõ **EMOCA đã deprecated**, tác giả hướng sang thư viện [`inferno`](https://github.com/radekd91/inferno) (module FaceReconstruction). EMOCA v2 là bản cuối trong repo này **[Mã]** `README.md:60-64` |
| Giấy phép | MPI — chỉ phi thương mại (xem [§9](#9-giấy-phép-và-tuân-thủ)) |

---

## 2. Vai trò trong MA3D-Net

MA3D-Net dùng EMOCA như một **bộ trích xuất 3DMM bị đóng băng**: từ một ảnh khuôn mặt, EMOCA ước lượng tham số FLAME; MA3D-Net lấy vector 334 chiều đó làm tín hiệu điều biến. Bài báo MA3D-Net nêu: *334 tham số = shape 100 + expression 50 + pose 6 + detail 128 + texture 50*.

### Đối chiếu bài báo ↔ mã EMOCA

| Nhóm | Số chiều | Cấu hình (`n_*`) | Khóa trong `codedict` | File `.npy` do `save_codes` ghi | Có trong 334? |
|---|---:|---|---|---|:-:|
| shape (danh tính) | 100 | `n_shape` | `shapecode` | `shape.npy` | ✔ |
| expression | 50 | `n_exp` | `expcode` | `exp.npy` | ✔ |
| pose (xoay toàn cục 3 + hàm 3) | 6 | `n_pose` | `posecode` | `pose.npy` | ✔ |
| detail | 128 | `n_detail` | `detailcode` | `detail.npy` | ✔ |
| texture | 50 | `n_tex` | `texcode` | `tex.npy` | ✔ |
| camera | 3 | `n_cam` | `cam` | — (không lưu) | ✘ |
| ánh sáng SH (9×3) | 27 | `n_light` | `lightcode` | — (không lưu) | ✘ |

- Cấu hình kích thước: `gdl_apps/EMOCA/emoca_conf/backend/model/sizes/default.yaml` **[Mã]**.
- `save_codes` ghi đúng 5 nhóm trên: `gdl_apps/EMOCA/utils/io.py:62-74` **[Mã]**.
- 100+50+6+128+50 = **334**; nếu tính cả `cam` (3) và `light` (27) thì tổng là 364. → Vector 334 chiều của MA3D-Net **khớp chính xác** với 5 file `.npy` mà `--save_codes True` xuất ra, tức là mọi thứ trừ `cam` và `light`.
- `detailcode` **chỉ tồn tại ở chế độ `detail`** (`gdl/models/DECA.py:889-891` **[Mã]**), trong khi `save_codes` truy cập `vals["detailcode"]` → chạy `--mode coarse --save_codes True` sẽ lỗi khóa **[Suy luận từ mã]**. Mặc định của cả hai demo là `detail`.

### Điều bài báo MA3D-Net không nói rõ (cần đối chiếu với repo MA3D-Net)

1. **Dùng checkpoint nào**: `EMOCA` (v1) hay `EMOCA_v2_*`. Bài báo chỉ trích dẫn EMOCA CVPR 2022. Notebook của bạn dùng `EMOCA_v2_lr_mse_20` **[Notebook]**.
2. **Thứ tự nối** 334 chiều: bài báo liệt kê shape → expression → pose → detail → texture, còn `save_codes` ghi shape → exp → tex → pose → detail.
3. **Cách xử lý ảnh đã được cắt sẵn** (RAF-DB, FERPlus…) — xem [§8](#8-áp-dụng-vào-ma3d-net--cheofamo).

---

## 3. Kiến trúc EMOCA (theo mã)

```
Ảnh mặt đã cắt 224×224, giá trị [0,1]
 │
 ├─► E_flame       ResNet50 + MLP(2048→1024→236)   ĐÓNG BĂNG (trọng số DECA)
 │      → shape(100) · tex(50) · exp(50) · pose(6) · cam(3) · light(27)
 │
 ├─► E_expression  ResNet50 + MLP → 50             ★ phần EMOCA huấn luyện lại
 │      (sao chép từ E_flame rồi huấn luyện toàn bộ) → thay thế `exp` của DECA
 │
 └─► E_detail      ResNet50 + MLP → 128            → detailcode
        │
        ▼
     D_detail (Generator conv: 128 + vector điều kiện → bản đồ dịch chuyển UV 256×256)

 FLAME (LBS): shape100 + exp50 + pose6 → mesh + landmark 68 (+ landmark MediaPipe ở v2)
 FLAMETex (BFM→FLAME): tex50 → albedo UV
 SRenderY (rasterizer PyTorch3D + ánh sáng SH) → ảnh render coarse / detail
```

Các điểm chính **[Mã]**:

- **`E_flame`**: `ResnetEncoder(outsize = n_param)` với `n_param = 100+50+50+6+3+27 = 236` (`gdl/models/DECA.py:2752`, `:2918`). Trong `ExpDECA` (chính là EMOCA) nó bị `requires_grad_(False)` (`:3167`).
- **`E_expression`** (`:3163-3186`) có bốn lựa chọn qua `expression_backbone`:
  - `deca_parallel` — thêm một đầu FC thứ hai trên backbone đóng băng (mặc định trong `default_*_expdeca.yaml`);
  - `deca_clone` — sao chép ResNet của DECA và huấn luyện toàn bộ; comment ở `:3174` ghi "This is in final EMOCA", và `training/emoca.yaml` cũng đặt `deca_clone`;
  - `emonet_trainable` / `emonet_static` — đã deprecated.
- **Nguồn của từng mã** trong cấu hình EMOCA cuối (`training/emoca.yaml`, `exp_deca_global_pose: false`, `exp_deca_jaw_pose: false`): `expression` (50) đến từ `E_expression`; `shape`, `tex`, `pose`, `cam`, `light` vẫn đến từ `E_flame` đóng băng của DECA. `E_expression` chỉ xuất thêm pose khi bật hai cờ trên (`n_exp_param += 3` hoặc `+= n_pose`, `:3195-3202`).
- **`E_detail`** → `detailcode` (128); **`D_detail`** là `Generator` (`gdl/models/DecaDecoder.py:27`): Linear → 128×8×8 → 5 lần upsample ×2 → 256×256, 1 kênh, `Tanh`, `out_scale=0.01` (`DECA.py:2901`).
- **FLAME** (`gdl/models/DecaFLAME.py:44`): dùng 100 thành phần shape đầu tiên trong 300 và 50 thành phần expression trong 100 (`:65`); pose 6 = xoay toàn cục 3 + hàm 3, cổ và mắt cố định bằng 0 (`:79-84`, `:241-242`).
- **Renderer** (`gdl/models/Renderer.py`): rasterize bằng `pytorch3d.renderer.mesh.rasterize_meshes` (`:14-16`, `Pytorch3dRasterizer` `:116`), ánh sáng SH (`add_SHlight` `:305`).
- **`decode()`** (`DECA.py:938`): trả về mesh (`verts`), landmark 2D/3D, albedo, ảnh render coarse; ở chế độ `detail` còn có `predicted_detailed_image`, `uv_detail_normals`, `displacement_map`… (`:1175-1203`).

### Cách EMOCA học biểu cảm (điểm mới so với DECA)

EMOCA không dùng nhãn cảm xúc trực tiếp; nó dùng **loss cảm xúc dạng perceptual**: đặc trưng EmoNet (`feat_2`) của ảnh đầu vào và ảnh render được so khớp bằng MSE, để mã `expression` phải khiến khuôn mặt render *mang cùng cảm xúc* với ảnh gốc (`training/emoca.yaml`: `use_emonet_loss: true`, `use_emonet_feat_2: true`, `emo_feat_loss: mse_loss`, `emonet_weight: 1.0`) **[Mã]**.

---

## 4. Bản đồ mã nguồn

| Đường dẫn (trong `backend/model/emoca/`) | Vai trò |
|---|---|
| `gdl/models/DECA.py` (3.428 dòng) | Lõi. `DecaModule` (LightningModule: `encode`/`decode`/loss/train), `DECA`, `ExpDECA` (=EMOCA), `EMICA` |
| `gdl/models/DecaEncoder.py` · `DecaDecoder.py` | `ResnetEncoder`, `SecondHeadResnet`, `SwinEncoder` · `Generator`, `GeneratorAdaIn` |
| `gdl/models/DecaFLAME.py` | `FLAME`, `FLAME_mediapipe`, `FLAMETex` |
| `gdl/models/Renderer.py` | `SRenderY`, `Pytorch3dRasterizer` |
| `gdl/layers/losses/` | `DecaLosses` (landmark, eye/lip, ID-MRF, VGGFace2…), `EmoNetLoss`, `MediaPipeLandmarkLosses` |
| `gdl/models/temporal/external/LipReadingLoss.py` | Lip-reading loss (v2, cần submodule `spectre`) |
| `gdl/datasets/` | Data module: `DecaDataModule`, `AffectNet*`, `FaceVideoDataModule` (158 KB), `ImageTestDataset` (`TestData`) |
| `gdl/utils/` | `FaceDetector` (FAN), `MediaPipeLandmarkDetector`, `DecaUtils`, … |
| `gdl_apps/EMOCA/demos/` | `test_emoca_on_images.py`, `test_emoca_on_video.py`, `download_assets.sh` |
| `gdl_apps/EMOCA/utils/` | `load.py` (`load_model`), `io.py` (`test`, `save_images/codes/obj`) |
| `gdl_apps/EMOCA/training/` | `train_expdeca.py`, `emoca.yaml`, … |
| `gdl_apps/EMOCA/emoca_conf/` | Cấu hình Hydra (backend/model/data/learning/inout) |
| `gdl_apps/EMOCA/evaluation/` | Script benchmark AffectNet, NoW |
| `gdl_apps/EmotionRecognition/` | Huấn luyện/đánh giá nhận diện cảm xúc (EmoNet, ResNet, Swin, dựa trên EMOCA) |
| `external/` | Submodule (**rỗng** trong bản clone này): `emonet`, `SwinTransformer`, `face-parsing.PyTorch`, `spectre`, `Deep3DFaceRecon_pytorch`, `TDDFA_V2` |

---

## 5. Quy trình suy luận (inference)

### 5.1 Trên ảnh — `gdl_apps/EMOCA/demos/test_emoca_on_images.py`

1. **`load_model(path_to_models, model_name, mode)`** (`utils/load.py:159`): đọc `<models>/<model_name>/cfg.yaml`, ghi đè đường dẫn tài sản qua `replace_asset_dirs` (`:134`), tìm checkpoint, gọi `DecaModule.load_from_checkpoint(..., strict=False)`.
2. **`TestData(input_folder, face_detector="fan", max_detection=20)`** (`gdl/datasets/ImageTestDataset.py:37`):
   - phát hiện mặt bằng **FAN** (SFD + 68 landmark, `gdl/utils/FaceDetector.py:84`);
   - `bbox2point(type='kpt68')`: `old_size = (w+h)/2 × 1.1`, tâm = tâm hộp (`gdl/datasets/ImageDatasetHelpers.py:25-32`), sau đó nhân `scale = 1.25`;
   - phép biến đổi *similarity* → **224×224**, pixel về `[0,1]`, không chuẩn hoá mean/std ở bước này.
3. **`test(emoca, batch)`** (`utils/io.py:77`): `encode(training=False)` rồi `decode`.
4. Lưu: `save_images` (`inputs`, `geometry_coarse/detail`, `out_im_coarse/detail`.png), `save_codes` (`.npy`), `save_obj` (`mesh_coarse.obj` và `_detail.obj` đã upsample).

| Tham số | Mặc định |
|---|---|
| `--model_name` | `EMOCA_v2_lr_mse_20` |
| `--mode` | `detail` (hoặc `coarse`) |
| `--save_images` / `--save_codes` / `--save_mesh` | `True` / `False` / `False` |
| `--path_to_models` | `assets/EMOCA/models` |

### 5.2 Trên video — `test_emoca_on_video.py`

`TestFaceVideoDM.prepare_data` (`FaceVideoDataModule.py:3181`): tách frame (ffmpeg) → FAN phát hiện mặt từng frame → lưu crop, landmark, metadata → `setup` gom crop thành `TestData(images, iscrop=False)` (`:3263-3269`) → chạy theo lô (`batch_size=4`, `test_emoca_on_video.py:60`) → dựng video kết quả có tiếng. Có thể tiếp tục từ lần chạy dở bằng `--processed_subfolder`.

Mỗi frame được xử lý **độc lập**, không có làm mượt theo thời gian **[Mã]**.

### 5.3 Các biến thể mô hình

Theo `gdl_apps/EMOCA/README.md:30-41` **[Mã]**:

| `--model_name` | Đặc điểm |
|---|---|
| `EMOCA` | Bản gốc (v1): landmark FAN, chỉ loss landmark tương đối |
| `EMOCA_v2_mp` | Landmark MediaPipe (mắt, miệng, lông mày) + loss landmark tuyệt đối, không lip-reading |
| `EMOCA_v2_lr_cos_1.5` | Thêm lip-reading loss (cosine); README cảnh báo có thể làm môi cường điệu |
| `EMOCA_v2_lr_mse_20` | **Khuyến nghị/mặc định**: lip-reading loss dạng MSE, ổn định hơn |

`cfg.yaml` và checkpoint của các mô hình nằm trong file zip tải về (`download_assets.sh`), **không có trong repo** → cấu hình v1/v2 chưa kiểm chứng được (ví dụ pose lấy từ nhánh nào).

---

## 6. Huấn luyện

Hai giai đoạn. Cấu hình EMOCA cuối: `gdl_apps/EMOCA/training/emoca.yaml` **[Mã]**.

| | Giai đoạn coarse | Giai đoạn detail |
|---|---|---|
| Dữ liệu | AffectNet (`AffectNetEmoNetSplitModule`, sampler `balanced_expr`) | `DecaDataModule`: VGGFace2-HQ + VoxCeleb2; val `now-val`, `celeb-val` |
| Tăng cường | Affine (±10°, tỉ lệ 0.9–1.1), JPEG, nhiễu, mờ, sắc nét | như coarse |
| Ảnh | 224, `scale=1.25`, FAN ngưỡng 0.9 | 224, `scale` 1.2–1.6 |
| Cái được huấn luyện | `E_expression` (`deca_clone`) | `E_detail` + `D_detail` (coarse cố định, `train_coarse: false`) |
| Tối ưu | Adam, lr 5e-5, batch 32, ≤30 epoch, early-stop 15 | Adam, lr 1e-4, batch 4, `train_K: 3` ảnh/danh tính, ≤12 epoch, early-stop 5 |
| Phần cứng ghi trong cấu hình | 1 GPU, `gpu_memory_min_gb: 30` | như coarse |

### Các thành phần loss (giá trị trong `emoca.yaml`)

| Loss | Khóa cấu hình | Giá trị |
|---|---|---|
| Photometric (ảnh render vs ảnh gốc) | `use_photometric`, `photow`, `photometric_normalization` | true · 2.0 · `mean` |
| **Emotion perceptual (EmoNet feat-2)** | `use_emonet_loss`, `use_emonet_feat_2`, `emo_feat_loss`, `emonet_weight` | true · true · MSE · **1.0** |
| Khoảng cách mắt / môi / khoé miệng (tương đối) | `use_eye_distance`+`eyed`, `use_lip_distance`+`lipd`, `use_mouth_corner_distance` | true · 0.5 · 0.5 |
| Landmark 68 điểm (tuyệt đối) | `use_landmarks`, `lmk_weight` | **false** (chỉ log như metric, `DECA.py:1857`), 1 |
| Danh tính (VGGFace2) | `idw` | 0 (tắt) |
| Regularizer | `shape_reg`, `exp_reg`, `tex_reg`, `light_reg` | 1e-4 · 1e-4 · 1e-5 · 12 |
| *Detail:* ID-MRF, L1 chi tiết | `mrfwr`, `use_detail_l1`, `use_detail_mrf` | 0.05 · true · true |
| *Detail:* z-reg / z-diff / z-sym | `zregw`, `zdiffw`, `zsymw` | 0.005 mỗi loại |

Cấu hình này khớp mô tả README rằng EMOCA v1 chỉ dùng loss landmark **tương đối**; v2 bổ sung loss landmark tuyệt đối (MediaPipe) và lip-reading loss (`DECA.py:1846-1893`, `:1451`) **[Mã]**.

### Hạn chế khi muốn tái huấn luyện

- `emoca.yaml` chứa đường dẫn tuyệt đối trên cụm của tác giả (`/ps/project/…`, `/home/rdanecek/…`).
- Cần AffectNet (xin cấp quyền), tập DECA (README nói VGGFace2 "không còn phát hành") và texture BFM (không được phân phối lại) **[Mã]** `gdl_apps/EMOCA/README.md:53,99-106`.
- **Lệnh trong README không khớp `main()`**: README ghi `python training/train_expdeca.py emoca.yaml`, nhưng `main()` (`train_expdeca.py:492-496`) coi `len(sys.argv) <= 2` là nhánh gỡ lỗi với cấu hình cứng, gồm `backend/model/settings=coarse_train_emica_emonet` — file này **không có** trong repo. Nhánh đọc file cấu hình cần **hai** đối số (coarse, detail), trong khi `emoca.yaml` gói cả `coarse:` và `detail:` trong một file. Nhiều khả năng lệnh như tài liệu sẽ không chạy được **[Mã; chưa chạy thử]**.
- Cấu hình mặc định `default_*_expdeca.yaml` dùng `deca_parallel`, còn EMOCA cuối dùng `deca_clone` — dễ nhầm khi tự huấn luyện.

---

## 7. Phụ thuộc và môi trường

### 7.1 Phụ thuộc cứng

- **PyTorch3D** là phụ thuộc bắt buộc **ngay cả khi chỉ suy luận**, vì `SRenderY` được dựng trong `DECA._setup_renderer` (`Renderer.py:14-16`, `:184`) **[Mã]**.
- **FLAME** (`generic_model.pkl`…): phải đăng ký và chấp nhận giấy phép tại flame.is.tue.mpg.de. **Checkpoint EMOCA**: đăng ký tại emoca.is.tue.mpg.de (`demos/download_assets.sh`).
- **Texture BFM→FLAME** (`FLAME_albedo_from_BFM.npz`): README ghi "Optional for inference" (`gdl_apps/EMOCA/README.md:53`), nhưng mã nạp nó **bắt buộc**: `replace_asset_dirs` đặt `tex_path` (`utils/load.py:142`), `FLAMETex.__init__` gọi `np.load(config.tex_path)` vô điều kiện với `tex_type: BFM` (`DecaFLAME.py:426-431`), và không cấu hình nào tắt texture (`uses_texture()` mặc định `True`, `DECA.py:2846-2849`) **[Mã]**. Khớp với việc notebook của bạn phải tải file này.
- **FAN** (`face_alignment`), **MediaPipe**, ffmpeg; `download_assets.sh` cần `bash`, `wget`, `unzip` và có hỏi xác nhận tương tác (`read -p`).

### 7.2 Phiên bản: repo so với môi trường đã chạy được

| Thành phần | Repo (README / yml) | Notebook chạy được **[Notebook]** |
|---|---|---|
| Python | 3.8 | 3.8.20 |
| PyTorch | README: 1.12.1 (đã phát triển với CUDA 11.1.1) | **1.11.0** (CUDA 11.5, cuDNN 8.3.2) |
| torchvision / torchaudio | yml ghim `0.11.3` / `0.10.2` (README lệnh dự phòng: `0.13.1` / `0.12.1`) | `0.12.0` / `0.11.0` |
| PyTorch3D | 0.6.2, **build từ nguồn** (`install_38.sh`) | 0.6.2 **bản dựng sẵn** `py38_cu115_pyt1110` (kênh conda `pytorch3d`) |
| cudatoolkit | — | 11.5.1 |
| numpy | `>=1.23.1` | 1.24.4 (pip) |
| Khác | `opencv-python~=4.5.5`, `onnx~=1.10.2`, … | `opencv-python-headless==4.5.5.64`, `chumpy 0.71`, `tensorboard 2.9.1`, `pyzmq 26.2.1`, `Cython==0.29`, `mmcv-full` từ index OpenMMLab `cu115/torch1.11` |

Ghi chú **[Suy luận]**: yml ghim `pytorch=1.12.1` cùng `torchvision=0.11.3`/`torchaudio=0.10.2`, là cặp phiên bản thuộc dòng PyTorch 1.10 → yml không nhất quán nội bộ. Bộ phiên bản trong notebook của bạn (torch 1.11 + torchvision 0.12 + pytorch3d `pyt1110`) là bộ nhất quán và đã chạy được.

### 7.3 Đối chiếu các bản vá trong notebook với mã

| Vị trí (trong `backend/model/emoca/`) | Bản vá trong notebook | Nguyên nhân **[Mã]** |
|---|---|---|
| `gdl/utils/DecaUtils.py:973-976` | `np.int` → `np.int32` | NumPy ≥1.24 đã gỡ alias `np.int`. Hàm `load_local_mask` được gọi ngay khi dựng model (`DECA.py:2781`) → chặn toàn bộ việc nạp model |
| `gdl/utils/FaceDetector.py:84` | `LandmarksType._2D` → `TWO_D` | `face_alignment` bản mới đổi tên enum (notebook không in phiên bản cụ thể) |
| `gdl/datasets/FaceVideoDataModule.py:1569` | thêm điều kiện `0 <= vi < len(annotation_list)` | `TestFaceVideoDM._gather_data` đặt `annotation_list = []` (`:3212`), còn mã gỡ video thực hiện `del self.annotation_list[vi]` khi `annotation_list` khác `None` → `IndexError` với danh sách rỗng |
| `gdl/datasets/FaceVideoDataModule.py:~3253` | `sequence_id = min(sequence_id, len(video_list)-1)` | Chặn chỉ số vượt ngoài `video_list` trong `_get_path_to_sequence_files` |
| `test_emoca_on_video.py:60` | đổi `batch_size` | `TestFaceVideoDM(..., batch_size=4, ...)` — chỉnh theo VRAM |
| `requirements38.txt` | Comment `onnx`, `youtube-dl`, `mediapipe`, `mmcv-full`, `pyzmq` rồi cài riêng | Cài riêng với phiên bản/wheel phù hợp (lý do chính xác: **[Suy luận]**, xung đột ghim phiên bản) |

**Patch `io.py` (ghi 2026-09-26, M0-03):** Không có patch io.py: mục ghi chú trong notebook gốc (cell 38) rỗng, sai đường dẫn (`gld_apps`; tệp thật `gdl_apps/EMOCA/utils/io.py`); demo ảnh chạy đúng không cần vá (smoke M0-04); `--save_mesh` chưa kiểm (đã bỏ theo quyết định M0-04).

**Còn sót, chưa cần cho demo**: `np.bool` trong `AffectNetDataModule`, `AfewVaDataModule`, `EmotioNetDataModule` và `np.float` trong `EmotioNetDataModule.py:410` sẽ lỗi trên NumPy ≥1.24 khi dùng các data module huấn luyện/đánh giá. `np.float` ở `DecaUtils.py:852` nằm trong hàm không ai gọi (`grep` không thấy nơi gọi) **[Chạy]**.

### 7.4 Điều notebook của bạn đã làm (tóm tắt) **[Notebook]**

Miniconda trên Colab (GPU) → `git clone` với `GIT_LFS_SKIP_SMUDGE=1` → ghi đè `environment.yml` bằng bản đã ghim → `pull_submodules.sh` → `mamba env update` → cài pip theo `requirements38.txt` đã sửa → `pip install -e .` → `yes | bash download_assets.sh` → vá mã → tải texture BFM → chạy `test_emoca_on_video.py` trên clip Chèo `cheo_1.mp4` (1280×2160, 30 fps, 891 frame ≈ 29,7 s) với `--model_name EMOCA_v2_lr_mse_20 --save_codes True --save_mesh True` → xuất `video_geometry_detail_with_sound.mp4`, nén zip và tải về.

Hai lưu ý:
- `yes | bash download_assets.sh` tự động trả lời "y" cho hai câu hỏi xác nhận giấy phép (FLAME, EMOCA). Chỉ hợp lệ khi bạn thực sự đã đăng ký và đồng ý điều khoản trên hai trang đó.
- Texture BFM được tải từ một liên kết Google Drive cá nhân (file id trong cell 40). README nói tác giả không được phép phân phối texture này vì giấy phép BFM; nên kiểm tra quyền sử dụng, hoặc lấy theo đường chính thức (đăng ký BFM rồi chuyển bằng công cụ `BFM_to_FLAME`).
- `template.mtl` mà notebook tạo trong `assets/FLAME/geometry/` không được mã nào của repo tham chiếu (`write_obj` tự ghi `.mtl` riêng, `DecaUtils.py:74,93,124`) → có lẽ phục vụ bước xuất mesh riêng của bạn **[Suy luận]**.

---

## 8. Áp dụng vào MA3D-Net / CheoFaMo

### 8.1 Trích vector 334 chiều

Bằng CLI (chỉ cần mã):

```bash
python gdl_apps/EMOCA/demos/test_emoca_on_images.py \
  --input_folder <thư_mục_ảnh> --output_folder <đầu_ra> \
  --model_name EMOCA_v2_lr_mse_20 --mode detail --save_codes True
```

Bằng API (chỉ cần mã; phác thảo suy ra từ mã, **chưa chạy thử**):

```python
from gdl_apps.EMOCA.utils.load import load_model

emoca, conf = load_model(path_to_models, "EMOCA_v2_lr_mse_20", "detail")
emoca.cuda().eval()
with torch.no_grad():
    code = emoca.encode({"image": batch}, training=False)   # batch: [B,3,224,224] trong [0,1]
x3dmm = torch.cat([code["shapecode"], code["expcode"], code["posecode"],
                   code["detailcode"], code["texcode"]], dim=1)   # [B, 334]
```

Chỉ gọi `encode` (không gọi `decode`) sẽ bỏ qua bước render (PyTorch3D) nên nhẹ hơn, và vẫn có đủ 5 mã.

### 8.2 Các quyết định cần thống nhất và ghi lại

| Vấn đề | Lý do / bằng chứng |
|---|---|
| **Cắt mặt hay dùng ảnh nguyên** | EMOCA được huấn luyện với crop từ FAN, `scale=1.25` → 224. Bộ dữ liệu đã căn chỉnh sẵn (RAF-DB, FERPlus, CAER-S) có thể đi qua đường `iscrop=True` (chạy lại FAN) hoặc `iscrop=False` (dùng nguyên ảnh, như nhánh video ở `FaceVideoDataModule.py:3269`) **[Suy luận]** |
| **Ảnh FAN không phát hiện mặt** | `TestData` chỉ `print('no face detected! run original image')` rồi dùng cả ảnh (`ImageTestDataset.py:107-113`), và hoán đổi `h`/`w` khi dựng hộp (`right = h-1`, `bottom = w-1`) **[Mã]** → kết quả 3DMM rất có thể vô nghĩa mà không báo lỗi. Đây là cơ chế cụ thể phía sau hạn chế "ảnh mờ/che khuất" của MA3D-Net → nên ghi log và đánh dấu các ảnh này |
| **Cache đặc trưng** | EMOCA đóng băng (bài báo MA3D-Net) → có thể trích một lần, lưu `.npy` theo id ảnh, khỏi nạp EMOCA (ba ResNet50 + PyTorch3D) khi huấn luyện trên GTX 1660 6 GB **[Suy luận]** |
| **Tính tái lập** | Ghi kèm: `model_name`, `mode`, `iscrop`, `scale`, commit `e0be0db`, phiên bản FAN/`face_alignment` |
| **Chuẩn hoá** | Bài báo chuẩn hoá từng chiều theo mean/std của tập train; hợp lý vì các nhóm có thang đo rất khác nhau (radian của pose so với detail 128 chiều) **[Suy luận]** |
| **Video / DFER (hướng tương lai)** | EMOCA suy luận độc lập từng frame, không có thành phần thời gian (lip-reading loss của v2 chỉ dùng lúc huấn luyện) → cần tự làm mượt/ràng buộc theo thời gian **[Mã]** |

---

## 9. Giấy phép và tuân thủ

Theo `LICENSE` (Max-Planck) **[Mã]** — đây là tóm tắt kỹ thuật, không phải tư vấn pháp lý:

- Cấp quyền cho **nghiên cứu khoa học phi thương mại, giáo dục phi thương mại, dự án nghệ thuật phi thương mại**.
- **Cấm** dùng cho thương mại, quân sự, giám sát, nội dung khiêu dâm; cấm tạo nội dung giả mạo/phỉ báng (ngoại lệ: phân tích trong nghiên cứu được bình duyệt); cấm dùng để **huấn luyện mô hình cho mục đích thương mại**.
- Phần mềm/mô hình **không được sao chép, sửa đổi hoặc chuyển cho bên thứ ba dưới bất kỳ hình thức nào nếu không có văn bản cho phép của Max-Planck** → khi nộp bài, nên **dẫn liên kết chính thức** thay vì đóng gói kèm repo, mã đã sửa hay checkpoint.
- Các giấy phép riêng: FLAME (đăng ký), checkpoint EMOCA (đăng ký), BFM (không phân phối lại texture), AffectNet (xin cấp quyền, nếu tái huấn luyện).

---

## 10. Đánh giá và rủi ro

### Điểm mạnh

- API rõ ràng: `encode` → `codedict` (các mã FLAME có nghĩa, diễn giải được) → `decode` → mesh/ảnh.
- Tách bạch backbone đóng băng (`E_flame`) và phần huấn luyện lại (`E_expression`) → phù hợp để tái sử dụng làm bộ trích xuất đặc trưng.
- Có sẵn script cho ảnh và video, cấu hình theo Hydra/OmegaConf, nhiều checkpoint (v1, v2).

### Phát hiện (sắp xếp theo mức ảnh hưởng tới dự án)

| # | Mức | Phát hiện | Bằng chứng |
|:-:|:-:|---|---|
| 1 | Cao | Repo **deprecated**; tác giả hướng sang `inferno` → khó kỳ vọng bản vá | `README.md:60-61` |
| 2 | Cao | **PyTorch3D bắt buộc ngay cả khi suy luận**, cùng bộ ghim Python 3.8 / PyTorch 1.11–1.12 → khó cài trực tiếp trên Windows; Colab/Linux/WSL là lựa chọn thực tế | `Renderer.py:14-16`; [§7](#7-phụ-thuộc-và-môi-trường) |
| 3 | Trung bình | README nói texture BFM "tuỳ chọn khi suy luận", nhưng mã nạp bắt buộc | `load.py:142`, `DecaFLAME.py:426-431`, `DECA.py:2846-2849` |
| 4 | Trung bình | `TestData` có nhiều lỗi (bảng bên dưới) | `ImageTestDataset.py` |
| 5 | Trung bình | Cờ `--save_images/--save_codes/--save_mesh` khai báo `type=bool` ở script ảnh → `--save_codes False` vẫn là `True` (chuỗi khác rỗng); script video dùng `str2bool` đúng | `test_emoca_on_images.py:41-43`, `test_emoca_on_video.py:28-36` |
| 6 | Trung bình | NumPy ≥1.24: `np.int` chặn nạp model; `np.bool`/`np.float` còn sót ở data module | `DecaUtils.py:973-976`; [§7.3](#73-đối-chiếu-các-bản-vá-trong-notebook-với-mã) |
| 7 | Thấp | Lệnh huấn luyện trong README không khớp `main()`; `emoca.yaml` chứa đường dẫn cụm của tác giả | [§6](#6-huấn-luyện) |
| 8 | Thấp | Mã khuyết/chết: `EMICA` phụ thuộc `gdl.models.mica` (không có trong repo); settings `coarse_train_emica_emonet` không có; `.gitmodules` khai `Deep3DFaceRecon_pytorch` hai lần, `pull_submodules.sh` không khởi tạo `Deep3DFaceRecon_pytorch` và `TDDFA_V2` | `DECA.py:3290-3302`; `.gitmodules`; `pull_submodules.sh` |
| 9 | Thông tin | Cấu hình các checkpoint v1/v2 nằm trong file zip, không có trong repo → chưa kiểm chứng | [§5.3](#53-các-biến-thể-mô-hình) |

**Lỗi trong `TestData`** (`gdl/datasets/ImageTestDataset.py`) **[Mã]**:

| Dòng | Lỗi | Hệ quả |
|---|---|---|
| `:126`, `:132-133` | Khi có `max_detection` (script ảnh truyền `20`, `test_emoca_on_images.py:65`), vòng lặp luôn lấy `bbox[0]`, và `old_size += [osz]`/`center += [c]` nằm **ngoài** vòng lặp | Mỗi ảnh chỉ xử lý **đúng một khuôn mặt** dù ảnh có nhiều mặt; tên đầu ra luôn có hậu tố `00` |
| `:78` | `image[:, :, None].repeat(1, 1, 3)` | Ảnh xám gây `TypeError: repeat() takes at most 2 arguments (3 given)` — **[Chạy]** đã xác nhận bằng numpy |
| `:186` | `util.check_mkdir` nhưng `util` không được import trong file | Truyền đường dẫn `.mp4` vào `TestData` sẽ `NameError` |
| `:74` | `imagepath.split('/')[-1]` | Trên Windows, `glob`/`Path` trả đường dẫn dùng dấu `\` nên tên ảnh có thể thành cả đường dẫn — **[Suy luận]**, chưa chạy trên Windows |
| `:107-113` | Fallback khi không thấy mặt hoán đổi `h`/`w` | Ảnh không vuông cho kết quả sai; xem [§8.2](#82-các-quyết-định-cần-thống-nhất-và-ghi-lại) |

---

## 11. Khuyến nghị cho dự án

1. **Cố định môi trường** theo bộ đã chạy được trong notebook (Python 3.8.20, torch 1.11.0+cu115, pytorch3d 0.6.2 `pyt1110`, numpy 1.24.4) cùng ba bản vá ở [§7.3](#73-đối-chiếu-các-bản-vá-trong-notebook-với-mã); lưu lại `env_work38.yml` (notebook đã xuất).
2. **Trích 334 chiều một lần và cache `.npy`**, kèm metadata (`model_name`, `mode`, `iscrop`, `scale`, commit `e0be0db`).
3. **Đối chiếu với repo MA3D-Net** ba điểm ở cuối [§2](#2-vai-trò-trong-ma3d-net): checkpoint (v1 hay v2), thứ tự nối, cách xử lý ảnh đã cắt sẵn.
4. **Ghi log/đánh dấu ảnh FAN không phát hiện mặt** thay vì để rơi vào fallback im lặng; liên hệ với phần "Limitations" của MA3D-Net.
5. Nếu dùng `TestData` cho nhiều mặt/ảnh xám/Windows: sửa các lỗi ở bảng trên (thay `bbox[0]` bằng `bbox[bbi]` và đưa `old_size += ...` vào trong vòng lặp; bỏ `repeat(1,1,3)`; import `util`; dùng `Path(...).stem`).
6. **Không phân phối lại** repo, mã đã sửa, checkpoint hay texture BFM; dẫn nguồn chính thức ([§9](#9-giấy-phép-và-tuân-thủ)).
7. Nếu cần bảo trì lâu dài, cân nhắc `inferno` — **nhưng phải kiểm tra trước** rằng tham số đầu ra vẫn cho vector 334 chiều tương thích, vì thay đổi sẽ ảnh hưởng toàn bộ MA3D-Net.

---

## 12. Những gì chưa kiểm chứng

- **Chưa chạy EMOCA** (cần GPU và các tài sản phải đăng ký: FLAME, checkpoint EMOCA, texture BFM). Mọi nhận định về hành vi runtime là từ đọc mã hoặc từ notebook của bạn; chỉ dòng `repeat(1,1,3)` là đã chạy thử bằng numpy.
- **Chưa khởi tạo submodule** (`emonet`, `spectre`, `SwinTransformer`, `face-parsing.PyTorch`…) nên chưa đọc mã của chúng; các phần loss cảm xúc và lip-reading chỉ được xét ở mức giao diện gọi.
- **Chưa thử trên Windows nguyên bản** (chỉ suy ra từ ghim phiên bản và cách xử lý đường dẫn).
- **Cấu hình của checkpoint v1/v2** (`cfg.yaml` trong zip) chưa xem.
- **Chưa đo** tốc độ hay VRAM.

---

## Phụ lục A — Chỉ mục vị trí chính (trong `backend/model/emoca/`)

| Nội dung | Vị trí |
|---|---|
| `DecaModule` (LightningModule) | `gdl/models/DECA.py:67` |
| `encode` · `decode` · `_compute_loss` | `gdl/models/DECA.py:469` · `:938` · `:1774` |
| Lắp ráp `codedict` | `gdl/models/DECA.py:883-913` |
| `DECA` · `_create_model` · `_reinitialize` | `gdl/models/DECA.py:2723` · `:2911` · `:2777` |
| `ExpDECA` (= EMOCA) · `_create_model` | `gdl/models/DECA.py:3157` · `:3163` |
| `EMICA` (phụ thuộc `mica` không có) | `gdl/models/DECA.py:3290` |
| `ResnetEncoder` · `SecondHeadResnet` | `gdl/models/DecaEncoder.py:76` · `:93` |
| `Generator` (detail) | `gdl/models/DecaDecoder.py:27` |
| `FLAME` · `FLAME_mediapipe` · `FLAMETex` | `gdl/models/DecaFLAME.py:44` · `:272` · `:415` |
| `SRenderY` · `Pytorch3dRasterizer` | `gdl/models/Renderer.py:174` · `:116` |
| `FAN` | `gdl/utils/FaceDetector.py:84` |
| `TestData` · `bbox2point` | `gdl/datasets/ImageTestDataset.py:37` · `gdl/datasets/ImageDatasetHelpers.py:25` |
| `TestFaceVideoDM` | `gdl/datasets/FaceVideoDataModule.py:3158` |
| `load_model` · `replace_asset_dirs` | `gdl_apps/EMOCA/utils/load.py:159` · `:134` |
| `test` · `save_codes` | `gdl_apps/EMOCA/utils/io.py:77` · `:62` |
| Kích thước tham số | `gdl_apps/EMOCA/emoca_conf/backend/model/sizes/default.yaml` |
| Cấu hình EMOCA cuối | `gdl_apps/EMOCA/training/emoca.yaml` |
| Điểm vào huấn luyện | `gdl_apps/EMOCA/training/train_expdeca.py:330` (`train_expdeca`), `:492` (`main`) |

## Phụ lục B — Lệnh cài đặt rút gọn theo notebook **[Notebook]**

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/radekd91/emoca.git && cd emoca
bash pull_submodules.sh
mamba create -n work38 python=3.8 -y
mamba env update -n work38 --file environment.yml --prune -y     # environment.yml = bản ghim của bạn (torch 1.11, pytorch3d 0.6.2 pyt1110)
pip install --upgrade pip==24.0
# sửa requirements38.txt: comment onnx, youtube-dl, mediapipe, mmcv-full, pyzmq
pip install -r requirements38.txt && pip install Cython==0.29
pip install onnx==1.10.2 onnxruntime-gpu "youtube-dl~=2021.6.6" mediapipe pyzmq==26.2.1
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html
pip install -e .
cd gdl_apps/EMOCA/demos && bash download_assets.sh              # cần đã đăng ký FLAME + EMOCA
pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless
pip install opencv-python-headless==4.5.5.64
pip install git+https://github.com/mattloper/chumpy             # chumpy 0.71
pip install tensorboard==2.9.1
# vá 3 chỗ (mục 7.3), thêm assets/FLAME/texture/FLAME_albedo_from_BFM.npz
python test_emoca_on_video.py --input_video <video> --output_folder <out> \
       --model_name EMOCA_v2_lr_mse_20 --save_codes True --save_mesh True
```
