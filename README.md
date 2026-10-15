# MA3D-Net Web Demo

Web demo nhận diện biểu cảm khuôn mặt từ một ảnh. Người dùng tải ảnh lên, YOLO tìm các khuôn mặt (tối đa 30), ảnh nghiêng
từ 15° trở lên được xoay thẳng. Sau đó **EMOCA** dựng landmark và mô hình 3D (FLAME) cho từng mặt, rồi giao diện so sánh
dự đoán của **POSTER** (mô hình SOTA trước đây) với **MA3D-Net** (mô hình của nhóm, có dùng thêm đặc trưng 3D).

```
ảnh ─► YOLO ─► sửa roll 2D ─► EMOCA (GPU, container riêng) ─► POSTER / MA3D-Net (CPU) ─► giao diện Streamlit
```

## Chạy bằng Docker (khuyến nghị)

### Yêu cầu

| | Chỉ xem giao diện (mock) | Đầy đủ (EMOCA thật) |
|---|---|---|
| Hệ điều hành | Windows 10/11, Linux, macOS | Windows 10/11 (Docker Desktop + WSL2) hoặc Linux |
| Phần mềm | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Docker Desktop **bật WSL2 backend**; Linux cần thêm [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| GPU | Không cần | NVIDIA, VRAM từ 4 GB, driver mới. RTX 20xx/30xx/40xx; **RTX 50xx chưa hỗ trợ** (EMOCA dùng torch 1.11/CUDA 11.5) |
| Đĩa trống | khoảng 6 GB | khoảng 30 GB |
| Tài khoản | Không | Đăng ký [FLAME](https://flame.is.tue.mpg.de) và [EMOCA](https://emoca.is.tue.mpg.de) |

Kiểm tra GPU đã vào được Docker: `docker run --rm --gpus all ubuntu nvidia-smi` phải in ra tên GPU.

### Cách 1: Chỉ xem giao diện (không cần GPU, không cần đăng ký)

```bash
git clone https://github.com/august0362/BTL-CVDHT-CNTT.git
cd BTL-CVDHT-CNTT
docker compose --profile mock up --build
```

Mở <http://localhost:8599>. Kết quả ở chế độ này là **dữ liệu giả lập**, chỉ để xem giao diện.

### Cách 2: Chạy đầy đủ với mô hình thật

**Bước 1. Tải trọng số cho web** (POSTER, MA3D-Net, YOLO, RealESRGAN, 3DDFA-V2; khoảng 850 MB, có kiểm SHA-256):

```bash
python environments/docker/download_weights.py
```

**Bước 2. Đồng ý giấy phép FLAME/EMOCA.** Đăng ký tài khoản và đồng ý điều khoản ở cả hai trang
[flame.is.tue.mpg.de](https://flame.is.tue.mpg.de) và [emoca.is.tue.mpg.de](https://emoca.is.tue.mpg.de). Giấy phép
yêu cầu mỗi người tự đồng ý, nên repo không chứa và không phân phối lại các file này. Sau đó tạo file `.env` cạnh
`docker-compose.yml`:

```
ACCEPT_FLAME_EMOCA_LICENSES=1
```

**Bước 3. Build và chạy:**

```bash
docker compose --profile real up --build
```

- Lần đầu: build image EMOCA mất khoảng 30–60 phút, sau đó container tự tải tài sản FLAME/EMOCA (vài GB) vào volume
  `emoca_assets`. Những lần sau chỉ cần `docker compose --profile real up`, khởi động khoảng 1 phút.
- Web chỉ mở khi EMOCA đã sẵn sàng. Mở <http://localhost:8599>.
- Dừng: `Ctrl+C` hoặc `docker compose --profile real down`. Xoá cả tài sản đã tải: thêm `-v`.

### Gỡ lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `emoca` thoát mã 10 | Chưa có `ACCEPT_FLAME_EMOCA_LICENSES=1` trong `.env` (Bước 2). |
| `emoca` thoát mã 11 | Tải texture từ Google Drive thất bại. Tải tay `FLAME_albedo_from_BFM.npz`, chạy `docker compose --profile real create emoca`, rồi `docker compose cp FLAME_albedo_from_BFM.npz emoca:/opt/emoca/assets/FLAME/texture/` và chạy lại. |
| `could not select device driver "nvidia"` | Docker chưa thấy GPU: cập nhật driver NVIDIA, bật WSL2 backend (Windows) hoặc cài NVIDIA Container Toolkit (Linux). |
| Web báo thiếu checkpoint | Chạy lại `python environments/docker/download_weights.py --check` để xem file nào thiếu hoặc lệch SHA-256. |
| Cổng 8599 đã bị dùng | Tắt chương trình đang chiếm cổng, hoặc đổi `"8599:8599"` trong `docker-compose.yml` thành `"8600:8599"`. |
| Xem log | `docker compose --profile real logs -f emoca` |

## Chạy không dùng Docker (Windows + WSL2)

Đây là cách nhóm dùng để phát triển: web chạy trong venv Windows, EMOCA chạy trong Ubuntu WSL2. Khởi động bằng
`run_demo.bat`, dừng bằng `stop_demo.bat`. Hướng dẫn chi tiết ở
[`documents/docs/RUNBOOK_DEMO.md`](documents/docs/RUNBOOK_DEMO.md).

```powershell
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8599
```

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| [`backend/`](backend/) | `web/` (điều phối, suy luận POSTER/MA3D, client EMOCA), `services/` (dịch vụ EMOCA, sửa roll, tiền xử lý ảnh), `model/` (mã MA3D-Net + chỗ đặt trọng số), `configs/`, `contracts/` |
| [`frontend/`](frontend/) | Giao diện Streamlit (`app.py`) và theme |
| [`environments/`](environments/) | `docker/` (Dockerfile, script tải trọng số), `web/` (requirements), `emoca_wsl/` (env conda của EMOCA) |
| [`database/`](database/) | Kết quả benchmark (CSV/JSON); ảnh dữ liệu không được commit |
| [`documents/`](documents/) | Runbook, ADR, báo cáo QA |
| [`another/`](another/) | Notebook Kaggle/Colab, test |

## Giấy phép và dữ liệu

- **FLAME, EMOCA:** giấy phép nghiên cứu phi thương mại của MPI-IS. Không có trong repo hay trong image; mỗi người tự
  đăng ký và tải.
- **3DDFA-V2:** mã MIT; các file BFM theo điều khoản Basel Face Model (phi thương mại).
- **Real-ESRGAN:** BSD-3-Clause. **YOLO26-face:** lấy từ `marceloeatworld/yolo26-training`.
- **Checkpoint POSTER/MA3D-Net:** do nhóm huấn luyện trên RAF-DB, chỉ dùng cho nghiên cứu.
- Ảnh tải lên chỉ xử lý trong bộ nhớ, không lưu xuống đĩa, không gửi ra ngoài máy.
- Kết quả trên giao diện chỉ là minh hoạ định tính, không phải đánh giá độ chính xác.

## Dành cho người duy trì repo: phát hành trọng số

`download_weights.py` tải checkpoint POSTER/MA3D-Net từ GitHub Release tag `weights-v1` của repo này. Tạo release một
lần trên trang GitHub (*Releases → Draft a new release*, tag `weights-v1`) và đính kèm 5 file từ
`backend/model/ma3d/checkpoints/`:

```
ir50.pth   mobilefacenet_model_best.pth.tar   raf_MA3D.pth   rafdb_POSTER.pth   rafdb_emoca_stats.npz
```

Nếu đổi checkpoint thì cập nhật SHA-256 trong `backend/configs/models.yaml` và `environments/docker/download_weights.py`,
rồi phát hành tag mới.
