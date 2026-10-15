"""AI — dữ liệu giữ chỗ cho chế độ mock của `emoca_client.py` và `engine.py`.

Gom MỘT chỗ mọi nhãn/metadata mô hình tạm thời, để sau này thay bằng
`configs/labels.yaml` + `configs/models.yaml` (CONTRACTS_ADR.md Phần C) mà không
phải sửa logic. Không import torch, không import Streamlit.

Mọi giá trị ở đây là GIẢ hoặc CHƯA XÁC MINH — không dùng để báo kết quả thật.
"""

from __future__ import annotations

import hashlib

CONTRACT_VERSION = "0.3.0"  # ADR-019: web_analyze thêm crop_image/crop_transform/landmarks_crop224 (0.2.0: ADR-013 feature_maps)

# CHỈ DÙNG CHO MOCK — quy ước công khai phổ biến của RAF-DB (thư mục 1..7). Thứ tự nhãn của
# checkpoint thật đã được 02_CONFIG_METRICS xác minh (2 nguồn, M2-01) và ghi ở configs/models.yaml
# (`labels`); chế độ real đọc từ đó, không đọc hằng này. Tên hằng giữ nguyên.
RAFDB_LABELS_UNVERIFIED = ("surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral")

# Registry giả. `uses_3dmm`: mô hình có cần mã 3DMM không (MA3D có; POSTER không — ADR-011, đã có
# bằng chứng trên checkpoint: mm_fusion chưa huấn luyện); `supports_attention`: có trả attention 7×7
# không (cả hai: POSTER cũng có transformer pyramid_fuse — ADR-013). `uses_3dmm` cũng quyết định có
# `feature_maps.fusion_delta` hay không.
MOCK_MODELS = (
    {
        "id": "poster_rafdb",
        "display_name": "POSTER (RAF-DB) [mock]",
        "dataset": "RAF-DB",
        "labels": RAFDB_LABELS_UNVERIFIED,
        "uses_3dmm": False,
        "supports_attention": True,
    },
    {
        "id": "ma3d_rafdb",
        "display_name": "MA3D-Net (RAF-DB) [mock]",
        "dataset": "RAF-DB",
        "labels": RAFDB_LABELS_UNVERIFIED,
        "uses_3dmm": True,
        "supports_attention": True,
    },
)

# Số chiều 5 nhóm mã FLAME của EMOCA — ĐÃ XÁC MINH (PROJECT_CONTEXT §3,
# contracts/emoca_analyze.schema.json). Chưa chuẩn hoá.
CODE_DIMS = {"shape": 100, "exp": 50, "tex": 50, "pose": 6, "detail": 128}

# Mô tả EMOCA giả (trả trong `model` của emoca_analyze).
MOCK_EMOCA_MODEL = {"name": "MOCK-EMOCA_v2_lr_mse_20", "mode": "detail", "upstream_commit": "e0be0db"}

MOCK_FACE_FRACTION = 0.6  # cạnh bbox mặt giả = 0.6 × cạnh ngắn của ảnh, đặt giữa ảnh
MOCK_CROP_SCALE = 1.25    # EMOCA: bbox FAN × 1.25 rồi resize (CONTRACTS_ADR Phần A)
MOCK_MIN_SIDE_PX = 32     # cạnh ngắn nhỏ hơn mức này → mock giả lập `no_face`
CROP_SIZE = 224           # kích thước crop đưa vào EMOCA/POSTER/MA3D (hệ `crop224`)

# Khoảng 1/3 số ảnh MA3D "chọn" lớp khác POSTER, để giao diện thử được trạng thái bất đồng.
MOCK_DISAGREE_MODULO = 3

MOCK_WARNING = "mock: dữ liệu giả sinh trong web/, không phải suy luận thật"


def image_digest(image) -> bytes:
    """SHA-256 của nội dung ảnh trong bộ nhớ (mode + kích thước + pixel); dùng làm seed mock.

    Không ghi ảnh ra đĩa, không log (ADR-009).
    """
    h = hashlib.sha256()
    h.update(image.mode.encode("ascii"))
    h.update(f"{image.width}x{image.height}".encode("ascii"))
    h.update(image.tobytes())
    return h.digest()
