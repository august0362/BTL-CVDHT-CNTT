# contracts/ — Hợp đồng giữa module (contract_version: 0.3.0 — ADR-019 thêm `crop_image`/`crop_transform`/`landmarks_crop224` vào `faces[]` của web_analyze; 0.2.0: ADR-013 thêm `feature_maps`)

> Chủ sở hữu: `GENERAL` (đổi tên từ `01_ML_ARCHITECT` — Session 03, 2026-09-24). Nguồn định nghĩa: `docs/context/CONTRACTS_ADR.md` Phần A–B.
> Đổi bất kỳ file `.schema.json` nào ở đây bắt buộc: (1) ADR mới trong `CONTRACTS_ADR.md`, (2) tăng `contract_version`, (3) cập nhật ví dụ + fixture vàng liên quan.

| File schema | Dùng cho | Ai điền dữ liệu thật |
|---|---|---|
| `emoca_analyze.schema.json` | `emoca_svc` — `POST /v1/analyze` (B1) | `01_AI` |
| `ma3d_predict.schema.json` | Kết quả suy luận POSTER/MA3D-Net (B2) — hiện tại là giá trị trả về nội bộ của `web/engine.py` (ADR-012), không phải HTTP riêng | `01_AI` |
| `web_analyze.schema.json` | Dữ liệu tổng hợp mà `frontend/app.py` dùng để vẽ giao diện (B3, rút gọn theo ADR-012 — không phải HTTP backend↔frontend vì đã gộp) | `03_BACKEND` (gộp response) + `04_FRONTEND` (tiêu thụ) |

Mỗi schema có 1 ví dụ hợp lệ (`*.example.valid.json`) và 1 ví dụ lỗi (`*.example.invalid.json`) để kiểm tự động không cần dựng mô hình thật.

Kiểm nhanh (cần `pip install jsonschema`):

```bash
python -c "
import json, jsonschema
schema = json.load(open('contracts/emoca_analyze.schema.json', encoding='utf-8'))
valid = json.load(open('contracts/emoca_analyze.example.valid.json', encoding='utf-8'))
invalid = json.load(open('contracts/emoca_analyze.example.invalid.json', encoding='utf-8'))
jsonschema.validate(valid, schema)
print('valid ví dụ: OK')
try:
    jsonschema.validate(invalid, schema)
    print('invalid ví dụ: SAI — lẽ ra phải bị từ chối')
except jsonschema.ValidationError as e:
    print('invalid ví dụ: bị từ chối đúng như kỳ vọng ->', e.message)
"
```
