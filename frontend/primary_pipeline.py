"""Run the existing main-page pipeline without calling Streamlit from a worker."""
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from PIL import Image, ImageOps


def run_primary(data: bytes, *, prepare_fn, roll_fn, analyze_fn, multiface_fn,
                emoca: dict, engine_mode: str, face_limit: int, report) -> dict:
    with Image.open(BytesIO(data)) as loaded:
        image = ImageOps.exif_transpose(loaded).convert("RGB")
    report(message="Đang phát hiện khuôn mặt…")
    try:
        views = prepare_fn(image, max_faces=face_limit,
                           progress_callback=lambda message: report(message=message))
    except Exception as exc:
        views = SimpleNamespace(source_image=image, faces=[], enhanced_image=None,
                                detection_source=None, detection_complete=False,
                                errors=[f"{type(exc).__name__}: {exc}"], stage_times_ms={})
    if len(views.faces) > 1:
        report(message=f"Đang phân tích {len(views.faces)} khuôn mặt…")
        multi = multiface_fn(image, emoca=emoca, engine_mode=engine_mode, max_faces=face_limit,
                             views=views, progress_callback=lambda message: report(message=message))
        return {"image": image, "views": views, "multi": multi,
                "corrected": image, "roll_info": None, "result": None}
    report(message="Đang kiểm tra góc nghiêng…")
    corrected, roll_info = roll_fn(image, views)
    buffer = BytesIO()
    corrected.convert("RGB").save(buffer, format="PNG")

    def progress(module, state, detail):
        report(message=detail or {"emoca": "Đang xử lý EMOCA…", "alignment": "Đang căn chỉnh…",
                                  "classifier": "Đang suy luận…"}.get(module, module))

    result = analyze_fn(buffer.getvalue(), emoca=emoca, engine_mode=engine_mode,
                        max_faces=1, progress_callback=progress)
    if not isinstance(result, dict):
        raise TypeError("Bộ điều phối không trả kết quả đúng định dạng.")
    return {"image": image, "views": views, "multi": None,
            "corrected": corrected, "roll_info": roll_info, "result": result}
