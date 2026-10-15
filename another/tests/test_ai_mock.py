"""Tự kiểm web/emoca_client.py + web/engine.py (chế độ mock). Chạy: python tests/test_ai_mock.py

Ảnh giả sinh trong bộ nhớ; không ghi file ảnh ra đĩa.
"""

import copy
import json
import math
import sys
from pathlib import Path

import jsonschema
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from web import emoca_client, engine, mock_data  # noqa: E402

EMOCA_SCHEMA = json.loads((ROOT / "backend" / "contracts" / "emoca_analyze.schema.json").read_text(encoding="utf-8"))
MA3D_SCHEMA = json.loads((ROOT / "backend" / "contracts" / "ma3d_predict.schema.json").read_text(encoding="utf-8"))


def make_image(size=(640, 480), color=(200, 150, 120)) -> Image.Image:
    img = Image.new("RGB", size, color)
    img.putpixel((1, 1), (0, 0, 0))
    return img


def check_predictions(result):
    jsonschema.validate(result, MA3D_SCHEMA)
    for p in result["predictions"]:
        if p["status"] != "ok":
            continue
        assert len(p["labels"]) == len(p["probs"]), p["model_id"]
        assert abs(sum(p["probs"]) - 1.0) < 1e-6, (p["model_id"], sum(p["probs"]))
        idx = max(range(len(p["probs"])), key=p["probs"].__getitem__)
        assert p["top1"] == {"index": idx, "label": p["labels"][idx], "prob": p["probs"][idx]}, p["top1"]
        assert "mock" in p["display_name"].lower(), p["display_name"]


def by_id(result):
    return {p["model_id"]: p for p in result["predictions"]}


def test_emoca_ok():
    img = make_image((640, 480))
    r = emoca_client.get_analysis(img)
    jsonschema.validate(r, EMOCA_SCHEMA)
    assert "error" not in r and r["image"] == {"width": 640, "height": 480}
    assert "MOCK" in r["model"]["name"]
    assert len(r["faces"]) == 1
    f = r["faces"][0]
    assert any("mock" in w for w in f["warnings"]), f["warnings"]
    assert {k: len(v) for k, v in f["codes"].items()} == mock_data.CODE_DIMS
    # original_px → crop224 qua matrix_3x3 phải khớp landmark crop224 (bất biến §7).
    m = f["crop_transform"]["matrix_3x3"]
    for (xo, yo), (xc, yc) in zip(f["landmarks"]["original_px"], f["landmarks"]["crop224"]):
        assert abs(m[0][0] * xo + m[0][1] * yo + m[0][2] - xc) < 0.01
        assert abs(m[1][0] * xo + m[1][1] * yo + m[1][2] - yc) < 0.01
    x0, y0, x1, y1 = f["bbox_original_px"]
    assert 0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480
    crop = emoca_client.decode_crop(f)
    assert crop is not None and crop.size == (224, 224)
    for key in f["renders"]:
        assert emoca_client.decode_crop({"crop_image": f["renders"][key]}).size == (224, 224)
    print("PASS emoca_ok: faces=1, codes 100/50/50/6/128, landmark crop224 == M * original_px")


def test_emoca_deterministic_and_modes():
    img = make_image()
    assert emoca_client.get_analysis(img)["faces"][0]["codes"] == emoca_client.get_analysis(img.copy())["faces"][0]["codes"]
    rgba = make_image((100, 80)).convert("RGBA")
    jsonschema.validate(emoca_client.get_analysis(rgba), EMOCA_SCHEMA)
    cases = [
        ("real", emoca_client.get_analysis(img, mode="real"), "model_unavailable"),
        ("bad_mode", emoca_client.get_analysis(img, mode="xyz"), "contract_mismatch"),
        ("tiny_no_face", emoca_client.get_analysis(make_image((20, 20))), "no_face"),
        ("not_image", emoca_client.get_analysis(b"bytes"), "image_invalid"),
        ("max_faces_0", emoca_client.get_analysis(img, max_faces=0), "contract_mismatch"),
    ]
    for name, r, code in cases:
        jsonschema.validate(r, EMOCA_SCHEMA)
        assert set(r) == {"error"} and r["error"]["code"] == code, (name, r)
        print(f"PASS emoca_{name}: error={code}")
    assert emoca_client.decode_crop({}) is None and emoca_client.decode_crop({"crop_image": {"b64": "!!"}}) is None
    print("PASS emoca_deterministic + decode_crop_bad -> None")


def test_engine_ok():
    img = make_image()
    face = emoca_client.get_analysis(img)["faces"][0]
    crop = emoca_client.decode_crop(face)
    r = engine.predict(crop, face["codes"], attention=True)
    check_predictions(r)
    p = by_id(r)
    assert list(p) == ["poster_rafdb", "ma3d_rafdb"] and all(x["status"] == "ok" for x in p.values())
    def is_map(m):
        return m["grid"] == [7, 7] and len(m["values"]) == 49 and min(m["values"]) == 0 and max(m["values"]) == 1

    # ADR-013: POSTER cũng có attention; feature_maps có mặc định, fusion_delta chỉ ở MA3D.
    assert is_map(p["poster_rafdb"]["attention"]) and is_map(p["ma3d_rafdb"]["attention"])
    assert set(p["poster_rafdb"]["feature_maps"]) == {"image_backbone", "landmark_backbone"}
    assert set(p["ma3d_rafdb"]["feature_maps"]) == {"image_backbone", "landmark_backbone", "fusion_delta"}
    assert all(is_map(m) for x in p.values() for m in x["feature_maps"].values())
    r2 = engine.predict(crop, face["codes"])
    check_predictions(r2)
    assert all(x["attention"] is None for x in r2["predictions"])
    r2b = engine.predict(crop, face["codes"], feature_maps=False)
    check_predictions(r2b)
    assert all(x["feature_maps"] is None for x in r2b["predictions"])
    assert [x["probs"] for x in r2b["predictions"]] == [x["probs"] for x in r["predictions"]], \
        "bật/tắt ảnh trung gian không được đổi xác suất"
    r3 = engine.predict(crop, face["codes"], models=["ma3d_rafdb"])
    check_predictions(r3)
    assert [x["model_id"] for x in r3["predictions"]] == ["ma3d_rafdb"]
    print("PASS engine_ok: 2 models, sum~1, top1==argmax, attention 7x7 cả hai, feature_maps (fusion_delta chỉ MA3D)")


def test_engine_codes_missing():
    img = make_image()
    good = emoca_client.get_analysis(img)["faces"][0]["codes"]
    missing = copy.deepcopy(good)
    del missing["exp"]
    short = copy.deepcopy(good)
    short["detail"] = short["detail"][:127]
    nan = copy.deepcopy(good)
    nan["pose"][0] = math.nan
    for name, codes in [("none", None), ("missing_exp", missing), ("short_detail", short), ("nan_pose", nan),
                        ("empty", {})]:
        r = engine.predict(img, codes)
        check_predictions(r)
        p = by_id(r)
        assert p["ma3d_rafdb"]["status"] == "error" and p["ma3d_rafdb"]["error"]["code"] == "codes_missing", (name, p)
        assert "probs" not in p["ma3d_rafdb"]
        # POSTER không dùng 3DMM nên vẫn chạy (một mô hình lỗi không làm hỏng mô hình kia).
        assert p["poster_rafdb"]["status"] == "ok", name
        print(f"PASS engine_codes_{name}: ma3d=codes_missing, poster=ok")


def test_engine_determinism_and_disagreement():
    img = make_image()
    codes = emoca_client.get_analysis(img)["faces"][0]["codes"]
    a = engine.predict(img, codes, attention=True)
    b = engine.predict(img.copy(), codes, attention=True)
    strip = lambda r: [{k: v for k, v in p.items() if k != "latency_ms"} for p in r["predictions"]]  # noqa: E731
    assert strip(a) == strip(b), "không xác định"
    states = set()
    for i in range(40):
        im = make_image((64, 64), (i * 6, 255 - i * 5, (i * 37) % 256))
        p = by_id(engine.predict(im, codes))
        states.add(p["poster_rafdb"]["top1"]["index"] == p["ma3d_rafdb"]["top1"]["index"])
    assert states == {True, False}, states
    print("PASS engine_deterministic + both agree/disagree states occur over 40 images")


def test_engine_modes():
    img = make_image()
    codes = emoca_client.get_analysis(img)["faces"][0]["codes"]
    cases = [
        ("bad_mode", lambda: engine.predict(img, codes, mode="xyz"), "contract_mismatch"),
        ("not_image", lambda: engine.predict(b"bytes", codes), "contract_mismatch"),
        ("unknown_model", lambda: engine.predict(img, codes, models=["nope"]), "model_unavailable"),
    ]
    try:
        import torch  # noqa: F401
    except ImportError:  # không có torch ⇒ chế độ real phải báo model_unavailable cho mọi mô hình
        cases.insert(0, ("real", lambda: engine.predict(img, codes, mode="real"), "model_unavailable"))
    else:
        print("  SKIP engine_real: có torch nên POSTER chạy thật — đã kiểm ở tests/test_engine_poster_real.py")
    for name, call, code in cases:
        r = call()
        check_predictions(r)
        assert r["predictions"] and all(p["status"] == "error" and p["error"]["code"] == code
                                        for p in r["predictions"]), (name, r)
        print(f"PASS engine_{name}: all predictions error={code}")


def main():
    test_emoca_ok()
    test_emoca_deterministic_and_modes()
    test_engine_ok()
    test_engine_codes_missing()
    test_engine_determinism_and_disagreement()
    test_engine_modes()
    print("ALL PASS")


if __name__ == "__main__":
    main()
