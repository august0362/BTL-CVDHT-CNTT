"""M8 benchmark: does 3DDFA-V2 frontalization help POSTER / MA3D-Net on RAF-DB test?

For each test image and each in-plane rotation angle, two branches are compared:
  raw     : image (rotated) -> EMOCA -> POSTER + MA3D-Net
  frontal : image (rotated) -> 3DDFA-V2 frontalize -> EMOCA -> POSTER + MA3D-Net
RealESRGAN and YOLO are deliberately skipped: RAF-DB test images are already
aligned face crops, and the goal is to isolate the frontalization step (ADR-026).
Angle 0 doubles as the natural-pose experiment (results binned by estimated |yaw|).

Not a test (not collected by run_all). Resumable: rows already in the JSONL are skipped.
Only numbers are written; no image is saved.

Usage (venv web, real EMOCA running):
  PYTHONPATH=".;web" python tests/bench_frontalization_rafdb.py --per-class 10 --angles 0,90
  PYTHONPATH=".;web" python tests/bench_frontalization_rafdb.py --summary
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
DATA = ROOT / "database" / "image" / "archive" / "DATASET" / "test"
OUT = ROOT / "documents" / "docs" / "qa" / "M8_RAFDB" / "results.jsonl"
# RAF-DB folder k -> label index k-1 (PROJECT_CONTEXT §4); names as in configs/labels.yaml.
CLASSES = ["surprise", "fear", "disgust", "happiness", "sadness", "anger", "neutral"]
MODELS = ("poster_rafdb", "ma3d_rafdb")
EMOCA = {"mode": "real", "url": "http://127.0.0.1:8765"}


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _rotate(image: Image.Image, angle: int) -> Image.Image:
    if angle == 0:
        return image
    # expand keeps the whole face; corners filled with the mean colour (no black wedges).
    mean = tuple(int(v) for v in image.resize((1, 1), Image.BILINEAR).getpixel((0, 0)))
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=mean)


def _predictions(result: dict | None) -> dict:
    out = {}
    for face in (result or {}).get("faces") or []:
        for pred in face.get("predictions") or []:
            mid = pred.get("model_id")
            if mid in MODELS and mid not in out:
                if pred.get("status") == "ok":
                    out[mid] = {"label": pred["top1"]["label"],
                                "probs": dict(zip(pred["labels"], [round(float(p), 5) for p in pred["probs"]]))}
                else:
                    out[mid] = {"error": str((pred.get("error") or {}).get("code") or pred.get("status"))}
    for mid in MODELS:
        out.setdefault(mid, {"error": str(((result or {}).get("error") or {}).get("code") or "no_prediction")})
    return out


def _items(per_class: int | None) -> list[tuple[str, int]]:
    items = []
    for k, name in enumerate(CLASSES, start=1):
        files = sorted((DATA / str(k)).glob("*.jpg"))
        items += [(f"{k}/{p.name}", k - 1) for p in (files[:per_class] if per_class else files)]
    return items


def run(per_class: int | None, angles: list[int]) -> None:
    from orchestrator import analyze
    from services.frontalization import get_tddfa_v2_frontalizer

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["image"], row["angle"]))
    frontalizer = get_tddfa_v2_frontalizer()
    todo = [(img, y, a) for img, y in _items(per_class) for a in angles if (img, a) not in done]
    print(f"todo {len(todo)} (done {len(done)})", flush=True)
    t_start = time.perf_counter()
    with OUT.open("a", encoding="utf-8") as sink:
        for n, (rel, y, angle) in enumerate(todo, start=1):
            with Image.open(DATA / rel) as src:
                image = _rotate(src.convert("RGB"), angle)
            row = {"image": rel, "label": CLASSES[y], "angle": angle}
            t0 = time.perf_counter()
            row["raw"] = _predictions(analyze(_png(image), emoca=EMOCA, engine_mode="real", max_faces=1))
            row["ms_raw"] = round((time.perf_counter() - t0) * 1000)
            try:
                result = frontalizer.frontalize(image)
                row["pose"] = [round(result.yaw_deg, 2), round(result.pitch_deg, 2), round(result.roll_deg, 2)]
                row["occluded"] = round(result.occluded_fraction, 4)
                row["ms_frontalize"] = round(result.time_ms)
                t0 = time.perf_counter()
                row["frontal"] = _predictions(analyze(_png(result.image), emoca=EMOCA, engine_mode="real", max_faces=1))
                row["ms_frontal"] = round((time.perf_counter() - t0) * 1000)
            except Exception as exc:  # recorded per row, never hidden
                row["frontal_error"] = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
                row["frontal"] = {mid: {"error": "frontalize_failed"} for mid in MODELS}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if n % 25 == 0 or n == len(todo):
                rate = (time.perf_counter() - t_start) / n
                print(f"{n}/{len(todo)} {rate:.2f}s/row eta {rate * (len(todo) - n) / 60:.1f} min", flush=True)


def summary() -> None:
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines()]

    def stat(subset, branch, mid):
        n = len(subset)
        valid = [r for r in subset if "label" in r[branch][mid]]
        correct = sum(r[branch][mid]["label"] == r["label"] for r in valid)
        conf_true = [r[branch][mid]["probs"].get(r["label"], 0.0) for r in valid]
        return n, len(valid), correct, (sum(conf_true) / len(conf_true) if conf_true else 0.0)

    def table(title, groups):
        print(f"\n### {title}")
        print("| nhóm | n | mô hình | coverage raw→frontal | accuracy (trên n) raw→frontal | p(nhãn đúng) TB raw→frontal |")
        print("|---|---|---|---|---|---|")
        for name, subset in groups:
            if not subset:
                continue
            for mid in MODELS:
                n, vr, cr, pr = stat(subset, "raw", mid)
                _, vf, cf, pf = stat(subset, "frontal", mid)
                print(f"| {name} | {n} | {mid} | {vr / n:.1%} → {vf / n:.1%} | {cr / n:.1%} → {cf / n:.1%} "
                      f"({(cf - cr) / n * 100:+.1f}) | {pr:.3f} → {pf:.3f} |")

    by_angle = defaultdict(list)
    for r in rows:
        by_angle[r["angle"]].append(r)
    table("Theo góc xoay nhân tạo (roll)", [(f"{a}°", by_angle[a]) for a in sorted(by_angle)])
    natural = by_angle.get(0, [])
    bins = [("|yaw|<15°", lambda y: y < 15), ("15–30°", lambda y: 15 <= y < 30), ("|yaw|≥30°", lambda y: y >= 30)]
    table("Góc tự nhiên (0°), chia theo |yaw| 3DDFA", [
        (name, [r for r in natural if "pose" in r and test(abs(r["pose"][0]))]) for name, test in bins])
    flips = defaultdict(lambda: [0, 0])
    for r in natural:
        for mid in MODELS:
            a, b = r["raw"][mid].get("label"), r["frontal"][mid].get("label")
            if a and b and a != b:
                flips[mid][0 if b == r["label"] else 1] += 1
    for mid, (good, bad) in flips.items():
        print(f"\n{mid} @0°: đổi nhãn sau frontalize → đúng hơn {good}, sai đi {bad}")
    errors = defaultdict(int)
    for r in rows:
        for branch in ("raw", "frontal"):
            for mid in MODELS:
                if "error" in r[branch][mid]:
                    errors[(r["angle"], branch, r[branch][mid]["error"])] += 1
    if errors:
        print("\nlỗi theo (góc, nhánh, mã):", dict(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=None, help="limit images per class (default: all)")
    parser.add_argument("--angles", default="0", help="comma-separated roll angles, e.g. 0,30,60,90")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1258
    if args.summary:
        summary()
    else:
        run(args.per_class, [int(a) for a in args.angles.split(",")])


if __name__ == "__main__":
    main()
