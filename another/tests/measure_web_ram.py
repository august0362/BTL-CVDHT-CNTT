"""Measure Windows working set through the real local web inference flow.

Run directly in the web venv; this filename is intentionally outside run_all.py's
test_*/check_* discovery. No image or measurement is written to disk.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
DATASET = ROOT / "database" / "image" / "archive" / "DATASET" / "test"
EMOCA_URL = "http://127.0.0.1:8765"
MIB = 1024 * 1024


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def windows_working_set_mb() -> dict[str, float]:
    """Current and process-lifetime peak working set, using PSAPI like M7 scripts."""
    if os.name != "nt":
        raise RuntimeError("Windows working-set measurement requires Windows.")
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"current_mb": round(counters.WorkingSetSize / MIB, 2),
            "peak_mb": round(counters.PeakWorkingSetSize / MIB, 2)}


def emoca_health() -> dict:
    with urllib.request.urlopen(f"{EMOCA_URL}/v1/health", timeout=3) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def dataset_images() -> list[Path]:
    if not DATASET.is_dir():
        return []
    return sorted(path for path in DATASET.rglob("*")
                  if path.is_file() and path.name.lower().endswith("_aligned.jpg"))


def compose_two_faces(first: Image.Image, second: Image.Image) -> Image.Image:
    face_a = first.convert("RGB").resize((160, 160), Image.Resampling.LANCZOS)
    face_b = second.convert("RGB").resize((160, 160), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (340, 180), (245, 245, 245))
    canvas.paste(face_a, (5, 10))
    canvas.paste(face_b, (175, 10))
    return canvas


def loaded_models(orchestrator, health: dict | None = None) -> list[str]:
    """Report loaded model objects from the lazy registries used by these flows."""
    loaded: list[str] = []
    try:
        from . import predata_experiment
    except ImportError:
        from web import predata_experiment
    stack = predata_experiment._STACK
    if stack is not None:
        if stack.general is not None:
            loaded.append("RealESRGAN")
        if stack.yolo is not None or stack.detector is not None:
            loaded.append("YOLO26-face")
    registry = getattr(orchestrator.engine, "_REAL_MODELS", None) or {}
    loaded.extend(f"FER:{name}" for name in sorted(registry))
    # Module name is tdfa_v2_wrapper; do not swallow ImportError (a typo here hid the 3DDFA load).
    from services.frontalization import tdfa_v2_wrapper
    frontalizer = tdfa_v2_wrapper._SINGLETON
    if frontalizer is not None and frontalizer._loaded:
        loaded.append("3DDFA-V2")
    if health is not None and health.get("status") == "up" and health.get("mode") == "real":
        model = health.get("model")
        name = model.get("name") if isinstance(model, dict) else None
        loaded.append(f"EMOCA service:{name or 'real/up'}")
    return loaded


def memory_snapshot() -> dict[str, float]:
    return windows_working_set_mb()


def execute_flow(orchestrator, memory: Callable[[], dict[str, float]], health: dict) -> dict:
    samples = dataset_images()
    if len(samples) < 2:
        raise FileNotFoundError(
            f"Cần ít nhất 2 RAF-DB aligned crop trong {DATASET}; tìm thấy {len(samples)}. Đo bị bỏ qua."
        )
    with Image.open(samples[0]) as src:
        single_image = src.convert("RGB")
    with Image.open(samples[0]) as a, Image.open(samples[1]) as b:
        two_face_image = compose_two_faces(a, b)

    settings = {"mode": "real", "url": EMOCA_URL, "timeout_s": 90}
    results: list[dict] = []

    def require_predictions(result: dict, expected: set[str], stage: str) -> None:
        if result.get("status") != "ok":
            raise RuntimeError(f"{stage} did not complete with status=ok: {result.get('error')}")
        seen = {prediction.get("model_id") for face in result.get("faces", [])
                for prediction in face.get("predictions", [])
                if prediction.get("status") == "ok"}
        if not expected.issubset(seen):
            raise RuntimeError(f"{stage} missing real predictions {sorted(expected - seen)}; got {sorted(seen)}")

    def mark(name: str, started: float) -> None:
        results.append({"step": name, "wall_seconds": round(time.perf_counter() - started, 3),
                        "working_set": memory(), "models_loaded": loaded_models(orchestrator, health)})

    started = time.perf_counter()
    views_single = orchestrator.prepare_multiface_views(single_image, max_faces=30)
    if len(views_single.faces) != 1:
        raise RuntimeError(f"Single-face fixture was detected as {len(views_single.faces)} faces: "
                           f"{samples[0].name}; choose an aligned crop YOLO detects once.")
    png = _png_bytes(single_image)
    single_result = orchestrator.analyze(png, emoca=settings, engine_mode="real", max_faces=1)
    require_predictions(single_result, {"poster_rafdb", "ma3d_rafdb"}, "single-face flow")
    mark("b_single_face", started)

    started = time.perf_counter()
    views_two = orchestrator.prepare_multiface_views(two_face_image, max_faces=30)
    if len(views_two.faces) != 2:
        raise RuntimeError(f"Two-face canvas yielded {len(views_two.faces)} detections; expected 2.")
    state_multi = orchestrator.analyze_multiface(two_face_image, emoca=settings,
                                                  engine_mode="real", max_faces=30, views=views_two)
    if state_multi.get("kind") != "faces":
        raise RuntimeError(f"Two-face analyze_multiface returned {state_multi.get('kind')!r}.")
    if len(state_multi.get("rows", [])) != 2:
        raise RuntimeError(f"Two-face analysis returned {len(state_multi.get('rows', []))} result rows.")
    for row in state_multi["rows"]:
        require_predictions(row.get("result") or {}, {"ma3d_rafdb"}, f"multi-face #{row.get('face_id')}")
    mark("c_multi_face", started)

    started = time.perf_counter()
    zero_image = Image.new("RGB", (320, 240), (190, 190, 190))
    zero_views = orchestrator.prepare_multiface_views(zero_image, max_faces=30)
    if zero_views.faces:
        raise RuntimeError(f"Blank zero-face fixture unexpectedly detected {len(zero_views.faces)} faces.")
    zero_result = orchestrator.analyze(_png_bytes(zero_image), emoca=settings, engine_mode="real", max_faces=1)
    if "RealESRGAN" not in loaded_models(orchestrator, health):
        raise RuntimeError("The zero-face preflight did not initialize the lazy RealESRGAN model.")
    mark("d_zero_face_lazy_realesrgan", started)

    started = time.perf_counter()
    frontal = orchestrator.run_frontalization_experiment(
        views_two, emoca=settings, engine_mode="real")
    mark("e_m8_roll_experiment", started)
    if "3DDFA-V2" not in results[-1]["models_loaded"]:
        raise RuntimeError("The M8 roll experiment did not load 3DDFA-V2.")

    return {"status": "ok", "emoca": {"url": EMOCA_URL, "health_status": health.get("status"),
                                         "mode": health.get("mode")},
            "inputs": {"single_face": samples[0].name, "two_face_sources": [samples[0].name, samples[1].name],
                       "two_face_canvas_px": list(two_face_image.size), "images_written": False},
            "steps": results,
            "frontalization": {"result_kind": frontal.get("kind") if isinstance(frontal, dict) else type(frontal).__name__}}


def _png_bytes(image: Image.Image) -> bytes:
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def self_test() -> dict:
    """Exercise sequence/arguments/output schema with model and memory stubs."""
    from types import SimpleNamespace
    global dataset_images, memory_snapshot, loaded_models

    events: list[tuple] = []
    fake_faces_one = [object()]
    fake_faces_two = [object(), object()]
    def fake_prediction_result(model_ids: list[str]) -> dict:
        return {"status": "ok", "faces": [{"predictions": [
            {"model_id": model_id, "status": "ok"} for model_id in model_ids]}]}

    fake_loaded: list[str] = []
    def fake_prepare(image, **kw):
        events.append(("prepare", image.size, kw))
        if image.size == (320, 240):
            fake_loaded.append("RealESRGAN")
            faces = []
        else:
            fake_loaded.append("YOLO26-face")
            faces = fake_faces_one if image.size == (80, 60) else fake_faces_two
        return SimpleNamespace(faces=faces)

    def fake_frontal(views, **kw):
        events.append(("frontal", kw))
        fake_loaded.append("3DDFA-V2")
        return {"kind": "ok"}

    fake = SimpleNamespace(
        prepare_multiface_views=fake_prepare,
        analyze=lambda image, **kw: events.append(("analyze", kw)) or fake_prediction_result(
            ["poster_rafdb", "ma3d_rafdb"]),
        analyze_multiface=lambda image, **kw: events.append(("multi", kw)) or {"kind": "faces", "rows": [
            {"face_id": i, "result": fake_prediction_result(["ma3d_rafdb"])} for i in (1, 2)]},
        run_frontalization_experiment=fake_frontal,
        engine=SimpleNamespace(_REAL_MODELS={}),
    )
    old_images, old_memory, old_loaded = dataset_images, memory_snapshot, loaded_models
    # Temporary input files are not needed: substitute only the path enumeration and PIL opens below.
    dataset_images = lambda: [Path("face_a_aligned.jpg"), Path("face_b_aligned.jpg")]
    memory_snapshot = lambda: {"current_mb": 100.0, "peak_mb": 105.0}
    loaded_models = lambda _orchestrator, _health=None: list(dict.fromkeys(fake_loaded))
    original_open = Image.open
    def fake_open(path, *args, **kwargs):
        marker = (20, 40, 60) if "face_a" in str(path) else (80, 100, 120)
        return Image.new("RGB", (80, 60), marker)
    Image.open = fake_open
    try:
        payload = execute_flow(fake, memory_snapshot, {"status": "up", "mode": "real"})
        payload["steps"].insert(0, {"step": "a_import_orchestrator_no_models_loaded", "wall_seconds": 0.0,
                                    "working_set": memory_snapshot(), "models_loaded": []})
    finally:
        Image.open = original_open
        dataset_images, memory_snapshot = old_images, old_memory
        loaded_models = old_loaded
    names = [step["step"] for step in payload["steps"]]
    expected = ["a_import_orchestrator_no_models_loaded", "b_single_face", "c_multi_face",
                "d_zero_face_lazy_realesrgan", "e_m8_roll_experiment"]
    assert names == expected, names
    assert [event[0] for event in events] == ["prepare", "analyze", "prepare", "multi", "prepare", "analyze", "frontal"]
    assert events[1][1] == {"emoca": {"mode": "real", "url": EMOCA_URL, "timeout_s": 90},
                            "engine_mode": "real", "max_faces": 1}
    assert events[3][1]["views"].faces == fake_faces_two
    assert all("working_set" in step and "models_loaded" in step and "wall_seconds" in step
               for step in payload["steps"])
    assert payload["inputs"]["images_written"] is False
    return {"status": "self_test_pass", "steps": payload["steps"], "events_checked": len(events)}


def print_report(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print("\n| Step | Wall time (s) | Working set current / peak (MB) | Models loaded |")
    print("|---|---:|---:|---|")
    for row in payload.get("steps", []):
        mem = row["working_set"]
        models = ", ".join(row["models_loaded"]) or "none"
        print(f"| {row['step']} | {row['wall_seconds']:.3f} | {mem['current_mb']:.2f} / {mem['peak_mb']:.2f} | {models} |")


def main() -> int:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="check flow and JSON schema with fakes; no models/network")
    args = parser.parse_args()
    if args.self_test:
        print_report(self_test())
        return 0
    try:
        health = emoca_health()
    except Exception as exc:
        print(f"REFUSED: local EMOCA health check failed at {EMOCA_URL}/v1/health: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    if health.get("status") != "up" or health.get("mode") != "real":
        print(f"REFUSED: EMOCA must report status=up and mode=real; received status={health.get('status')!r}, "
              f"mode={health.get('mode')!r}.", file=sys.stderr)
        return 2
    if os.name != "nt":
        print("REFUSED: Windows process working-set measurement requires Windows.", file=sys.stderr)
        return 2
    samples = dataset_images()
    if len(samples) < 2:
        print(f"SKIP: RAF-DB aligned crops missing; expected at least 2 under {DATASET}, found {len(samples)}.")
        return 0
    # Import after the initial health/data checks. The first recorded point is immediately after import.
    try:
        from web import orchestrator
        initial = {"step": "a_import_orchestrator_no_models_loaded", "wall_seconds": 0.0,
                   "working_set": memory_snapshot(), "models_loaded": loaded_models(orchestrator, health)}
        locally_loaded = [name for name in initial["models_loaded"] if not name.startswith("EMOCA service:")]
        if locally_loaded:
            raise RuntimeError(f"Unexpected local model loaded by importing orchestrator: {locally_loaded}")
        payload = execute_flow(orchestrator, memory_snapshot, health)
        payload["steps"].insert(0, initial)
        print_report(payload)
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
