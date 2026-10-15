"""Print in-memory 3DDFA-V2 pose corrections for image files or a folder."""
from __future__ import annotations
import argparse
import ctypes
import os
from pathlib import Path
from PIL import Image
from services.frontalization import get_tddfa_v2_frontalizer
from web.predata_experiment import build_multiface_views


def peak_rss_kb() -> int:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_=[("cb",ctypes.c_ulong),("PageFaultCount",ctypes.c_ulong),
                      ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                      ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
                      ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                      ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        kernel32=ctypes.WinDLL("kernel32",use_last_error=True)
        psapi=ctypes.WinDLL("psapi",use_last_error=True)
        kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype=ctypes.c_int
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),ctypes.byref(counters),counters.cb)
        return int(counters.PeakWorkingSetSize//1024)
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if value>100000 else value*1024)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("paths",nargs="+",help="image files and/or folders")
    args=parser.parse_args()
    paths=[]
    for value in args.paths:
        path=Path(value)
        paths.extend(sorted(p for p in path.iterdir() if p.is_file()) if path.is_dir() else [path])
    frontalizer=get_tddfa_v2_frontalizer()
    for path in paths:
        try:
            with Image.open(path) as src: image=src.convert("RGB")
            views=build_multiface_views(image)
            for item in views.faces:
                result=frontalizer.frontalize(item.detection.cropped_face)
                _,pose=frontalizer.estimate(result.image)
                print(f"face={item.face_id} yaw={result.yaw_deg:.2f}->{pose[0]:.2f} "
                      f"pitch={result.pitch_deg:.2f}->{pose[1]:.2f} roll={result.roll_deg:.2f}->{pose[2]:.2f} "
                      f"occluded={result.occluded_fraction:.4f} ms={result.time_ms:.2f}")
        except Exception as exc:
            print(f"ERROR {path.name}: {type(exc).__name__}: {getattr(exc, 'message', exc)}")
    print(f"peak_rss_kb={peak_rss_kb()}")

if __name__=="__main__":main()
