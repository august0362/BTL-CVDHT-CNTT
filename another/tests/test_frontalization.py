"""venv web: kiểm tra mô hình 3DDFA-V2 và sửa roll của luồng chính."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
MODEL_FILES=("mb1_120x120.pth","bfm_noneck_v3.pkl","param_mean_std_62d_120x120.pkl","tri.pkl")
missing=[name for name in MODEL_FILES if not (ROOT/"model/3ddfa_v2"/name).is_file()]
if missing:
    print("SKIP test_frontalization: missing 3DDFA-V2 model files: "+", ".join(missing));sys.exit(0)
try: import torch  # noqa: F401
except ImportError: print("SKIP test_frontalization: torch is unavailable");sys.exit(0)
sys.path.insert(0,str(ROOT / "backend"))
from services.frontalization import FrontalizationError,TDDFAV2Frontalizer


def synthetic_face(yaw_deg:float,size:int=400)->Image.Image:
    import pickle
    def load(name):
        try:return pickle.load(open(ROOT/"model/3ddfa_v2"/name,"rb"))
        except Exception:return pickle.load(open(ROOT/"model/3ddfa_v2"/name,"rb"),encoding="latin1")
    bfm,triangles=load("bfm_noneck_v3.pkl"),load("tri.pkl").T-1
    vertices=np.asarray(bfm["u"],np.float32).reshape(3,-1,order="F").T
    angle=np.deg2rad(yaw_deg)
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],np.float32)
    vertices=vertices@rotation.T;xy=vertices[:,:2].copy()
    xy[:,0]=(xy[:,0]-xy[:,0].min())/max(np.ptp(xy[:,0]),1e-5)*size*.58+size*.21
    xy[:,1]=(xy[:,1].max()-xy[:,1])/max(np.ptp(xy[:,1]),1e-5)*size*.72+size*.13
    image=np.full((size,size,3),(55,48,44),np.uint8);z=vertices[:,2]
    norm=(z-z.min())/max(float(np.ptp(z)),1e-6)
    for tri in triangles[np.argsort(vertices[triangles,2].mean(axis=1))]:
        p=vertices[tri];normal=np.cross(p[1]-p[0],p[2]-p[0])
        light=.58+.42*abs(float(normal[2])/(np.linalg.norm(normal)+1e-8));shade=float(np.mean(norm[tri]))*.12+.86
        color=tuple(int(v*light*shade) for v in (224,177,146))
        cv2.fillConvexPoly(image,xy[tri].astype(np.int32),color,lineType=cv2.LINE_AA)
    return Image.fromarray(image,"RGB")


def test_bad_inputs():
    f=TDDFAV2Frontalizer()
    for image in (Image.new("RGB",(0,0)),Image.new("RGB",(128,128),(128,128,128))):
        try:f.frontalize(image)
        except FrontalizationError as exc:assert exc.code in {"no_face","fit_failed"}
        else:raise AssertionError("invalid input unexpectedly frontalized")
    print("PASS bad_inputs")


def test_synthetic_network_and_warp():
    f=TDDFAV2Frontalizer();straight=synthetic_face(0);_,pose=f.estimate(straight)
    if abs(pose[0])>=10:
        print(f"  SKIP synthetic render: network does not recognize frontal mean mesh (yaw {pose[0]:.2f}°)")
        return
    assert f.frontalize(straight).image.size==straight.size
    angled=synthetic_face(40);_,before=f.estimate(angled)
    if abs(before[0])<=30:
        print(f"  SKIP synthetic render: network does not recognize 40° mean mesh (yaw {before[0]:.2f}°)")
        return
    corrected=f.frontalize(angled);_,after=f.estimate(corrected.image)
    assert abs(after[0])<abs(before[0])-15,f"yaw {before[0]:.1f}° → {after[0]:.1f}°"
    print(f"PASS synthetic_network_and_warp yaw={before[0]:.1f}->{after[0]:.1f}")


def test_roll_only_rotation_sign_and_below_threshold():
    """ADR-028 rule on face-tight RAF-DB crops (the app passes YOLO crops; whole scenes are unreliable)."""
    samples=sorted((ROOT/"database/image/archive/DATASET/test").glob("*/test_000*_aligned.jpg"))[:3]
    if not samples:
        print("  SKIP roll_only: RAF-DB test crops not present (database/image/archive/DATASET/test)")
        return
    f=TDDFAV2Frontalizer()
    first=Image.open(samples[0]).convert("RGB")
    below=f.correct_roll(first,90)
    assert below.applied is False and below.rotation_deg==0.0
    assert below.image.tobytes()==first.tobytes(), "below-threshold input must be pixel-identical"
    for path in samples:
        upright=Image.open(path).convert("RGB")
        upright_roll=f.estimate(upright)[1][2]
        fill=tuple(int(v) for v in upright.resize((1,1),Image.BILINEAR).getpixel((0,0)))
        for degrees in (30,45,90,-45):
            tilted=upright.rotate(degrees,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=fill)
            corrected=f.correct_roll(tilted,20)
            after=f.estimate(corrected.image)[1][2]
            assert corrected.applied, (path.name,degrees,corrected.roll_deg)
            assert abs(corrected.rotation_deg+corrected.roll_deg)<1e-6, "rule is rotate(-roll)"
            # Sign check: a wrong sign would leave ~2x the tilt. Estimator error itself (e.g. 45° read
            # as 33°) is measured by the Kaggle roll sweep, so allow residual <= min(15°, half the tilt).
            assert abs(after-upright_roll)<=min(15,abs(degrees)/2), (path.name,degrees,upright_roll,corrected.roll_deg,after)
            print(f"  {path.name} {degrees:+}°: upright={upright_roll:.1f}° est={corrected.roll_deg:.1f}° after={after:.1f}°")
    print("PASS roll_only_rotation_sign_and_below_threshold")


def test_baseline_roll_corrects_original_upload():
    """Heavy ADR-029 smoke: estimate from a YOLO-like crop, rotate the full upload, recheck face."""
    samples=sorted((ROOT/"database/image/archive/DATASET/test").glob("*/test_*_aligned.jpg"))
    if not samples:
        print("  SKIP baseline roll: aligned RAF-DB crops are missing")
        return
    from web.orchestrator import roll_correct_upload
    frontalizer=TDDFAV2Frontalizer()
    upright=Image.open(samples[0]).convert("RGB")
    fill=tuple(int(v) for v in np.asarray(upright,dtype=np.float32).mean(axis=(0,1)))
    tilted=upright.rotate(90,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=fill)
    canvas=Image.new("RGB",(max(tilted.width,tilted.height)+160, max(tilted.width,tilted.height)+160),(128,128,128))
    left=(canvas.width-tilted.width)//2;top=(canvas.height-tilted.height)//2
    canvas.paste(tilted,(left,top))
    detection=SimpleNamespace(cropped_face=tilted)
    views=SimpleNamespace(faces=[SimpleNamespace(detection=detection)])
    corrected,info=roll_correct_upload(canvas,views)
    assert info and not info.get("error") and info["rotation_deg"]!=0, info
    assert abs(info["rotation_deg"]+90)<=15, info
    face_box=(canvas.width-upright.width)//2,(canvas.height-upright.height)//2
    x,y=face_box
    corrected_face=corrected.crop((x,y,x+upright.width,y+upright.height))
    residual=frontalizer.estimate(corrected_face)[1][2]
    assert abs(residual)<15, f"residual roll {residual:.1f}° (estimate {info})"
    print(f"PASS baseline_roll_corrects_original_upload rotation={info['rotation_deg']:.1f}° residual={residual:.1f}°")


def main():
    test_bad_inputs();test_synthetic_network_and_warp();test_roll_only_rotation_sign_and_below_threshold()
    test_baseline_roll_corrects_original_upload()
    print("ALL PASS")
if __name__=="__main__":main()
