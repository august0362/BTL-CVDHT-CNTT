"""Ước lượng roll bằng 3DDFA-V2 cho bước sửa roll của luồng chính."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
import importlib.util
import pickle

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model" / "3ddfa_v2"
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "3DDFA_V2"


class FrontalizationError(RuntimeError):
    """A model load, face-fit, or warp failure."""
    def __init__(self, code: str, message: str):
        if code not in {"no_face", "fit_failed", "load_failed"}:
            raise ValueError("invalid frontalization error code")
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class FrontalizationResult:
    image: Image.Image
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    occluded_fraction: float
    time_ms: float
    method: str = "3ddfa_v2"


@dataclass
class RollCorrectionResult:
    image: Image.Image
    roll_deg: float
    yaw_deg: float
    pitch_deg: float
    applied: bool
    rotation_deg: float
    threshold_deg: float
    time_ms: float
    method: str = "3ddfa_v2_roll2d"


def _pickle(path: Path):
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except Exception:
        with path.open("rb") as stream:
            return pickle.load(stream, encoding="latin1")


def _pose_angles(rotation: np.ndarray) -> tuple[float, float, float]:
    # Port of third_party/3DDFA_V2/utils/pose.py:18-61 (P2sRt and matrix2angle).
    r1, r2 = rotation[0:1], rotation[1:2]
    r1 = r1 / max(float(np.linalg.norm(r1)), 1e-12)
    r2 = r2 / max(float(np.linalg.norm(r2)), 1e-12)
    r = np.concatenate((r1, r2, np.cross(r1, r2)), axis=0)
    if r[2, 0] > .998:
        z, x, y = 0.0, np.pi / 2, np.arctan2(-r[0, 1], -r[0, 2])
    elif r[2, 0] < -.998:
        z, x, y = 0.0, -np.pi / 2, np.arctan2(r[0, 1], r[0, 2])
    else:
        x = np.arcsin(r[2, 0])
        y = np.arctan2(r[2, 1] / np.cos(x), r[2, 2] / np.cos(x))
        z = np.arctan2(r[1, 0] / np.cos(x), r[0, 0] / np.cos(x))
    return tuple(float(v * 180 / np.pi) for v in (x, y, z))


class TDDFAV2Frontalizer:
    """Lazy CPU-only MobileNet 3DMM fitter and image-space frontalizer."""
    def __init__(self, model_dir: str | Path = MODEL_DIR,
                 upstream_root: str | Path = UPSTREAM_ROOT, device: str = "cpu"):
        if device != "cpu":
            raise FrontalizationError("load_failed", "3DDFA-V2 frontalizer only supports CPU")
        self.model_dir = Path(model_dir)
        self.upstream_root = Path(upstream_root)
        self._loaded = False
        self._lock = Lock()

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import torch
                module_path = self.upstream_root / "models" / "mobilenet_v1.py"
                spec = importlib.util.spec_from_file_location("ma3d_tddfa_mobilenet_v1", module_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot load {module_path}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.model = module.mobilenet(widen_factor=1.0, num_classes=62).cpu().eval()
                checkpoint = torch.load(self.model_dir / "mb1_120x120.pth", map_location="cpu", weights_only=False)
                state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                # Upstream utils/tddfa_util.py:33-38 drops fc_lm and maps fc_param to fc.
                state = {k.removeprefix("module."): v for k, v in state.items()
                         if not k.removeprefix("module.").startswith("fc_lm.")}
                state = {k.replace("fc_param.", "fc.", 1): v for k, v in state.items()}
                self.model.load_state_dict(state, strict=True)
                bfm = _pickle(self.model_dir / "bfm_noneck_v3.pkl")
                self.u = np.asarray(bfm["u"], dtype=np.float32).reshape(-1)
                self.w_shp = np.asarray(bfm["w_shp"], dtype=np.float32)[:, :40]
                self.w_exp = np.asarray(bfm["w_exp"], dtype=np.float32)[:, :10]
                self.tri = np.asarray(_pickle(self.model_dir / "tri.pkl"), dtype=np.int32)
                if self.tri.ndim == 2 and self.tri.shape[0] == 3:
                    self.tri = self.tri.T
                # upstream BFM triangles are one-based; normalize once for NumPy.
                if self.tri.min() == 1:
                    self.tri = self.tri - 1
                stats = _pickle(self.model_dir / "param_mean_std_62d_120x120.pkl")
                self.param_mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
                self.param_std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
                self.torch = torch
                self._loaded = True
            except FrontalizationError:
                raise
            except Exception as exc:
                raise FrontalizationError("load_failed", f"Could not load 3DDFA-V2: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _roi(width: int, height: int) -> tuple[float, float, float, float]:
        # Port of third_party/3DDFA_V2/utils/functions.py:85-100.
        old_size = (width + height) / 2
        cx = width / 2
        cy = height / 2 + old_size * .14
        size = int(old_size * 1.58)
        return cx-size/2, cy-size/2, cx+size/2, cy+size/2

    @staticmethod
    def _crop(image: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
        # Port of third_party/3DDFA_V2/utils/functions.py:23-47, including zero padding.
        x1, y1, x2, y2 = (int(round(x)) for x in roi)
        h, w = image.shape[:2]
        out = np.zeros((y2-y1, x2-x1, 3), np.uint8)
        sx, sy, ex, ey = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
        if ex > sx and ey > sy:
            out[sy-y1:ey-y1, sx-x1:ex-x1] = image[sy:ey, sx:ex]
        return out

    def estimate(self, image: Image.Image) -> tuple[np.ndarray, tuple[float, float, float]]:
        if not isinstance(image, Image.Image) or image.width < 2 or image.height < 2:
            raise FrontalizationError("no_face", "Input crop is empty or too small")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if float(rgb.std()) < .5:
            raise FrontalizationError("no_face", "Uniform crop has no usable face detail")
        self._load()
        roi = self._roi(image.width, image.height)
        # Upstream uses cv2 input (BGR) and leaves that channel order for the net.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        crop = cv2.resize(self._crop(bgr, roi), (120,120), interpolation=cv2.INTER_LINEAR)
        inp = self.torch.from_numpy(crop.transpose(2,0,1).copy()).float().sub_(127.5).div_(128).unsqueeze(0)
        with self.torch.inference_mode():
            pred = self.model(inp).squeeze(0).cpu().numpy().astype(np.float32)
        param = pred * self.param_std + self.param_mean
        if param.shape != (62,) or not np.isfinite(param).all():
            raise FrontalizationError("no_face", "3DDFA-V2 returned invalid parameters")
        rotation = param[:12].reshape(3,4)[:,:3]
        return param, _pose_angles(rotation)

    def _reconstruct(self, param: np.ndarray, roi: tuple[float,float,float,float], size: tuple[int,int]) -> np.ndarray:
        # Port of third_party/3DDFA_V2/TDDFA.py:124-139 and utils/tddfa_util.py:64-77.
        camera = param[:12].reshape(3,4)
        rotation, offset = camera[:,:3], camera[:,3:4]
        shape = param[12:52].reshape(40,1)
        exp = param[52:62].reshape(10,1)
        model = self.u[:,None] + self.w_shp @ shape + self.w_exp @ exp
        points = (rotation @ model.reshape(3,-1,order="F") + offset).astype(np.float32)
        n = 120
        points[0] -= 1
        points[2] -= 1
        points[1] = n - points[1]
        sx,sy,ex,ey = roi
        scale_x,scale_y=(ex-sx)/n,(ey-sy)/n
        points[0] = points[0]*scale_x+sx
        points[1] = points[1]*scale_y+sy
        points[2] *= (scale_x+scale_y)/2
        points[2] -= points[2].min()
        return points

    def frontalize(self, image: Image.Image) -> FrontalizationResult:
        """Keep the 3D warp for benchmark reproducibility; ADR-028 replaced it in the app."""
        started = perf_counter()
        param, pose = self.estimate(image)
        width,height=image.size
        roi=self._roi(width,height)
        pts=self._reconstruct(param,roi,image.size)
        xyz=pts.T
        x,y,z=xyz[:,0],xyz[:,1],xyz[:,2]
        if len(x)<3 or not np.isfinite(xyz).all():
            raise FrontalizationError("no_face", "Projected 3D face is invalid")
        inside=(x>=0)&(x<width)&(y>=0)&(y<height)
        if float(inside.mean()) < .20:
            raise FrontalizationError("no_face", "Projected face is mostly outside the crop")
        tri=self.tri
        if tri.size == 0 or tri.max() >= len(x):
            raise FrontalizationError("fit_failed", "BFM topology is invalid")
        faces=xyz[tri]
        normals=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
        # After similar_transform (y flipped, z shifted to >= 0) camera-facing normals have
        # +z and the nose tip has the largest z (measured on real crops: 85-92 % of
        # triangles face +z on frontal faces).
        facing=normals[:,2]>0
        degree=np.bincount(tri.reshape(-1),minlength=len(x)).astype(np.float32)
        def vertex_mean(per_triangle: np.ndarray) -> np.ndarray:
            total=np.zeros(len(x),np.float32)
            np.add.at(total,tri.reshape(-1),np.repeat(per_triangle.astype(np.float32),3))
            return total/np.maximum(degree,1)
        # A vertex is visible when most of its triangles face the camera (not just one).
        visible=(vertex_mean(facing)>=.5)&inside
        # Coarse z-buffer (larger z = nearer) only for clear self-occlusion (e.g. the nose at
        # large yaw). Cell ~3 % of face width, tolerance 20 % of face depth: tighter tests
        # flag sloped skin as hidden and speckle the warp (seen in QA, 2026-10-03).
        cell=max(2,int(round(float(np.ptp(x[inside]))*.03)))
        gx=np.clip((x//cell).astype(int),0,(width-1)//cell); gy=np.clip((y//cell).astype(int),0,(height-1)//cell)
        zbuf=np.full(((height-1)//cell+1,(width-1)//cell+1),-np.inf,np.float32)
        np.maximum.at(zbuf,(gy[inside],gx[inside]),z[inside])
        visible &= z >= zbuf[gy,gx] - max(2.,float(np.ptp(z))*.20)
        # One smoothing pass over the mesh so hidden regions are contiguous patches.
        visible=(vertex_mean(visible[tri].mean(axis=1))>=.5)&inside
        occluded=1-float(visible.sum()/max(1,len(x)))
        # Reproject the same shape/expression with identity rotation; port of TDDFA.py:recon_vers.
        camera = param[:12].reshape(3,4)
        rotation, offset = camera[:,:3], camera[:,3:4]
        shape = param[12:52].reshape(40,1)
        exp = param[52:62].reshape(10,1)
        model = self.u[:,None] + self.w_shp @ shape + self.w_exp @ exp
        model = model.reshape(3,-1,order="F")
        camera_scale = (np.linalg.norm(rotation[0,:3]) + np.linalg.norm(rotation[1,:3])) / 2
        frontal_roi = (camera_scale*model[0]+offset[0], camera_scale*model[1]+offset[1])
        fx0,fy0=frontal_roi
        sx,sy,ex,ey=roi
        scale_x,scale_y=(ex-sx)/120,(ey-sy)/120
        fx0=(fx0-1)*scale_x+sx
        fy0=(120-fy0)*scale_y+sy  # same y flip as similar_transform
        # Keep the camera scale (a yawed face is narrower, not smaller); only re-centre it
        # on the visible face so the frontal face stays inside the crop.
        fx=fx0-float(fx0.mean())+float(x[inside].mean())
        fy=fy0-float(fy0.mean())+float(y[inside].mean())
        # Sample visible vertices from input. Hidden vertices source their bilateral BFM partner.
        source_x=x.copy(); source_y=y.copy()
        mirror=self._mirror_pairs()
        hidden=~visible
        pair=mirror[hidden]
        valid=pair>=0
        ids=np.flatnonzero(hidden)[valid]; partners=pair[valid]
        source_x[ids]=x[partners]; source_y[ids]=y[partners]
        points=np.column_stack((fx,fy))
        source=np.column_stack((source_x,source_y))
        # Decimate deterministically for a bounded Delaunay backward map.
        stride=max(1,int(np.ceil(len(points)/9000)))
        chosen=np.arange(0,len(points),stride)
        chosen=chosen[np.isfinite(points[chosen]).all(axis=1)]
        from scipy.interpolate import LinearNDInterpolator
        try:
            yy,xx=np.mgrid[0:height,0:width]
            interp=LinearNDInterpolator(points[chosen],source[chosen],fill_value=-1)
            mapping=np.asarray(interp(xx,yy),dtype=np.float32)
        except Exception as exc:
            raise FrontalizationError("fit_failed", f"Could not build frontal warp: {exc}") from exc
        map_x,map_y=mapping[...,0],mapping[...,1]
        valid_map=(map_x>=0)&(map_y>=0)&(map_x<width)&(map_y<height)
        if valid_map.mean()<.01:
            raise FrontalizationError("fit_failed", "Frontal warp is degenerate")
        source_img=np.asarray(image.convert("RGB"))
        warped=cv2.remap(source_img,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        mask=valid_map.astype(np.uint8)*255
        sigma=max(1,round(min(width,height)*.03))
        mask=cv2.GaussianBlur(mask,(0,0),sigmaX=sigma,sigmaY=sigma).astype(np.float32)/255
        out=np.asarray(image.convert("RGB"),dtype=np.float32)*(1-mask[...,None])+warped.astype(np.float32)*mask[...,None]
        result=Image.fromarray(np.clip(out,0,255).astype(np.uint8),"RGB")
        return FrontalizationResult(result,*pose,occluded,(perf_counter()-started)*1000)

    def correct_roll(self, image: Image.Image, threshold_deg: float) -> RollCorrectionResult:
        """ADR-028: estimate roll on a face-tight crop and undo it with a plain 2D rotation.

        Sign measured on RAF-DB aligned crops (2026-10-03): PIL rotate(+a) gives an estimated
        roll of about +a, so rotate(-roll) restores the upright face. This is the exact rule
        measured by the Kaggle roll sweep (kaggle/dataset/scripts/run_frontalize_worker.py
        --mode roll2d). Estimates are unreliable on whole-scene photos; pass a face crop.
        """
        started = perf_counter()
        if isinstance(threshold_deg, bool) or not isinstance(threshold_deg, (int, float)) or not 0 < threshold_deg <= 90:
            raise ValueError("threshold_deg must be in (0, 90]")
        source = image.convert("RGB")
        _, (yaw_deg, pitch_deg, roll_deg) = self.estimate(source)
        applied = abs(roll_deg) >= threshold_deg
        rotation_deg = 0.0
        corrected = image
        if applied:
            rotation_deg = -float(roll_deg)
            fill = tuple(int(v) for v in source.resize((1, 1), Image.BILINEAR).getpixel((0, 0)))
            corrected = source.rotate(rotation_deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
        return RollCorrectionResult(
            image=corrected, roll_deg=float(roll_deg), yaw_deg=float(yaw_deg), pitch_deg=float(pitch_deg),
            applied=applied, rotation_deg=float(rotation_deg), threshold_deg=float(threshold_deg),
            time_ms=(perf_counter() - started) * 1000,
        )

    def _mirror_pairs(self) -> np.ndarray:
        cached=getattr(self,"_mirror_cache",None)
        if cached is not None:return cached
        mean=self.u.reshape(3,-1,order="F").T
        from scipy.spatial import cKDTree
        mirrored=mean.copy(); mirrored[:,0]*=-1
        pairs=cKDTree(mean).query(mirrored,k=1)[1].astype(np.int32)
        self._mirror_cache=pairs
        return pairs


_SINGLETON: TDDFAV2Frontalizer | None = None
_SINGLETON_LOCK=Lock()
def get_tddfa_v2_frontalizer() -> TDDFAV2Frontalizer:
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:_SINGLETON=TDDFAV2Frontalizer()
        return _SINGLETON
