"""Đọc checkpoint PyTorch (định dạng zip ≥1.6 và định dạng legacy) KHÔNG cần torch.

Dùng cho kiểm chứng artifact (sha256, shape đầu phân loại, so sánh trọng số) khi môi trường
chưa cài torch. An toàn: `find_class` KHÔNG import class tuỳ ý — mọi class lạ thay bằng stub,
nên không thực thi mã trong file pickle.

Chỉ đọc bytes thô của tensor; không giải mã sang số thực trừ khi gọi `tensor_values` (float32).
"""

from __future__ import annotations

import array
import collections
import math
import pickle
import struct
import zipfile
from pathlib import Path

ELEMENT_SIZE = {
    "FloatStorage": 4, "LongStorage": 8, "IntStorage": 4, "DoubleStorage": 8,
    "HalfStorage": 2, "BFloat16Storage": 2, "ByteStorage": 1, "CharStorage": 1,
    "ShortStorage": 2, "BoolStorage": 1,
}


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class TensorRef:
    """Tham chiếu tới một tensor: storage key, kiểu storage, offset (phần tử), shape."""

    __slots__ = ("key", "storage_type", "offset", "shape")

    def __init__(self, key, storage_type, offset, shape):
        self.key, self.storage_type, self.offset, self.shape = key, storage_type, offset, tuple(shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh):
        super().__init__(fh)
        self.storage_types: dict[str, str] = {}

    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return lambda storage, offset, size, *rest: TensorRef(storage[0], storage[1], offset, size)
        if name == "_rebuild_parameter":
            return lambda data, *rest: data
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return name
        return _Stub

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, numel[, view_metadata])
        storage_type = pid[1] if isinstance(pid[1], str) else getattr(pid[1], "__name__", str(pid[1]))
        self.storage_types[pid[2]] = storage_type
        return (pid[2], storage_type)


class Checkpoint:
    """Mở một checkpoint; `obj` là đối tượng gốc (thường dict), tensor là `TensorRef`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if zipfile.is_zipfile(self.path):
            self._zip = zipfile.ZipFile(self.path)
            pkl = next(n for n in self._zip.namelist() if n.endswith("/data.pkl"))
            self.archive_name = pkl.split("/")[0]
            with self._zip.open(pkl) as fh:
                self.obj = _Unpickler(fh).load()
            self._legacy_raw = None
        else:
            self._zip = None
            self.archive_name = None
            with open(self.path, "rb") as fh:
                for _ in range(3):  # magic number, protocol version, sys_info
                    pickle.load(fh)
                up = _Unpickler(fh)
                self.obj = up.load()
                keys = pickle.load(fh)
                self._legacy_raw = {}
                for k in keys:
                    n = struct.unpack("<q", fh.read(8))[0]
                    self._legacy_raw[k] = fh.read(n * ELEMENT_SIZE[up.storage_types[k]])

    def state_dict(self, key: str | None = None) -> dict[str, TensorRef]:
        sd = self.obj[key] if key else self.obj
        return {k: v for k, v in sd.items() if isinstance(v, TensorRef)}

    def tensor_bytes(self, t: TensorRef) -> bytes:
        es = ELEMENT_SIZE[t.storage_type]
        if self._zip is not None:
            raw = self._zip.read(f"{self.archive_name}/data/{t.key}")
        else:
            raw = self._legacy_raw[t.key]
        return raw[t.offset * es:(t.offset + t.numel) * es]

    def tensor_values(self, t: TensorRef) -> array.array:
        if t.storage_type != "FloatStorage":
            raise TypeError(f"chỉ giải mã float32, gặp {t.storage_type}")
        a = array.array("f")
        a.frombytes(self.tensor_bytes(t))
        return a
