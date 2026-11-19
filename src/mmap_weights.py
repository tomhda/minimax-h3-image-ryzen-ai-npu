"""Read only the required safetensors ranges on Windows."""
import ctypes as C
import json
import struct
import warnings
from pathlib import Path

import numpy as np
import torch

from windows_runtime import KERNEL

KERNEL.CreateFileW.argtypes = [C.c_wchar_p, C.c_ulong, C.c_ulong, C.c_void_p, C.c_ulong,
                              C.c_ulong, C.c_void_p]
KERNEL.CreateFileW.restype = C.c_void_p
KERNEL.ReadFile.argtypes = [C.c_void_p, C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong), C.c_void_p]
KERNEL.SetFilePointerEx.argtypes = [C.c_void_p, C.c_longlong, C.c_void_p, C.c_ulong]
KERNEL.CloseHandle.argtypes = [C.c_void_p]


def unbuffered_read(path, offset, size):
    alignment = 4096
    begin = offset // alignment * alignment
    shift = offset - begin
    request = (size + shift + alignment - 1) // alignment * alignment
    owner = np.empty(request + alignment - 1, np.uint8)
    start = -owner.ctypes.data % alignment
    buffer = owner[start:start + request]
    handle = KERNEL.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x20000000, None)
    if handle == C.c_void_p(-1).value:
        raise C.WinError(C.get_last_error())
    try:
        if not KERNEL.SetFilePointerEx(handle, begin, None, 0):
            raise C.WinError(C.get_last_error())
        done = 0
        while done < size + shift:
            count = min(request - done, 8 * 1024**2)
            received = C.c_ulong()
            if not KERNEL.ReadFile(handle, buffer.ctypes.data + done, count, C.byref(received), None):
                raise C.WinError(C.get_last_error())
            if not received.value:
                raise EOFError(path)
            done += received.value
    finally:
        KERNEL.CloseHandle(handle)
    return buffer[shift:shift + size]


class Weights:
    def __init__(self, folder, *, shard=None):
        self.folder = Path(folder)
        self.headers = {}
        if shard is None:
            index = self.folder / "model.safetensors.index.json"
            self.mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            _, header = self.header(shard)
            self.mapping = {name: shard for name in header if name != "__metadata__"}

    def preload_headers(self, keys):
        for shard in sorted({self.mapping[key] for key in keys}):
            self.header(shard)

    def header(self, shard):
        if shard not in self.headers:
            with (self.folder / shard).open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + size, json.loads(handle.read(size)))
        return self.headers[shard]

    def read(self, key, rows=None, direct=False):
        shard = self.mapping[key]
        base, header = self.header(shard)
        info = header[key]
        shape = tuple(info["shape"])
        dtype = np.dtype({"BF16": "<u2", "F32": "<f4", "F16": "<f2", "I64": "<i8"}[info["dtype"]])
        offset = base + info["data_offsets"][0]
        path = self.folder / shard
        if direct:
            try:
                array = unbuffered_read(path, offset, int(np.prod(shape)) * dtype.itemsize)
                array = array.view(dtype).reshape(shape)
            except OSError as exc:
                # Filesystems without compatible direct I/O still support read-only mapping.
                if exc.winerror not in (1, 50, 87):
                    raise
                array = np.array(np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape))
        else:
            array = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)
        if rows is not None:
            array = np.array(array[rows], copy=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            value = torch.from_numpy(array)
        return value.view(torch.bfloat16) if info["dtype"] == "BF16" else value

    def fp32(self, key):
        return self.read(key).float().contiguous()

