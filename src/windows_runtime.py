"""Windows process affinity, execution state and process-tree lifetime."""
import atexit
import ctypes as C
import os
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from ctypes import wintypes as W
import struct

import psutil

KERNEL = C.WinDLL("kernel32", use_last_error=True)


@contextmanager
def ascii_directory(path):
    """Give the native partitioner an ASCII path without copying any parts."""
    if str(path).isascii():
        yield path
        return
    candidates = (Path(value) for value in (tempfile.gettempdir(), os.environ.get("PUBLIC")) if value)
    base = next((folder for folder in candidates if str(folder).isascii() and folder.is_dir()), None)
    if base is None:
        raise RuntimeError("An ASCII TEMP or Public directory is needed for the Ryzen AI runtime")
    with tempfile.TemporaryDirectory(prefix="h3npu-", dir=base) as temp:
        alias = Path(temp) / "build"
        # Token values cannot include Windows cmd metacharacters.
        if any(char in str(path) + str(alias) for char in '"&|<>^%!\r\n'):
            raise ValueError("The path contains unsupported junction characters")
        subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(path)],
                       check=True, stdout=subprocess.DEVNULL)
        try:
            yield alias
        finally:
            alias.rmdir()


_NATIVE_STACK = ExitStack()
_NATIVE_PATHS = {}
atexit.register(_NATIVE_STACK.close)


def native_directory(path):
    path = Path(path)
    if str(path).isascii():
        return path
    if path not in _NATIVE_PATHS:
        _NATIVE_PATHS[path] = _NATIVE_STACK.enter_context(ascii_directory(path))
    return _NATIVE_PATHS[path]


def cpu_sets():
    query = KERNEL.GetSystemCpuSetInformation
    query.argtypes = [C.c_void_p, W.DWORD, C.POINTER(W.DWORD), W.HANDLE, W.DWORD]
    size = W.DWORD()
    query(None, 0, C.byref(size), None, 0)
    if not size.value:
        raise C.WinError(C.get_last_error())
    buf = C.create_string_buffer(size.value)
    if not query(buf, size, C.byref(size), None, 0):
        raise C.WinError(C.get_last_error())
    offset = 0
    while offset < size.value:
        length, kind = struct.unpack_from("II", buf, offset)
        if length < 8 or offset + length > size.value:
            raise ValueError("Invalid CPU set record")
        if kind == 0:
            _, group, logical, _, _, _, efficiency = struct.unpack_from("IHBBBBB", buf, offset + 8)
            yield group, logical, efficiency
        offset += length


def set_affinity(mode):
    process = psutil.Process()
    record = {"mode": mode, "status": "off"}
    if mode == "off":
        return record
    try:
        rows = list(cpu_sets())
        if not rows or {group for group, _, _ in rows} != {0}:
            raise ValueError("Expected one processor group")
        classes = {efficiency for _, _, efficiency in rows}
        best = max(classes)
        eligible = set(process.cpu_affinity())
        selected = sorted(cpu for _, cpu, efficiency in rows if efficiency == best and cpu in eligible)
        if len(classes) > 1:
            if not selected:
                raise ValueError("No eligible processor in the highest efficiency class")
            process.cpu_affinity(selected)
        record.update(status="applied" if len(classes) > 1 else "uniform",
                      processors=process.cpu_affinity(),
                      mask=hex(sum(1 << cpu for cpu in process.cpu_affinity())))
    except (OSError, ValueError, psutil.Error) as exc:
        record.update(status="unavailable", error=str(exc))
    return record


class NpuBusy(RuntimeError):
    pass


class NpuLease:
    """Serialize NPU use by the repository's public entry points."""

    def __enter__(self):
        KERNEL.CreateMutexW.argtypes = [C.c_void_p, W.BOOL, W.LPCWSTR]
        KERNEL.CreateMutexW.restype = W.HANDLE
        KERNEL.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
        KERNEL.WaitForSingleObject.restype = W.DWORD
        KERNEL.CloseHandle.argtypes = [W.HANDLE]
        KERNEL.ReleaseMutex.argtypes = [W.HANDLE]
        self.handle = KERNEL.CreateMutexW(None, False, r"Local\MiniMax-H3-Image-RyzenAI-NPU")
        if not self.handle:
            raise C.WinError(C.get_last_error())
        result = KERNEL.WaitForSingleObject(self.handle, 0)
        if result in (0, 0x80):
            return self
        error = C.get_last_error()
        KERNEL.CloseHandle(self.handle)
        if result == 0x102:
            raise NpuBusy("Another generator, converter or doctor is using the NPU")
        raise C.WinError(error)

    def __exit__(self, *_):
        KERNEL.ReleaseMutex(self.handle)
        KERNEL.CloseHandle(self.handle)


class KeepAwake:
    def __enter__(self):
        KERNEL.SetThreadExecutionState.argtypes = [W.DWORD]
        KERNEL.SetThreadExecutionState.restype = W.DWORD
        self.state = KERNEL.SetThreadExecutionState(0x80000003)
        return bool(self.state)

    def __exit__(self, *_):
        if self.state:
            KERNEL.SetThreadExecutionState(self.state)


class KillOnClose:
    """All subprocesses inherit membership in a job that ends with its owner."""

    def __enter__(self):
        kernel = KERNEL
        kernel.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
        kernel.CreateJobObjectW.restype = W.HANDLE
        kernel.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        kernel.CloseHandle.argtypes = [W.HANDLE]
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise C.WinError(C.get_last_error())
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION on Windows x64.
        info = C.create_string_buffer(144)
        struct.pack_into("I", info, 16, 0x2000)
        if not kernel.SetInformationJobObject(self.handle, 9, info, len(info)):
            error = C.WinError(C.get_last_error())
            kernel.CloseHandle(self.handle)
            raise error
        return self

    def assign(self, process):
        if not KERNEL.AssignProcessToJobObject(self.handle, int(process._handle)):
            error = C.WinError(C.get_last_error())
            process.kill()
            process.wait()
            raise error

    def __exit__(self, *_):
        KERNEL.CloseHandle(self.handle)


def process_times(process):
    values = [W.FILETIME() for _ in range(4)]
    function = KERNEL.GetProcessTimes
    function.argtypes = [W.HANDLE] + [C.POINTER(W.FILETIME)] * 4
    if not function(int(process._handle), *(C.byref(value) for value in values)):
        raise C.WinError(C.get_last_error())
    ticks = [(value.dwHighDateTime << 32) | value.dwLowDateTime for value in values]
    return {"pid": process.pid, "cpu_s": sum(ticks[2:]) / 10**7,
            "wall_s": (ticks[1] - ticks[0]) / 10**7}


def finish_npu(code=0):
    # Some Ryzen AI Software 1.8 sessions fault when native libraries unload.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    _NATIVE_STACK.close()
    KERNEL.GetCurrentProcess.restype = W.HANDLE
    KERNEL.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
    if not KERNEL.TerminateProcess(KERNEL.GetCurrentProcess(), code):
        raise C.WinError(C.get_last_error())

