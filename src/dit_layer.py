"""One diffusion transformer layer and its reusable host buffers."""
import numpy as np
import torch
import torch.nn.functional as F

from common import PROJECTIONS, PROJECTION_SHAPES, clock, duration
from model import FFN_WIDTH, FRAME_ROWS, HEAD_DIM, HEADS, IMAGE_ROWS, WIDTH

TILE_ROWS = 256
MATRIX_WIDTHS = dict(zip(PROJECTIONS, (21504, WIDTH, 2 * FFN_WIDTH, WIDTH)))
HEAD_GROUPS = ((0, 24), (24, 24), (48, 8))


def aligned(shape, dtype, zero=False):
    dtype = np.dtype(dtype)
    size = int(np.prod(shape)) * dtype.itemsize
    owner = np.empty(size + 4095, np.uint8)
    start = -owner.ctypes.data % 4096
    result = owner[start:start + size].view(dtype).reshape(shape)
    if zero:
        result.fill(0)
    return result


class Casts:
    def __init__(self, npu, parts):
        self.to_bf = npu.session(parts.cast("to_bf"), "cast")
        self.to_fp = npu.session(parts.cast("to_fp"), "cast")

    def convert(self, source, target, bf16):
        session = self.to_bf if bf16 else self.to_fp
        binding = session.io_binding()
        binding.bind_input("A", "cpu", 0, np.float32 if bf16 else 16, source.shape, source.ctypes.data)
        binding.bind_output("Y", "cpu", 0, 16 if bf16 else np.float32, target.shape, target.ctypes.data)
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()


class Matrix:
    def __init__(self, npu, casts, directory, name):
        self.input_width, self.output_width, pieces = PROJECTION_SHAPES[name]
        self.casts = casts
        self.session = npu.session(directory / "replaced.onnx", "dit")
        cap = max(k0 + width for k0, _, width, _ in pieces)
        shape = (1, TILE_ROWS, cap)
        self.fp = aligned(shape, np.float32, zero=True)
        self.bf = aligned(shape, np.uint16)
        self.binding = self.session.io_binding()
        self.binding.bind_input("A", "cpu", 0, 16, shape, self.bf.ctypes.data)
        self.outputs = []
        for index, (_, offset, _, width) in enumerate(pieces):
            shape = (1, TILE_ROWS, width)
            raw = aligned(shape, np.uint16)
            fp = aligned(shape, np.float32)
            self.binding.bind_output(f"b{index}_Y", "cpu", 0, 16, shape, raw.ctypes.data)
            self.outputs.append((offset, raw, fp))

    def into(self, array, target):
        np.copyto(self.fp[:, :, :self.input_width], array)
        self.casts.convert(self.fp, self.bf, bf16=True)
        self.binding.synchronize_inputs()
        self.session.run_with_iobinding(self.binding)
        self.binding.synchronize_outputs()
        used = set()
        for offset, raw, fp in self.outputs:
            self.casts.convert(raw, fp, bf16=False)
            width = min(fp.shape[2], self.output_width - offset)
            destination = target[:, :, offset:offset + width]
            source = fp[:, :, :width]
            if offset in used:
                np.add(destination, source, out=destination)
            else:
                np.copyto(destination, source)
                used.add(offset)

    def close(self):
        self.binding = None
        self.session = None


class LayerBuffers:
    def __init__(self, geometry):
        self.geometry = geometry
        tags = geometry["tags"]
        self.rows = len(tags)
        nt = int((tags == tags[0]).sum())
        self.slices = ((slice(0, nt), int(tags[0])), (slice(nt, self.rows), int(tags[-1])))
        self.input = {}
        self.output = {}
        cos = np.ones((FRAME_ROWS, 64), np.float32)
        sin = np.zeros((FRAME_ROWS, 64), np.float32)
        cos[:self.rows, :48] = geometry["cos"][:, :48].numpy()
        sin[:self.rows, :48] = geometry["sin"][:, :48].numpy()
        self.feeds = []
        for _ in HEAD_GROUPS:
            feed = {name: np.zeros((1, FRAME_ROWS, 3072), np.float32) for name in ("q", "k", "v", "mark")}
            feed["mark"][:, :self.rows] = 1
            for name, array in (("cos", cos), ("sin", sin)):
                feed[f"{name}_image"] = array[:IMAGE_ROWS]
                feed[f"{name}_text"] = array[IMAGE_ROWS:]
            self.feeds.append(feed)

    def matrix(self, name, matrix, value, times):
        array = value.numpy()
        if name not in self.input:
            self.input[name] = np.zeros((FRAME_ROWS, array.shape[1]), np.float32)
            self.output[name] = np.empty((FRAME_ROWS, MATRIX_WIDTHS[name]), np.float32)
        padded = self.input[name]
        output = self.output[name]
        np.copyto(padded[:self.rows], array)
        calls = []
        for start in range(0, FRAME_ROWS, TILE_ROWS):
            stop = start + TILE_ROWS
            stamp = clock()
            matrix.into(padded[None, start:stop], output[None, start:stop])
            calls.append(duration(stamp))
        times[name] = calls
        return torch.from_numpy(output[:self.rows])

    def attention(self, session, qkv, times):
        q, k, v = qkv.reshape(-1, HEADS, 3, HEAD_DIM).unbind(2)
        outputs = []
        attention_times = []
        for feed, (head, count) in zip(self.feeds, HEAD_GROUPS):
            for name, value in (("q", q), ("k", k)):
                target = feed[name].reshape(1, FRAME_ROWS, 24, HEAD_DIM)[0, :self.rows, :count]
                source = value.numpy()[:, head:head + count]
                np.copyto(target[..., :96:2], source[..., :48])
                np.copyto(target[..., 1:96:2], source[..., 48:96])
                np.copyto(target[..., 96:], source[..., 96:])
            target = feed["v"].reshape(1, FRAME_ROWS, 24, HEAD_DIM)[0, :self.rows, :count]
            np.copyto(target, v.numpy()[:, head:head + count])
        for feed in self.feeds:
            stamp = clock()
            outputs.append(session.run(None, feed)[0])
            attention_times.append(duration(stamp))
        times["attention"] = attention_times
        chunks = [array[0, :self.rows].reshape(self.rows, 24, HEAD_DIM)[:, :count]
                  for array, (_, count) in zip(outputs, HEAD_GROUPS)]
        return torch.from_numpy(np.concatenate(chunks, axis=1).reshape(self.rows, HEADS * HEAD_DIM))


class Layer:
    def __init__(self, npu, casts, parts, index):
        stamp = clock()
        with np.load(parts.small(index)) as weights:
            self.weights = {name: torch.from_numpy(weights[name].copy())
                            for name in ("norm1.weight", "norm2.weight")}
        self.weight_load = duration(stamp)
        stamp = clock()
        self.attention = npu.session(parts.attention(index) / "replaced.onnx", "attention")
        self.matrices = {name: Matrix(npu, casts, parts.dit_matrix(index, name), name) for name in PROJECTIONS}
        self.session_create = duration(stamp)

    def run(self, hidden, mods, buffers):
        def affine(value, scale, shift):
            for rows, tag in buffers.slices:
                value[rows].mul_(1 + mods[scale, tag]).add_(mods[shift, tag])
            return value

        def residual(value, gate, base):
            for rows, tag in buffers.slices:
                value[rows].mul_(mods[gate, tag]).add_(base[rows])
            return value

        times = {}

        def project(name, value):
            return buffers.matrix(name, self.matrices[name], value, times)

        def norm(value, weight):
            return F.rms_norm(value, (value.shape[-1],), self.weights[weight], 1e-5)

        qkv = project("qkv", affine(norm(hidden, "norm1.weight"), 1, 0))
        attention = project("out", buffers.attention(self.attention, qkv, times))
        hidden = residual(attention, 2, hidden)
        combined = project("fc1", affine(norm(hidden, "norm2.weight"), 4, 3))
        active = F.silu(combined[:, :FFN_WIDTH]).mul_(combined[:, FFN_WIDTH:])
        result = residual(project("fc2", active), 5, hidden)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Non-finite transformer layer output")
        return result, times

    def close(self):
        for matrix in self.matrices.values():
            matrix.close()
        self.attention = None

