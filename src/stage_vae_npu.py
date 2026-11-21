"""Use the NPU for the four linear projections in each decoder layer."""
import gc

import numpy as np
import torch
import torch.nn.functional as F

from common import PROJECTIONS, VAE_LAYERS, clock, duration


def norm(value, weight=None):
    result = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    return result if weight is None else result * weight


class Decoder:
    def __init__(self, npu, parts):
        self.npu = npu
        self.parts = parts
        with np.load(parts.vae_small) as arrays:
            self.small = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
        self.anchor = npu.session(parts.vae_anchor / "replaced.onnx", "attention")
        self.sessions = {}
        self.biases = {}

    def load_layer(self, index):
        for name in PROJECTIONS:
            path = self.parts.vae_matrix(index, name)
            self.sessions[name] = self.npu.session(path / "replaced.onnx", "vae")
            self.biases[name] = torch.from_numpy(np.load(path / "bias.npy"))

    def release_layer(self):
        self.sessions.clear()
        self.biases.clear()
        gc.collect()

    def linear(self, value, name):
        array = value.float().contiguous().numpy().reshape(-1, value.shape[-1])
        outputs = []
        for start in range(0, len(array), 256):
            chunk = array[start:start + 256]
            padded = np.zeros((1, 256, array.shape[-1]), np.float32)
            padded[0, :len(chunk)] = chunk
            result = self.sessions[name].run(None, {"A": padded})[0]
            outputs.append(result[0, :len(chunk)].copy())
        return torch.from_numpy(np.concatenate(outputs)).reshape(*value.shape[:-1], -1) + self.biases[name]

    def block(self, hidden, index, rotary):
        p = f"transformer_blocks.{index}."
        s = self.small
        qkv = self.linear(norm(hidden, s[p + "norm1.weight"]), "qkv")
        q, k, v = (value.reshape(*value.shape[:2], 16, 64) for value in qkv.chunk(3, -1))
        cos, sin = rotary
        width = cos.shape[-1]

        def rotate(value):
            left, right = value[..., :width].chunk(2, -1)
            rotated = value[..., :width] * cos + torch.cat((-right, left), -1) * sin
            return torch.cat((rotated, value[..., width:]), -1)

        q, k = rotate(norm(q)), rotate(norm(k))
        attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        attended = attended.transpose(1, 2).reshape(*hidden.shape[:2], 1024)
        hidden = hidden + self.linear(attended, "out") * s[p + "scale1"]
        value, gate = self.linear(norm(hidden, s[p + "norm2.weight"]), "fc1").chunk(2, -1)
        return hidden + self.linear(value * F.silu(gate), "fc2") * s[p + "scale2"]

    def decode(self, latent):
        if latent.ndim != 4 or latent.shape[:2] != (1, 24) or min(latent.shape[-2:]) < 16:
            raise ValueError("The decoder expects shape (1, 24, height, width), with both dimensions at least 16")
        y_starts, y_overlaps = tile_layout(latent.shape[-2])
        x_starts, x_overlaps = tile_layout(latent.shape[-1])
        s = self.small
        stamp = clock()
        tiles = []
        for y in y_starts:
            for x in x_starts:
                z = F.conv2d(latent[:, :, y:y + 16, x:x + 16],
                             s["post_quant_conv.weight"], s["post_quant_conv.bias"])
                z = F.linear(z.permute(0, 2, 3, 1).reshape(1, 256, 24), s["proj_in.weight"], s["proj_in.bias"])
                tiles.append(torch.cat((z, s["register_tokens"], torch.zeros_like(z[:, :1])), 1))
        grid = [2 * (torch.arange(0.5, 16) / 16) - 1 for _ in range(2)]
        pos = torch.stack(torch.meshgrid(*grid, indexing="ij"), -1).flatten(0, 1).unsqueeze(0)
        pos = torch.cat((pos, torch.zeros((1, 5, 2))), 1)
        inv = 1 / 100**torch.arange(0, 1, 4 / 32, dtype=torch.float32)
        angles = (2 * torch.pi * pos[:, :, :, None] * inv[None, None, None, :]).flatten(2, 3).tile(2).unsqueeze(2)
        rotary = angles.cos(), angles.sin()
        hidden = torch.cat(tiles, dim=0)
        pre = duration(stamp)

        records = []
        for index in range(VAE_LAYERS):
            stamp = clock()
            self.load_layer(index)
            create = duration(stamp)
            stamp = clock()
            hidden = self.block(hidden, index, rotary)
            infer = duration(stamp)
            stamp = clock()
            self.release_layer()
            records.append({"layer": index, "session_create": create, "inference": infer,
                            "session_release": duration(stamp)})

        stamp = clock()
        tiles = []
        for tile in hidden.split(1, dim=0):
            value = F.layer_norm(tile, (1024,), s["norm_out.weight"], s["norm_out.bias"], 1e-5)
            value = F.linear(value, s["proj_out.weight"], s["proj_out.bias"])[:, :256]
            value = value.view(1, 16, 16, 3, 16, 16).permute(0, 3, 1, 4, 2, 5)
            tiles.append(value.contiguous().reshape(1, 3, 256, 256))
        image = stitch(tiles, y_overlaps, x_overlaps)
        return image, records, pre, duration(stamp)


def tile_layout(length):
    # Match Diffusers' round-robin overlap distribution in latent units.
    tile, min_overlap = 16, 4
    if length == tile:
        return [0], []
    stride = tile - min_overlap
    gaps = (length - tile + stride - 1) // stride
    shared, extra = divmod(tile + gaps * stride - length, gaps)
    overlaps = [min_overlap + shared + (i < extra) for i in range(gaps)]
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile - overlap)
    return starts, [overlap * 16 for overlap in overlaps]


def blend(before, after, overlap, dim):
    positions = torch.arange(overlap, dtype=after.dtype)
    shape = [1] * after.ndim
    shape[dim] = overlap
    a_weight = (1 - positions / overlap).view(shape)
    b_weight = (positions / overlap).view(shape)
    a = [slice(None)] * after.ndim
    b = a.copy()
    a[dim] = slice(-overlap, None)
    b[dim] = slice(0, overlap)
    boundary = before[tuple(a)] * a_weight + after[tuple(b)] * b_weight
    b[dim] = slice(overlap, None)
    return torch.cat((boundary, after[tuple(b)]), dim)


def stitch(tiles, y_overlaps, x_overlaps):
    columns_per_row = len(x_overlaps) + 1
    grid = [tiles[i:i + columns_per_row] for i in range(0, len(tiles), columns_per_row)]
    rows = []
    for y, row in enumerate(grid):
        columns = []
        for x, tile in enumerate(row):
            if y:
                tile = blend(grid[y - 1][x], tile, y_overlaps[y - 1], -2)
            if x:
                tile = blend(row[x - 1], tile, x_overlaps[x - 1], -1)
            if y < len(y_overlaps):
                tile = tile[:, :, :-y_overlaps[y], :]
            if x < len(x_overlaps):
                tile = tile[:, :, :, :-x_overlaps[x]]
            columns.append(tile)
        rows.append(torch.cat(columns, -1))
    return torch.cat(rows, -2)

