"""Quantize transformer weights and export CPU-side tensors."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import DIT_LAYERS, STEPS, load_config, read_json
from mmap_weights import Weights
from model import PROJECTION_KEYS, SMALL_KEYS, WIDTH

CLIP = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)


def quantize(weight):
    groups = weight.reshape(weight.shape[0], -1, 128)
    lo = groups.amin(2, keepdim=True)
    hi = groups.amax(2, keepdim=True)
    center = (lo + hi) / 2
    best = None
    for alpha in CLIP:
        half = (hi - lo) / 2 * alpha
        lower, upper = center - half, center + half
        scale = (upper - lower) / 15
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = torch.round(-lower / scale).clamp(0, 15)
        values = torch.round(groups / scale + zero).clamp(0, 15)
        error = (((values - zero) * scale - groups)**2).sum(2, keepdim=True)
        candidate = (values, scale, zero)
        if best is None:
            best = candidate
            best_error = error
        else:
            improved = error < best_error
            best = tuple(torch.where(improved, a, b) for a, b in zip(candidate, best))
            best_error = torch.where(improved, error, best_error)
    return best


def export_dit(reader, index, destination, scratch):
    prefix = f"blocks.{index}."
    np.savez(destination / "small.npz", **{name: reader.fp32(prefix + name).numpy() for name in SMALL_KEYS})
    for name, key in PROJECTION_KEYS.items():
        weight = reader.read(prefix + key)
        n, k = weight.shape
        values = np.lib.format.open_memmap(scratch / f"{name}_q.npy", mode="w+", dtype=np.uint8, shape=(n, k))
        scales = np.lib.format.open_memmap(scratch / f"{name}_scale.npy", mode="w+", dtype=np.float32,
                                           shape=(n, k // 128))
        zeros = np.lib.format.open_memmap(scratch / f"{name}_zero.npy", mode="w+", dtype=np.uint8,
                                          shape=(n, k // 128))
        for start in range(0, n, 128):
            raw = weight[start:start + 128].float().contiguous()
            quantized, scale, zero = quantize(raw)
            values[start:start + len(raw)] = quantized.reshape(-1, k).numpy().astype(np.uint8)
            scales[start:start + len(raw)] = scale[..., 0].numpy()
            zeros[start:start + len(raw)] = zero[..., 0].numpy().astype(np.uint8)
        for array in (values, scales, zeros):
            array.flush()
        del values, scales, zeros, array
        print(f"Quantized DiT layer {index} {name}", flush=True)


def export_outer(model, reader, destination):
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    keys = {"proj.weight": "video_patch_proj.weight", "proj.bias": "video_patch_proj.bias",
            "norm.norm.weight": "final_layer.norm.weight",
            "norm.linear.weight": "final_layer.adaln_proj.linear.weight",
            "norm.linear.bias": "final_layer.adaln_proj.linear.bias"}
    weights = {name: reader.fp32(key).numpy() for name, key in keys.items()}
    with safe_open(str(model / "pdd_fused_heads.safetensors"), framework="pt", device="cpu") as heads:
        for step in range(STEPS):
            for kind in ("weight", "bias"):
                weights[f"head{step}.{kind}"] = heads.get_slice(f"video_out.{kind}")[step].float().numpy()
    np.savez(destination / "modules.npz", **weights)

    projection = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
    embedding = TimestepEmbedding(in_channels=256, time_embed_dim=WIDTH, out_dim=2688)
    embedding.load_state_dict({f"linear_{index}.{name}": reader.fp32(f"time_embedder.proj_{part}.{name}")
                               for index, part in ((1, "in"), (2, "out")) for name in ("weight", "bias")})
    schedule = read_json(model / "pdd_config.json")
    ts = torch.tensor([1 - sigma for sigma in schedule["sigmas"][:-1]], dtype=torch.float32)
    scheduler = MiniMaxH3Scheduler(shift=12.0, stochastic_sampling=False, shift_provided_sigmas=False,
                                  euler_update_form="velocity")
    scheduler.set_timesteps(sigmas=torch.cat((1 - ts, torch.zeros(1))), device="cpu")
    tembs = [embedding(projection(t.reshape(1))) for t in scheduler.timesteps]
    np.save(destination / "tembs.npy", torch.stack(tembs).numpy())
    np.save(destination / "sigmas.npy", scheduler.sigmas.numpy())
    modulations = np.empty((STEPS, DIT_LAYERS, 6, 3, WIDTH), np.float32)
    for layer in range(DIT_LAYERS):
        base = f"blocks.{layer}.adaln_proj.linear."
        weight, bias = (reader.fp32(base + name) for name in ("weight", "bias"))
        for step, temb in enumerate(tembs):
            values = F.linear(F.silu(temb), weight, bias).view(-1, 6 * WIDTH).chunk(6, -1)
            modulations[step, layer] = np.stack([v.numpy() for v in values])
    for step in range(STEPS):
        np.save(destination / f"modulations{step}.npy", modulations[step])


def export_refiner(reader, destination):
    weights = {name: reader.read(name).contiguous() for name in sorted(reader.mapping)
               if name.startswith(("condition_proj.", "token_refiner."))}
    save_file(weights, destination / "weights.safetensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--scratch", type=Path)
    unit = parser.add_mutually_exclusive_group()
    unit.add_argument("--dit-layer", type=int)
    unit.add_argument("--refiner", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    model = load_config(args.config)["model_path"]
    reader = Weights(model / "transformer")
    with torch.no_grad():
        if args.refiner:
            export_refiner(reader, args.destination)
        elif args.dit_layer is None:
            export_outer(model, reader, args.destination)
        else:
            export_dit(reader, args.dit_layer, args.destination, args.scratch)


if __name__ == "__main__":
    main()

