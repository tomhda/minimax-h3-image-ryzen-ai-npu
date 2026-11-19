"""Decode the final latent and write a 1024 by 1024 RGB PNG."""
import argparse
import time
from pathlib import Path

START = time.perf_counter()

import numpy as np
import torch
from PIL import Image

from common import Parts, add_times, clock, duration, load_config, read_json, save_json
from windows_runtime import finish_npu


def decode_cpu(model, latent):
    ratio = model.spatial_compression_ratio
    y_starts, y_sizes, y_overlaps = model._split_tiles(latent.shape[-2] * ratio, 256, 64)
    x_starts, x_sizes, x_overlaps = model._split_tiles(latent.shape[-1] * ratio, 256, 64)
    tiles = [latent[..., y // ratio:(y + h) // ratio, x // ratio:(x + w) // ratio]
             for y, h in zip(y_starts, y_sizes) for x, w in zip(x_starts, x_sizes)]
    result = [None] * len(tiles)
    groups = {}
    for index, tile in enumerate(tiles):
        groups.setdefault(tuple(tile.shape), []).append(index)
    for indices in groups.values():
        batch = torch.cat([tiles[index] for index in indices])
        decoded = model.decoder(model.post_quant_conv(batch))
        for position, index in enumerate(indices):
            result[index] = decoded[position:position + 1]
    grid = [result[i:i + len(x_starts)] for i in range(0, len(result), len(x_starts))]
    return model._stitch_tiles(grid, y_overlaps, x_overlaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--kind", choices=("cpu", "npu"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.kind == "npu":
        from npu_config import Npu
        from stage_vae_npu import Decoder
    else:
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import AutoencoderKLMiniMaxH3Image

    record = {"import_s": time.perf_counter() - START, "kind": args.kind}
    folder = config["model_path"] / "vae_diffusers"
    stamp = clock()
    vae_config = read_json(folder / "config.json")
    if args.kind == "cpu":
        model = AutoencoderKLMiniMaxH3Image.from_pretrained(str(folder), torch_dtype=torch.float32, local_files_only=True)
        model.eval()
    record["weights_load"] = duration(stamp)

    stamp = clock()
    while not (args.out / "dit.ready").exists():
        time.sleep(0.1)
    record["wait_for_latent"] = duration(stamp)
    stamp = clock()
    raw = np.load(args.out / "vae_input.npy")
    if not np.isfinite(raw).all():
        raise RuntimeError("Non-finite VAE input")
    latent = torch.from_numpy(raw).float()
    latent = latent * torch.tensor(vae_config["latents_std"]).view(1, -1, 1, 1)
    latent = latent + torch.tensor(vae_config["latents_mean"]).view(1, -1, 1, 1)
    prep = duration(stamp)

    with torch.inference_mode():
        stamp = clock()
        if args.kind == "npu":
            model = Decoder(Npu(config, args.threads), Parts(config))
            init = duration(stamp)
            image, layers, pre, post = model.decode(latent)
            record["layers"] = layers
            record["session_create"] = add_times([init] + [row["session_create"] for row in layers])
            record["session_release"] = add_times([row["session_release"] for row in layers])
            record["inference"] = add_times([row["inference"] for row in layers])
        else:
            image = decode_cpu(model, latent)
            record["inference"] = duration(stamp)
            pre = post = {"wall_s": 0.0, "cpu_s": 0.0}
    stamp = clock()
    if not bool(torch.isfinite(image).all()):
        raise RuntimeError("Non-finite VAE output")
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1)
    pixels = (((image * std + mean).clamp(0, 1) * 255).clamp(0, 255).to(torch.uint8))[0]
    Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(args.out / "image.png")
    record["pre_post"] = add_times([prep, pre, post, duration(stamp)])
    save_json(args.out / "vae_log.json", record | {"status": "complete"})
    if args.kind == "npu":
        finish_npu()


if __name__ == "__main__":
    main()

