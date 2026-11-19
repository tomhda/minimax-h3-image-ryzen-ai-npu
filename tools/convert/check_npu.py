"""Check finite output for converted transformer and decoder projections."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import DIT_LAYERS, VAE_LAYERS, Parts, load_config
from doctor_probe import probe
from npu_config import Npu
from stage_vae_npu import Decoder
from windows_runtime import finish_npu, set_affinity


def vae_layer(config, index):
    parts = Parts(config)
    decoder = Decoder(Npu(config), parts)
    decoder.load_layer(index)
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn((1, 261, 1024), generator=generator) * 0.1
    rotary = torch.ones((1, 261, 1, 32)), torch.zeros((1, 261, 1, 32))
    with torch.no_grad():
        output = decoder.block(hidden, index, rotary).numpy()
    decoder.release_layer()
    return output, decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--kind", choices=("dit", "vae"), required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    count = DIT_LAYERS if args.kind == "dit" else VAE_LAYERS
    if not 0 <= args.layer < count:
        parser.error(f"Layer index must be in 0..{count - 1}")
    config = load_config(args.config)
    if args.parts is not None:
        config.update(parts_path=args.parts, part_paths={})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    set_affinity("auto")
    if args.kind == "dit":
        output, _, anchor = probe(config, args.layer)
    else:
        output, anchor = vae_layer(config, args.layer)
    finite = bool(np.isfinite(output).all())
    if args.output:
        np.save(args.output, output)
    print(f"{args.kind} layer {args.layer}: finite={finite}, shape={output.shape}", flush=True)
    finish_npu(0 if finite else 1)


if __name__ == "__main__":
    main()

