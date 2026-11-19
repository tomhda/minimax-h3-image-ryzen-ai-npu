"""Run a deterministic finite-output check of a complete transformer layer."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import DEFAULT_CONFIG, Parts, clock, duration, load_config, save_json
from dit_layer import Casts, Layer, LayerBuffers
from model import IMAGE_ROWS, WIDTH
from npu_config import Npu, PROVIDER
from windows_runtime import NpuLease, finish_npu, set_affinity


def probe(config, index, seed=7):
    parts = Parts(config)
    npu = Npu(config)
    anchor = npu.session(parts.attention(0) / "replaced.onnx", "attention")
    casts = Casts(npu, parts)
    layer = Layer(npu, casts, parts, index)
    size = IMAGE_ROWS + 31
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn((size, WIDTH), generator=generator) * 0.1
    tags = torch.cat((torch.ones(31, dtype=torch.long), torch.zeros(IMAGE_ROWS, dtype=torch.long)))
    geometry = {"tags": tags, "cos": torch.ones((size, 96)), "sin": torch.zeros((size, 96))}
    mods = torch.zeros((6, 3, WIDTH))
    for gate in (2, 5):
        mods[gate].fill_(0.5)
    with torch.no_grad():
        stamp = clock()
        output, _ = layer.run(hidden, mods, LayerBuffers(geometry))
        inference = duration(stamp)
    layer.close()
    return output.numpy(), inference, anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--layer", default=0, type=int)
    parser.add_argument("--output", required=True, type=Path)
    opts = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    affinity = set_affinity("auto")
    with NpuLease():
        output, timing, anchor = probe(load_config(opts.config), opts.layer)
        finite = bool(np.isfinite(output).all())
        save_json(opts.output, {"ok": finite, "provider": PROVIDER, "layer": opts.layer,
                                "shape": list(output.shape), "inference": timing, "affinity": affinity})
        finish_npu(0 if finite else 1)



if __name__ == "__main__":
    main()

