"""Run a consecutive range of denoising steps in one process."""
import argparse
import time
from pathlib import Path

START = time.perf_counter()

import numpy as np
import torch

from common import DIT_LAYERS, Parts, add_times, clock, duration, load_config, read_json, save_json
from dit_layer import Casts, Layer, LayerBuffers
from model import Outer
from npu_config import Npu
from windows_runtime import finish_npu

IMPORT_S = time.perf_counter() - START


def wait_for(path):
    while not path.exists():
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--item", required=True, type=Path)
    parser.add_argument("--first-step", required=True, type=int)
    parser.add_argument("--end-step", required=True, type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    parts = Parts(config)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    wait_for(args.out / "npu.start")
    gate_time = time.perf_counter()
    wait_for(args.item.parent / "text.ready")
    item = read_json(args.item)
    start = clock()
    outer = Outer(parts, args.item.parent, item["seed"],
                  args.previous / "checkpoint.npy" if args.previous else None)
    outer_load = duration(start)
    start = clock()
    npu = Npu(config, args.threads)
    # Keep a diffusion DynamicDispatch session alive for the duration of this process.
    anchor = npu.session(parts.attention(0) / "replaced.onnx", "attention")
    anchor_time = duration(start)
    casts = None
    cast_time = {"wall_s": 0.0, "cpu_s": 0.0}
    rows = []
    steps = []
    with torch.no_grad():
        for step in range(args.first_step, args.end_step):
            if step == 5:
                (args.item.parent / "vae.preload").touch()
            begin = clock()
            stamp = clock()
            hidden, mods = outer.begin(step)
            entry = duration(stamp)
            buffers = LayerBuffers(outer.geometry)
            for index in range(DIT_LAYERS):
                if casts is None:
                    stamp = clock()
                    casts = Casts(npu, parts)
                    cast_time = duration(stamp)
                layer = Layer(npu, casts, parts, index)
                stamp = clock()
                hidden, calls = layer.run(hidden, mods[index], buffers)
                inference = duration(stamp)
                stamp = clock()
                layer.close()
                release = duration(stamp)
                rows.append({"step": step, "layer": index, "small_weights": layer.weight_load,
                             "session_create": layer.session_create, "inference": inference,
                             "session_release": release, "calls": calls})
                del layer
            stamp = clock()
            outer.end(step, hidden)
            steps.append({"step": step, "entry": entry, "exit": duration(stamp), **duration(begin)})
            print(f"Completed denoising step {step + 1}", flush=True)

    if args.end_step == 6:
        latent = outer.vae_input().numpy()
        if not np.isfinite(latent).all():
            raise RuntimeError("Non-finite VAE input")
        np.save(args.item.parent / "vae_input.npy", latent)
    else:
        np.save(args.out / "checkpoint.npy", outer.latents.numpy())
    totals = {key: add_times([row[key] for row in rows])
              for key in ("small_weights", "session_create", "inference", "session_release")}
    totals["session_create"] = add_times([totals["session_create"], anchor_time, cast_time])
    totals["inference"] = add_times([totals["inference"]] + [row[key] for row in steps for key in ("entry", "exit")])
    save_json(args.out / "result.json", totals | {"status": "complete", "import_s": IMPORT_S,
                                                 "gate_time": gate_time, "outer_load": outer_load,
                                                 "steps": steps, "layers": rows})
    # The process owns the anchor until it exits.
    finish_npu()


if __name__ == "__main__":
    main()

