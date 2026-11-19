"""Pack weights, partition attention and compile one local conversion unit."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import PROJECTIONS, load_config, save_json
from graph_build import dit_attention, dit_projection, save_casts, save_packed, vae_attention, vae_projection
from npu_config import Npu
from windows_runtime import finish_npu

LINEARS = {"out": "attn.to_out.0", "fc1": "ff.net.0.proj", "fc2": "ff.net.2"}
ORDER = np.concatenate((np.stack((np.arange(48), np.arange(48, 96)), axis=-1).reshape(-1),
                        np.arange(96, 128)))


def save_vae(source, index, destination):
    with safe_open(str(source), framework="np") as weights:
        if index is None:
            decoder = "decoder."
            small = {key.removeprefix(decoder): weights.get_tensor(key) for key in weights.keys()
                     if key.startswith(decoder) and not any(
                         f".{kind}." in key for kind in (*LINEARS.values(), "attn.to_q", "attn.to_k", "attn.to_v"))}
            for key in weights.keys():
                if key.startswith("post_quant_conv."):
                    small[key] = weights.get_tensor(key)
            np.savez(destination / "small.npz", **small)
            return

        prefix = f"decoder.transformer_blocks.{index}."
        for name in PROJECTIONS:
            if name == "qkv":
                weight, bias = [np.concatenate([weights.get_tensor(prefix + f"attn.to_{part}.{kind}")
                                                for part in ("q", "k", "v")]) for kind in ("weight", "bias")]
            else:
                weight, bias = (weights.get_tensor(prefix + LINEARS[name] + "." + kind)
                                for kind in ("weight", "bias"))
            directory = destination / name
            save_packed(vae_projection(weight), directory)
            np.save(directory / "bias.npy", bias)
            print(f"Packed VAE layer {index} {name}", flush=True)


def save_dit(index, destination, scratch):
    for name in PROJECTIONS:
        packed = [np.load(scratch / f"{name}_{suffix}.npy", mmap_mode="r") for suffix in ("q", "scale", "zero")]
        model, metadata = dit_projection(name, packed)
        del packed
        directory = destination / name
        save_packed(model, directory)
        save_json(directory / "host.json", metadata)
        # The provider expects a cache directory for int4 session creation.
        (directory / "cache").mkdir()
        for suffix in ("q", "scale", "zero"):
            (scratch / f"{name}_{suffix}.npy").unlink()
        print(f"Packed DiT layer {index} {name}", flush=True)


def partition(model, directory, scratch, strategy):
    from ryzenai_onnx_utils.preprocess import infer_symbolic_shapes

    directory.mkdir()
    # The SDK recognizes its RMSNorm pattern before the ONNX opset checker does.
    shaped = infer_symbolic_shapes(model)
    names = {value.name for value in shaped.graph.value_info}
    shaped.graph.value_info.extend(value for value in shaped.graph.input if value.name not in names)
    path = scratch / "attention.onnx"
    onnx.save_model(shaped, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=path.name + ".data", size_threshold=1024)
    target = directory / "replaced.onnx"
    cli = Path(sys.executable).parent / "Scripts" / "onnx_utils.exe"
    partition_work = scratch / "partition"
    partition_work.mkdir()
    local_strategy = partition_work / "attention.yaml"
    shutil.copyfile(strategy, local_strategy)
    subprocess.run([str(cli), "partition", str(path), str(target), str(local_strategy), "--attributes",
                    "whitebox_checker_level=2", "--save-as-external", "--force", "-v"],
                   cwd=partition_work, check=True)
    if target.is_dir():
        temporary = directory / "graph.tmp"
        (target / target.name).replace(temporary)
        for entry in target.iterdir():
            entry.replace(directory / entry.name)
        target.rmdir()
        temporary.replace(target)
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--unit", choices=("dit", "vae", "vae-common", "casts"), required=True)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    args.destination.mkdir(exist_ok=True, parents=True)
    source = config["model_path"] / "vae_diffusers" / "diffusion_pytorch_model.safetensors"
    if args.unit == "casts":
        save_casts(args.destination)
        return
    if args.unit == "vae":
        os.environ["DD_ROOT"] = str(config["ryzen_ai_path"] / "deployment")
        os.environ.pop("DD_PLUGINS_ROOT", None)
        save_vae(source, args.layer, args.destination)
        return

    os.environ["DD_ROOT"] = str(config["ryzen_ai_path"] / "deployment")
    os.environ.pop("DD_PLUGINS_ROOT", None)
    start = time.perf_counter()
    if args.unit == "dit":
        save_dit(args.layer, args.destination, args.scratch)
        with np.load(args.destination / "small.npz") as small:
            model = dit_attention(small["attn.q_norm.weight"], small["attn.k_norm.weight"], ORDER)
    else:
        save_vae(source, None, args.destination)
        model = vae_attention()
    weights_s = time.perf_counter() - start

    start = time.perf_counter()
    lib = config["genai_sd_path"] / "lib"
    os.environ.update(DD_ROOT=str(lib), DD_PLUGINS_ROOT=str(lib / "transaction" / "stx"))
    target = partition(model, args.destination / "attention", args.scratch, args.strategy)
    npu = Npu(config)
    session = npu.session(target, "attention", compile_part=True)
    dims = {inp.name: tuple(inp.shape) for inp in session.get_inputs()}
    inputs = {name: np.zeros(shape, np.float32) for name, shape in dims.items()}
    inputs["mark"].fill(1)
    for name in inputs:
        if name.startswith("cos"):
            inputs[name].fill(1)
    if not np.isfinite(session.run(None, inputs)[0]).all():
        raise RuntimeError(f"Non-finite attention output: {target}")
    save_json(args.scratch / "timings.json", {"weight_export_s": weights_s,
                                             "attention_build_s": time.perf_counter() - start})
    (args.scratch / "attention.ok").touch()
    print(f"Compiled and checked {target}", flush=True)
    finish_npu()


if __name__ == "__main__":
    main()

