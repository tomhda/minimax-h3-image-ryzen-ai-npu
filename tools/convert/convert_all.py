"""Build the complete set of local NPU parts from the pinned model inputs."""
import argparse
import msvcrt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))

from common import (DEFAULT_CONFIG, DIT_LAYERS, PROJECTIONS, ROOT, STEPS, VAE_LAYERS,
                    Parts, load_config, process_env, read_json, save_json)
from parts_check import attention_files, expected_paths, projection_files, require, verified_manifest
from download_models import ensure_rebuild_inputs
from windows_runtime import KillOnClose, NpuLease, ascii_directory


def selected(value, count):
    if value == "all":
        return list(range(count))
    if value == "none":
        return []
    if not re.fullmatch(r"\d+(,\d+)*", value):
        raise argparse.ArgumentTypeError("Expected 'all', 'none' or a comma-separated layer list")
    result = sorted({int(number) for number in value.split(",")})
    if any(number >= count for number in result):
        raise argparse.ArgumentTypeError(f"Layer indices must be in 0..{count - 1}")
    return result


def check_unit(directory, kind):
    if kind == "dit":
        require(directory / "small.npz", 40_000)
        attention_files(directory / "attention")
    elif kind == "vae-common":
        require(directory / "small.npz", 50_000)
        attention_files(directory / "attention")
    elif kind == "outer":
        for name, minimum in (("modules.npz", 100_000_000), ("tembs.npy", 64_000), ("sigmas.npy", 150)):
            require(directory / name, minimum)
        for step in range(STEPS):
            require(directory / f"modulations{step}.npy", 19_000_000)
    elif kind == "casts":
        for name in ("to_bf", "to_fp"):
            require(directory / f"{name}.onnx", 50)
    elif kind == "refiner":
        require(directory / "weights.safetensors", 100_000_000)
    if kind in ("dit", "vae"):
        for name in PROJECTIONS:
            projection_files(directory / name, decoder=kind == "vae")


def build_all(args, config, out):
    full = args.dit_layers == "all" and args.vae_layers == "all"
    dit = selected(args.dit_layers, DIT_LAYERS)
    vae = selected(args.vae_layers, VAE_LAYERS)
    records = []
    log = out / "convert_log.json"
    signature = {key: str(config[key]) for key in ("model_path", "image_revision", "diffusers_revision",
                                                   "ryzen_ai_path", "genai_sd_path")}
    work = out / ".build"
    work.mkdir(exist_ok=True)
    for name in ("scratch", "logs"):
        (work / name).mkdir(exist_ok=True)
    # Temporary units can be left behind if Windows terminates a conversion.
    for leftover in (work / "scratch").iterdir():
        if leftover.is_dir() and not leftover.is_junction():
            shutil.rmtree(leftover)
    floor = float(config.get("min_disk_free_gb", 0)) * 10**9

    def run(kind, script, log_file, parameters, scratch):
        python = config[f"{kind}_python"]
        subprocess_env = process_env(python, scratch, config)
        command = [str(python), "-B", "-u", str(HERE / script), "--config", str(args.config), *map(str, parameters)]
        with log_file.open("wb") as handle, KillOnClose() as job:
            start = time.perf_counter()
            child = subprocess.Popen(command, env=subprocess_env, cwd=scratch, stdout=handle,
                                     stderr=subprocess.STDOUT)
            job.assign(child)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                child.kill()
                child.wait()
                raise
        if code:
            raise RuntimeError(f"{script} failed with exit {code}. See {log_file}")
        return time.perf_counter() - start

    def build(relative, kind, layer=None):
        destination = out / relative
        start = time.perf_counter()
        if destination.is_dir():
            manifest = verified_manifest(destination)
            if manifest["source"] != signature:
                raise ValueError(f"Input configuration changed for {destination}. Choose another output directory.")
            check_unit(destination, kind)
            return {"part": relative, "status": "verified", "bytes": sum(manifest["files"].values()),
                    "wall_s": time.perf_counter() - start}

        reserve = {"dit": 900_000_000, "refiner": 1_700_000_000}.get(kind, 500_000_000)
        if shutil.disk_usage(out).free < floor + reserve:
            raise RuntimeError(f"Insufficient disk space to build {relative}")
        label = relative.replace("/", "-")
        durations = {}
        with tempfile.TemporaryDirectory(prefix=label + "-", dir=work / "scratch") as temp:
            scratch = Path(temp)
            pending = scratch / "part"
            pending.mkdir()
            if kind in ("dit", "outer", "refiner"):
                cpu_args = ["--destination", pending]
                if layer is not None:
                    cpu_args += ["--dit-layer", layer, "--scratch", scratch]
                if kind == "refiner":
                    cpu_args += ["--refiner"]
                durations["cpu_build_s"] = run(
                    "text", "build_cpu.py", work / "logs" / f"{label}-cpu.log", cpu_args, scratch
                )
            if kind not in ("outer", "refiner"):
                npu_args = ["--unit", kind, "--destination", pending, "--scratch", scratch,
                            "--strategy", args.strategy]
                if layer is not None:
                    npu_args += ["--layer", layer]
                durations["npu_build_s"] = run(
                    "npu", "build_npu.py", work / "logs" / f"{label}-npu.log", npu_args, scratch
                )
                timings = scratch / "timings.json"
                if timings.is_file():
                    durations.update(read_json(timings))
            check_unit(pending, kind)
            if kind in ("dit", "vae-common") and not (scratch / "attention.ok").is_file():
                raise RuntimeError(f"Missing finite-output confirmation for {relative}")
            files = {path.relative_to(pending).as_posix(): path.stat().st_size
                     for path in sorted(pending.rglob("*")) if path.is_file()}
            save_json(pending / "manifest.json", {"part": relative, "source": signature, "files": files})
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending.rename(destination)
        return {"part": relative, "status": "built", "bytes": sum(files.values()),
                "wall_s": time.perf_counter() - start, **durations}

    # Write independent units directly under the layout used by the runtime.
    jobs = []
    if dit:
        jobs.append(("casts", "casts", None))
    jobs.extend((f"dit/layer{layer:02d}", "dit", layer) for layer in dit)
    if full:
        jobs.append(("outer", "outer", None))
        jobs.append(("refiner", "refiner", None))
    if full or args.vae_common:
        jobs.append(("vae-common", "vae-common", None))
    jobs.extend((f"vae/l{layer:02d}", "vae", layer) for layer in vae)
    for relative, kind, index in jobs:
        record = build(relative, kind, index)
        records.append(record)
        save_json(log, {"parts": records})
        action = "Built" if record["status"] == "built" else "Reused"
        print(f"{action} {relative} in {record['wall_s']:.2f} s", flush=True)
    if full:
        expected_paths(Parts(config | {"parts_path": out, "part_paths": {}}))
    if args.verify:
        for kind, layers in (("dit", dit), ("vae", vae)):
            if not layers:
                continue
            if kind == "dit" and not (out / "dit" / "layer00" / "attention").is_dir():
                raise ValueError("Build DiT layer 0 before validating DiT projections")
            if kind == "vae" and not (out / "vae-common").is_dir():
                raise ValueError("Build the shared VAE parts before validating VAE projections")
            run("npu", "check_npu.py", work / "logs" / f"verify-{kind}.log",
                ["--parts", out, "--kind", kind, "--layer", layers[0]], work)
            print(f"Verified finite output for {kind} layer {layers[0]}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dit-layers", default="all")
    parser.add_argument("--vae-layers", default="all")
    parser.add_argument("--vae-common", action="store_true", help="Also build the shared VAE parts in a partial run")
    parser.add_argument("--verify", action="store_true", help="Run one complete layer from each selected stage")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.strategy = HERE / "strategies" / "attention.yaml"
    config = load_config(args.config)
    try:
        ensure_rebuild_inputs(config, download=False)
    except FileNotFoundError as exc:
        parser.error(f"{exc}. Run setup.ps1 -WithDiTWeights to download the inputs for rebuilding.")
    out = (args.out or ROOT / "rebuild").resolve()
    if (out / "manifest.json").is_file():
        parser.error("Choose a build directory separate from the installed parts")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".conversion.lock"
    with lock.open("a+b") as handle:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"Conversion output is in use: {out}") from exc
        try:
            with NpuLease(), ascii_directory(out) as aliased:
                build_all(args, config, aliased)
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()

