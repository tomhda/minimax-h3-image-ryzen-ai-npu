"""Generate 1024 by 1024 images from a JSON item list."""
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

import psutil

from common import DEFAULT_CONFIG, ROOT, load_config, process_env, read_json, save_json
from windows_runtime import KeepAwake, KillOnClose, NpuBusy, NpuLease, process_times, set_affinity

VALID_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")


def read_items(path):
    items = read_json(path)
    if not isinstance(items, list) or not items:
        raise ValueError("Items must be a nonempty JSON array")
    names = set()
    for item in items:
        if not isinstance(item, dict) or not {"id", "prompt", "seed"} <= item.keys():
            raise ValueError("Each item requires id, prompt and seed")
        name, prompt, seed = (item[key] for key in ("id", "prompt", "seed"))
        if not isinstance(name, str) or not VALID_ID.fullmatch(name) or name.endswith("."):
            raise ValueError(f"Invalid item id: {name!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Empty prompt for {name}")
        if type(seed) is not int or not -(2**63) <= seed < 2**64:
            raise ValueError(f"Invalid seed for {name}")
        filename = f"{name}_s{seed}".lower()
        if filename in names:
            raise ValueError(f"Duplicate image filename for {name} seed {seed}")
        names.add(filename)
    return items


def generate_item(item, work, config, args):
    temp = work / "tmp"
    temp.mkdir()
    item_file = work / "item.json"
    save_json(item_file, item)
    children = {}
    records = {}
    minimum = psutil.virtual_memory().available
    floor = int(config.get("min_available_gb", 5) * 10**9)
    disk_floor = int(config.get("min_disk_free_gb", 0) * 10**9)
    start, cpu = time.perf_counter(), time.process_time()
    start_unix = time.time()

    def spawn(stage, python, script, *options):
        cmd = [str(python), "-B", "-u", str(ROOT / "src" / script), "--config", str(args.config), *map(str, options)]
        with (work / f"{stage}.log").open("wb") as log:
            child = subprocess.Popen(cmd, cwd=work, env=process_env(python, temp, config), stdout=log,
                                     stderr=subprocess.STDOUT)
        job.assign(child)
        children[stage] = child
        try:
            affinity = psutil.Process(child.pid).cpu_affinity()
        except psutil.Error:
            affinity = None
        records[stage] = {"affinity": affinity}

    text_python = config["text_python"]
    status = "complete"
    with KillOnClose() as job:
        spawn("text", text_python, "stage_text.py", "--item", item_file, "--out", work,
              "--threads", args.text_threads, "--read-workers", args.read_workers)
        finished = set()
        while True:
            available = psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < floor:
                raise RuntimeError(f"Available physical memory is {available / 10**9:.2f} GB. "
                                   f"The configured minimum is {floor / 10**9:g} GB.")
            if disk_floor and shutil.disk_usage(work).free < disk_floor:
                raise RuntimeError(f"Free output disk space is below {disk_floor / 10**9:g} GB")
            for stage, child in children.items():
                if stage in finished or child.poll() is None:
                    continue
                records[stage].update(process_times(child), exit_code=child.returncode)
                if stage == "text" and child.returncode == 3:
                    status = "skipped"
                    break
                if child.returncode:
                    raise RuntimeError(f"{stage} exited with code {child.returncode}. See {work / (stage + '.log')}")
                finished.add(stage)
                (work / f"{stage}.ready").touch()
            if status == "skipped" or "vae" in finished:
                break
            if "text" in finished and "dit" not in children:
                # Let the text worker release the refiner weights before importing the NPU workers.
                spawn("dit", text_python, "stage_dit.py", "--item", item_file, "--restart", args.dit_restart,
                      "--threads", args.dit_threads)
            if "vae" not in children and ((work / "vae.preload").exists() or "dit" in finished):
                spawn("vae", config["npu_python"] if args.vae == "npu" else text_python,
                      "stage_vae.py", "--out", work, "--kind", args.vae, "--threads", args.vae_threads)
            time.sleep(0.2)

    record = {"id": item["id"], "seed": item["seed"], "status": status,
              "start_unix_s": start_unix, "end_unix_s": time.time(),
              "wall_s": time.perf_counter() - start, "supervisor_cpu_s": time.process_time() - cpu,
              "min_available_bytes": minimum, "stages": records}
    if status == "complete":
        dit = read_json(work / "dit_log.json")
        record["child_cpu_s"] = sum(row["cpu_s"] for row in records.values()) + dit["worker_cpu_s"]
        record["cpu_wall_ratio"] = record["child_cpu_s"] / record["wall_s"]
        name = f"{item['id']}_s{item['seed']}"
        os.replace(work / "image.png", args.out / f"{name}.png")
        shutil.copyfile(work / "vae_input.npy", args.out / "latents" / f"{name}.npy")
        shutil.copyfile(work / "text_embeds.npy", args.out / "text" / f"{name}.npy")
        print(f"Saved {name}.png", flush=True)
    else:
        tokens = read_json(work / "text_log.json")["tokens"]
        print(f"SKIP id={item['id']} tokens={tokens}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--vae", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--affinity", choices=("auto", "off"), default="auto")
    parser.add_argument("--dit-restart", choices=("3", "1", "off"), default="3")
    parser.add_argument("--text-threads", type=int, default=4)
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--dit-threads", type=int, default=2)
    parser.add_argument("--vae-threads", type=int, default=2)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.out = args.out.resolve()
    try:
        if min(args.text_threads, args.read_workers, args.dit_threads, args.vae_threads) < 1:
            raise ValueError("Thread counts must be positive")
        items = read_items(args.items)
        config = load_config(args.config)
        args.out.mkdir(parents=True, exist_ok=True)
        lock = args.out / ".generation.lock"
        lock_handle = lock.open("a+b")
        try:
            lock_handle.seek(0)
            msvcrt.locking(lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock_handle.close()
            raise ValueError(f"Output folder is in use: {args.out}") from None
    except (OSError, ValueError, KeyError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    records = []
    stage_logs = {stage: [] for stage in ("text", "dit", "vae")}
    summary = {"affinity": set_affinity(args.affinity), "vae": args.vae, "dit_restart": args.dit_restart,
               "items": records}
    try:
        for folder in ("latents", "text"):
            (args.out / folder).mkdir(exist_ok=True)
        with NpuLease(), KeepAwake() as awake:
            summary["keep_awake"] = awake
            for item in items:
                work = Path(tempfile.mkdtemp(prefix="run-", dir=args.out))
                record = generate_item(item, work, config, args)
                records.append(record)
                for stage, rows in stage_logs.items():
                    path = work / f"{stage}_log.json"
                    if path.is_file():
                        rows.append({"id": item["id"], "seed": item["seed"], **read_json(path)})
                work = work.resolve()
                if work.parent != args.out:
                    raise RuntimeError("Unexpected temporary output directory")
                shutil.rmtree(work)
        return 3 if any(row["status"] == "skipped" for row in records) else 0
    except NpuBusy as exc:
        summary["error"] = str(exc)
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        summary["error"] = str(exc)
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            save_json(args.out / "gen_summary.json", summary)
            for stage, rows in stage_logs.items():
                save_json(args.out / f"{stage}_log.json", rows)
        finally:
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

