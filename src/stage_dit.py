"""Start the next denoising process while the current process uses the NPU."""
import argparse
import subprocess
import time
from pathlib import Path

import psutil

from common import ROOT, STEPS, add_times, load_config, process_env, read_json, save_json
from windows_runtime import process_times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--item", required=True, type=Path)
    parser.add_argument("--restart", required=True, choices=("1", "3", "off"))
    parser.add_argument("--threads", required=True, type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    out = args.item.parent
    python = config["npu_python"]
    env = process_env(python, out / "tmp", config)
    size = STEPS if args.restart == "off" else int(args.restart)
    starts = list(range(0, STEPS, size))
    workers = {}
    records = []

    def launch(index):
        first = starts[index]
        folder = out / f"segment{index}"
        folder.mkdir()
        command = [str(python), "-B", "-u", str(ROOT / "src" / "stage_dit_worker.py"),
                   "--config", str(args.config), "--item", str(args.item), "--out", str(folder),
                   "--first-step", str(first), "--end-step", str(min(first + size, STEPS)),
                   "--threads", str(args.threads)]
        if index:
            command += ["--previous", str(workers[index - 1][1])]
        with (folder / "process.log").open("wb") as log:
            worker = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
        workers[index] = worker, folder

    start = time.perf_counter()
    try:
        launch(0)
        for index in range(len(starts)):
            worker, folder = workers[index]
            try:
                affinity = psutil.Process(worker.pid).cpu_affinity()
            except psutil.Error:
                affinity = None
            (folder / "npu.start").touch()
            if index + 1 < len(starts):
                launch(index + 1)
            if worker.wait():
                raise RuntimeError(f"Transformer segment {index} failed. See {folder / 'process.log'}")
            previous_exit = records[-1]["exit_observed_at"] if records else None
            exit_observed_at = time.perf_counter()
            record = read_json(folder / "result.json")
            handoff_s = max(0.0, record.pop("gate_time") - previous_exit) if previous_exit is not None else 0.0
            records.append(record | process_times(worker) | {"affinity": affinity,
                            "exit_observed_at": exit_observed_at, "handoff_s": handoff_s})
    finally:
        for worker, _ in workers.values():
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    metric_keys = ("small_weights", "outer_load", "session_create", "inference", "session_release")
    totals = {key: add_times([record[key] for record in records]) for key in metric_keys}
    for record in records:
        del record["exit_observed_at"]
    save_json(out / "dit_log.json", totals | {
        "status": "complete", "wall_s": time.perf_counter() - start,
        "worker_cpu_s": sum(record["cpu_s"] for record in records),
        "handoff_s": sum(record["handoff_s"] for record in records), "segments": records,
    })


if __name__ == "__main__":
    main()

