"""Local configuration, subprocess environments and stage timing."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "local.json"
PROJECTIONS = ("qkv", "out", "fc1", "fc2")
DIT_LAYERS = 50
VAE_LAYERS = 24
STEPS = 6
# Entries are input offset, output offset, hardware K and hardware N.
PROJECTION_SHAPES = {
    "qkv": (5376, 21504, ((0, 0, 2048, 22016), (2048, 0, 4096, 22016))),
    "out": (7168, 5376, ((0, 0, 7168, 2560), (0, 2560, 8192, 3072))),
    "fc1": (5376, 28672, ((0, 0, 3072, 32064), (3072, 0, 3072, 32064))),
    "fc2": (14336, 5376, ((0, 0, 14336, 4096), (0, 4096, 14336, 4096))),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("npu_python", "text_python", "ryzen_ai_path", "genai_sd_path",
                "model_path", "text_encoder_path", "parts_path", "diffusers_source"):
        if config.get(key):
            config[key] = (path.parent / config[key]).resolve()
    for key, value in config.get("part_paths", {}).items():
        config["part_paths"][key] = (path.parent / value).resolve()
    return config


class Parts:
    def __init__(self, config):
        self.root = config["parts_path"]
        self.overrides = config.get("part_paths", {})

    def dit_matrix(self, layer, name):
        return self.overrides.get("dit_matrices", self.root / "dit") / f"layer{layer:02d}" / name

    def attention(self, layer):
        if "dit_attention" in self.overrides:
            return self.overrides["dit_attention"] / f"heads24norm_m1280_l{layer:02d}" / "dd"
        return self.root / "dit" / f"layer{layer:02d}" / "attention"

    def small(self, layer):
        return self.overrides.get("dit_small", self.root / "dit") / f"layer{layer:02d}" / "small.npz"

    def cast(self, name):
        return self.overrides.get("casts", self.root / "casts") / f"{name}.onnx"

    @property
    def refiner(self):
        return self.root / "refiner" / "weights.safetensors"

    @property
    def outer(self):
        return self.overrides.get("dit_outer", self.root / "outer")

    def vae_matrix(self, layer, name):
        return self.overrides.get("vae_matrices", self.root / "vae") / f"l{layer:02d}" / name

    @property
    def vae_small(self):
        return self.overrides.get("vae_small", self.root / "vae-common" / "small.npz")

    @property
    def vae_anchor(self):
        return self.overrides.get("vae_anchor", self.root / "vae-common" / "attention")


def process_env(python, temp, config):
    prefix = Path(python).parent
    env = os.environ.copy()
    env.update(
        PATH=os.pathsep.join(map(str, (prefix, prefix / "Library" / "bin",
                                      prefix / "Scripts", env["PATH"]))),
        TEMP=str(temp), TMP=str(temp), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_WAIT_POLICY="PASSIVE",
        KMP_BLOCKTIME="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", RYZEN_AI_INSTALLATION_PATH=str(config["ryzen_ai_path"]),
    )
    if config.get("diffusers_source"):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(config["diffusers_source"]), env.get("PYTHONPATH"))))
    return env


def clock():
    return time.perf_counter(), time.process_time()


def duration(start):
    return {"wall_s": time.perf_counter() - start[0], "cpu_s": time.process_time() - start[1]}


def add_times(records):
    return {key: sum(row[key] for row in records) for key in ("wall_s", "cpu_s")}

