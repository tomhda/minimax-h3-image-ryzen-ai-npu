"""Check the set and recorded size of each local model part."""
from pathlib import Path

from common import DIT_LAYERS, PROJECTIONS, STEPS, VAE_LAYERS, read_json


def require(path, minimum=1):
    if not path.is_file() or path.stat().st_size < minimum:
        raise ValueError(f"Missing or truncated part: {path}")


def projection_files(directory, decoder=False):
    require(directory / "replaced.onnx", 200)
    require(directory / "replaced.pb.bin", 40)
    if decoder:
        require(directory / "bias.npy", 128)
    else:
        require(directory / "host.json", 40)
    refs = directory / "source_refs.json"
    data = list(read_json(refs).values()) if refs.is_file() else [directory / "replaced.onnx.data"]
    for path in data:
        require(Path(path), 100_000)


def attention_files(directory):
    require(directory / "replaced.onnx", 500)
    require(directory / "cache" / "ops-config.json", 2)
    for extension in ("ddbundle", "fconst", "state"):
        matches = list(directory.glob(f"dd_metastate_*.{extension}"))
        if not matches:
            raise ValueError(f"Missing DynamicDispatch {extension} in {directory}")
        for path in matches:
            require(path)


def expected_paths(parts):
    if not parts.overrides:
        require(parts.refiner, 100_000_000)
    for name, minimum in (("modules.npz", 100_000_000), ("tembs.npy", 64_000), ("sigmas.npy", 150)):
        require(parts.outer / name, minimum)
    for step in range(STEPS):
        require(parts.outer / f"modulations{step}.npy", 19_000_000)
    for name in ("to_bf", "to_fp"):
        require(parts.cast(name), 50)
    for layer in range(DIT_LAYERS):
        require(parts.small(layer), 40_000)
        attention_files(parts.attention(layer))
        for name in PROJECTIONS:
            projection_files(parts.dit_matrix(layer, name))
    require(parts.vae_small, 50_000)
    attention_files(parts.vae_anchor)
    for layer in range(VAE_LAYERS):
        for name in PROJECTIONS:
            projection_files(parts.vae_matrix(layer, name), decoder=True)


def verified_manifest(directory):
    manifest = read_json(directory / "manifest.json")
    for relative, size in manifest["files"].items():
        base = directory.resolve()
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"Invalid manifest entry: {relative}")
        require(path, size)
        if path.stat().st_size != size:
            raise ValueError(f"Size mismatch: {path}")
    return manifest

