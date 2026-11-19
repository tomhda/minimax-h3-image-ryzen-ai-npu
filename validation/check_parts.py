"""Check the expected local parts and any conversion manifest."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import DEFAULT_CONFIG, DIT_LAYERS, VAE_LAYERS, Parts, load_config
from parts_check import expected_paths, verified_manifest
from parts_manifest import MANIFEST, verify_parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    parts = Parts(config)
    if (parts.root / MANIFEST).is_file():
        count = verify_parts(parts.root, config["parts_repo"])
        expected_paths(parts)
        print(f"Parts manifest verified: {count} files.")
        return
    expected_paths(parts)
    count = 0
    if not parts.overrides:
        units = [parts.root / name for name in ("casts", "outer", "refiner", "vae-common")]
        units += [parts.root / "dit" / f"layer{index:02d}" for index in range(DIT_LAYERS)]
        units += [parts.root / "vae" / f"l{index:02d}" for index in range(VAE_LAYERS)]
        for directory in units:
            verified_manifest(directory)
            count += 1
    print(f"All required local parts found. Manifests verified: {count}.")


if __name__ == "__main__":
    main()

