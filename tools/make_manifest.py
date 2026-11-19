"""Write the release manifest with the SHA256 and size of each distributed file."""
import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ROOT, read_json, save_json
from parts_manifest import MANIFEST, PINNED_COMMIT, part_files, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--github-commit", help="Commit of the matching GitHub release")
    parser.add_argument("--sdk-version", default="1.8.0")
    args = parser.parse_args()
    if args.github_commit and not PINNED_COMMIT.fullmatch(args.github_commit):
        parser.error("--github-commit must be a full commit hash")
    config = read_json(ROOT / "configs" / "example.json")
    wheels = {}
    for name in ("ryzenai_onnx_utils", "ryzenai_dynamic_dispatch",
                 "onnxruntime_providers_ryzenai", "onnxruntime_vitisai"):
        try:
            wheels[name] = version(name)
        except PackageNotFoundError:
            raise SystemExit(f"Run with the NPU environment's Python: missing {name}") from None
    files = [{"path": name, "size": path.stat().st_size, "sha256": sha256(path)}
             for name, path in sorted(part_files(args.parts_root))]
    if not files:
        parser.error("--parts-root contains no files")
    total = sum(entry["size"] for entry in files)
    save_json(args.parts_root / MANIFEST, {
        "github_commit": args.github_commit,
        "hf_repo": config["parts_repo"],
        "upstream": {
            "minimax_image": {"repo": config["image_repo"], "revision": config["image_revision"]},
            "minimax_text": {"repo": config["text_repo"], "revision": config["text_revision"]},
            "diffusers": {"repo": "huggingface/diffusers", "revision": config["diffusers_revision"]},
        },
        "ryzen_ai": {"sdk_version": args.sdk_version, "wheels": wheels},
        "parts": {
            "dit": {"shape": "1,280 rows (1,024 image and up to 256 text tokens)", "layers": 50},
            "vae": {"shape": "25 overlapping tiles of 16 by 16 latent positions", "layers": 24},
            "vae-common": {"description": "Small decoder tensors and DynamicDispatch attention session"},
            "casts": {"description": "CastAvx graphs"},
            "outer": {"description": "Host tensors for the six DiT steps"},
            "refiner": {"description": "Unquantized condition projection and token refiner tensors"},
        },
        "total_size": total,
        "files": files,
    })
    print(f"Manifest written for {len(files)} files, {total} bytes.")


if __name__ == "__main__":
    main()
