"""Fetch and verify the pinned model files used for conversion and generation."""
import argparse
import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import load_config, read_json
from parts_manifest import MANIFEST, PINNED_COMMIT, load_manifest, sha256, verify_parts

IMAGE_RUNTIME = (
    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    "vae_diffusers/config.json", "vae_diffusers/diffusion_pytorch_model.safetensors",
)
DIT_INDEX = "transformer/model.safetensors.index.json"
DIT_EXTRAS = ("pdd_config.json", "pdd_fused_heads.safetensors")


class ModelFiles:
    def __init__(self, folder, repo, revision, prefix=""):
        self.folder = folder
        self.repo = repo
        self.revision = revision
        self.prefix = prefix
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            from huggingface_hub import HfApi

            model = HfApi().model_info(self.repo, revision=self.revision, files_metadata=True)
            if model.sha != self.revision:
                raise ValueError(f"Unexpected revision for {self.repo}: {model.sha}")
            self._metadata = {file.rfilename: file for file in model.siblings}
        return self._metadata

    def download(self, name, destination):
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repo, self.prefix + name, revision=self.revision,
                                    local_dir=destination))

    def verify(self, name, reference_dir):
        key = self.prefix + name
        file = self.metadata[key]
        local = self.folder / name
        if local.stat().st_size != file.size:
            raise ValueError(f"Size mismatch for {local}: expected {file.size} bytes")
        if file.lfs:
            if file.lfs.size != file.size:
                raise ValueError(f"Inconsistent LFS size for {key}")
            expected = file.lfs.sha256
        else:
            reference = self.download(name, reference_dir)
            data = reference.read_bytes()
            git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if len(data) != file.size or git_blob != file.blob_id:
                raise ValueError(f"Pinned Git blob mismatch for {key}")
            expected = hashlib.sha256(data).hexdigest()
        actual = sha256(local)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {local}: expected {expected}, got {actual}")
        print(f"Verified {self.repo}@{self.revision} {key} {file.size} {actual}", flush=True)

    def ensure(self, names, download, reference_dir=None):
        for name in names:
            local = self.folder / name
            if not local.is_file() or local.stat().st_size == 0:
                if not download:
                    raise FileNotFoundError(f"Missing model input: {local}")
                if self.prefix and self.folder.name != self.prefix.rstrip("/"):
                    self.folder.mkdir(parents=True, exist_ok=True)
                    with TemporaryDirectory(dir=self.folder) as scratch:
                        fetched = self.download(name, scratch)
                        local.parent.mkdir(parents=True, exist_ok=True)
                        fetched.replace(local)
                else:
                    target = self.folder.parent if self.prefix else self.folder
                    self.download(name, target)
            if reference_dir is not None:
                self.verify(name, reference_dir)


def ensure_rebuild_inputs(config, download, reference_dir=None):
    model = config["model_path"]
    image = ModelFiles(model, config["image_repo"], config["image_revision"])
    image.ensure((DIT_INDEX, *DIT_EXTRAS), download, reference_dir)
    index = read_json(model / DIT_INDEX)
    image.ensure([f"transformer/{name}" for name in sorted(set(index["weight_map"].values()))],
                 download, reference_dir)
    image.ensure(("vae_diffusers/diffusion_pytorch_model.safetensors",), download)


def required_files(config, download, reference_dir=None, with_dit_weights=False):
    model = config["model_path"]
    image = ModelFiles(model, config["image_repo"], config["image_revision"])
    image.ensure(IMAGE_RUNTIME, download, reference_dir)
    if with_dit_weights:
        ensure_rebuild_inputs(config, download, reference_dir)

    text = config["text_encoder_path"]
    text_source = ModelFiles(text, config["text_repo"], config["text_revision"], "text_encoder/")
    text_source.ensure(("config.json", "model.safetensors.index.json"), download, reference_dir)
    index = read_json(text / "model.safetensors.index.json")["weight_map"]
    layers = "model.language_model.layers."
    shards = sorted({shard for name, shard in index.items()
                     if name == "model.language_model.embed_tokens.weight"
                     or (name.startswith(layers) and int(name[len(layers):].split(".")[0]) < 50)})
    text_source.ensure(shards, download, reference_dir)


def install_parts(config, download):
    root, repo = config["parts_path"], config["parts_repo"]
    if download:
        from huggingface_hub import hf_hub_download, snapshot_download

        revision = config["parts_revision"]
        if not PINNED_COMMIT.fullmatch(revision):
            raise ValueError("Parts revision must be a full Hugging Face commit hash")
        hf_hub_download(repo, MANIFEST, revision=revision, local_dir=root)
        entries = load_manifest(root, repo)
        snapshot_download(repo, revision=revision, local_dir=root,
                          allow_patterns=[entry["path"] for entry in entries])
    count = verify_parts(root, repo)
    print(f"Parts manifest verified: {count} files.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="Require the local inputs without downloading weights")
    parser.add_argument("--verify-existing", action="store_true", help="Check size and SHA256 against the pinned revisions")
    parser.add_argument("--with-dit-weights", action="store_true", help="Include the original DiT files for rebuilding")
    parser.add_argument("--install-parts", action="store_true", help="Prepare and verify the parts directory")
    parser.add_argument("--skip-parts", action="store_true", help="Verify parts without downloading them")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.install_parts:
        install_parts(config, not (args.skip_parts or args.check_only))
    if args.verify_existing:
        with TemporaryDirectory(prefix="model-verification-", dir=args.config.resolve().parent) as scratch:
            required_files(config, not args.check_only, Path(scratch), args.with_dit_weights)
    else:
        required_files(config, not args.check_only, with_dit_weights=args.with_dit_weights)
    print("Model inputs ready.")


if __name__ == "__main__":
    main()
