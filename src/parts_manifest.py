"""Build and verify the SHA256 manifest for a complete parts directory."""
import hashlib
import re
from pathlib import PurePosixPath

from common import read_json

MANIFEST = "manifest.json"
PINNED_COMMIT = re.compile(r"[0-9a-f]{40}")
PORTABLE_PATH = re.compile(r"[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*")
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_repository_asset(relative):
    name = relative.casefold()
    return name in ("readme.md", ".gitattributes") or name.startswith("images/")


def part_files(root, *, ignore_hub_cache=False):
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if is_repository_asset(relative):
            continue
        if ignore_hub_cache and relative.startswith(".cache/huggingface/"):
            continue
        if path.is_symlink() or path.is_junction():
            raise ValueError(f"Unexpected link in parts: {relative}")
        if relative != MANIFEST and path.is_file():
            yield relative, path


def load_manifest(root, repo):
    manifest = read_json(root / MANIFEST)
    if manifest["hf_repo"] != repo:
        raise ValueError(f"Parts manifest is for {manifest['hf_repo']}, expected {repo}")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("No files listed in the parts manifest")
    paths = set()
    for entry in entries:
        name = entry["path"]
        if (not PORTABLE_PATH.fullmatch(name) or name.casefold() == MANIFEST
                or any(part in (".", "..") or part.endswith(".") for part in PurePosixPath(name).parts)
                or name.casefold().startswith(".cache/") or name.casefold() in paths):
            raise ValueError(f"Invalid or duplicate parts manifest path: {name}")
        paths.add(name.casefold())
        if type(entry["size"]) is not int or entry["size"] < 0 or not SHA256.fullmatch(entry["sha256"]):
            raise ValueError(f"Invalid size or SHA256 for {name}")
    if manifest["total_size"] != sum(entry["size"] for entry in entries):
        raise ValueError("Inconsistent total size in the parts manifest")
    # Older manifests can list repository assets that are now excluded.
    parts = [entry for entry in entries if not is_repository_asset(entry["path"])]
    if not parts:
        raise ValueError("No files listed in the parts manifest")
    return parts


def verify_parts(root, repo):
    entries = load_manifest(root, repo)
    files = dict(part_files(root, ignore_hub_cache=True))
    expected = {entry["path"] for entry in entries}
    if missing := sorted(expected - files.keys()):
        raise ValueError(f"Missing parts: {', '.join(missing[:10])}")
    if extra := sorted(files.keys() - expected):
        raise ValueError(f"Unexpected files in parts: {', '.join(extra[:10])}")
    for entry in entries:
        name = entry["path"]
        path = files[name]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Parts size mismatch: {name}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Parts SHA256 mismatch: {name}")
    return len(entries)
