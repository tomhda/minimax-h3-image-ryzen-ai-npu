"""Report installed versions without importing model or NPU libraries."""
import importlib.metadata
import json
import sys

versions = {"python": sys.version.split()[0]}
for name in sys.argv[1:]:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = "missing"
print(json.dumps(versions, indent=2))

