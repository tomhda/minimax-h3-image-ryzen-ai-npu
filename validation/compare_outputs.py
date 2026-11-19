"""Compare final latents, text embeddings and PNGs in two output folders."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def array_comparison(left, right):
    a, b = np.load(left), np.load(right)
    if a.shape != b.shape:
        return {"same_shape": False, "left": list(a.shape), "right": list(b.shape)}
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {"bit_equal": a.dtype == b.dtype and a.tobytes() == b.tobytes(),
            "max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-30))}


def png_comparison(left, right):
    a = np.asarray(Image.open(left).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(right).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return {"same_shape": False, "left": list(a.shape), "right": list(b.shape)}
    delta = a - b
    mse = float(np.mean(delta**2))
    return {"byte_equal": hashlib.sha256(left.read_bytes()).digest() == hashlib.sha256(right.read_bytes()).digest(),
            "pixel_equal": mse == 0, "max_abs": float(np.abs(delta).max()),
            "psnr_db": float(10 * np.log10(255**2 / mse)) if mse else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    report = {}
    equal = True
    for pattern, compare in (("*.png", png_comparison), ("latents/*.npy", array_comparison),
                             ("text/*.npy", array_comparison)):
        left = {path.relative_to(args.left) for path in args.left.glob(pattern)}
        right = {path.relative_to(args.right) for path in args.right.glob(pattern)}
        for relative in sorted(left | right):
            if relative not in left or relative not in right:
                report[relative.as_posix()] = {"missing": "right" if relative in left else "left"}
                equal = False
                continue
            result = compare(args.left / relative, args.right / relative)
            report[relative.as_posix()] = result
            equal &= result.get("bit_equal", result.get("byte_equal", False))
    print(json.dumps(report, indent=2))
    return 0 if equal and report else 1


if __name__ == "__main__":
    raise SystemExit(main())

