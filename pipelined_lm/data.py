"""Portable, document-contained uint32 token rows used by the training recipes."""

import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_rows(directory, split, *, verify=False):
    root = Path(directory)
    spec = json.loads((root / "manifest.json").read_text())
    entry = spec["splits"][split]
    width, n = spec["row_width"], entry["rows"]
    path = root / f"{split}.bin"
    if (
        sys.byteorder != "little"
        or spec["dtype"] != "<u4"
        or not spec["packing"].startswith("none;")
        or width < 2
        or n < 1
        or entry["targets"] != n * (width - 1)
        or path.stat().st_size != 4 * n * width
    ):
        raise ValueError(f"incompatible or incomplete {split} token rows")
    if verify and sha256(path) != entry["sha256"]:
        raise ValueError(f"{split} token row checksum mismatch")
    return torch.from_file(str(path), size=n * width, dtype=torch.int32).view(n, width)
