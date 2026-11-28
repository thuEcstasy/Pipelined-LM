"""Build independent-document train/val rows from pinned public data or local JSONL."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from tokenizers import Tokenizer

from pipelined_lm.data import sha256

TOKENIZER = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
SOURCES = {
    "web": (
        "HuggingFaceFW/fineweb-edu",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "sample/100BT",
    ),
    "math": (
        "HuggingFaceTB/finemath",
        "e92b25a616738fe95dc186b64dfb19f9c8525594",
        "finemath-3plus",
    ),
}


def documents(name, local):
    if local is not None:
        with local.open() as stream:
            for line in stream:
                yield json.loads(line)["text"]
        return
    import pyarrow.parquet as pq

    repo, rev, folder = SOURCES[name]
    files = sorted(
        entry.path
        for entry in HfApi().list_repo_tree(
            repo, path_in_repo=folder, repo_type="dataset", revision=rev
        )
        if entry.path.endswith(".parquet")
    )
    if not files:
        raise ValueError(f"no parquet shards found for {name}")
    fs = HfFileSystem()
    for file in files:
        uri = f"datasets/{repo}@{rev}/{file}"
        with fs.open(uri, "rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(
                batch_size=256, columns=["text"]
            ):
                yield from batch.column(0).to_pylist()


def token_rows(name, local, tokenizer, length, split):
    for text in documents(name, local):
        if not isinstance(text, str) or not text.strip():
            continue
        digest = hashlib.sha256(text.encode()).digest()
        held_out = int.from_bytes(digest[:8], "little") % 100 < 5
        if held_out != (split == "val"):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        for start in range(0, len(ids) - length, length):
            yield ids[start : start + length + 1]


def build(args, root, tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    local = {"web": args.web_jsonl, "math": args.math_jsonl}
    names = ["web", "math"] if args.corpus == "mix" else ["web"]
    sources = {
        k: ({"local_jsonl_sha256": sha256(local[k])} if local[k] else SOURCES[k])
        for k in names
    }
    result = dict(
        dtype="<u4",
        row_width=args.seq_len + 1,
        packing="none; one document per row; stride seq_len",
        tokenizer_sha256=sha256(tokenizer_path),
        vocab=151936,
        corpus=args.corpus,
        sources=sources,
        split_rule="little-endian sha256(text)[:8] mod 100 < 5 selects validation",
        splits={},
    )
    for split, count in [("train", args.train_rows), ("val", args.val_rows)]:
        streams = {
            k: token_rows(k, local[k], tokenizer, args.seq_len, split) for k in names
        }
        counts = dict.fromkeys(names, 0)
        path = root / f"{split}.bin"
        try:
            with path.open("xb") as dest:
                for i in range(count):
                    name = (
                        "math" if args.corpus == "mix" and (i + 1) % 10 == 0 else "web"
                    )
                    try:
                        ids = next(streams[name])
                    except StopIteration:
                        raise ValueError(
                            f"{name} ran out of {split} documents at row {i}"
                        ) from None
                    dest.write(np.asarray(ids, dtype="<u4").tobytes())
                    counts[name] += 1
        finally:
            for stream in streams.values():
                stream.close()
        result["splits"][split] = dict(
            rows=count,
            targets=count * args.seq_len,
            sha256=sha256(path),
            source_rows=counts,
        )
        print(split, result["splits"][split], flush=True)
    (root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", choices=["web", "mix"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-rows", type=int, required=True)
    p.add_argument("--val-rows", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument(
        "--web-jsonl", type=Path, help='optional local documents with a "text" field'
    )
    p.add_argument(
        "--math-jsonl",
        type=Path,
        help='optional local math documents with a "text" field',
    )
    p.add_argument(
        "--tokenizer-json", type=Path, help="local copy of the pinned Qwen tokenizer"
    )
    args = p.parse_args()
    if min(args.train_rows, args.val_rows, args.seq_len) < 1 or args.output.exists():
        p.error(
            "use positive row counts and sequence length, and a new output directory"
        )
    tokenizer_path = args.tokenizer_json or hf_hub_download(
        TOKENIZER[0], "tokenizer.json", revision=TOKENIZER[1]
    )
    if sha256(tokenizer_path) != TOKENIZER_SHA:
        raise ValueError("Qwen tokenizer checksum mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        staging = Path(tmp) / "rows"
        staging.mkdir()
        build(args, staging, tokenizer_path)
        os.rename(staging, args.output)


if __name__ == "__main__":
    main()
