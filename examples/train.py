"""Single-device next-token training example with optional pretokenized document rows."""

import argparse
from dataclasses import asdict
from pathlib import Path
import os

import torch
from pipelined_lm import Backbone, POLICIES, PipelinedLM, tiny_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, help="a torch.save long tensor [documents, tokens_per_row]"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--size", choices=["tiny", "qwen06"], default="tiny")
    parser.add_argument("--policy", choices=POLICIES, default="direct")
    parser.add_argument("--subblock-residuals", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1:
        parser.error("steps and batch size must be positive")
    if args.output and args.output.exists():
        parser.error("choose a new output path")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    shape = tiny_backbone() if args.size == "tiny" else Backbone()
    if args.data:
        rows = torch.load(args.data, map_location="cpu", weights_only=True, mmap=True)
        if (
            not isinstance(rows, torch.Tensor)
            or rows.ndim != 2
            or rows.dtype != torch.long
            or not len(rows)
            or rows.shape[1] < 2
        ):
            parser.error(
                "data must be a nonempty torch.long matrix with at least two tokens per row"
            )
    else:
        rows = torch.randint(
            shape.vocab, (32, 17), generator=torch.Generator().manual_seed(11)
        )
    kwargs = dict(policy=args.policy, subblock_residuals=args.subblock_residuals)
    model = PipelinedLM(shape, device=args.device, **kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    count = 0
    for step in range(args.steps):
        indices = (torch.arange(args.batch_size) + step * args.batch_size) % len(rows)
        tokens = rows[indices].to(args.device)
        if (tokens < 0).any() or (tokens >= shape.vocab).any():
            parser.error("token ID outside the selected backbone vocabulary")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"
        ):
            loss = model.loss(tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        count += tokens.shape[0] * (tokens.shape[1] - 1)
        print(f"step {step + 1:4d}  targets {count:8d}  nll {loss.item():.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(".tmp")
        torch.save(
            dict(
                backbone=asdict(shape),
                options=kwargs,
                model=model.state_dict(),
                targets=count,
            ),
            tmp,
        )
        os.replace(tmp, args.output)


if __name__ == "__main__":
    main()
