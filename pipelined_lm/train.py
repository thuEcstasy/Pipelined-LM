"""Single-device or torchrun pretraining with a sequential row cursor and atomic resume."""

import argparse
import contextlib
from dataclasses import asdict, replace
import fcntl
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from . import Backbone, FoldedPLT4, PLT4, PipelinedLM, tiny_backbone
from .data import open_rows, sha256

MODELS = {"plt4": PLT4, "folded-plt4": FoldedPLT4, "pipeline": PipelinedLM}


def learning_rate(recipe, completed):
    peak = 3e-4
    if recipe != "pipeline":
        return peak * min((completed + 1) / 128, 1.0)
    if completed < 100:
        return peak * (completed + 1) / 100
    splice, original, extended = 22889, 45777, 152588
    pos = (
        completed
        if completed <= splice
        else splice + (completed - splice) * (original - splice) / (extended - splice)
    )
    phase = min(1.0, (pos - 100) / (original - 100))
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * phase)))


class Objective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rows):
        return self.model.loss(rows)


def optimizer_for(model, cuda):
    groups = {0.0: [], 0.1: []}
    for name, param in model.named_parameters():
        if param.requires_grad:
            wd = 0.1 if param.ndim >= 2 and not name.startswith("cross_part.") else 0.0
            groups[wd].append(param)
    return torch.optim.AdamW(
        [dict(params=g, weight_decay=w) for w, g in groups.items()],
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=cuda,
    )


def check_state(model, opt):
    buffers = list(model.parameters()) + [
        v
        for state in opt.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor)
    ]
    by_device = {}
    for tensor in buffers:
        by_device.setdefault((tensor.device, tensor.dtype), []).append(tensor.detach())
    for tensors in by_device.values():
        if not torch.stack(torch._foreach_norm(tensors, float("inf"))).isfinite().all():
            raise FloatingPointError("nonfinite parameter or optimizer state")


def log(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def args_from_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", choices=MODELS, required=True)
    p.add_argument("--output", type=Path, required=True)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument(
        "--data",
        type=Path,
        help="directory containing manifest.json, train.bin and val.bin",
    )
    data.add_argument(
        "--smoke", action="store_true", help="tiny model and generated token IDs"
    )
    p.add_argument(
        "--size", choices=["tiny", "qwen06"], help="research data defaults to qwen06"
    )
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="defaults to CPU for smoke, CUDA for data",
    )
    p.add_argument(
        "--steps",
        type=int,
        help="cumulative update target (required for folded-plt4 on real data)",
    )
    p.add_argument("--global-batch", type=int, help="global sequences per update")
    p.add_argument(
        "--micro-batch",
        type=int,
        default=1,
        help="sequences per rank and backward pass",
    )
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-rows", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--initial-weights", type=Path, help="optional original backbone state_dict"
    )
    a = p.parse_args()
    if a.resume and a.initial_weights:
        p.error("resume reads initialization from latest.pt")
    if a.smoke and a.size == "qwen06":
        p.error("--smoke uses the tiny model")
    a.device = a.device or ("cpu" if a.smoke else "cuda")
    a.size = a.size or ("tiny" if a.smoke else "qwen06")
    if a.steps is None:
        if not a.smoke and a.recipe == "folded-plt4":
            p.error("specify the folded experiment budget with --steps")
        a.steps = 2 if a.smoke else (76294 if a.recipe == "pipeline" else 45777)
    if a.global_batch is None:
        a.global_batch = 4 if a.smoke else (64 if a.recipe == "pipeline" else 128)
    if (
        min(
            a.micro_batch,
            a.global_batch,
            a.steps,
            a.save_every,
            a.eval_every,
            a.eval_rows,
        )
        < 1
    ):
        p.error("batch sizes, periods, eval rows and step target must be positive")
    return a


def run(a, rank, world, device):
    cuda = device.type == "cuda"
    if a.global_batch % (world * a.micro_batch):
        raise ValueError(
            "global batch must be divisible by world size times micro batch"
        )
    shape = tiny_backbone() if a.size == "tiny" else Backbone()
    options = (
        {"policy": "direct", "foresight": True}
        if a.recipe == "pipeline"
        else {"checkpoint_layers": True}
    )
    shape = replace(
        shape,
        window={"plt4": 64, "folded-plt4": 32, "pipeline": shape.window}[a.recipe],
    )
    if a.smoke:

        def generate(n, seed):
            return torch.randint(
                shape.vocab, (n, 17), generator=torch.Generator().manual_seed(seed)
            )

        train, val = generate(128, 11), generate(16, 12)
        identity = "synthetic-v1"
    else:
        train, val = (
            open_rows(a.data, split, verify=rank == 0) for split in ("train", "val")
        )
        identity = sha256(a.data / "manifest.json")
        # A narrow pilot may still use the full Qwen tokenizer.
        shape = replace(shape, vocab=151936)
    if a.steps * a.global_batch > len(train):
        raise ValueError(
            f"need {a.steps*a.global_batch} unique train rows, found {len(train)}"
        )
    cpu_seed = 20261112 if a.recipe == "pipeline" else 20261017
    torch.manual_seed(cpu_seed)
    model = MODELS[a.recipe](shape, device=device, **options)
    run_id = dict(
        schema=1,
        recipe=a.recipe,
        backbone=asdict(shape),
        options=options,
        data=identity,
        global_batch=a.global_batch,
        micro_batch=a.micro_batch,
        world=world,
        device=device.type,
        targets_per_update=a.global_batch * (train.size(1) - 1),
    )
    out = a.output / "latest.pt"
    start = 0
    if a.resume:
        saved = torch.load(out, map_location="cpu", weights_only=True, mmap=True)
        if (
            saved["run"] != run_id
            or saved["targets"] != saved["step"] * run_id["targets_per_update"]
        ):
            raise ValueError(
                "checkpoint recipe, data, topology or token counter mismatch"
            )
        start = saved["step"]
        if not 0 <= start <= a.steps or saved["next_row"] != start * a.global_batch:
            raise ValueError("checkpoint row cursor or target step mismatch")
        model.load_state_dict(saved["model"], strict=True)
    elif a.initial_weights:
        model.load_state_dict(
            torch.load(
                a.initial_weights, weights_only=True, map_location="cpu", mmap=True
            ),
            strict=True,
        )
    opt = optimizer_for(model, cuda)
    if a.resume:
        opt.load_state_dict(saved["optimizer"])
        rng = saved["rng"][rank]
        torch.set_rng_state(rng["cpu"])
        if cuda:
            torch.cuda.set_rng_state(rng["cuda"], device=device)
        del saved
    policy = Objective(model)
    if world > 1:
        policy = DistributedDataParallel(
            policy, device_ids=[device.index] if cuda else None
        )

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda)

    def batch(rows):
        x = rows.to(device=device, dtype=torch.long)
        if x.amin() < 0 or x.amax() >= shape.vocab:
            raise ValueError("token ID outside model vocabulary")
        return x

    with torch.no_grad(), amp():
        sample = batch(val[:1, : min(35, val.size(1))])
        reference = model(sample)
        streamed = model.decode(sample).logits
        relative = float(
            (reference.float() - streamed.float()).norm()
            / reference.float().norm().clamp_min(1e-8)
        )
        if not math.isfinite(relative) or relative > (0.02 if cuda else 0.0001):
            raise ValueError(f"training / streaming logits mismatch: {relative}")
    check_state(model, opt)
    if rank == 0:
        log(
            "ready",
            recipe=a.recipe,
            next_row=start * a.global_batch,
            step=start,
            world=world,
            relative_logits_l2=relative,
            targets_per_update=run_id["targets_per_update"],
        )

    @torch.no_grad()
    def validate():
        n = min(a.eval_rows, len(val))
        total = torch.zeros((), device=device, dtype=torch.float64)
        model.eval()
        for idx in range(rank, n, world):
            with amp():
                total += model.loss(batch(val[idx : idx + 1])).double()
        if world > 1:
            dist.all_reduce(total)
        model.train()
        nll = float(total / n)
        if not math.isfinite(nll):
            raise FloatingPointError("nonfinite validation NLL")
        return nll, n * (val.size(1) - 1)

    def save(step):
        check_state(model, opt)
        state = {"cpu": torch.get_rng_state()}
        if cuda:
            state["cuda"] = torch.cuda.get_rng_state(device)
        rng = [state]
        if world > 1:
            rng = [None] * world
            dist.all_gather_object(rng, state)
        if rank == 0:
            record = dict(
                run=run_id,
                step=step,
                targets=step * run_id["targets_per_update"],
                next_row=step * a.global_batch,
                backbone=asdict(shape),
                options=options,
                architecture=a.recipe,
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                rng=rng,
            )
            tmp = out.with_suffix(".tmp")
            with tmp.open("wb") as f:
                torch.save(record, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, out)
            log("checkpoint", step=step, targets=record["targets"], path=str(out))
        if world > 1:
            dist.barrier()

    local = a.global_batch // world
    accum = local // a.micro_batch
    for completed in range(start, a.steps):
        step = completed + 1
        lr = learning_rate(a.recipe, completed)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device, dtype=torch.float64)
        t0 = time.monotonic()
        begin = completed * a.global_batch + rank * local
        for micro in range(accum):
            offset = begin + micro * a.micro_batch
            sync = (
                policy.no_sync()
                if world > 1 and micro + 1 < accum
                else contextlib.nullcontext()
            )
            with sync:
                with amp():
                    loss = policy(batch(train[offset : offset + a.micro_batch]))
                (loss / accum).backward()
            total += loss.detach().double()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if world > 1:
            dist.all_reduce(total)
        nll = float(total / (world * accum))
        if not math.isfinite(nll):
            raise FloatingPointError("nonfinite training NLL")
        opt.step()
        if cuda:
            torch.cuda.synchronize(device)
        if rank == 0:
            log(
                "train",
                step=step,
                targets=step * run_id["targets_per_update"],
                nll=nll,
                lr=lr,
                grad_norm=float(norm),
                targets_per_second=run_id["targets_per_update"]
                / (time.monotonic() - t0),
            )
        if step % a.eval_every == 0 or step == a.steps:
            val_nll, val_targets = validate()
            if rank == 0:
                log("validation", step=step, nll=val_nll, targets=val_targets)
        if step % a.save_every == 0 or step == a.steps:
            save(step)


def main():
    a = args_from_cli()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if a.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the research recipes require BF16 support")
    else:
        torch.set_num_threads(min(2, torch.get_num_threads()))
    device = (
        torch.device("cuda", local_rank) if a.device == "cuda" else torch.device("cpu")
    )
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        with contextlib.ExitStack() as stack:
            if rank == 0:
                a.output.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context((a.output / ".train.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not a.resume and (a.output / "latest.pt").exists():
                    raise FileExistsError(
                        "checkpoint exists: use --resume or a fresh output directory"
                    )
            if world > 1:
                dist.barrier()
            run(a, rank, world, device)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
