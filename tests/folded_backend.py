"""Train/serve parity on the selected backend, at full stage/layer depth and small width."""

from dataclasses import replace

import torch

from pipelined_lm import FoldedPLT4, tiny_backbone
from pipelined_lm.train import optimizer_for


def configure_cuda():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "the requested CUDA check requires an available BF16-capable GPU"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, 128
    )


def check_backend(device, foresight, metrics):
    cuda = device == "cuda"
    limit = 0.02 if cuda else 1e-4
    shape = replace(tiny_backbone(), layers=28, window=32, head_dim=64)
    m = FoldedPLT4(shape, device=device, foresight=foresight, checkpoint_layers=True)
    ids = torch.arange(80, device=device).view(2, 40) * 11 % shape.vocab
    metrics.update(
        shape=[16, 7], batch=2, length=40, prefix=33, relative_l2_limit=limit
    )
    opt = optimizer_for(m, cuda)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=cuda):
        loss = m.loss(ids, chunk=16)
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is None or not p.grad.isfinite().all():
            raise AssertionError(f"missing or nonfinite gradient: {name}")
    metrics["nll"] = loss.item()
    if not torch.isfinite(loss):
        raise AssertionError("nonfinite next-token NLL")
    metrics["grad_norm"] = float(
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1, error_if_nonfinite=True)
    )
    opt.step()
    opt.zero_grad(set_to_none=True)
    m.eval()

    def parity(name, actual, expected):
        if actual.shape != expected.shape:
            raise AssertionError(f"{name}: {actual.shape} != {expected.shape}")
        a, b = actual.float(), expected.float()
        error = ((a - b).norm() / b.norm().clamp_min(1e-8)).item()
        metrics.setdefault("relative_l2", {})[name] = (
            error if error < float("inf") else None
        )
        if not 0 <= error < limit:
            raise AssertionError(f"{name}: relative L2 {error} (limit {limit})")

    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16, enabled=cuda),
    ):
        graph = m.hidden(ids, return_details=True)
        logits = m.head(graph.hidden)
        stream = m.decode(ids)
        prefix = m.prefill(ids[:, :33])
        suffix = m.decode(ids[:, 33:], cache=prefix.cache)
        parity("logits/stream", stream.logits, logits)
        parity("logits/prefill", prefix.logits, logits[:, :33])
        parity("logits/continuation", suffix.logits, logits[:, 33:])
        if set(graph.kv) != {(j, ell) for j in range(16) for ell in range(7)}:
            raise AssertionError("training graph must publish all 112 logical KV pairs")
        for path, cache in [("stream", stream.cache), ("prefill_decode", suffix.cache)]:
            if (
                cache.time != 40
                or len(cache.global_banks) != 7
                or len(cache.local_banks) != 7
            ):
                raise AssertionError(
                    f"{path}: wrong cache clock or physical layer count"
                )
            parity(
                f"{path}/previous_states",
                cache.previous,
                torch.stack([x[:, -1:] for x in graph.loops[:-1]]),
            )
            for (r, ell), pair in graph.kv.items():
                for side, expected in enumerate(pair):
                    if r == 0:
                        cached = cache.global_banks[ell][side][:, :, :40]
                    else:
                        ring = cache.local_banks[ell][side]
                        if ring.shape[2:4] != (15, 32):
                            raise AssertionError(f"{path}: incorrect local KV capacity")
                        slots = torch.arange(8, 40, device=device) % 32
                        cached = ring[:, :, r - 1].index_select(2, slots)
                        expected = expected[:, :, 8:]
                    parity(
                        f"{path}/stage{r + 1}/offset{ell}/{'KV'[side]}",
                        cached,
                        expected,
                    )
