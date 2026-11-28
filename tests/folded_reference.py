"""Folded PLT-4 scalar FP32 specification. Only PyTorch and raw weights are needed."""

import math

import torch
from torch.nn import functional as F


def rms(x, scale, eps):
    return F.rms_norm(x, (x.size(-1),), weight=scale, eps=eps)


def readout(weights, config, x):
    return F.linear(
        rms(x, weights["final_norm.weight"], config.eps), weights["embedding.weight"]
    )


def visible(stage, target, ell):
    """List (producer, layer offset, prediction position), using zero-based stages."""
    # A state at (j,s) was produced at u=s-16+j. A receiver at (r,t)
    # runs at c=t-16+r. Always require 0 <= u <= c, plus c-u < 32 for locals.
    clock = target - 16 + stage
    global_keys = [(0, ell, 16 + u) for u in range(clock + 1)]
    local_keys = [
        (producer, ell, u + 16 - producer)
        for producer in range(1, stage + 1)
        for u in range(max(0, clock - 31), clock + 1)
    ]
    return global_keys, local_keys


def scalar_trajectory(weights, config, ids, *, foresight):
    """One unpadded CPU document. Return states, stage inputs and every K/V by trajectory."""
    if ids.ndim != 1 or ids.numel() == 0 or config.window != 32 or config.layers % 4:
        raise ValueError(
            "expected one nonempty token row, four equal slices and window 32"
        )
    depth = config.layers // 4
    gamma = 0.5 / (4 * math.sqrt(config.layers))
    groups = config.heads // config.kv_heads
    banks, states, entries = {}, {}, {}
    layers = []
    for index in range(config.layers):
        prefix = f"stack.{index}."
        layers.append(
            {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
        )

    def heads(z, weight, count):
        return F.linear(z, weight).reshape(1, count, 1, config.head_dim)

    def rotate(x, cycle):
        half = config.head_dim // 2
        freq = config.theta ** (-torch.arange(half, device=x.device).float() / half)
        phase = cycle * freq
        a, b = x.chunk(2, -1)
        return x * phase.cos().repeat(2) + torch.cat((-b, a), -1) * phase.sin().repeat(
            2
        )

    def retrieve(query, keys):
        key, value = (torch.cat(side, 2) for side in zip(*(banks[k] for k in keys)))
        scores = query @ key.repeat_interleave(groups, 1).transpose(-2, -1)
        return (scores / math.sqrt(config.head_dim)).softmax(
            -1
        ) @ value.repeat_interleave(groups, 1)

    for stage in range(16):
        for cycle, token in enumerate(ids):
            target = cycle + 16 - stage
            x = F.embedding(token.view(1, 1), weights["embedding.weight"])
            if stage and cycle:
                if foresight:
                    previous = [states[j, target + stage - j - 1] for j in range(stage)]
                    values = rms(torch.stack(previous, -2), None, config.eps)
                    query = weights[f"cross_part.query.{stage - 1}"]
                    scores = (values @ query) / math.sqrt(config.dim)
                    probs = (scores + weights[f"cross_part.bias.{stage - 1}"]).softmax(
                        -1
                    )
                    x = x + weights[f"cross_part.alpha.{stage - 1}"] * (
                        probs[..., None] * values
                    ).sum(-2)
                else:
                    x = x + states[stage - 1, target]
            entries[stage, target] = x
            for ell in range(depth):
                w = layers[(stage % 4) * depth + ell]
                z = rms(x, w["n1.weight"], config.eps)
                q = rms(
                    heads(z, w["q.weight"], config.heads),
                    w["q_norm.weight"],
                    config.eps,
                )
                k = rms(
                    heads(z, w["k.weight"], config.kv_heads),
                    w["k_norm.weight"],
                    config.eps,
                )
                v = heads(z, w["v.weight"], config.kv_heads)
                gate = (
                    (q * w["gate_weight"][None, :, None]).sum(-1)
                    + w["gate_bias"][None, :, None]
                ).sigmoid()[..., None]
                q = rotate(q, cycle)
                banks[stage, ell, target] = rotate(k, cycle), v
                glob, local = visible(stage, target, ell)
                out = retrieve(q, glob)
                if local:
                    out = (1 - gate) * out + gate * retrieve(q, local)
                x = x + gamma * F.linear(out.transpose(1, 2).flatten(-2), w["o.weight"])
                z = rms(x, w["n2.weight"], config.eps)
                mlp = F.silu(F.linear(z, w["mgate.weight"])) * F.linear(
                    z, w["up.weight"]
                )
                x = x + gamma * rms(
                    F.linear(mlp, w["down.weight"]),
                    w["ffn_output_norm.weight"],
                    config.eps,
                )
            states[stage, target] = x
    return states, entries, banks
