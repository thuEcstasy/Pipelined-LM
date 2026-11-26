"""Aligned previous-cycle Foresight for the attention and FFN residual streams."""

import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F


def aligned_training(residual, values, query, bias, gain, valid):
    if len(values) == 1:
        message = values[0]
    else:
        scores = torch.stack([(v * query).sum(-1) for v in values], dim=-1)
        weights = (scores * residual.shape[-1] ** -0.5 + bias).softmax(-1)
        message = sum(v * weights[..., j : j + 1] for j, v in enumerate(values))
    message = F.pad(message[:, :-1], (0, 0, 1, 0)) * valid[..., None]
    return (residual.float() + gain * message).to(residual.dtype)


@lru_cache(maxsize=1)
def fused_aligned_training():
    return torch.compile(aligned_training, fullgraph=True, dynamic=False)


class SourceRead(nn.Module):
    def __init__(self, sources, dim, device):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros((), device=device))
        if sources > 1:
            self.query = nn.Parameter(torch.zeros(dim, device=device))
            self.bias = nn.Parameter(torch.zeros(sources, device=device))
        else:
            self.register_parameter("query", None)

    def forward(self, residual, values, previous_valid=None):
        if previous_valid is not None:
            fn = fused_aligned_training() if residual.is_cuda else aligned_training
            return fn(
                residual,
                tuple(values),
                self.query,
                getattr(self, "bias", None),
                self.gain,
                previous_valid,
            )
        if len(values) == 1:
            message = values[0]
        else:
            # Each value is a source-RMS-normalized hidden state. Compute the
            # small softmax in FP32 in both the training and the cached graphs.
            scores = torch.stack([(v * self.query).sum(-1) for v in values], dim=-1)
            weights = (scores / math.sqrt(residual.shape[-1]) + self.bias).softmax(-1)
            message = sum(v * weights[..., j : j + 1] for j, v in enumerate(values))
        return (residual.float() + self.gain * message).to(residual.dtype)


def normalized(x, eps):
    v = x.float()
    return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)


def install(m):
    device = next(m.parameters()).device
    for r in range(1, m.H):
        for layer in m.layers_for_loop(r):
            layer.subblock_attention = SourceRead(r, m.c.dim, device)
            layer.subblock_ffn = SourceRead(r, m.c.dim, device)


def receive(layer, kind, residual, sources, previous_valid=None):
    if not sources:
        return residual
    return getattr(layer, "subblock_" + kind)(residual, sources, previous_valid)


def after_attention(layer, residual, attention):
    return residual + layer.gamma * layer.o(attention.transpose(1, 2).flatten(-2))


def after_ffn(layer, residual):
    from .core.precision import swiglu

    z = layer.n2(residual)
    return residual + layer.gamma * layer.ffn_output_norm(
        layer.down(swiglu(layer, layer.mgate(z), layer.up(z)))
    )
