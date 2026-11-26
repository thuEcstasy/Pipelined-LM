"""Qwen-shaped layer used by the non-looped pipeline."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Backbone:
    vocab: int = 151936  # Qwen3-0.6B padded vocabulary
    dim: int = 1024
    layers: int = 28
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 3072
    window: int = 64
    eps: float = 1e-6
    theta: float = 1000000.0


def positions(docs):
    t = torch.arange(docs.shape[1], device=docs.device).expand_as(docs)
    starts = torch.cat(
        (torch.ones_like(docs[:, :1], dtype=torch.bool), docs[:, 1:] != docs[:, :-1]),
        dim=1,
    )
    return t - torch.where(starts, t, 0).cummax(dim=1).values


def shift(x, docs):
    return torch.cat(
        (
            torch.zeros_like(x[:, :1]),
            x[:, :-1] * (docs[:, 1:] == docs[:, :-1]).unsqueeze(-1),
        ),
        dim=1,
    )


def rope(x, pos, theta):
    freq = theta ** (
        -torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1]
    )
    angle = pos[:, None, :, None].float() * freq
    a, b = x.chunk(2, dim=-1)
    return torch.cat(
        (a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1
    ).to(x.dtype)


def attend(q, kv, mask, groups):
    k, v = (x.repeat_interleave(groups, dim=1) for x in kv)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[:, None], dropout_p=0.0
    )


class OutputNorm(nn.RMSNorm):
    def forward(self, x):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            * self.weight.float()
        ).to(x.dtype)


class Layer(nn.Module):
    def __init__(self, c, gamma):
        super().__init__()
        self.c = c
        self.gamma = gamma
        self.ffn_output_norm = OutputNorm(c.dim, eps=c.eps)
        self.n1 = nn.RMSNorm(c.dim, eps=c.eps)
        self.n2 = nn.RMSNorm(c.dim, eps=c.eps)
        self.q_norm = nn.RMSNorm(c.head_dim, eps=c.eps)
        self.k_norm = nn.RMSNorm(c.head_dim, eps=c.eps)
        self.q = nn.Linear(c.dim, c.heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.dim, c.kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.dim, c.kv_heads * c.head_dim, bias=False)
        self.o = nn.Linear(c.heads * c.head_dim, c.dim, bias=False)
        self.up = nn.Linear(c.dim, c.intermediate, bias=False)
        self.mgate = nn.Linear(c.dim, c.intermediate, bias=False)
        self.down = nn.Linear(c.intermediate, c.dim, bias=False)
        self.gate_weight = nn.Parameter(torch.zeros(c.heads, c.head_dim))
        self.gate_bias = nn.Parameter(torch.zeros(c.heads))

    def project(self, x, pos):
        z = self.n1(x)

        def reshape(y, heads):
            return y.reshape(*y.shape[:2], heads, self.c.head_dim).transpose(1, 2)

        q = self.q_norm(reshape(self.q(z), self.c.heads))
        k = self.k_norm(reshape(self.k(z), self.c.kv_heads))
        v = reshape(self.v(z), self.c.kv_heads)
        # Gate is applied to the projected Q before rotary positional rotation.
        gate = torch.sigmoid(
            (q * self.gate_weight[None, :, None]).sum(-1)
            + self.gate_bias[None, :, None]
        ).unsqueeze(-1)
        q, k = rope(q, pos, self.c.theta), rope(k, pos, self.c.theta)
        if getattr(self, "round_rotary_output", False):
            q, k = q.to(v.dtype), k.to(v.dtype)
        return q, (k, v), gate

    def finish(self, x, q, gate, global_kv, gm, local_kv=None, lm=None):
        groups = self.c.heads // self.c.kv_heads
        out = attend(q, global_kv, gm, groups)
        if local_kv is not None:
            local = attend(q, local_kv, lm, groups)
            out = gate * local + (1 - gate) * out
        x = x + self.gamma * self.o(out.transpose(1, 2).flatten(-2))
        z = self.n2(x)
        from .precision import swiglu

        return x + self.gamma * self.ffn_output_norm(
            self.down(swiglu(self, self.mgate(z), self.up(z)))
        )
