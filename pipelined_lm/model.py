"""Untied, four-stage Pipelined LM with training and incremental execution."""

import copy
import hashlib
import math
from types import MethodType

import torch
from torch import nn

from . import depth_read, full_kv, gswa, subblock
from .config import Backbone, Partition, PipelineConfig
from .core.aggregation import CrossLoopAggregation
from .core.layers import Layer
from .core.linear_ce import linear_ce, project
from .core.precision import enable
from .core.sequence import layout

POLICIES = {
    "full_1d": "full1d",
    "full_2d": "full2d",
    "gswa_1d": "gswa1d",
    "history": "gswa2d",
    "direct": "gswa2d_direct",
    "gated": "gswa2d_frontier",
}


class PipelinedLM(nn.Module):
    """Build the Qwen3-0.6B-shaped research model, or pass a smaller Backbone."""

    def __init__(
        self,
        backbone=None,
        *,
        policy="direct",
        foresight=True,
        aligned_foresight=False,
        subblock_residuals=False,
        block_size=4,
        seed=20261112,
        device="cpu",
    ):
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(
                f"unknown cache policy {policy!r}; choose from {tuple(POLICIES)}"
            )
        aligned_foresight = aligned_foresight or subblock_residuals
        if aligned_foresight and (policy != "direct" or not foresight):
            raise ValueError(
                "the subblock experiment uses direct frontier KV and Foresight"
            )
        self.c = backbone or Backbone()
        c = self.c
        if (
            c.layers < 8
            or (c.layers - 4) % 4
            or min(
                c.dim,
                c.window,
                c.heads,
                c.kv_heads,
                c.intermediate,
                c.head_dim,
                c.vocab,
            )
            < 1
            or c.heads % c.kv_heads
            or c.head_dim % 2
        ):
            raise ValueError("invalid backbone size or four-stage partition")
        self.gswa2d_mode = POLICIES[policy]
        self.gswa = policy not in ("full_1d", "full_2d")
        self.H = 4
        self.gamma = 1.0
        self.source_rmsnorm = True
        self.cross_loop_source = "previous_token"
        self.cross_loop_aggregation = "attnres"
        self.aligned_foresight = aligned_foresight
        self.subblock_residuals = subblock_residuals
        self.frontier_backend = "eager"
        p = Partition(depth=(c.layers - 4) // 4)
        self.cfg = PipelineConfig(p, "gswa" if self.gswa else "global")
        self.mixing_layers = (
            {i for r in range(1, 4) for i in p.branch_indices(r)}
            if self.gswa
            else set()
        )
        self.kv_publishers = set(range(c.layers))
        self.kv_source = {i: i for i in range(c.layers)}
        self.global_source = dict(self.kv_source)
        for r in range(1, 4):
            if self.gswa:
                self.global_source.update(zip(p.branch_indices(r), p.branch_indices(0)))
        with torch.device("meta"):
            self.embedding = nn.Embedding(c.vocab, c.dim)
            self.stack = nn.ModuleList(Layer(c, 1.0) for _ in range(c.layers))
            for i, layer in enumerate(self.stack):
                layer.index = i
                layer.ffn_output_norm = nn.Identity()
                for param in (layer.gate_weight, layer.gate_bias):
                    param.requires_grad_(i in self.mixing_layers)
            self.final_norm = nn.RMSNorm(c.dim, eps=c.eps)
            self.cross_part = CrossLoopAggregation(c.dim, 4, "attnres", 1e-6)
            self.cross_part.predecessor_only = not foresight
        self.to_empty(device=device)
        self._shared_init(seed)
        enable(self)
        if policy == "gated":
            for r in range(1, 4):
                for layer in self.layers_for_loop(r):
                    layer.frontier_weight = nn.Parameter(
                        torch.zeros(c.heads, c.head_dim, device=device)
                    )
                    layer.frontier_bias = nn.Parameter(
                        torch.zeros(c.heads, device=device)
                    )
        if aligned_foresight:
            subblock.install(self)
        if subblock_residuals:
            depth_read.install(self, block_size)
        if self.gswa:
            self.hidden = MethodType(gswa.hidden, self)
            self.prefill = MethodType(gswa.stream, self)
            self.decode = MethodType(gswa.stream, self)
        else:
            full_kv.attach(
                self, "private" if policy == "full_1d" else "wavefront_same_layer"
            )

    @torch.no_grad()
    def _shared_init(self, seed):
        for name, param in self.named_parameters():
            if name.startswith("cross_part.") or name.endswith(
                ("gate_weight", "gate_bias")
            ):
                param.fill_(float(name.startswith("cross_part.alpha")))
            elif param.ndim == 1:
                param.fill_(1.0)
            else:
                digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
                gen = torch.Generator(device=param.device).manual_seed(
                    int.from_bytes(digest[:8], "little") % (2**63 - 1)
                )
                param.normal_(0, 0.02, generator=gen)
                if name.endswith((".o.weight", ".down.weight")):
                    param.div_(math.sqrt(2 * self.c.layers))

    @property
    def cross_loop(self):
        return self.cross_part

    def layers_for_loop(self, r):
        return tuple(self.stack[i] for i in self.cfg.partition.branch_indices(r))

    def prelude_layers(self):
        return tuple(self.stack[i] for i in self.cfg.partition.front_indices())

    def coda_layers(self):
        return tuple(self.stack[i] for i in self.cfg.partition.tail_indices())

    def head(self, x):
        return project(
            self.final_norm(x.float()),
            getattr(self, "inference_head_weight", self.embedding.weight),
        )

    def forward(self, tokens, docs=None, *, valid=None):
        """Return next-token logits [batch, sequence, vocab]."""
        return self.head(self.hidden(tokens, docs, valid=valid))

    def loss(self, tokens, docs=None, *, valid=None, chunk=512):
        """Mean next-token NLL. Skip padding and document transitions."""
        docs = torch.zeros_like(tokens) if docs is None else docs
        ids, segments, mask = layout(tokens, docs, valid)
        keep = mask[:, 1:] & mask[:, :-1] & (segments[:, 1:] == segments[:, :-1])
        if not keep.any():
            raise ValueError("at least one within-document target is required")
        h = self.hidden(ids[:, :-1], docs[:, :-1], valid=mask[:, :-1])
        features = self.final_norm(h[keep].float())
        return (
            linear_ce(features, self.embedding.weight, ids[:, 1:][keep], chunk=chunk)
            / keep.sum()
        )

    def cuda_decoder(self, batch_size, capacity):
        """Build a separate inference copy with paged BF16 KV and CUDA Graph support."""
        if self.embedding.weight.device.type != "cuda":
            raise ValueError("move the model to a CUDA device first")
        from .engine import Decoder

        return Decoder(copy.deepcopy(self).eval(), batch_size, capacity)
