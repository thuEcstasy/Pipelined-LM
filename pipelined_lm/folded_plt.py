"""Four passes of a shared PLT backbone, each split into four cross-token stages."""

from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import Backbone
from .core.layers import Layer, positions, shift
from .core.precision import enable
from .core.sequence import GraphOutput, layout
from .full_kv import finish
from .gswa import attend
from .model import AutoregressiveLM


@dataclass
class FoldCache:
    """One cache per decoding batch. Time counts published input columns."""

    time: int = 0
    last_docs: torch.Tensor | None = None
    last_valid: torch.Tensor | None = None
    starts: torch.Tensor | None = None
    previous: torch.Tensor | None = None
    global_banks: dict = field(default_factory=dict)
    local_banks: dict = field(default_factory=dict)

    def publish(self, ell, keys, values, window):
        # Global [B,H,T,D] grows geometrically. The joint local [B,H,15,W,D]
        # allocation doubles as the attention operand, so a full ring needs no repack.
        bank = self.global_banks.get(ell)
        if bank is None or self.time == bank[0].size(2):
            capacity = 2 * bank[0].size(2) if bank is not None else 1
            shape = (*keys.shape[1:3], capacity, keys.size(-1))
            new = tuple(x.new_empty(shape) for x in (keys, values))
            if bank is not None:
                for target, old in zip(new, bank):
                    target[:, :, : self.time].copy_(old)
            self.global_banks[ell] = bank = new
        ring = self.local_banks.get(ell)
        if ring is None:
            shape = (*keys.shape[1:3], keys.size(0) - 1, window, keys.size(-1))
            self.local_banks[ell] = ring = tuple(
                x.new_zeros(shape) for x in (keys, values)
            )
        for x, full, local in zip((keys, values), bank, ring):
            full[:, :, self.time : self.time + 1].copy_(x[0])
            local[:, :, :, self.time % window].copy_(
                x[1:].squeeze(3).permute(1, 2, 0, 3)
            )
        return (
            tuple(x[:, :, : self.time + 1] for x in bank),
            tuple(x.flatten(2, 3) for x in ring),
        )


@dataclass
class FoldDecodeOutput:
    logits: torch.Tensor
    cache: FoldCache


class _LoopedPipeline(AutoregressiveLM):
    """Stage r runs physical slice r % 4, using the previous cycle's r-1 output."""

    outer_loops = 4

    def __init__(
        self, backbone=None, *, seed=20261017, device="cpu", checkpoint_layers=False
    ):
        super().__init__()
        self.c = c = backbone or Backbone(window=self.window)
        self.stages = self.outer_loops * self.folds
        if (
            c.window != self.window
            or c.layers <= 0
            or c.layers % self.folds
            or min(c.dim, c.vocab, c.heads, c.kv_heads, c.head_dim, c.intermediate) <= 0
            or c.heads % c.kv_heads
            or c.head_dim % 2
        ):
            raise ValueError(
                f"this PLT needs a {self.window}-cycle window, {self.folds} equal layer slices and valid GQA dimensions"
            )
        self.depth = c.layers // self.folds
        self.checkpoint_layers = checkpoint_layers
        self.gamma = 0.5 / (self.outer_loops * math.sqrt(c.layers))
        # Legacy PLT-4 state_dict names and residual scaling enable strict weight loading.
        with torch.device("meta"):
            self.embedding = nn.Embedding(c.vocab, c.dim)
            self.stack = nn.ModuleList(Layer(c, self.gamma) for _ in range(c.layers))
            self.final_norm = nn.RMSNorm(c.dim, eps=c.eps)
        enable(self)
        if torch.device(device).type == "meta":
            return
        self.to_empty(device=device)
        rng = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.endswith(("gate_weight", "gate_bias")):
                    p.zero_()
                elif p.ndim == 1:
                    p.fill_(1.0)
                else:
                    p.normal_(0, 0.02, generator=rng)
                    if name.endswith((".o.weight", ".down.weight")):
                        p.div_(math.sqrt(2 * c.layers))

    def layer(self, stage, offset):
        return self.stack[(stage % self.folds) * self.depth + offset]

    def _training_layer(
        self, layer, x, pos, valid, earlier, global_mask, local_mask, fast
    ):
        q, kv, gate = layer.project(x, pos)
        kv = tuple(y.masked_fill(~valid[:, None, :, None], 0) for y in kv)
        out = attend(q, [earlier[0] if earlier else kv], global_mask, fast)
        if earlier:
            local = attend(q, [*earlier[1:], kv], local_mask, fast, self.c.window)
            out = (1 - gate) * out + gate * local
        return finish(layer, x, out).masked_fill(~valid[..., None], 0), kv

    def hidden(self, tokens, docs=None, *, valid=None, return_details=False):
        docs = torch.zeros_like(tokens) if docs is None else docs
        ids, segments, valid = layout(tokens, docs, valid)
        pos = positions(segments)
        e = self.embedding(ids).masked_fill(~valid[..., None], 0)
        fast = (
            tokens.is_cuda
            and torch.is_autocast_enabled("cuda")
            and bool(valid.all())
            and bool((segments == segments[:, :1]).all())
        )
        global_mask = local_mask = None
        if not fast:
            clock = torch.arange(ids.size(1), device=ids.device)
            age = clock[:, None] - clock
            global_mask = (
                (age >= 0)
                & (segments[:, :, None] == segments[:, None, :])
                & valid[:, :, None]
                & valid[:, None, :]
            )[:, None]
            local_mask = global_mask & (age < self.c.window)
        banks = {}
        inputs, outputs = [], []
        for stage in range(self.stages):
            x = e if stage == 0 else e + shift(x, segments)
            if return_details:
                inputs.append(x)
            for ell in range(self.depth):
                layer = self.layer(stage, ell)
                source_stages = (
                    range(stage) if self.cross_stage_local else range(min(stage, 1))
                )
                earlier = tuple(banks[j, ell] for j in source_stages)
                args = (layer, x, pos, valid, earlier, global_mask, local_mask, fast)
                if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                    x, kv = checkpoint(self._training_layer, *args, use_reentrant=False)
                else:
                    x, kv = self._training_layer(*args)
                banks[stage, ell] = kv
            if return_details:
                outputs.append(x)
        if return_details:
            return GraphOutput(x, tuple(outputs), tuple(inputs), banks)
        return x

    def _cycle(self, token, docs, valid, cache):
        # All 16 inputs use one immutable snapshot before any stage publishes.
        e = self.embedding(token).masked_fill(~valid[:, None, None], 0)
        time = cache.time
        if cache.previous is None:
            current = e.unsqueeze(0).expand(self.stages, -1, -1, -1)
            starts = torch.zeros_like(docs)
        else:
            same = valid & cache.last_valid & (docs == cache.last_docs)
            starts = torch.where(same, cache.starts, time)
            incoming = e + cache.previous * same[None, :, None, None]
            current = torch.cat((e.unsqueeze(0), incoming), 0)
        pos = (time - starts)[:, None].repeat(self.outer_loops, 1)

        # In one cycle all queries have the same real-token prefix. On the local
        # axis the lower triangle gates producer stages, including the diagonal.
        device = token.device
        slots = torch.arange(self.c.window, device=device)
        slot_times = time - (time - slots).remainder(self.c.window)
        global_mask = (torch.arange(time + 1, device=device) >= starts[:, None])[
            :, None, None
        ]
        local_valid = slot_times >= starts[:, None]
        stages = torch.arange(1, self.stages, device=device)
        if self.cross_stage_local:
            producer_visible = stages[:, None] >= stages[None, :]
        else:
            producer_visible = stages[:, None] == stages[None, :]
        local_mask = producer_visible[None, :, :, None] & local_valid[:, None, None, :]
        local_mask = local_mask.flatten(-2)[:, None]

        for ell in range(self.depth):
            # The four outer passes of each physical slice share one projection
            # batch. Publish every stage's KV before reading the current frontier.
            queries, keys, values, gates = [], [], [], []
            for fold in range(self.folds):
                layer = self.layer(fold, ell)
                q, (k, v), g = layer.project(
                    current[fold :: self.folds].flatten(0, 1), pos
                )
                queries.append(q)
                keys.append(k)
                values.append(v)
                gates.append(g)

            def in_stage_order(parts):
                # [outer * B, fold, ...] -> [outer * fold, B, ...]
                return (
                    torch.stack(parts, 1)
                    .unflatten(0, (self.outer_loops, -1))
                    .transpose(1, 2)
                    .flatten(0, 1)
                )

            q, k, v, gate = map(in_stage_order, (queries, keys, values, gates))
            masked = ~valid[None, :, None, None, None]
            k, v = k.masked_fill(masked, 0), v.masked_fill(masked, 0)
            (kg, vg), (kl, vl) = cache.publish(ell, k, v, self.c.window)
            # [stage,B,H,1,D] -> [B,H,stage,D], sharing the actual global KV.
            qq = q.squeeze(3).permute(1, 2, 0, 3).to(v.dtype)
            glob = F.scaled_dot_product_attention(
                qq, kg.to(v.dtype), vg, attn_mask=global_mask, enable_gqa=True
            )
            loc = F.scaled_dot_product_attention(
                qq[:, :, 1:], kl.to(v.dtype), vl, attn_mask=local_mask, enable_gqa=True
            )
            blend = gate[1:].squeeze(3).permute(1, 2, 0, 3)
            out = torch.cat(
                (glob[:, :, :1], (1 - blend) * glob[:, :, 1:] + blend * loc), 2
            )
            out = out.permute(2, 0, 1, 3).unsqueeze(3)
            groups = []
            for fold in range(self.folds):
                groups.append(
                    finish(
                        self.layer(fold, ell),
                        current[fold :: self.folds].flatten(0, 1),
                        out[fold :: self.folds].flatten(0, 1),
                    )
                )
            current = in_stage_order(groups).masked_fill(~valid[None, :, None, None], 0)
        cache.previous = current[:-1]
        cache.last_docs, cache.last_valid, cache.starts = (
            docs.clone(),
            valid.clone(),
            starts,
        )
        cache.time += 1
        return current[-1]

    @torch.no_grad()
    def decode(self, tokens, docs=None, *, valid=None, cache=None):
        """Consume one or more real tokens and keep bounded, per-stage local rings."""
        docs = torch.zeros_like(tokens) if docs is None else docs
        ids, _, valid = layout(tokens, docs, valid)
        if not ids.size(1) or not ids.size(0):
            raise ValueError("decode requires a nonempty batch and prefix")
        cache = FoldCache() if cache is None else cache
        if cache.previous is not None and cache.previous.size(1) != ids.size(0):
            raise ValueError("decode batch size differs from the cached batch")
        states = [
            self._cycle(ids[:, t : t + 1], docs[:, t], valid[:, t], cache)
            for t in range(ids.size(1))
        ]
        return FoldDecodeOutput(self.head(torch.cat(states, 1)), cache)

    @torch.no_grad()
    def prefill(self, tokens, docs=None, *, valid=None, cache=None):
        """Build the incremental state from a sequence-parallel prefix."""
        if cache is not None:
            return self.decode(tokens, docs, valid=valid, cache=cache)
        docs = torch.zeros_like(tokens) if docs is None else docs
        _, segments, valid = layout(tokens, docs, valid)
        if not tokens.size(1) or not tokens.size(0):
            raise ValueError("prefill requires a nonempty batch and prefix")
        graph = self.hidden(tokens, docs, valid=valid, return_details=True)
        length = tokens.size(1)
        cache = FoldCache(
            time=length,
            last_docs=docs[:, -1].clone(),
            last_valid=valid[:, -1].clone(),
            starts=length - 1 - positions(segments)[:, -1],
            previous=torch.stack([h[:, -1:] for h in graph.loops[:-1]]),
        )
        span = min(length, self.c.window)
        slots = (
            torch.arange(length - span, length, device=tokens.device) % self.c.window
        )
        for ell in range(self.depth):
            cache.global_banks[ell] = graph.kv[0, ell]
            ring = []
            for side in (0, 1):
                src = torch.stack(
                    [
                        graph.kv[r, ell][side][:, :, -span:]
                        for r in range(1, self.stages)
                    ],
                    2,
                )
                buf = src.new_zeros((*src.shape[:3], self.c.window, src.size(-1)))
                buf.index_copy_(3, slots, src)
                ring.append(buf)
            cache.local_banks[ell] = tuple(ring)
        return FoldDecodeOutput(self.head(graph.hidden), cache)


class FoldedPLT4(_LoopedPipeline):
    """Four slices per loop, with directly indexed frontier KV across all 16 stages."""

    folds = 4
    window = 32
    cross_stage_local = True


class PLT4(_LoopedPipeline):
    """Original plain PLT-4. Each pass uses the whole shared stack and its own window."""

    folds = 1
    window = 64
    cross_stage_local = False
