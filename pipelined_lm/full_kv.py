"""Full raw KV, causal in token time and the declared stage/layer source order.

Training is stage-major and token-parallel. Decode is layer-major within each
cycle, with all same-offset projections published before receivers consume them.
Foresight inputs use the immutable previous-cycle snapshot. No trained checkpoint
or auxiliary projections are introduced by this module.
"""

from functools import lru_cache
from types import MethodType, SimpleNamespace
import torch
import torch.nn.functional as F

MODES = ("wavefront_same_layer", "private")


def sources(r, ell, mode):
    if mode == "private":
        return [(r, ell)]
    assert mode == "wavefront_same_layer", mode
    return [(j, ell) for j in range(r + 1)]


def logical_layer_pairs(m):
    return (
        m.cfg.partition.front
        + m.cfg.partition.tail
        + sum(
            len(sources(r, k, m.kv2d_mode))
            for r in range(m.H)
            for k in range(m.cfg.partition.depth)
        )
    )


@lru_cache(maxsize=64)
def block_mask(length, banks, device):
    from torch.nn.attention.flex_attention import create_block_mask

    def visible(b, h, q, k):
        return (q < length) & (k < length * banks) & ((k % length) <= q)

    return create_block_mask(
        visible, None, None, length, length * banks, device=device, _compile=True
    )


@lru_cache(maxsize=1)
def flex():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, fullgraph=True, dynamic=False)


def attention(q, kvs, mask=None, use_flex=False):
    kk = torch.cat([kv[0] for kv in kvs], 2)
    vv = torch.cat([kv[1] for kv in kvs], 2)
    q = q.to(vv.dtype)
    kk = kk.to(vv.dtype)
    if use_flex:
        return flex()(
            q,
            kk,
            vv,
            block_mask=block_mask(q.shape[2], len(kvs), str(q.device)),
            enable_gqa=True,
        )
    gm = None if mask is None else mask.repeat(1, 1, 1, len(kvs))
    return F.scaled_dot_product_attention(
        q, kk, vv, attn_mask=gm, dropout_p=0.0, enable_gqa=True
    )


def finish(layer, x, out):
    from .core.precision import swiglu

    u = x + layer.gamma * layer.o(out.transpose(1, 2).flatten(-2))
    z = layer.n2(u)
    return u + layer.gamma * layer.ffn_output_norm(
        layer.down(swiglu(layer, layer.mgate(z), layer.up(z)))
    )


def hidden(m, tokens, docs=None, *, valid=None, return_details=False):
    from .core.sequence import layout, GraphOutput
    from .core.layers import positions

    single = docs is None
    if docs is None:
        docs = torch.zeros_like(tokens)
    tokens, segments, valid = layout(tokens, docs, valid)
    pos = positions(segments)
    B, T = tokens.shape
    fast = (
        tokens.is_cuda
        and m.kv2d_backend == "flex"
        and bool(valid.all())
        and (single or bool((segments == segments[:, :1]).all()))
    )
    causal = None
    if not fast:
        t = torch.arange(T, device=tokens.device)
        causal = (
            (t[:, None] >= t[None, :])
            & (segments[:, :, None] == segments[:, None, :])
            & valid[:, :, None]
            & valid[:, None, :]
        )[:, None]
    x = m.embedding(tokens)
    trace = {}
    middle = {}
    outputs = []
    inputs = []

    def project(layer, x):
        q, kv, _ = layer.project(x, pos)
        kv = tuple(v.masked_fill(~valid[:, None, :, None], 0.0) for v in kv)
        return q, kv

    def outer(x, layers, lane):
        lane_kv = []
        for layer in layers:
            q, kv = project(layer, x)
            lane_kv.append(kv)
            x = finish(layer, x, attention(q, [kv], causal, fast)).masked_fill(
                ~valid[:, :, None], 0.0
            )
        trace[lane] = lane_kv
        return x

    e = outer(x, m.prelude_layers(), "prelude")
    for r in range(m.H):
        x = e if r == 0 else m.cross_loop.training_input(e, outputs, segments, valid)
        inputs.append(x)
        lane = []
        for ell, layer in enumerate(m.layers_for_loop(r)):
            q, kv = project(layer, x)
            middle[r, ell] = kv
            lane.append(kv)
            kvs = [middle[j, k] for j, k in sources(r, ell, m.kv2d_mode)]
            x = finish(layer, x, attention(q, kvs, causal, fast)).masked_fill(
                ~valid[:, :, None], 0.0
            )
        outputs.append(x)
        trace["G" if r == 0 else f"L{r + 1}"] = lane
    x = outer(outputs[-1], m.coda_layers(), "coda")
    return GraphOutput(x, tuple(outputs), tuple(inputs), trace) if return_details else x


def new_row(m):
    return dict(previous=None, doc=None, position=0, banks={}, mode=m.kv2d_mode)


def stream(
    m, tokens, docs=None, *, valid=None, cache=None, return_trace=False, reverse=False
):
    from .core.sequence import layout

    if docs is None:
        docs = torch.zeros_like(tokens)
    safe, _, valid = layout(tokens, docs, valid)
    B, T = tokens.shape
    if cache is None:
        cache = SimpleNamespace(rows=[new_row(m) for _ in range(B)], mode=m.kv2d_mode)
    assert len(cache.rows) == B and cache.mode == m.kv2d_mode
    hs = []
    ls = []
    allkv = []
    allloops = []
    allinputs = []
    for b in range(B):
        state = cache.rows[b]
        outputs = []
        logits = []
        archive = {}
        loop_rows = []
        input_rows = []
        keys = (
            [("prelude", k) for k in range(m.cfg.partition.front)]
            + [(r, k) for r in range(m.H) for k in range(m.cfg.partition.depth)]
            + [("coda", k) for k in range(m.cfg.partition.tail)]
        )
        for c in range(T):
            if not bool(valid[b, c]):
                zero = m.embedding.weight.new_zeros((1, 1, m.c.dim))
                outputs.append(zero)
                logits.append(m.embedding.weight.new_zeros((1, 1, m.c.vocab)))
                loop_rows.append(torch.stack([zero] * m.H, 2))
                input_rows.append(torch.stack([zero] * m.H, 2))
                dtype = (
                    torch.get_autocast_dtype(tokens.device.type)
                    if torch.is_autocast_enabled(tokens.device.type)
                    else m.embedding.weight.dtype
                )
                for key in keys:
                    archive.setdefault(key, []).append(
                        tuple(
                            torch.zeros(
                                (1, m.c.kv_heads, 1, m.c.head_dim),
                                device=tokens.device,
                                dtype=dtype,
                            )
                            for _ in range(2)
                        )
                    )
                continue
            doc = int(docs[b, c])
            assert state["mode"] == m.kv2d_mode
            if state["doc"] != doc:
                state = new_row(m)
                state["doc"] = doc
                cache.rows[b] = state
            pos = torch.tensor([[state["position"]]], device=tokens.device)
            x = m.embedding(safe[b : b + 1, c : c + 1])
            banks = state["banks"]

            def publish(key, kv):
                archive.setdefault(key, []).append(kv)
                old = banks.get(key)
                banks[key] = (
                    kv
                    if old is None
                    else tuple(torch.cat((u, v), 2) for u, v in zip(old, kv))
                )

            def outside(x, layers, lane):
                for k, layer in enumerate(layers):
                    q, kv, _ = layer.project(x, pos)
                    publish((lane, k), kv)
                    x = finish(layer, x, attention(q, [banks[lane, k]]))
                return x

            x = outside(x, m.prelude_layers(), "prelude")
            prev = state["previous"]
            if prev is None:
                prev = [torch.zeros_like(x) for _ in range(m.H)]
            current = list(
                m.cross_loop.prepared_inputs(x, prev, has_history=state["position"] > 0)
            )
            input_rows.append(torch.stack(current, 2))
            for ell in range(m.cfg.partition.depth):
                qs = []
                for r in range(m.H):
                    q, kv, _ = m.layers_for_loop(r)[ell].project(current[r], pos)
                    qs.append(q)
                    publish((r, ell), kv)
                next_states = [None] * m.H
                for r in reversed(range(m.H)) if reverse else range(m.H):
                    kvs = [banks[key] for key in sources(r, ell, m.kv2d_mode)]
                    next_states[r] = finish(
                        m.layers_for_loop(r)[ell], current[r], attention(qs[r], kvs)
                    )
                current = next_states
            state["previous"] = current
            state["position"] += 1
            loop_rows.append(torch.stack(current, 2))
            out = outside(current[-1], m.coda_layers(), "coda")
            outputs.append(out)
            logits.append(m.head(out))
        hs.append(torch.cat(outputs, 1))
        ls.append(torch.cat(logits, 1))
        allloops.append(torch.cat(loop_rows, 1))
        allinputs.append(torch.cat(input_rows, 1))
        allkv.append(
            {
                key: tuple(torch.cat([v[n] for v in vals], 2) for n in (0, 1))
                for key, vals in archive.items()
            }
        )
    flat = {
        key: tuple(torch.cat([v[key][n] for v in allkv]) for n in (0, 1))
        for key in allkv[0]
    }
    grouped = {
        "prelude": [flat["prelude", k] for k in range(m.cfg.partition.front)],
        "coda": [flat["coda", k] for k in range(m.cfg.partition.tail)],
    }
    for r in range(m.H):
        grouped["G" if r == 0 else f"L{r + 1}"] = [
            flat[r, k] for k in range(m.cfg.partition.depth)
        ]
    return SimpleNamespace(
        hidden=torch.cat(hs),
        logits=torch.cat(ls),
        cache=cache,
        kv=grouped if return_trace else None,
        loops=torch.cat(allloops),
        inputs=torch.cat(allinputs),
    )


def attach(m, mode, backend="flex"):
    assert mode in MODES, mode
    m.kv2d_mode = mode
    m.kv2d_backend = backend
    m.hidden = MethodType(hidden, m)
    m.prefill = MethodType(stream, m)
    m.decode = MethodType(stream, m)
    return m
