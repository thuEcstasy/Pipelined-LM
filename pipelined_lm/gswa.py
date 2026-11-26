"""Training, prefill and streaming decode for global/sliding-window KV policies."""

from functools import lru_cache
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from . import depth_read as D, subblock as S
from .full_kv import finish


def local_sources(m, r):
    return list(range(1, r + 1)) if m.gswa2d_mode != "gswa1d" else [r]


@lru_cache(maxsize=64)
def local_mask(length, lags, window, device):
    from torch.nn.attention.flex_attention import create_block_mask

    banks = len(lags)
    first = lags[0]
    same = len(set(lags)) == 1
    assert same or lags == tuple(first - j for j in range(banks))

    def visible(b, h, q, k):
        delta = q - k % length
        lag = first if same else first - k // length
        return (q < length) & (k < length * banks) & (delta >= lag) & (delta < window)

    return create_block_mask(
        visible, None, None, length, length * banks, device=device, _compile=True
    )


@lru_cache(maxsize=1)
def flex():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, fullgraph=True, dynamic=False)


def attend(q, kvs, mask=None, fast=False, window=None, lags=None):
    k = torch.cat([v[0] for v in kvs], 2) if len(kvs) > 1 else kvs[0][0]
    v = torch.cat([v[1] for v in kvs], 2) if len(kvs) > 1 else kvs[0][1]
    q = q.to(v.dtype)
    k = k.to(v.dtype)
    if k.shape[2] == 0:
        return q * 0
    lags = tuple(lags or [0] * len(kvs))
    if fast and window is None:
        from .core.flash_backend import global_sdpa

        lag = lags[0]
        if lag == 0:
            return global_sdpa(q, k, v, is_causal=True, enable_gqa=True)
        if q.shape[2] <= lag:
            return q * 0
        out = global_sdpa(
            q[:, :, lag:],
            k[:, :, :-lag],
            v[:, :, :-lag],
            is_causal=True,
            enable_gqa=True,
        )
        return F.pad(out, (0, 0, lag, 0))
    if fast and len(kvs) == 1 and lags == (0,):
        from .core.flash_backend import Window

        group = q.shape[1] // k.shape[1]
        return Window.apply(
            q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1), window
        )
    if fast:
        return flex()(
            q,
            k,
            v,
            block_mask=local_mask(q.shape[2], lags, window, str(q.device)),
            enable_gqa=True,
        )
    if mask is not None:
        T = q.shape[2]
        delta = (
            torch.arange(T, device=q.device)[:, None]
            - torch.arange(T, device=q.device)[None, :]
        )
        mask = torch.cat([mask & (delta >= lag)[None, None] for lag in lags], -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)


def frontier(q, kvs, segments, valid):
    # At most R(R-1)/2 keys/query, no sequence-square intermediate.
    B, H, T, D = q.shape
    HK = kvs[0][0].shape[1]
    keys = []
    vals = []
    masks = []
    r = len(kvs)
    for j, (k, v) in enumerate(kvs):
        for age in range(r - j):
            if age >= T:
                keys.append(k * 0)
                vals.append(v * 0)
                masks.append(torch.zeros_like(valid))
                continue
            keys.append(F.pad(k[:, :, : T - age], (0, 0, age, 0)))
            vals.append(F.pad(v[:, :, : T - age], (0, 0, age, 0)))
            sm = F.pad(segments[:, : T - age], (age, 0), value=-1)
            vm = F.pad(valid[:, : T - age], (age, 0), value=False)
            masks.append(valid & vm & (segments == sm))
    k = torch.stack(keys, -2)
    v = torch.stack(vals, -2)
    mask = torch.stack(masks, -1)[:, None, None]
    with torch.autocast(q.device.type, enabled=False):
        scores = torch.einsum(
            "bhgtd,bhtnd->bhgtn", q.float().reshape(B, HK, H // HK, T, D), k.float()
        ) * (D**-0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    safe = mask.any(-1, keepdim=True)
    prob = torch.softmax(torch.where(safe, scores, 0.0), -1).masked_fill(~mask, 0.0)
    out = torch.einsum("bhgtn,bhtnd->bhgtd", prob.to(v.dtype), v)
    return out.reshape(B, H, T, D)


def add_frontier(layer, base, front):
    alpha = torch.tanh(
        (base.float() * layer.frontier_weight[None, :, None, :]).sum(-1, keepdim=True)
        * (base.shape[-1] ** -0.5)
        + layer.frontier_bias[None, :, None, None]
    )
    return base + alpha * front


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
        and bool(valid.all())
        and (single or bool((segments == segments[:, :1]).all()))
    )
    gm = lm = None
    if not fast:
        t = torch.arange(T, device=tokens.device)
        delta = t[:, None] - t[None, :]
        gm = (
            (delta >= 0)
            & (segments[:, :, None] == segments[:, None, :])
            & valid[:, :, None]
            & valid[:, None, :]
        )[:, None]
        lm = gm & (delta < m.c.window)[None, None]
    trace = {}
    middle = {}
    outputs = []
    inputs = []

    def project(layer, x):
        q, kv, gate = layer.project(x, pos)
        return q, tuple(v.masked_fill(~valid[:, None, :, None], 0.0) for v in kv), gate

    def outer(x, layers, lane):
        banks = []
        for layer in layers:
            q, kv, _ = project(layer, x)
            banks.append(kv)
            x = finish(layer, x, attend(q, [kv], gm, fast)).masked_fill(
                ~valid[:, :, None], 0.0
            )
        trace[lane] = banks
        return x

    e = outer(m.embedding(tokens), m.prelude_layers(), "prelude")
    aligned = {}
    endpoints = []
    same = F.pad(
        valid[:, 1:] & valid[:, :-1] & (segments[:, 1:] == segments[:, :-1]), (1, 0)
    )
    for r in range(m.H):
        x = e if r == 0 else m.cross_loop.training_input(e, outputs, segments, valid)
        inputs.append(x)
        lane = []
        if m.subblock_residuals:
            sources = [
                D.shift_endpoint(endpoints[j], r - j, segments, valid)
                for j in range(r - 1)
            ]
            depth = D.Stage(m.depth_reads[r], x, sources)
        for ell, layer in enumerate(m.layers_for_loop(r)):
            if m.subblock_residuals:
                x = depth.read(2 * ell, x)
            if m.aligned_foresight:
                x = S.receive(
                    layer, "attention", x, [aligned[j, ell, 0] for j in range(r)], same
                )
            q, kv, gate = project(layer, x)
            middle[r, ell] = kv
            lane.append(kv)
            out = attend(
                q,
                [middle[0, ell]],
                gm,
                fast,
                lags=[0 if m.gswa2d_mode in ("gswa1d", "gswa2d_direct") else r],
            )
            if r:
                js = local_sources(m, r)
                local = attend(
                    q,
                    [middle[j, ell] for j in js],
                    lm,
                    fast,
                    m.c.window,
                    [0 if m.gswa2d_mode == "gswa2d_direct" else r - j for j in js],
                )
                out = gate * local + (1 - gate) * out
                if m.gswa2d_mode == "gswa2d_frontier":
                    sources = [middle[j, ell] for j in range(r)]
                    if fast and m.frontier_backend == "flex":
                        from .kernels.sparse_frontier import run

                        f = run(q, sources)
                    else:
                        f = frontier(q, sources, segments, valid)
                    out = add_frontier(layer, out, f)
            if m.aligned_foresight:
                u = S.after_attention(layer, x, out)
                if r < m.H - 1:
                    aligned[r, ell, 0] = S.normalized(u, m.c.eps)
                if m.subblock_residuals:
                    u = depth.read(2 * ell + 1, u)
                u = S.receive(
                    layer, "ffn", u, [aligned[j, ell, 1] for j in range(r)], same
                )
                x = S.after_ffn(layer, u).masked_fill(~valid[..., None], 0.0)
                if r < m.H - 1:
                    aligned[r, ell, 1] = S.normalized(x, m.c.eps)
            else:
                x = finish(layer, x, out).masked_fill(~valid[:, :, None], 0.0)
        if m.subblock_residuals and r < m.H - 2:
            endpoints.append(aligned[r, m.cfg.partition.depth - 1, 1])
        outputs.append(x)
        trace["G" if r == 0 else f"L{r + 1}"] = lane
    x = outer(outputs[-1], m.coda_layers(), "coda")
    if not return_details:
        return x
    result = GraphOutput(x, tuple(outputs), tuple(inputs), trace)
    result.subblocks = aligned
    return result


def new_row(m):
    return dict(
        previous=None,
        older=None,
        subblocks={},
        doc=None,
        position=0,
        banks={},
        mode=m.gswa2d_mode,
    )


def stream(
    m, tokens, docs=None, *, valid=None, cache=None, return_trace=False, reverse=False
):
    from .core.sequence import layout

    if docs is None:
        docs = torch.zeros_like(tokens)
    safe, _, valid = layout(tokens, docs, valid)
    B, T = tokens.shape
    if cache is None:
        cache = SimpleNamespace(rows=[new_row(m) for _ in range(B)], mode=m.gswa2d_mode)
    assert len(cache.rows) == B and cache.mode == m.gswa2d_mode
    hs = []
    ls = []
    allkv = []
    allloops = []
    allinputs = []
    keys = (
        [("prelude", k) for k in range(m.cfg.partition.front)]
        + [(r, k) for r in range(m.H) for k in range(m.cfg.partition.depth)]
        + [("coda", k) for k in range(m.cfg.partition.tail)]
    )
    for b in range(B):
        state = cache.rows[b]
        outputs = []
        logits = []
        archive = {}
        loop_rows = []
        input_rows = []
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
            if state["doc"] != doc:
                state = new_row(m)
                state["doc"] = doc
                cache.rows[b] = state
            pos = torch.tensor([[state["position"]]], device=tokens.device)
            banks = state["banks"]

            def publish(key, kv):
                archive.setdefault(key, []).append(kv)
                old = banks.get(key)
                joined = (
                    kv
                    if old is None
                    else tuple(torch.cat((u, v), 2) for u, v in zip(old, kv))
                )
                banks[key] = (
                    tuple(v[:, :, -m.c.window :] for v in joined)
                    if isinstance(key[0], int) and key[0] > 0
                    else joined
                )

            def outer(x, layers, lane):
                for k, layer in enumerate(layers):
                    q, kv, _ = layer.project(x, pos)
                    publish((lane, k), kv)
                    x = finish(layer, x, attend(q, [banks[lane, k]]))
                return x

            x = outer(
                m.embedding(safe[b : b + 1, c : c + 1]), m.prelude_layers(), "prelude"
            )
            prev = state["previous"]
            if prev is None:
                prev = [torch.zeros_like(x) for _ in range(m.H)]
            current = list(
                m.cross_loop.prepared_inputs(x, prev, has_history=state["position"] > 0)
            )
            input_rows.append(torch.stack(current, 2))
            earlier = state["subblocks"]
            published = {}
            if m.subblock_residuals:
                if state["older"] is None:
                    state["older"] = D.allocate_older(prev)
                depth = [
                    D.Stage(
                        m.depth_reads[r],
                        v,
                        D.cached_endpoints(state["older"], pos, r, m.c.eps),
                    )
                    for r, v in enumerate(current)
                ]
            for ell in range(m.cfg.partition.depth):
                if m.subblock_residuals:
                    current = [d.read(2 * ell, v) for d, v in zip(depth, current)]
                if m.aligned_foresight and earlier:
                    current = [
                        S.receive(
                            m.layers_for_loop(r)[ell],
                            "attention",
                            x,
                            [earlier[j, ell, 0] for j in range(r)],
                        )
                        for r, x in enumerate(current)
                    ]
                projected = []
                for r in range(m.H):
                    q, kv, gate = m.layers_for_loop(r)[ell].project(current[r], pos)
                    projected.append((q, gate))
                    publish((r, ell), kv)
                next_states = [None] * m.H
                for r in reversed(range(m.H)) if reverse else range(m.H):
                    q, gate = projected[r]

                    def past(pair, lag):
                        return tuple(v[:, :, : max(0, v.shape[2] - lag)] for v in pair)

                    out = attend(
                        q,
                        [
                            past(
                                banks[0, ell],
                                0
                                if m.gswa2d_mode in ("gswa1d", "gswa2d_direct")
                                else r,
                            )
                        ],
                    )
                    if r:
                        out = (
                            gate
                            * attend(
                                q,
                                [
                                    past(
                                        banks[j, ell],
                                        0
                                        if m.gswa2d_mode == "gswa2d_direct"
                                        else r - j,
                                    )
                                    for j in local_sources(m, r)
                                ],
                            )
                            + (1 - gate) * out
                        )
                        if m.gswa2d_mode == "gswa2d_frontier":
                            f = [
                                tuple(v[:, :, -(r - j) :] for v in banks[j, ell])
                                for j in range(r)
                            ]
                            out = add_frontier(
                                m.layers_for_loop(r)[ell], out, attend(q, f)
                            )
                    layer = m.layers_for_loop(r)[ell]
                    if m.aligned_foresight:
                        u = S.after_attention(layer, current[r], out)
                        if r < m.H - 1:
                            published[r, ell, 0] = S.normalized(u, m.c.eps)
                        if m.subblock_residuals:
                            u = depth[r].read(2 * ell + 1, u)
                        if earlier:
                            u = S.receive(
                                layer, "ffn", u, [earlier[j, ell, 1] for j in range(r)]
                            )
                        y = S.after_ffn(layer, u)
                        if r < m.H - 1:
                            published[r, ell, 1] = S.normalized(y, m.c.eps)
                        next_states[r] = y
                    else:
                        next_states[r] = finish(layer, current[r], out)
                current = next_states
            if m.subblock_residuals:
                state["older"] = [
                    torch.cat((prev[j].unsqueeze(0), bank[:-1]), dim=0)
                    for j, bank in enumerate(state["older"])
                ]
            state["subblocks"] = published
            state["previous"] = current
            state["position"] += 1
            loop_rows.append(torch.stack(current, 2))
            out = outer(current[-1], m.coda_layers(), "coda")
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
        lane: [flat[lane, k] for k in range(count)]
        for lane, count in [
            ("prelude", m.cfg.partition.front),
            ("coda", m.cfg.partition.tail),
        ]
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
