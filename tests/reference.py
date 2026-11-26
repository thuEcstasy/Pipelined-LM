"""Differentiable scalar-token oracle independent of all source/cache adapters.

Reuses only the unchanged backbone Layer equations. It never imports aggregation,
sequence, engine, serial_decode, trajectory, or their indexing helpers.
"""

import math
import torch


def mix(model, candidates, receiver):
    """Candidates are explicit (zero-based loop index, full-stack output) pairs."""
    if model.cross_loop.predecessor_only:
        candidates = candidates[-1:]
    m = model.cross_loop
    vals = [
        (
            index,
            (
                h.float() / (h.float().square().mean(-1, keepdim=True) + m.eps).sqrt()
                if model.source_rmsnorm
                else h.float()
            ),
        )
        for index, h in candidates
    ]
    if not vals:
        raise ValueError("Caller must handle absent historical sources")
    q, bias = m.query[receiver - 2].float(), m.bias[receiver - 2].float()
    scores = torch.stack(
        [(q * v).sum(-1) / math.sqrt(m.dim) + bias[index] for index, v in vals], -1
    )
    weights = scores.softmax(-1)
    return m.alpha[receiver - 2].float() * sum(
        weights[..., i : i + 1] * v for i, (_, v) in enumerate(vals)
    )


def depth_input(h, reader, site, entry, completed_deltas, partial, past):
    """Enumerate all valid sources separately for this token and reader."""

    def norm(v):
        x = v.float()
        return x / (x.square().mean(-1, keepdim=True) + reader.eps).sqrt()

    def attend(values, query, use_raw=False):
        keys = [norm(v) for v in values]
        values = values if use_raw else keys
        if len(values) == 1:
            return values[0]
        scale = math.sqrt(h.shape[-1])
        logits = torch.stack([(v * query.float()).sum(-1) / scale for v in keys], -1)
        weights = logits.softmax(-1)
        return sum(value * weights[..., j : j + 1] for j, value in enumerate(values))

    sources = [entry, *completed_deltas]
    if partial is not None:
        sources.append(partial)
    local = attend(sources, reader.local_query[site], use_raw=True)

    def gate(channel):
        w = reader.write[site, channel].float()
        return (
            (h.float() * w).sum(-1, keepdim=True) / math.sqrt(h.shape[-1])
            + reader.bias[site, channel].float()
        ).tanh()

    result = h.float() + gate(0) * local
    if past:
        q = reader.pre_query[site] if reader.num_pre > 1 else None
        result = result + gate(1) * attend(past, q)
    return result.to(h.dtype)


def run(model, tokens, docs, valid=None):
    if model.gswa2d_mode in ("full1d", "full2d"):
        from . import full_reference as kv2d_reference

        return kv2d_reference.run(model, tokens, docs, valid)
    assert model.gswa
    if valid is None:
        valid = torch.ones_like(tokens, dtype=torch.bool)
    B, T = tokens.shape
    R = model.H
    c = model.c
    # Deliberately derive the partition from flat Qwen indices here, without
    # importing production structure hooks, aggregation, or cache adapters.
    cfg = getattr(model, "cfg", None)
    untied = cfg is not None and cfg.execution_core == "plt"
    if untied:
        part = cfg.partition
        stacks = [
            tuple(
                model.stack[
                    part.front + r * part.depth : part.front + (r + 1) * part.depth
                ]
            )
            for r in range(R)
        ]
        front = tuple(model.stack[: part.front])
        tail = tuple(model.stack[-part.tail :])
        use_gswa = cfg.branch_attention == "gswa"
    else:
        stacks = [model.stack] * R
        front = tail = ()
        use_gswa = True
    depth = len(stacks[0])
    all_logits, all_hidden, all_inputs, all_kv = [], [], [], []
    outer_archives = []
    source_map = []
    for b in range(B):
        history = [[[] for _ in range(depth)] for _ in range(R)]
        archive = [[[] for _ in range(depth)] for _ in range(R)]
        outside = {"prelude": front, "coda": tail}
        oh = {k: [[] for _ in v] for k, v in outside.items()}
        oa = {k: [[] for _ in v] for k, v in outside.items()}
        sublayers_by_cycle = {}
        states_by_cycle = {}
        last_document = None
        position = 0
        logits, hidden, inputs = [], [], []
        for t in range(T):
            if not bool(valid[b, t]):
                z = model.embedding.weight.new_zeros(1, 1, R, c.dim)
                hidden.append(z)
                inputs.append(z)
                logits.append(model.embedding.weight.new_zeros(1, 1, c.vocab))
                for r in range(R):
                    for i in range(depth):
                        kv_zero = model.embedding.weight.new_zeros(
                            1, c.kv_heads, 1, c.head_dim
                        )
                        archive[r][i].append((kv_zero, kv_zero))
                for banks in oa.values():
                    for bank in banks:
                        zero = model.embedding.weight.new_zeros(
                            1, c.kv_heads, 1, c.head_dim
                        )
                        bank.append((zero, zero))
                continue
            doc = int(docs[b, t])
            if doc != last_document:
                history = [[[] for _ in range(depth)] for _ in range(R)]
                oh = {k: [[] for _ in v] for k, v in outside.items()}
                states_by_cycle = {}
                sublayers_by_cycle = {}
                position = 0
            e = model.embedding(tokens[b : b + 1, t : t + 1])
            p = torch.tensor([[position]], device=e.device)

            def run_outer(x, name):
                for i, layer in enumerate(outside[name]):
                    q, kv, gate = layer.project(x, p)
                    oh[name][i].append(kv)
                    oa[name][i].append(kv)
                    joined = tuple(
                        torch.cat([v[k] for v in oh[name][i]], 2) for k in (0, 1)
                    )
                    mask = torch.ones(
                        1, 1, joined[0].shape[2], device=x.device, dtype=torch.bool
                    )
                    x = layer.finish(x, q, gate, joined, mask)
                return x

            e = run_outer(e, "prelude")
            completed, prepared = [], []
            saved = {}

            def sublayer_input(h, receiver, depth_id, kind, layer):
                if not model.aligned_foresight or receiver == 1 or position == 0:
                    return h
                previous = sublayers_by_cycle[position - 1]
                raw = [previous[j, depth_id, kind].float() for j in range(1, receiver)]
                values = [
                    u / (u.square().mean(-1, keepdim=True) + model.c.eps).sqrt()
                    for u in raw
                ]
                reader = getattr(layer, "subblock_" + kind)
                if receiver > 2:
                    scores = torch.stack(
                        [
                            (v * reader.query).sum(-1) / math.sqrt(c.dim)
                            + reader.bias[j]
                            for j, v in enumerate(values)
                        ],
                        -1,
                    )
                    weights = scores.softmax(-1)
                    msg = sum(v * weights[..., j : j + 1] for j, v in enumerate(values))
                else:
                    msg = values[0]
                return (h.float() + reader.gain * msg).to(h.dtype)

            for r in range(1, R + 1):
                pairs, indices = [], []
                if r == 1:
                    h = e
                elif model.cross_loop_aggregation == "legacy":
                    if model.cross_loop_source == "same_token":
                        h = e + completed[-1]
                        indices = [(position, r - 1)]
                    else:
                        old = states_by_cycle.get(position - 1)
                        h = e if old is None else e + old[r - 2]
                        if old is not None:
                            indices = [(position - 1, r - 1)]
                else:
                    for j in range(1, r):
                        if (
                            getattr(model, "cross_loop_source", "previous_token")
                            == "same_token"
                        ):
                            pairs.append((j - 1, completed[j - 1]))
                            indices.append((position, j))
                        else:
                            offset = getattr(model, "source_trajectory_offset", None)
                            cycle = (
                                position - 1
                                if offset is None
                                else position - r + j - offset
                            )
                            old = states_by_cycle.get(cycle)
                            if old is not None:
                                pairs.append((j - 1, old[j - 1]))
                                indices.append((cycle, j))
                    h = (e.float() + mix(model, pairs, r)).to(e.dtype) if pairs else e
                prepared.append(h)
                source_map.append(
                    dict(batch=b, token=t, cycle=position, receiver=r, sources=indices)
                )
                p = torch.tensor([[position]], device=e.device)
                if model.subblock_residuals:
                    reader = model.depth_reads[r - 1]
                    entry = h
                    start = h
                    deltas = []
                    # Enumerate earlier segments of this same trajectory.
                    birth = position - r + 1
                    past = [
                        states_by_cycle[birth + j][j]
                        for j in range(r - 2)
                        if birth + j in states_by_cycle
                    ]

                    def read(site, residual):
                        nonlocal start
                        if site % reader.block == 0:
                            if site:
                                deltas.append(residual.float() - start.float())
                            start = residual
                            partial = None
                        else:
                            partial = residual.float() - start.float()
                        return depth_input(
                            residual, reader, site, entry, deltas, partial, past
                        )

                for i, layer in enumerate(stacks[r - 1]):
                    if model.subblock_residuals:
                        h = read(2 * i, h)
                    h = sublayer_input(h, r, i, "attention", layer)
                    q, kv, gate = layer.project(h, p)
                    history[r - 1][i].append(kv)
                    if r > 1 and use_gswa:
                        history[r - 1][i] = history[r - 1][i][-c.window :]
                    archive[r - 1][i].append(kv)

                    def evaluate(entries):
                        if not entries:
                            return q * 0
                        kk = torch.cat([v[0] for v in entries], 2).to(q.dtype)
                        vv = torch.cat([v[1] for v in entries], 2)
                        return torch.nn.functional.scaled_dot_product_attention(
                            q.to(vv.dtype), kk.to(vv.dtype), vv, enable_gqa=True
                        )

                    prefix = history[0][i]
                    if model.gswa2d_mode not in ("gswa1d", "gswa2d_direct"):
                        prefix = prefix[: max(0, len(prefix) - (r - 1))]
                    out = evaluate(prefix)
                    if r > 1:
                        local = []
                        for producer in (
                            [r - 1] if model.gswa2d_mode == "gswa1d" else range(1, r)
                        ):
                            entries = history[producer][i]
                            lag = (
                                0
                                if model.gswa2d_mode == "gswa2d_direct"
                                else r - 1 - producer
                            )
                            local.extend(entries[: max(0, len(entries) - lag)])
                        out = gate * evaluate(local) + (1 - gate) * out
                        if model.gswa2d_mode == "gswa2d_frontier":
                            small = []
                            for producer in range(r - 1):
                                small.extend(
                                    history[producer][i][-(r - 1 - producer) :]
                                )
                            alpha = (
                                (
                                    out.float()
                                    * layer.frontier_weight[None, :, None, :]
                                ).sum(-1, keepdim=True)
                                / math.sqrt(c.head_dim)
                                + layer.frontier_bias[None, :, None, None]
                            ).tanh()
                            out = out + alpha * evaluate(small)
                    from pipelined_lm.core.precision import swiglu

                    h = h + layer.gamma * layer.o(out.transpose(1, 2).flatten(-2))
                    saved[r, i, "attention"] = h
                    if model.subblock_residuals:
                        h = read(2 * i + 1, h)
                    h = sublayer_input(h, r, i, "ffn", layer)
                    z = layer.n2(h)
                    h = h + layer.gamma * layer.ffn_output_norm(
                        layer.down(swiglu(layer, layer.mgate(z), layer.up(z)))
                    )
                    saved[r, i, "ffn"] = h
                completed.append(h)
            sublayers_by_cycle[position] = saved
            states_by_cycle[position] = completed
            last_document = doc
            position += 1
            hidden.append(torch.stack(completed, 2))
            inputs.append(torch.stack(prepared, 2))
            logits.append(model.head(run_outer(completed[-1], "coda")))
        all_logits.append(torch.cat(logits, 1))
        all_hidden.append(torch.cat(hidden, 1))
        all_inputs.append(torch.cat(inputs, 1))
        all_kv.append(
            [
                [
                    tuple(torch.cat([v[k] for v in lane], 2) for k in (0, 1))
                    for lane in layers
                ]
                for layers in archive
            ]
        )
        outer_archives.append(
            {
                name: [
                    tuple(torch.cat([v[k] for v in bank], 2) for k in (0, 1))
                    for bank in banks
                ]
                for name, banks in oa.items()
                if banks
            }
        )
    kv = {
        ("G" if r == 0 else f"L{r + 1}"): [
            tuple(torch.cat([v[r][i][k] for v in all_kv], 0) for k in (0, 1))
            for i in range(depth)
        ]
        for r in range(R)
    }
    for name in outer_archives[0]:
        kv[name] = [
            tuple(torch.cat([v[name][i][k] for v in outer_archives], 0) for k in (0, 1))
            for i in range(len(outer_archives[0][name]))
        ]
    return dict(
        logits=torch.cat(all_logits),
        hidden=torch.cat(all_hidden),
        inputs=torch.cat(all_inputs),
        kv=kv,
        sources=source_map,
    )
