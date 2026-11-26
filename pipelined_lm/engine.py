"""Optimized 1D/2D G-SWA: shared global FA4, fused local reader, CUDA Graph."""

from . import gswa as G
from .kernels import local_kernel
from . import subblock as S
from . import depth_read as D
from .fast_decode import FastDecoder, prepare_for_train_numerics

torch = G.torch


class GSWADecoder(FastDecoder):
    def __init__(self, m, batch, capacity):
        assert m.gswa and m.H == 4
        head = prepare_for_train_numerics(m)
        m.inference_head_weight = head
        super().__init__(
            m, batch, capacity, numerics="train", head_weight=head, attn="flash"
        )
        self.twod = m.gswa2d_mode != "gswa1d"
        self.frontier = m.gswa2d_mode == "gswa2d_frontier"
        self.direct = m.gswa2d_mode == "gswa2d_direct"
        if self.m.aligned_foresight:
            self.older = D.allocate_older(self.prev) if m.subblock_residuals else []
            self.aligned = torch.zeros(
                (m.H - 1, m.cfg.partition.depth, 2, batch, 1, m.c.dim),
                device=self.device,
                dtype=torch.float32,
            )
        self.query_cu = (
            torch.arange(batch + 1, device=self.device, dtype=torch.int32) * 4
        )
        # Drop padded per-layer local caches; one compact set of 3 producer rings.
        self.kv = {i: kv for i, kv in self.kv.items() if i not in m.mixing_layers}
        self.local = []
        self.out_local = []
        for _ in range(m.cfg.partition.depth):
            k = torch.zeros(
                (batch, 3, m.c.kv_heads, m.c.window, m.c.head_dim),
                device=self.device,
                dtype=torch.bfloat16,
            )
            self.local.append((k, torch.zeros_like(k)))
            self.out_local.append(
                torch.empty(
                    (batch, 3, m.c.heads, m.c.head_dim),
                    device=self.device,
                    dtype=torch.bfloat16,
                )
            )

    def step_branches(self, e):
        current = list(self.branch_inputs(e))
        pos = self.t.expand(self.B, 1)
        c = self.c
        groups = c.heads // c.kv_heads
        if self.m.subblock_residuals:
            depth = [
                D.Stage(
                    self.m.depth_reads[r],
                    x,
                    D.cached_endpoints(self.older, pos, r, c.eps),
                )
                for r, x in enumerate(current)
            ]
        for ell in range(self.m.cfg.partition.depth):
            if self.m.subblock_residuals:
                current = [d.read(2 * ell, x) for d, x in zip(depth, current)]
            if self.m.aligned_foresight:
                current = [
                    S.receive(
                        self.m.layers_for_loop(r)[ell],
                        "attention",
                        x,
                        list(self.aligned[:r, ell, 0].unbind(0)),
                    )
                    for r, x in enumerate(current)
                ]
            qs = []
            ks = []
            vs = []
            gates = []
            for r in range(4):
                q, kv, gate = self.m.layers_for_loop(r)[ell].project(current[r], pos)
                qs.append(q[:, :, 0].to(torch.bfloat16))
                ks.append(kv[0].to(torch.bfloat16))
                vs.append(kv[1].to(torch.bfloat16))
                gates.append(gate)
            i = self.cfg.partition.branch_indices(0)[ell]
            global_cache = self.kv[i]
            global_cache.write(ks[0][:, :, 0], vs[0][:, :, 0], self.t)
            lk, lv = self.local[ell]
            slot = (self.t % c.window).view(1)
            lk.index_copy_(3, slot, torch.stack(ks[1:], 1))
            lv.index_copy_(3, slot, torch.stack(vs[1:], 1))
            q = torch.stack(qs, 1).contiguous()
            if self.twod and not self.direct:
                # Reverse receiver order makes FA4's bottom-right causal mask
                # exactly u<=c-(r-j), j=0, for the four history-global queries.
                qg = q.flip(1).reshape(self.B * 4, c.heads, c.head_dim).contiguous()
                og = (
                    global_cache.flash(
                        qg,
                        global_cache.k,
                        global_cache.v,
                        max_seqlen_q=4,
                        cu_seqlens_q=self.query_cu,
                        max_seqlen_k=global_cache.capacity,
                        seqused_k=global_cache.length(self.t),
                        causal=True,
                        block_table=global_cache.table,
                        fa_version=4,
                    )
                    .reshape(self.B, 4, c.heads, c.head_dim)
                    .flip(1)
                )
            else:
                qg = (
                    q.reshape(self.B, 4, c.kv_heads, groups, c.head_dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(self.B, 4 * c.heads, c.head_dim)
                    .contiguous()
                )
                og = (
                    global_cache.attend(qg, self.t)
                    .reshape(self.B, c.kv_heads, 4, groups, c.head_dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(self.B, 4, c.heads, c.head_dim)
                )
            ol = local_kernel.run(
                q[:, 1:].contiguous(),
                lk,
                lv,
                self.t,
                self.twod,
                self.out_local[ell],
                direct=self.direct,
            )
            if self.frontier:
                from .kernels import frontier_kernel

                of = frontier_kernel.run(
                    q[:, 1:].contiguous(),
                    lk,
                    lv,
                    global_cache.k,
                    global_cache.v,
                    self.t,
                )
            updated = []
            published = []
            for r in range(4):
                out = og[:, r, :, None, :]
                if r:
                    out = gates[r] * ol[:, r - 1, :, None, :] + (1 - gates[r]) * out
                    if self.frontier:
                        out = G.add_frontier(
                            self.m.layers_for_loop(r)[ell],
                            out,
                            of[:, r - 1, :, None, :],
                        )
                layer = self.m.layers_for_loop(r)[ell]
                if self.m.aligned_foresight:
                    u = S.after_attention(layer, current[r], out)
                    v = S.receive(
                        layer,
                        "ffn",
                        depth[r].read(2 * ell + 1, u)
                        if self.m.subblock_residuals
                        else u,
                        list(self.aligned[:r, ell, 1].unbind(0)),
                    )
                    y = S.after_ffn(layer, v)
                    updated.append(y)
                    if r < self.m.H - 1:
                        published.append(
                            torch.stack(
                                (
                                    S.normalized(u, self.c.eps),
                                    S.normalized(y, self.c.eps),
                                )
                            )
                        )
                else:
                    updated.append(G.finish(layer, current[r], out))
            if self.m.aligned_foresight:
                # All receivers at this offset have consumed the previous cycle.
                self.aligned[:, ell].copy_(torch.stack(published))
            current = updated
        if self.m.subblock_residuals:
            D.advance_older(self.older, self.prev)
        return dict(enumerate(current))

    def _caches(self):
        return (
            super()._caches()
            + [v for pair in self.local for v in pair]
            + ([self.aligned, *self.older] if self.m.aligned_foresight else [])
        )

    @torch.no_grad()
    def prefill_parallel(self, prompt, all_logits=True):
        assert prompt.shape[0] == self.B and 0 < prompt.shape[1] < self.L
        with self._ctx():
            details = self.m.hidden(prompt, return_details=True)
            logits = self.m.head(
                details.hidden if all_logits else details.hidden[:, -1:]
            )
        n = prompt.shape[1]
        W = self.c.window
        lanes = [
            ("prelude", self.cfg.partition.front_indices()),
            ("G", self.cfg.partition.branch_indices(0)),
            ("coda", self.cfg.partition.tail_indices()),
        ]
        for lane, indices in lanes:
            for ell, i in enumerate(indices):
                cache = self.kv[i]
                for dest, src in zip((cache.k, cache.v), details.kv[lane][ell]):
                    dest.view(
                        self.B, cache.pages * 128, self.c.kv_heads, self.c.head_dim
                    )[:, :n].copy_(src.transpose(1, 2))
        positions = torch.arange(max(0, n - W), n, device=self.device)
        for ell, pair in enumerate(self.local):
            for r in range(1, 4):
                for dest, src in zip(pair, details.kv[f"L{r + 1}"][ell]):
                    dest[:, r - 1].index_copy_(
                        2, positions % W, src[:, :, max(0, n - W) :]
                    )
        if self.m.aligned_foresight:
            D.prefill_older(self.older, details.loops)
            for (r, ell, kind), rows in details.subblocks.items():
                self.aligned[r, ell, kind].copy_(rows[:, -1:])
        for r in range(4):
            self.prev[r].copy_(details.loops[r][:, -1:])
        self.t.fill_(n)
        self.token.copy_(logits[:, -1:].argmax(-1))
        return logits

    def cache_accounting(self):
        outer = sum(
            x.numel() * x.element_size()
            for kv in self.kv.values()
            for x in (kv.k, kv.v)
        )
        local = sum(x.numel() * x.element_size() for pair in self.local for x in pair)
        output = sum(x.numel() * x.element_size() for x in self.out_local)
        return dict(
            KV_bytes=outer + local,
            full_cache_bytes=outer,
            local_cache_bytes=local,
            local_output_bytes=output,
            previous_hidden_bytes=self.prev.numel() * self.prev.element_size()
            + (
                self.aligned.nbytes + sum(bank.nbytes for bank in self.older)
                if self.m.aligned_foresight
                else 0
            ),
            cache_dtype="BF16",
            full_history_reconstruction=False,
        )


def Decoder(m, batch, capacity):
    if m.gswa2d_mode in ("full1d", "full2d"):
        from .full_engine import WavefrontDecoder

        return WavefrontDecoder(m, batch, capacity)
    return GSWADecoder(m, batch, capacity)
