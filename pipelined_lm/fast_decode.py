"""Static-shape decoder for the folded Parallel Transformer and its dense ablation.

Everything a decode step touches is preallocated, and the position counter
lives on the device, so one step can be captured as a CUDA graph and replayed.

  full-length caches  [B, kv_heads, max_len, head_dim], masked by position
  branch caches       [B, kv_heads, window, head_dim] ring buffers
  branch state        the previous token's output of every branch

The math is the same as Layer.project/finish, written out so attention can use
enable_gqa instead of materializing repeated K/V. check_fast_decode verifies it
against the reference decoder at every position rather than assuming it.

The optional `branches_on` argument restricts which branches this process
computes; that is the hook the multi-GPU branch-parallel decoder uses.
"""

import torch
import torch.nn.functional as F

from .core.layers import rope


def prepare_for_train_numerics(model, compute_dtype=torch.bfloat16):
    """Make an fp32 model decode with exactly the dtype flow of autocast training.

    Training keeps fp32 parameters and runs under autocast, so every linear sees
    its weight rounded to bf16 and the residual stream stays fp32. Rounding fp32
    to bf16 is deterministic, so casting the linear weights once here gives the
    same matmul operands autocast would produce on every call, without paying
    that cast again on every decode step. Norm weights and the embedding stay
    fp32, as in training. Returns the bf16 head weight for the tied LM head.
    """
    for layer in model.stack:
        for name in ("q", "k", "v", "o", "up", "mgate", "down"):
            lin = getattr(layer, name, None)
            if lin is not None:
                lin.weight.data = lin.weight.data.to(compute_dtype)
    return model.embedding.weight.detach().to(compute_dtype)


class FastDecoder:
    def __init__(
        self,
        model,
        batch,
        max_len,
        dtype=torch.bfloat16,
        device=None,
        branches_on=None,
        numerics="bf16",
        head_weight=None,
        attn="sdpa",
    ):
        """numerics='bf16'  : everything in `dtype`; fastest, drifts from training.
        numerics='train' : residual stream and norms fp32, linears bf16 under
                           autocast; the model must have been passed through
                           prepare_for_train_numerics. Matches training.
        """
        assert numerics in ("bf16", "train")
        assert attn in ("sdpa", "flash", "ref")
        self.attn = attn
        self.m = model
        self.cfg = model.cfg
        self.c = model.c
        self.B, self.L = batch, max_len
        self.numerics = numerics
        self.dtype = dtype  # K/V cache and matmul dtype
        self.rdtype = torch.float32 if numerics == "train" else dtype  # residual stream
        self.head_w = head_weight
        self.device = device or next(model.parameters()).device
        self.dense = self.cfg.layout == "dense"
        p = self.cfg.partition
        c = self.c
        kvshape = (batch, c.kv_heads, max_len, c.head_dim)
        mk = lambda shape: torch.zeros(shape, dtype=dtype, device=self.device)
        if self.dense:
            self.full_layers = list(range(c.layers))
        else:
            assert self.cfg.branch_attention in ("local_only", "global", "gswa")
            assert (
                self.cfg.front_attention == "global"
                and self.cfg.tail_attention == "global"
            )
            self.full_layers = p.front_indices() + p.tail_indices()
            self.branches_on = (
                list(range(p.branches)) if branches_on is None else list(branches_on)
            )
            W = c.window
            self.bk = {
                b: mk((batch, c.kv_heads, W, c.head_dim)) for b in range(p.branches)
            }
            self.bv = {
                b: mk((batch, c.kv_heads, W, c.head_dim)) for b in range(p.branches)
            }
            self.prev = torch.zeros(
                (p.branches, batch, 1, c.dim), dtype=self.rdtype, device=self.device
            )
        self.unified = attn in ("flash", "ref") or self.cfg.execution_core == "plt"
        if (
            model.gswa
            and self.cfg.execution_core == "plt"
            and set(self.branches_on) != set(range(p.branches))
        ):
            raise ValueError(
                "G-SWA needs loop-1 KV published at each depth; use the layer-synchronous model.decode()"
            )
        if self.unified:
            # One cache per publishing layer; every layer reads the cache of its
            # kv_source. Folded branch publishers are window rings, everything
            # else is full length. This one table covers all four cells of
            # {dense, folded} x {shared, not shared}.
            from .kv_flash import FlashKV, RefKV

            KV = FlashKV if attn == "flash" else RefKV
            # Branch caches are window rings only when the branches use SWA;
            # with global branch attention they are full length like the rest.
            self.kvg = {}
            if getattr(model, "gswa", False):
                # G-SWA: front, loop 1 and tail layers own full-length caches;
                # each layer of loops 2..R owns a window ring for its local part,
                # and, when not sharing, also a full-length cache for its global part.
                self.kv = {}
                for i in range(c.layers):
                    if i in model.mixing_layers:
                        self.kv[i] = KV(
                            batch,
                            c.kv_heads,
                            c.head_dim,
                            c.window,
                            dtype,
                            self.device,
                            ring=True,
                        )
                        if not model.cfg.shares_kv():
                            self.kvg[i] = KV(
                                batch,
                                c.kv_heads,
                                c.head_dim,
                                max_len,
                                dtype,
                                self.device,
                            )
                    else:
                        self.kv[i] = KV(
                            batch, c.kv_heads, c.head_dim, max_len, dtype, self.device
                        )
            else:
                windowed = (
                    not self.dense
                ) and self.cfg.branch_attention == "local_only"
                ring_layers = (
                    {i for b in range(p.branches) for i in p.branch_indices(b)}
                    if windowed
                    else set()
                )
                self.kv = {
                    i: KV(
                        batch,
                        c.kv_heads,
                        c.head_dim,
                        c.window if i in ring_layers else max_len,
                        dtype,
                        self.device,
                        ring=i in ring_layers,
                    )
                    for i in sorted(model.kv_publishers)
                }
            if not self.dense:
                del self.bk, self.bv
            self.fk = self.fv = {}
        else:
            assert model.cfg.shares_kv() == (model.cfg.layout == "folded"), (
                "the sdpa path only implements the default sharing of each layout"
            )
            assert self.dense or self.cfg.branch_attention == "local_only", (
                "the sdpa path only implements windowed branches"
            )
            self.fk = {i: mk(kvshape) for i in self.full_layers}
            self.fv = {i: mk(kvshape) for i in self.full_layers}
        self.t = torch.zeros((), dtype=torch.long, device=self.device)
        self.token = torch.zeros((batch, 1), dtype=torch.long, device=self.device)
        self.logits = None
        self.ar = torch.arange(max_len, device=self.device)
        self.arw = None if self.dense else torch.arange(c.window, device=self.device)

    # ------------------------------------------------------------ primitives
    def _project(self, layer, x, pos, with_kv=True):
        c = self.c
        z = layer.n1(x)
        q = layer.q(z).view(self.B, 1, c.heads, c.head_dim).transpose(1, 2)
        q = rope(layer.q_norm(q), pos, c.theta)
        if not with_kv:
            return q, None, None
        k = layer.k(z).view(self.B, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        k = rope(layer.k_norm(k), pos, c.theta)
        v = layer.v(z).view(self.B, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        # The cache holds exactly what autocast would feed SDPA in training.
        if getattr(layer, "round_rotary_output", False):
            q = q.to(v.dtype)
        return q, k.to(self.dtype), v.to(self.dtype)

    def _finish(self, layer, x, q, k, v, mask):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        g = layer.gamma
        x = x + g * layer.o(out.transpose(1, 2).reshape(self.B, 1, -1))
        z = layer.n2(x)
        from .core.precision import swiglu

        return x + g * layer.ffn_output_norm(
            layer.down(swiglu(layer, layer.mgate(z), layer.up(z)))
        )

    def _finish_flash(self, layer, x, q, kv):
        out = kv.attend(q[:, :, 0, :].to(self.dtype), self.t)  # [B, H, hd]
        g = layer.gamma
        x = x + g * layer.o(out.reshape(self.B, 1, -1))
        z = layer.n2(x)
        from .core.precision import swiglu

        return x + g * layer.ffn_output_norm(
            layer.down(swiglu(layer, layer.mgate(z), layer.up(z)))
        )

    def _project_gate(self, layer, x, pos):
        """q (with RoPE), k, v and the per-head gate, as Layer.project computes them."""
        c = self.c
        z = layer.n1(x)
        qp = layer.q_norm(
            layer.q(z).view(self.B, 1, c.heads, c.head_dim).transpose(1, 2)
        )
        gate = torch.sigmoid(
            (qp * layer.gate_weight[None, :, None]).sum(-1)
            + layer.gate_bias[None, :, None]
        ).unsqueeze(-1)  # [B,H,1,1]
        q = rope(qp, pos, c.theta)
        k = rope(
            layer.k_norm(
                layer.k(z).view(self.B, 1, c.kv_heads, c.head_dim).transpose(1, 2)
            ),
            pos,
            c.theta,
        )
        v = layer.v(z).view(self.B, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        if getattr(layer, "round_rotary_output", False):
            q = q.to(v.dtype)
        return q, k.to(self.dtype), v.to(self.dtype), gate

    def _gswa_layer(self, i, x, pos):
        """PLT's later-loop call: finish(q, global_kv, gm, own_kv, lm) at one token."""
        layer = self.m.stack[i]
        q, k, v, gate = self._project_gate(layer, x, pos)
        self.kv[i].write(k[:, :, 0, :], v[:, :, 0, :], self.t)
        if i in self.kvg:
            self.kvg[i].write(k[:, :, 0, :], v[:, :, 0, :], self.t)
            gcache = self.kvg[i]
        else:
            gcache = self.kv[self.m.global_source[i]]
        qq = q[:, :, 0, :].to(self.dtype)
        out_g = gcache.attend(qq, self.t)
        out_l = self.kv[i].attend(qq, self.t)
        g0 = gate[:, :, 0]  # [B,H,1]
        out = g0 * out_l + (1 - g0) * out_g
        gm = layer.gamma
        x = x + gm * layer.o(out.reshape(self.B, 1, -1))
        z = layer.n2(x)
        from .core.precision import swiglu

        return x + gm * layer.ffn_output_norm(
            layer.down(swiglu(layer, layer.mgate(z), layer.up(z)))
        )

    def _unified_layer(self, i, x, pos):
        layer = self.m.stack[i]
        src = self.m.kv_source[i]
        if src == i:
            q, k, v = self._project(layer, x, pos)
            self.kv[i].write(k[:, :, 0, :], v[:, :, 0, :], self.t)
        else:
            q, _, _ = self._project(layer, x, pos, with_kv=False)
        return self._finish_flash(layer, x, q, self.kv[src])

    def _full_layer(self, i, x, pos):
        if self.unified:
            return self._unified_layer(i, x, pos)
        layer = self.m.stack[i]
        q, k, v = self._project(layer, x, pos)
        self.fk[i].index_copy_(2, self.t.view(1), k)
        self.fv[i].index_copy_(2, self.t.view(1), v)
        mask = (self.ar <= self.t).view(1, 1, 1, self.L)
        return self._finish(layer, x, q, self.fk[i], self.fv[i], mask)

    def _branch(self, b, x, pos):
        if self.unified:
            for i in self.cfg.partition.branch_indices(b):
                x = (
                    self._gswa_layer(i, x, pos)
                    if i in getattr(self.m, "mixing_layers", ())
                    else self._unified_layer(i, x, pos)
                )
            return x
        p, W = self.cfg.partition, self.c.window
        slot = torch.remainder(self.t, W).view(1)
        mask = (self.arw < torch.clamp(self.t + 1, max=W)).view(1, 1, 1, W)
        for n, i in enumerate(p.branch_indices(b)):
            layer = self.m.stack[i]
            q, k, v = self._project(layer, x, pos, with_kv=(n == 0))
            if n == 0:
                self.bk[b].index_copy_(2, slot, k)
                self.bv[b].index_copy_(2, slot, v)
            x = self._finish(layer, x, q, self.bk[b], self.bv[b], mask)
        return x

    def branch_inputs(self, front_out):
        """Lag-1 Foresight inputs for every branch, from the device-side state."""
        p = self.cfg.partition
        cp = self.m.cross_part
        valid = (self.t > 0).expand(self.B, 1)
        ins = [front_out]
        for r in range(2, p.branches + 1):
            src = self.prev[: r - 1].permute(1, 2, 0, 3)  # [B,1,r-1,d]
            msg, _ = cp.message(src, r, history_valid=valid)
            ins.append((front_out.float() + msg).to(front_out.dtype))
        return ins

    # ------------------------------------------------------------------ step
    def _ctx(self):
        if self.numerics == "train":
            return torch.autocast(
                "cuda" if self.device.type == "cuda" else "cpu", dtype=self.dtype
            )
        import contextlib

        return contextlib.nullcontext()

    def step_front(self):
        pos = self.t.expand(self.B, 1)
        x = self.m.embedding(self.token).to(self.rdtype)
        if self.dense:
            for i in self.full_layers:
                x = self._full_layer(i, x, pos)
            return x
        for i in self.cfg.partition.front_indices():
            x = self._full_layer(i, x, pos)
        return x

    def step_branches(self, front_out):
        pos = self.t.expand(self.B, 1)
        if self.cfg.cross_part_source == "same_token":
            if set(self.branches_on) != set(range(self.cfg.partition.branches)):
                raise ValueError(
                    "Serial AttnRes requires every shallower loop on this executor"
                )
            outputs = {}
            for b in range(self.cfg.partition.branches):
                x = front_out
                if b:
                    src = torch.stack([outputs[j] for j in range(b)], -2)
                    msg, _ = self.m.cross_part.message(src, b + 1)
                    x = (front_out.float() + msg).to(front_out.dtype)
                outputs[b] = self._branch(b, x, pos)
            return outputs
        ins = self.branch_inputs(front_out)
        # Under G-SWA loops 2..R read loop 1's caches at this token, so loop 1
        # must have written them first; sorted order puts branch 0 first.
        return {b: self._branch(b, ins[b], pos) for b in sorted(self.branches_on)}

    def step_tail(self, last_branch_out):
        pos = self.t.expand(self.B, 1)
        x = last_branch_out
        for i in self.cfg.partition.tail_indices():
            x = self._full_layer(i, x, pos)
        return x

    def head_logits(self, x):
        if self.numerics == "train":
            return self.m.head(x)
        if self.head_w is not None:
            return F.linear(self.m.final_norm(x), self.head_w)
        return self.m.head(x)

    def finish_token(self, x):
        logits = self.head_logits(x)
        self.token.copy_(logits.argmax(-1))
        self.t.add_(1)
        return logits

    def step(self):
        """One full token step on a single device. Graph-capturable."""
        with self._ctx():
            x = self.step_front()
            if not self.dense:
                outs = self.step_branches(x)
                for b, o in outs.items():
                    self.prev[b].copy_(o)
                x = self.step_tail(outs[self.cfg.partition.branches - 1])
            self.logits = self.finish_token(x)
        return self.logits

    # --------------------------------------------------------------- driving
    @torch.no_grad()
    def prefill(self, prompt):
        """Token by token through the same step, so prefill and decode share code."""
        logits = []
        for t in range(prompt.shape[1]):
            self.token.copy_(prompt[:, t : t + 1])
            logits.append(self.step().clone())
        return torch.cat(logits, 1)

    def set_context(self, n):
        """Jump to context length n with random cache contents, for timing only."""
        if self.unified:
            for c in list(self.kv.values()) + list(self.kvg.values()):
                c.fill_random()
        else:
            for d in (self.fk, self.fv):
                [x.normal_() for x in d.values()]
            if not self.dense:
                for d in (self.bk, self.bv):
                    [x.normal_() for x in d.values()]
        if not self.dense:
            self.prev.normal_()
        self.t.fill_(n)

    @torch.no_grad()
    def capture(self, warmup=3, keep_state=True):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        snap = self._snapshot() if keep_state else None
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        if keep_state:
            self._restore(snap)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()
        if keep_state:
            self._restore(snap)
        return self.graph

    def _caches(self):
        if self.unified:
            cs = list(self.kv.values()) + list(self.kvg.values())
            return [c.k for c in cs] + [c.v for c in cs]
        return None

    def _snapshot(self):
        if self.unified:
            st = {
                "t": self.t.clone(),
                "token": self.token.clone(),
                "caches": [x.clone() for x in self._caches()],
            }
            if not self.dense:
                st["prev"] = self.prev.clone()
            return st
        st = {"t": self.t.clone(), "token": self.token.clone()}
        st["fk"] = {i: v.clone() for i, v in self.fk.items()}
        st["fv"] = {i: v.clone() for i, v in self.fv.items()}
        if not self.dense:
            st["bk"] = {i: v.clone() for i, v in self.bk.items()}
            st["bv"] = {i: v.clone() for i, v in self.bv.items()}
            st["prev"] = self.prev.clone()
        return st

    def _restore(self, st):
        if self.unified:
            self.t.copy_(st["t"])
            self.token.copy_(st["token"])
            for a, b in zip(self._caches(), st["caches"]):
                a.copy_(b)
            if not self.dense:
                self.prev.copy_(st["prev"])
            return
        self.t.copy_(st["t"])
        self.token.copy_(st["token"])
        for i in self.fk:
            self.fk[i].copy_(st["fk"][i])
            self.fv[i].copy_(st["fv"][i])
        if not self.dense:
            for i in self.bk:
                self.bk[i].copy_(st["bk"][i])
                self.bv[i].copy_(st["bv"][i])
            self.prev.copy_(st["prev"])
