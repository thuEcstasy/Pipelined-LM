"""CUDA-graph wavefront executor, preserving the complete Foresight trajectory.
Outer layers use existing FlashKV. Middle producer K/V is either full or native
structural latent. All producers publish before any receiver consumes the
current same-offset frontier. No cache history is reconstructed.
"""

import torch
from .fast_decode import FastDecoder, prepare_for_train_numerics
from .full_kv import finish
from .kernels.shared_latent_kernel import run, workspace


class WavefrontDecoder(FastDecoder):
    def __init__(self, model, batch, max_len, *, splits=None):
        assert not model.gswa and model.kv2d_mode in ("private", "wavefront_same_layer")
        self.latent = hasattr(model, "shared_basis")
        self.private = model.kv2d_mode == "private"
        head = prepare_for_train_numerics(model)
        model.inference_head_weight = head
        if self.latent:
            # Same BF16 matmul operands as autocast; do not round norm/mean/residual.
            for ell in range(model.cfg.partition.depth):
                for name in ("K", "V"):
                    p = getattr(model.shared_basis, f"{name}_{ell}")
                    p.data = p.data.to(torch.bfloat16)
        super().__init__(
            model, batch, max_len, numerics="train", head_weight=head, attn="flash"
        )
        keep = set(
            model.cfg.partition.front_indices() + model.cfg.partition.tail_indices()
        )
        self.kv = {i: kv for i, kv in self.kv.items() if i in keep}
        self.middle = []
        self.work = []
        self.attn_out = []
        self.splits = splits or (16 if batch == 1 else (4 if self.latent else 1))
        HK = 1 if self.latent else model.c.kv_heads
        D = model.shared_basis.K_0.shape[-1] if self.latent else model.c.head_dim
        for _ in range(model.cfg.partition.depth):
            k = torch.zeros(
                (batch, model.H, HK, max_len, D),
                device=self.device,
                dtype=torch.bfloat16,
            )
            v = torch.zeros_like(k)
            q = torch.empty(
                (batch, model.H, model.c.heads, D),
                device=self.device,
                dtype=torch.bfloat16,
            )
            self.middle.append((k, v))
            self.work.append(workspace(q, k, self.splits))
            self.attn_out.append(q)

    def step_branches(self, front_out):
        current = list(self.branch_inputs(front_out))
        pos = self.t.expand(self.B, 1)
        lengths = (self.t + 1).to(torch.int32).expand(self.B).clone()
        for ell in range(self.m.cfg.partition.depth):
            queries = []
            keys = []
            values = []
            for r in range(self.m.H):
                q, kv, _ = self.m.layers_for_loop(r)[ell].project(current[r], pos)
                if self.latent:
                    q = self.m.shared_basis.query(ell, q)
                    kv = self.m.shared_basis.encode(ell, kv)
                queries.append(q[:, :, 0].to(torch.bfloat16))
                keys.append(kv[0].to(torch.bfloat16))
                values.append(kv[1].to(torch.bfloat16))
            q = torch.stack(queries, 1).contiguous()
            k, v = self.middle[ell]
            k.index_copy_(3, self.t.view(1), torch.stack(keys, 1))
            v.index_copy_(3, self.t.view(1), torch.stack(values, 1))
            out = run(
                q,
                k,
                v,
                splits=self.splits,
                private=self.private,
                lengths=lengths,
                out=self.attn_out[ell],
                work=self.work[ell],
            )
            updated = []
            for r in range(self.m.H):
                value = out[:, r, :, None, :]
                if self.latent:
                    value = self.m.shared_basis.output(ell, value)
                updated.append(
                    finish(self.m.layers_for_loop(r)[ell], current[r], value)
                )
            current = updated
        return dict(enumerate(current))

    def _caches(self):
        outer = super()._caches()
        return outer + [z for kv in self.middle for z in kv]

    def cache_accounting(self):
        outer = sum(
            z.numel() * z.element_size() for c in self.kv.values() for z in (c.k, c.v)
        )
        middle = sum(z.numel() * z.element_size() for kv in self.middle for z in kv)
        workspace_bytes = sum(
            z.numel() * z.element_size() for pair in self.work for z in pair
        )
        return dict(
            outer_cache_bytes=outer,
            middle_cache_bytes=middle,
            KV_bytes=outer + middle,
            partial_output_workspace_bytes=workspace_bytes,
            previous_hidden_bytes=self.prev.numel() * self.prev.element_size(),
            state_dtype=str(self.prev.dtype),
            cache_dtype="BF16",
            history_reconstruction=False,
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
        for lane, indices in [
            ("prelude", self.cfg.partition.front_indices()),
            ("coda", self.cfg.partition.tail_indices()),
        ]:
            for ell, i in enumerate(indices):
                cache = self.kv[i]
                for dest, src in zip((cache.k, cache.v), details.kv[lane][ell]):
                    dest.view(
                        self.B, cache.pages * 128, self.c.kv_heads, self.c.head_dim
                    )[:, :n].copy_(src.transpose(1, 2))
        for ell, (k, v) in enumerate(self.middle):
            for r in range(self.m.H):
                for dest, src in zip(
                    (k[:, r, :, :n], v[:, r, :, :n]),
                    details.kv["G" if r == 0 else f"L{r + 1}"][ell],
                ):
                    dest.copy_(src)
        for r in range(self.m.H):
            self.prev[r].copy_(details.loops[r][:, -1:])
        self.t.fill_(n)
        self.token.copy_(logits[:, -1:].argmax(-1))
        return logits
