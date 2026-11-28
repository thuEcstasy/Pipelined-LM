"""Trajectory oracle for the 4 x 4 tied model and its directly indexed frontier."""

import copy
import io
import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from pipelined_lm import FoldedPLT4, tiny_backbone


def scalar_trajectory(model, ids):
    """Single document. Index all states by their final prediction, not the clock."""
    depth = len(model.stack) // 4
    banks, states, entries = {}, {}, {}
    for stage in range(16):
        for c in range(len(ids)):
            target = c + 16 - stage
            e = model.embedding(ids[c : c + 1].view(1, 1))
            if model.cross_part is not None and stage and c:
                mix = model.cross_part
                # Fix the receiving trajectory. Earlier producers at c-1
                # aim progressively further ahead than its own predecessor.
                previous = [states[j, target + stage - j - 1] for j in range(stage)]
                values = torch.stack(
                    [F.rms_norm(h.float(), (model.c.dim,), eps=mix.eps) for h in previous],
                    -2,
                )
                scores = (values @ mix.query[stage - 1]) / model.c.dim**0.5
                weights = (scores + mix.bias[stage - 1]).softmax(-1)
                x = e + mix.alpha[stage - 1] * (weights[..., None] * values).sum(-2)
            else:
                x = e + states.get((stage - 1, target), 0)
            entries[stage, target] = x
            for ell in range(depth):
                layer = model.stack[(stage * depth + ell) % len(model.stack)]
                q, kv, gate = layer.project(x, ids.new_tensor([[c]]))
                banks[stage, ell, target] = kv

                def retrieve(keys):
                    k, v = (
                        torch.cat(parts, 2)
                        for parts in zip(*(banks[key] for key in keys))
                    )
                    groups = model.c.heads // model.c.kv_heads
                    scores = q @ k.repeat_interleave(groups, 1).transpose(-2, -1)
                    return (scores / q.size(-1) ** 0.5).softmax(
                        -1
                    ) @ v.repeat_interleave(groups, 1)

                out = retrieve([(0, ell, s) for s in range(16, target + stage + 1)])
                if stage:
                    keys = []
                    for producer in range(1, stage + 1):
                        last = target + stage - producer
                        first = max(16 - producer, last - 31)
                        keys.extend((producer, ell, s) for s in range(first, last + 1))
                    out = (1 - gate) * out + gate * retrieve(keys)
                u = x + layer.gamma * layer.o(out.transpose(1, 2).flatten(-2))
                z = layer.n2(u)
                x = u + layer.gamma * layer.ffn_output_norm(
                    layer.down(F.silu(layer.mgate(z)) * layer.up(z))
                )
            states[stage, target] = x
    return states, entries, banks


class FoldedPLTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.shape = replace(
            tiny_backbone(), layers=8, window=32, dim=16, intermediate=24
        )
        self.model = FoldedPLT4(self.shape)
        self.close = lambda a, b: torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-5)

    def foresight_model(self, *, device="cpu", shape=None):
        model = FoldedPLT4(shape or self.shape, device=device, foresight=True)
        # Nonuniform, receiver-dependent weights exercise more than the initial mean.
        with torch.no_grad():
            mix = model.cross_part
            for i, (q, b, a) in enumerate(zip(mix.query, mix.bias, mix.alpha)):
                q.copy_(torch.linspace(-0.6, 0.3, q.numel(), device=device))
                b.copy_(torch.linspace(0.4, -0.7, b.numel(), device=device))
                a.fill_(0.6 + i / 20)
        return model

    def test_foresight_initialization_and_legacy_weights(self):
        enabled = FoldedPLT4(self.shape, foresight=True)
        shared = self.model.state_dict()
        self.assertEqual(
            set(shared), {k for k in enabled.state_dict() if not k.startswith("cross_part.")}
        )
        for k, p in shared.items():
            self.assertTrue(torch.equal(p, enabled.state_dict()[k]), k)
        mix = enabled.cross_part
        for q, b, a in zip(mix.query, mix.bias, mix.alpha):
            self.assertTrue(torch.all(q == 0) and torch.all(b == 0) and a.item() == 1)
        other = FoldedPLT4(self.shape, seed=11, foresight=True)
        other.load_initial_weights(shared)
        for k, p in enabled.state_dict().items():
            self.assertTrue(torch.equal(p, other.state_dict()[k]), k)
        for missing in ("cross_part.alpha.0", "embedding.weight"):
            weights = dict(enabled.state_dict())
            del weights[missing]
            with self.assertRaises(RuntimeError):
                other.load_initial_weights(weights)
        with self.assertRaises(RuntimeError):
            other.load_state_dict(shared, strict=True)

    def test_trajectory_frontier_and_full_gradients(self):
        self.check_trajectory(self.model)

    def test_foresight_trajectory_frontier_and_full_gradients(self):
        self.check_trajectory(self.foresight_model())

    def check_trajectory(self, m):
        oracle = copy.deepcopy(m)
        for r in range(12):
            for ell in range(2):
                self.assertIs(m.layer(r, ell), m.layer(r + 4, ell))
        self.assertIsNot(m.layer(0, 0), m.layer(1, 0))
        length = 18 if m.cross_part is not None else 5
        ids = (torch.arange(length)[None] * 13 + 2) % m.c.vocab
        graph = m.hidden(ids, return_details=True)
        h, entries, banks = scalar_trajectory(oracle, ids[0])
        for r in range(16):
            for c in range(ids.size(1)):
                p = c + 16 - r
                self.close(graph.inputs[r][:, c : c + 1], entries[r, p])
                self.close(graph.loops[r][:, c : c + 1], h[r, p])
                for ell in range(2):
                    for actual, expected in zip(graph.kv[r, ell], banks[r, ell, p]):
                        self.close(actual[:, :, c : c + 1], expected)
        ref = oracle.head(torch.cat([h[15, p] for p in range(1, ids.size(1) + 1)], 1))
        self.close(m.head(graph.hidden), ref)
        ref_loss = F.cross_entropy(ref[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
        m.checkpoint_layers = True
        observed = m.loss(ids, chunk=2)
        self.close(observed, ref_loss)
        observed.backward()
        ref_loss.backward()
        for (name, p), (other, q) in zip(
            m.named_parameters(), oracle.named_parameters(), strict=True
        ):
            self.assertEqual(name, other)
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(p.grad.isfinite().all(), name)
            self.close(p.grad, q.grad)
        if m.cross_part is not None:
            self.assertTrue(all(a.grad.abs().sum() > 0 for a in m.cross_part.alpha))
            self.assertGreater(m.cross_part.query[-1].grad.abs().sum().item(), 0)

    def test_wrap_chunking_packing_and_prefix(self):
        self.check_wrap(self.model)

    def test_foresight_wrap_chunking_packing_and_prefix(self):
        self.check_wrap(self.foresight_model())

    @torch.no_grad()
    def check_wrap(self, m):
        ids = (torch.arange(2 * 68).view(2, 68) * 7) % m.c.vocab
        docs = torch.zeros_like(ids)
        docs[1, 20:37] = 1  # recurrent external ID denotes a new document
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[1, :4] = False
        valid[1, -2:] = False
        out = m.hidden(ids, docs, valid=valid, return_details=True)
        z = m.head(out.hidden)
        cache, logits = None, []
        for a, b in [(0, 17), (17, 32), (32, 33), (33, 65), (65, 68)]:
            step = m.decode(ids[:, a:b], docs[:, a:b], valid=valid[:, a:b], cache=cache)
            cache = step.cache
            logits.append(step.logits)
        self.close(torch.cat(logits, 1), z)
        for cut in (1, 32, 37):
            prefix = m.prefill(ids[:, :cut], docs[:, :cut], valid=valid[:, :cut])
            continuation = m.decode(
                ids[:, cut:], docs[:, cut:], valid=valid[:, cut:], cache=prefix.cache
            )
            self.close(torch.cat((prefix.logits, continuation.logits), 1), z)
        self.assertEqual(len(cache.global_banks), m.depth)
        self.assertEqual(len(cache.local_banks), m.depth)
        for r, ell in out.kv:
            kv = (
                tuple(x[:, :, r - 1] for x in cache.local_banks[ell])
                if r
                else cache.global_banks[ell]
            )
            start = 36 if r else 0
            slots = torch.arange(start, 68)
            if r:
                self.assertEqual(kv[0].size(2), 32)
                slots %= 32
            for actual, expected in zip(kv, out.kv[r, ell]):
                self.close(actual.index_select(2, slots), expected[:, :, start:])
        keep = valid[:, 1:] & valid[:, :-1] & (docs[:, 1:] == docs[:, :-1])
        self.close(
            m.loss(ids, docs, valid=valid),
            F.cross_entropy(z[:, :-1][keep], ids[:, 1:][keep]),
        )
        self.close(m(ids[1:2, 37:66]), z[1:2, 37:66])
        altered = ids.clone()
        altered[:, 40:] = (ids[:, 40:] + 13) % m.c.vocab
        self.assertTrue(torch.equal(m(altered, docs, valid=valid)[:, :40], z[:, :40]))
        self.assertEqual(cache.previous.shape, (15, 2, 1, m.c.dim))

    @torch.no_grad()
    def test_legacy_depth_and_frontier_after_window_wrap(self):
        m = FoldedPLT4(replace(self.shape, layers=28)).eval()
        self.assertEqual((m.stages, m.depth, len(m.stack)), (16, 7, 28))
        ids = (torch.arange(36) * 13 % m.c.vocab)[None]
        # Exercise a full-width ring and a frontier up to 15 prediction positions.
        h, _, _ = scalar_trajectory(m, ids[0])
        self.close(m(ids)[:, -2:], m.head(torch.cat([h[15, 35], h[15, 36]], 1)))

    def test_checkpoint_next_update(self):
        self.check_next_update(self.model)

    def test_foresight_checkpoint_next_update(self):
        self.check_next_update(self.foresight_model())

    def check_next_update(self, m):
        ids = torch.tensor([[11, 4, 8, 14, 2], [6, 7, 1, 9, 3]])
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

        def step(model, optim, row):
            optim.zero_grad(set_to_none=True)
            loss = model.loss(ids[row : row + 1])
            loss.backward()
            optim.step()
            return loss.detach()

        step(m, opt, 0)
        file = io.BytesIO()
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), row=1), file)
        file.seek(0)
        saved = torch.load(file, weights_only=True)
        other = FoldedPLT4(self.shape, seed=11, foresight=m.cross_part is not None)
        other.load_state_dict(saved["model"], strict=True)
        other_opt = torch.optim.AdamW(other.parameters(), lr=3e-4)
        other_opt.load_state_dict(saved["optimizer"])
        self.assertTrue(
            torch.equal(step(m, opt, 1), step(other, other_opt, saved["row"]))
        )
        for p, q in zip(m.parameters(), other.parameters(), strict=True):
            self.assertTrue(torch.equal(p, q))
            for key in opt.state[p]:
                self.assertTrue(torch.equal(opt.state[p][key], other_opt.state[q][key]))

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA device required for BF16 backend checks"
    )
    def test_cuda_training_and_streaming(self):
        shape = replace(self.shape, layers=4, head_dim=32)
        for foresight in (False, True):
            with self.subTest(foresight=foresight):
                model = (
                    self.foresight_model(device="cuda", shape=shape)
                    if foresight else FoldedPLT4(shape, device="cuda")
                )
                self.check_cuda(model)

    def check_cuda(self, m):
        ids = (torch.arange(80, device="cuda").view(2, 40) * 11) % m.c.vocab
        with torch.autocast("cuda", dtype=torch.bfloat16):
            graph = m.hidden(ids, return_details=True)
            logits = m.head(graph.hidden)
            loss = m.loss(ids, chunk=16)
            continued = m.decode(ids[:, 33:], cache=m.prefill(ids[:, :33]).cache)
            streamed = m.decode(ids)

        def parity(a, b):
            self.assertTrue(a.isfinite().all() and b.isfinite().all())
            self.assertLess(
                (
                    (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
                ).item(),
                0.02,
            )

        parity(streamed.logits, logits)
        parity(continued.logits, logits[:, 33:])
        for (r, ell), pair in graph.kv.items():
            if r == 0:
                for cached, expected in zip(streamed.cache.global_banks[ell], pair):
                    parity(cached[:, :, :40], expected)
            else:
                slots = torch.arange(8, 40, device="cuda") % 32
                for ring, expected in zip(streamed.cache.local_banks[ell], pair):
                    parity(ring[:, :, r - 1].index_select(2, slots), expected[:, :, 8:])
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
        )


if __name__ == "__main__":
    unittest.main()
