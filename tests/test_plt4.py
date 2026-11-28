"""Independent plain PLT-4 recurrence using one full shared stack per stage."""

import copy
import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from pipelined_lm import PLT4, tiny_backbone
from pipelined_lm.train import learning_rate


class PlainPLTTest(unittest.TestCase):
    def test_original_recurrence_window_and_gradients(self):
        torch.set_num_threads(2)
        m = PLT4(replace(tiny_backbone(), layers=2, window=64))
        ref = copy.deepcopy(m)
        ids = torch.arange(68)[None] % m.c.vocab
        e = ref.embedding(ids)
        clock = torch.arange(68)
        causal = clock[:, None] >= clock
        local = causal & ((clock[:, None] - clock) < 64)
        banks = []
        for r in range(4):
            h = e if r == 0 else e + F.pad(h[:, :-1], (0, 0, 1, 0))
            for ell, layer in enumerate(ref.stack):
                q, kv, gate = layer.project(h, clock[None])
                if not r:
                    banks.append(kv)
                h = layer.finish(
                    h,
                    q,
                    gate,
                    banks[ell],
                    causal[None],
                    kv if r else None,
                    local[None] if r else None,
                )
        z = ref.head(h)
        torch.testing.assert_close(m(ids), z, rtol=1e-4, atol=2e-5)
        torch.testing.assert_close(
            m.decode(ids[:, 64:], cache=m.prefill(ids[:, :64]).cache).logits,
            z[:, 64:],
            rtol=1e-4,
            atol=2e-5,
        )
        F.cross_entropy(
            z[:, :-1].reshape(-1, m.c.vocab), ids[:, 1:].flatten()
        ).backward()
        m.loss(ids, chunk=7).backward()
        for p, q in zip(m.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-5)

    def test_rates(self):
        for recipe in ("plt4", "folded-plt4"):
            self.assertEqual(learning_rate(recipe, 0), 3e-4 / 128)
            self.assertEqual(learning_rate(recipe, 127), 3e-4)
            self.assertEqual(learning_rate(recipe, 45776), 3e-4)
        self.assertAlmostEqual(learning_rate("pipeline", 99), 3e-4)
        self.assertAlmostEqual(learning_rate("pipeline", 152588), 3e-5)
        self.assertGreater(
            learning_rate("pipeline", 22889), learning_rate("pipeline", 22890)
        )
