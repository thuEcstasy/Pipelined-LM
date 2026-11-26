import copy
import unittest

import torch
import torch.nn.functional as F

from pipelined_lm import POLICIES, PipelinedLM, tiny_backbone
from tests import reference


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def exercise(self, policy, **options):
        model = PipelinedLM(tiny_backbone(), policy=policy, **options)
        # Make new paths active even when their production gains initialize to zero.
        rng = torch.Generator().manual_seed(19)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if any(
                    k in name
                    for k in ("cross_part.", "frontier_", "subblock_", "depth_reads.")
                ):
                    p.uniform_(-0.17, 0.17, generator=rng)
        oracle = copy.deepcopy(model)
        x = torch.arange(18).reshape(2, 9)
        docs = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 0], [2] * 9])
        valid = torch.ones_like(x, dtype=torch.bool)
        valid[1, 0] = False
        valid[1, -1] = False
        tr = model.hidden(x, docs, valid=valid, return_details=True)
        z = model.head(tr.hidden)
        gold = reference.run(oracle, x, docs, valid)

        def close(a, b):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)

        close(z, gold["logits"])
        for key in tr.kv:
            for kv, expected in zip(tr.kv[key], gold["kv"][key], strict=True):
                for v, ref in zip(kv, expected, strict=True):
                    close(v, ref)
        weights = torch.randn(z.shape, generator=rng)
        (weights * z).sum().backward()
        (weights * gold["logits"]).sum().backward()
        for (name, p), (other, q) in zip(
            model.named_parameters(), oracle.named_parameters(), strict=True
        ):
            self.assertEqual(name, other)
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertIsNotNone(q.grad, name)
                close(p.grad, q.grad)
        with torch.inference_mode():
            prefix = model.prefill(x[:, :4], docs[:, :4], valid=valid[:, :4])
            rest = model.decode(
                x[:, 4:], docs[:, 4:], valid=valid[:, 4:], cache=prefix.cache
            )
            close(torch.cat([prefix.logits, rest.logits], 1), z)
            changed = x.clone()
            changed[:, 6:] += 11
            self.assertTrue(
                torch.equal(model(changed, docs, valid=valid)[:, :6], z[:, :6])
            )
            keep = valid[:, 1:] & valid[:, :-1] & (docs[:, 1:] == docs[:, :-1])
            close(
                model.loss(x, docs, valid=valid),
                F.cross_entropy(z[:, :-1][keep], x[:, 1:][keep]),
            )

    def test_cache_policies(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                self.exercise(policy)

    def test_subblock_and_predecessor_paths(self):
        self.exercise("direct", subblock_residuals=True, block_size=3)
        self.exercise("direct", aligned_foresight=True)
        self.exercise("direct", foresight=False)

    def test_closed_subblock_gates_preserve_boundary_model(self):
        plain = PipelinedLM(tiny_backbone(), policy="direct")
        deeper = PipelinedLM(tiny_backbone(), policy="direct", subblock_residuals=True)
        x = torch.arange(10)[None]
        self.assertTrue(torch.equal(plain(x), deeper(x)))


if __name__ == "__main__":
    unittest.main()
