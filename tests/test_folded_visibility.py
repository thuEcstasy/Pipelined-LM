"""Recover the actual (producer, cycle) attention weights with tagged V vectors."""

import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from pipelined_lm import FoldedPLT4, tiny_backbone


@torch.no_grad()
def check_visibility(device, metrics):
    length, stages, heads, width = 35, 16, 4, 64
    cuda = device == "cuda"
    dtype = torch.bfloat16 if cuda else torch.float32
    shape = replace(
        tiny_backbone(), layers=4, window=32, heads=heads, kv_heads=1, head_dim=width
    )
    m = FoldedPLT4(shape, device=device).eval()
    ids = torch.zeros(stages, length, dtype=torch.long, device=device)
    # Each row selects one producer; each V component selects one publication cycle.
    # The four query heads read pure global, pure local and two different mixtures.
    gates = torch.tensor([0, 1, 0.25, 0.75], device=device)

    def check_path(path):
        outputs = [[] for _ in range(4)]

        def tagged_project(fold):
            def project(x, pos):
                batch, seq = pos.shape
                row = torch.arange(batch, device=device)
                if path == "training":
                    producer = fold + 4 * len(outputs[fold])
                else:
                    producer = fold + 4 * (row // stages)
                marker = (row % stages == producer)[:, None, None]
                v = (F.one_hot(pos, width) * marker).to(dtype)[:, None]
                q = v.new_zeros(batch, heads, seq, width)
                return (
                    q,
                    (torch.zeros_like(v), v),
                    gates.view(1, heads, 1, 1).expand(batch, -1, seq, -1),
                )

            return project

        def capture(fold):
            def record(module, args):
                outputs[fold].append(
                    args[0].detach().float().cpu().unflatten(-1, (heads, width))
                )

            return record

        with ExitStack() as stack:
            for fold, layer in enumerate(m.stack):
                stack.enter_context(
                    patch.object(layer, "project", tagged_project(fold))
                )
                stack.enter_context(layer.o.register_forward_pre_hook(capture(fold)))
            with torch.autocast(device, dtype=torch.bfloat16, enabled=cuda):
                if path == "training":
                    m.hidden(ids)
                else:
                    m.decode(ids)

        errors = metrics[path] = {"max_abs": 0.0, "max_abs_forbidden": 0.0}
        for r in range(stages):
            pieces = outputs[r % 4]
            if path == "training":
                actual = pieces[r // 4]
            else:
                actual = torch.cat(
                    [p.unflatten(0, (4, stages))[r // 4] for p in pieces], 1
                )
            expected = torch.zeros_like(actual)
            for c in range(length):
                # Stage 0 is always the only global producer. The oldest legal local
                # cycle is c-31 inclusive. The current cycle is visible through stage r.
                local_gate = gates.cpu() if r else torch.zeros(heads)
                expected[0, c, :, : c + 1] = ((1 - local_gate) / (c + 1))[:, None]
                if r:
                    first = max(0, c - 31)
                    expected[1 : r + 1, c, :, first : c + 1] = (
                        local_gate / (r * (c - first + 1))
                    )[:, None]
            error = (actual - expected).abs()
            errors["max_abs"] = max(errors["max_abs"], error.max().item())
            forbidden = error[expected == 0].max().item()
            errors["max_abs_forbidden"] = max(errors["max_abs_forbidden"], forbidden)
            if forbidden != 0:
                raise AssertionError(
                    f"{path}, receiver {r + 1}: forbidden source weight {forbidden}"
                )
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.02 if cuda else 1e-5,
                atol=0,
                msg=lambda error, site=f"{path}, receiver {r + 1}": f"{site}: {error}",
            )

    check_path("training")
    check_path("decode")


class FoldedVisibilityTest(unittest.TestCase):
    def test_exact_training_and_decode_source_weights(self):
        torch.set_num_threads(2)
        self.metrics = {}
        check_visibility("cpu", self.metrics)
