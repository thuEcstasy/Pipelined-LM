"""Per-sublayer depth reads on one prediction trajectory.

Immutable depth sources are scored for a block of readers together. Each reader
then merges the still-changing within-block update by an exact softmax merge.
Pre endpoints travel with the trajectory, whereas aligned Foresight is separate.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .subblock import normalized


DEFAULT_BLOCK = 4


class DepthRead(nn.Module):
    def __init__(self, sites, dim, stage, eps, block, device):
        super().__init__()
        if not isinstance(block, int) or block <= 0:
            raise ValueError("block size must be a positive number of sublayers")
        self.block = block
        self.sites = sites
        self.eps = eps
        self.scale = dim**-0.5
        self.num_pre = max(0, stage - 1)
        self.local_query = nn.Parameter(torch.zeros(sites, dim, device=device))
        if self.num_pre > 1:
            self.pre_query = nn.Parameter(torch.zeros(sites, dim, device=device))
        channels = 1 + int(self.num_pre > 0)
        self.write = nn.Parameter(torch.zeros(sites, channels, dim, device=device))
        self.bias = nn.Parameter(torch.zeros(sites, channels, device=device))


def install(model, block):
    model.depth_reads = nn.ModuleList(
        DepthRead(
            2 * model.cfg.partition.depth,
            model.c.dim,
            r,
            model.c.eps,
            block,
            model.embedding.weight.device,
        )
        for r in range(model.H)
    )


def shift_endpoint(value, lag, segments, valid):
    """Both ends must be valid and belong to the same contiguous document."""
    if lag >= value.shape[1]:
        return value * 0, torch.zeros_like(valid)
    source = F.pad(value[:, :-lag], (0, 0, lag, 0))
    mask = valid[:, lag:] & valid[:, :-lag] & (segments[:, lag:] == segments[:, :-lag])
    return source, F.pad(mask, (lag, 0))


def cached_endpoints(older, cycle, r, eps):
    """older[j][0] is producer j two cycles ago (prev[j] is one ago)."""
    return [
        (normalized(older[j][r - j - 2], eps), cycle >= r - j)
        for j in range(max(0, r - 1))
    ]


def allocate_older(prev):
    return [
        prev[j].new_zeros((len(prev) - j - 2, *prev[j].shape))
        for j in range(len(prev) - 2)
    ]


def advance_older(older, prev):
    for j, bank in enumerate(older):
        for slot in reversed(range(1, len(bank))):
            bank[slot].copy_(bank[slot - 1])
        bank[0].copy_(prev[j])


def prefill_older(older, loops):
    for j, bank in enumerate(older):
        for slot, target in enumerate(bank):
            lag = slot + 2
            if loops[j].shape[1] >= lag:
                target.copy_(loops[j][:, -lag].unsqueeze(1))
            else:
                target.zero_()


class PreBank:
    def __init__(self, reader, sources):
        if len(sources) == 1:
            value, mask = sources[0]
            self.single = value * mask[..., None]
            self.weights = None
            return
        # In the four-stage pipeline only D4 has two older, non-adjacent
        # endpoints. A two-key softmax is a single sigmoid of the score gap.
        if len(sources) != 2:
            raise ValueError("expected one or two non-adjacent Pre endpoints")
        (first, valid_first), (second, valid_second) = sources
        with torch.autocast(first.device.type, enabled=False):
            self.delta = second - first
            scores = F.linear(self.delta, reader.pre_query.float()) * reader.scale
            both = (valid_first & valid_second)[..., None]
            self.weights = torch.where(
                both, scores.sigmoid(), valid_second[..., None].float()
            )
            self.base = first * (valid_first | valid_second)[..., None]

    def message(self, site):
        if self.weights is None:
            return self.single
        return torch.addcmul(self.base, self.weights[..., site : site + 1], self.delta)


class Stage:
    """Forward-local state. Site order is attention, FFN, attention, FFN, ..."""

    def __init__(self, reader, entry, sources):
        self.reader = reader
        self.start = entry
        self.completed = [entry.float()]
        self.keys = [normalized(entry, reader.eps)]
        self.pre = PreBank(reader, sources) if sources else None

    def read(self, site, residual):
        rd = self.reader
        offset = site % rd.block
        with torch.autocast(residual.device.type, enabled=False):
            x = residual.float()
            if offset == 0:
                if site:
                    delta = x - self.start.float()
                    self.completed.append(delta)
                    self.keys.append(normalized(delta, rd.eps))
                self.start = residual
                queries = rd.local_query[site : site + rd.block].float()
                if len(self.completed) == 1:
                    self.log_z = F.linear(self.keys[0], queries) * rd.scale
                    self.messages = (
                        self.completed[0]
                        .unsqueeze(-2)
                        .expand(*x.shape[:-1], len(queries), x.shape[-1])
                    )
                else:
                    keys = torch.stack(self.keys, dim=-2)
                    values = torch.stack(self.completed, dim=-2)
                    scores = (keys @ queries.T).transpose(-1, -2) * rd.scale
                    self.log_z = scores.logsumexp(-1)
                    self.messages = scores.softmax(-1) @ values
            local = self.messages[..., offset, :]
            if offset:
                partial = x - self.start.float()
                key = normalized(partial, rd.eps)
                score = (key * rd.local_query[site].float()).sum(-1) * rd.scale
                alpha = (score - self.log_z[..., offset]).sigmoid().unsqueeze(-1)
                local = torch.lerp(local, partial, alpha)
            gates = (
                F.linear(x, rd.write[site].float()) * rd.scale + rd.bias[site].float()
            ).tanh()
            out = torch.addcmul(x, gates[..., :1], local)
            if self.pre is not None:
                out = torch.addcmul(out, gates[..., 1:], self.pre.message(site))
            return out.to(residual.dtype)
