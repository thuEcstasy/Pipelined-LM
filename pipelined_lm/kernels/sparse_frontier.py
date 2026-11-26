"""Candidate sparse training reader: index mask, no shifted KV expansion.

All sources are same-offset producer layers. For receiver r (zero based),
producer j is eligible at age 0 <= c-u < r-j. No K/V precision change.
The packed/doc fallback remains the independently checked eager implementation.
This backend must pass BF16 forward/backward parity before being selected.
"""

from functools import lru_cache
import torch


@lru_cache(maxsize=32)
def mask(length, receiver, device):
    from torch.nn.attention.flex_attention import create_block_mask

    def visible(b, h, q, k):
        producer = k // length
        age = q - k % length
        return (
            (q < length)
            & (k < receiver * length)
            & (age >= 0)
            & (age < receiver - producer)
        )

    return create_block_mask(
        visible, None, None, length, receiver * length, device=device, _compile=True
    )


@lru_cache(maxsize=1)
def compiled_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, fullgraph=True, dynamic=False)


def run(q, kvs):
    r = len(kvs)
    assert 1 <= r <= 3 and q.is_cuda
    k = torch.cat([p[0] for p in kvs], 2) if r > 1 else kvs[0][0]
    v = torch.cat([p[1] for p in kvs], 2) if r > 1 else kvs[0][1]
    return compiled_attention()(
        q.to(v.dtype),
        k.to(v.dtype),
        v,
        block_mask=mask(q.shape[2], r, str(q.device)),
        enable_gqa=True,
    )
