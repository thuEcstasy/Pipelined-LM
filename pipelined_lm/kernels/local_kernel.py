"""One tiled read of local producer rings for all legal receiver query heads."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _local(
    Q,
    K,
    V,
    O,
    T,
    HK: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    TWOD: tl.constexpr,
    DIRECT: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // HK
    h = bh % HK
    G: tl.constexpr = HQ // HK
    M: tl.constexpr = R * G
    mi = tl.arange(0, BM)
    di = tl.arange(0, D)
    rr = mi // G
    qh = h * G + mi % G
    q = tl.load(
        Q + ((b * R + rr[:, None]) * HQ + qh[:, None]) * D + di[None, :],
        mi[:, None] < M,
        0,
    )
    length = tl.minimum(tl.load(T) + 1, W)
    accum = tl.full((BM, D), 0, tl.float32)
    den = tl.full((BM,), 0, tl.float32)
    mx = tl.full((BM,), float("-inf"), tl.float32)
    for j in range(R):
        tt = tl.arange(0, W)
        valid = tt < length
        ptr = ((b * R + j) * HK + h) * W * D + tt * D
        k = tl.load(K + ptr[None, :] + di[:, None], valid[None, :], 0)
        v = tl.load(V + ptr[:, None] + di[None, :], valid[:, None], 0)
        allowed = (mi[:, None] < M) & valid[None, :]
        if TWOD:
            allowed = allowed & (j <= rr[:, None])
            if not DIRECT:
                allowed = allowed & (
                    ((tl.load(T) - tt[None, :]) % W) >= rr[:, None] - j
                )
        else:
            allowed = allowed & (j == rr[:, None])
        score = tl.where(allowed, tl.dot(q, k) * (D**-0.5), float("-inf"))
        nm = tl.maximum(mx, tl.max(score, 1))
        safe = tl.where(nm == float("-inf"), 0, nm)
        alpha = tl.where(mx == float("-inf"), 0, tl.exp(mx - safe))
        p = tl.where(allowed, tl.exp(score - safe[:, None]), 0)
        accum = accum * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        den = den * alpha + tl.sum(p, 1)
        mx = nm
    result = accum / tl.where(den > 0, den, 1)[:, None]
    tl.store(
        O + ((b * R + rr[:, None]) * HQ + qh[:, None]) * D + di[None, :],
        result,
        mi[:, None] < M,
    )


def run(q, k, v, t, twod, out, direct=False):
    B, R, HQ, D = q.shape
    HK = k.shape[2]
    W = k.shape[3]
    assert R == 3 and k.shape == (B, R, HK, W, D) and k.shape == v.shape
    assert W in (32, 64, 128) and D in (64, 128) and HQ % HK == 0
    assert all(x.is_contiguous() and x.dtype == torch.bfloat16 for x in (q, k, v, out))
    _local[(B * HK,)](
        q,
        k,
        v,
        out,
        t,
        HK,
        HQ,
        D,
        W,
        R,
        max(16, tr.next_power_of_2(R * (HQ // HK))),
        twod,
        direct,
        num_warps=4,
    )
    return out


def reference(q, k, v, t, twod, direct=False):
    B, R, HQ, D = q.shape
    HK = k.shape[2]
    n = min(int(t) + 1, k.shape[3])
    out = torch.zeros_like(q)
    for b in range(B):
        for r in range(R):
            for h in range(HQ):
                producers = range(r + 1) if twod else [r]
                indices = torch.arange(n, device=q.device)
                masks = [
                    ((int(t) - indices) % k.shape[3] >= r - j)
                    if twod and not direct
                    else torch.ones_like(indices, dtype=torch.bool)
                    for j in producers
                ]
                kk = torch.cat(
                    [
                        k[b, j, h // (HQ // HK), :n][mask]
                        for j, mask in zip(producers, masks)
                    ]
                ).float()
                vv = torch.cat(
                    [
                        v[b, j, h // (HQ // HK), :n][mask]
                        for j, mask in zip(producers, masks)
                    ]
                ).float()
                out[b, r, h] = ((q[b, r, h].float() @ kk.T) * (D**-0.5)).softmax(
                    -1
                ) @ vv
    return out
