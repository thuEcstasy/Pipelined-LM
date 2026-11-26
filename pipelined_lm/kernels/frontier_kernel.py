"""Read <=6 currently available frontier states; no extra KV bank or copies."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _front(
    Q,
    K,
    V,
    GK,
    GV,
    O,
    T,
    HK: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    CAP: tl.constexpr,
    BM: tl.constexpr,
):
    bh = tl.program_id(0)
    b = bh // HK
    h = bh % HK
    group: tl.constexpr = HQ // HK
    mi = tl.arange(0, BM)
    di = tl.arange(0, D)
    n = tl.arange(0, 16)
    r = mi // group + 1
    qh = h * group + mi % group
    producer = tl.where(n < 3, 0, tl.where(n < 5, 1, 2))
    age = tl.where(n < 3, n, tl.where(n < 5, n - 3, 0))
    pos = tl.load(T) - age
    gptr = ((b * CAP + pos) * HK + h) * D
    lptr = ((b * 3 + producer - 1) * HK + h) * W * D + (pos % W) * D
    kg = tl.load(GK + gptr[None, :] + di[:, None], ((n < 3) & (pos >= 0))[None, :], 0)
    kl = tl.load(
        K + lptr[None, :] + di[:, None], ((n >= 3) & (n < 6) & (pos >= 0))[None, :], 0
    )
    vg = tl.load(GV + gptr[:, None] + di[None, :], ((n < 3) & (pos >= 0))[:, None], 0)
    vl = tl.load(
        V + lptr[:, None] + di[None, :], ((n >= 3) & (n < 6) & (pos >= 0))[:, None], 0
    )
    k = kg + kl
    v = vg + vl
    q = tl.load(
        Q + ((b * 3 + r[:, None] - 1) * HQ + qh[:, None]) * D + di[None, :],
        r[:, None] <= 3,
        0,
    )
    allowed = (
        (r[:, None] <= 3)
        & (n[None, :] < 6)
        & (pos[None, :] >= 0)
        & (producer[None, :] < r[:, None])
        & (age[None, :] < r[:, None] - producer[None, :])
    )
    scores = tl.where(allowed, tl.dot(q, k) * (D**-0.5), float("-inf"))
    mx = tl.max(scores, 1)
    safe = tl.where(mx == float("-inf"), 0, mx)
    p = tl.where(allowed, tl.exp(scores - safe[:, None]), 0)
    den = tl.sum(p, 1)
    p = p / tl.where(den > 0, den, 1)[:, None]
    out = tl.dot(p.to(v.dtype), v)
    tl.store(
        O + ((b * 3 + r[:, None] - 1) * HQ + qh[:, None]) * D + di[None, :],
        out,
        r[:, None] <= 3,
    )


def run(q, k, v, gk, gv, t):
    B, R, HQ, D = q.shape
    HK = k.shape[2]
    W = k.shape[3]
    assert R == 3 and D in (64, 128) and W in (32, 64, 128)
    out = torch.empty_like(q)
    _front[(B * HK,)](
        q,
        k,
        v,
        gk,
        gv,
        out,
        t,
        HK,
        HQ,
        D,
        W,
        gk.shape[0] * 128 // B,
        max(16, tr.next_power_of_2(3 * HQ // HK)),
        num_warps=4,
    )
    return out


def reference(q, k, v, gk, gv, t):
    B, R, HQ, D = q.shape
    HK = k.shape[2]
    W = k.shape[3]
    gk = gk.view(B, -1, HK, D)
    gv = gv.view(B, -1, HK, D)
    out = torch.zeros_like(q)
    for b in range(B):
        for r in range(1, 4):
            for h in range(HQ):
                pairs = [
                    (j, int(t) - age)
                    for j in range(r)
                    for age in range(r - j)
                    if int(t) >= age
                ]
                if not pairs:
                    continue
                kh = h // (HQ // HK)
                kk = torch.stack(
                    [
                        gk[b, p, kh] if j == 0 else k[b, j - 1, kh, p % W]
                        for j, p in pairs
                    ]
                ).float()
                vv = torch.stack(
                    [
                        gv[b, p, kh] if j == 0 else v[b, j - 1, kh, p % W]
                        for j, p in pairs
                    ]
                ).float()
                out[b, r - 1, h] = (
                    (q[b, r - 1, h].float() @ kk.T) * (D**-0.5)
                ).softmax(-1) @ vv
    return out
