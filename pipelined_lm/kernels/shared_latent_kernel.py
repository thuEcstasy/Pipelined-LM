"""BF16 native latent decode. Read each producer bank once per KV-head/split.
Supports full-KV and latent-KV with the same 2D producer mask and launch family.
No reconstructed history, quantization, or approximate softmax. Split merging
includes its FP32 partial-output storage in workspace accounting.
"""

import math
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _partial(
    Q,
    K,
    V,
    LENGTHS,
    P,
    LSE,
    C: tl.constexpr,
    R: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    PRIVATE: tl.constexpr,
):
    bg = tl.program_id(0)
    si = tl.program_id(1)
    b = bg // HK
    h = bg % HK
    G: tl.constexpr = HQ // HK
    M: tl.constexpr = R * G
    mi = tl.arange(0, BM)
    di = tl.arange(0, D)
    rr = mi // G
    hh = h * G + mi % G
    q = tl.load(
        Q + b * R * HQ * D + rr[:, None] * HQ * D + hh[:, None] * D + di[None, :],
        mi[:, None] < M,
        0,
    )
    length = tl.load(LENGTHS + b)
    acc = tl.full((BM, D), 0, tl.float32)
    l = tl.full((BM,), 0, tl.float32)
    m = tl.full((BM,), float("-inf"), tl.float32)
    CHUNK: tl.constexpr = tr.cdiv(R * C, S * BN) * BN
    for begin in range(si * CHUNK, tl.minimum((si + 1) * CHUNK, R * C), BN):
        ni = begin + tl.arange(0, BN)
        jj = ni // C
        tt = ni % C
        nk = (ni < R * C) & (tt < length)
        ptr = (b * R * HK * C + jj * HK * C + h * C + tt) * D
        k = tl.load(K + ptr[None, :] + di[:, None], nk[None, :], 0)
        v = tl.load(V + ptr[:, None] + di[None, :], nk[:, None], 0)
        score = tl.dot(q, k) * (D**-0.5)
        mask = (mi[:, None] < M) & nk[None, :]
        if PRIVATE:
            mask = mask & (jj[None, :] == rr[:, None])
        else:
            mask = mask & (jj[None, :] <= rr[:, None])
        score = tl.where(mask, score, float("-inf"))
        nm = tl.maximum(m, tl.max(score, 1))
        safe = tl.where(nm == float("-inf"), 0, nm)
        alpha = tl.where(m == float("-inf"), 0, tl.exp(m - safe))
        p = tl.where(mask, tl.exp(score - safe[:, None]), 0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l = l * alpha + tl.sum(p, 1)
        m = nm
    denom = tl.where(l > 0, l, 1)
    acc = acc / denom[:, None]
    ptr = ((bg * S + si) * M + mi) * D
    tl.store(P + ptr[:, None] + di[None, :], acc, mi[:, None] < M)
    logz = tl.where(l > 0, m + tl.log(l), float("-inf"))
    tl.store(LSE + (bg * S + si) * M + mi, logz, mi < M)


@tr.jit
def _merge(
    P,
    LSE,
    O,
    R: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
):
    bg = tl.program_id(0)
    mi = tl.program_id(1)
    b = bg // HK
    h = bg % HK
    G: tl.constexpr = HQ // HK
    M: tl.constexpr = R * G
    ss = tl.arange(0, BS)
    dd = tl.arange(0, D)
    lz = tl.load(LSE + (bg * S + ss) * M + mi, ss < S, float("-inf"))
    mx = tl.max(lz, 0)
    safe = tl.where(mx == float("-inf"), 0, mx)
    w = tl.where(ss < S, tl.exp(lz - safe), 0)
    den = tl.sum(w, 0)
    w = w / tl.where(den > 0, den, 1)
    p = tl.load(
        P + ((bg * S + ss[:, None]) * M + mi) * D + dd[None, :], ss[:, None] < S, 0
    )
    val = tl.sum(p * w[:, None], 0)
    r = mi // G
    qh = h * G + mi % G
    tl.store(O + ((b * R + r) * HQ + qh) * D + dd, val)


def workspace(q, k, splits):
    B, R, HQ, D = q.shape
    HK = k.shape[2]
    M = R * (HQ // HK)
    return (
        torch.empty((B, HK, splits, M, D), device=q.device, dtype=torch.float32),
        torch.empty((B, HK, splits, M), device=q.device, dtype=torch.float32),
    )


def run(
    q, k, v, *, splits=4, private=False, lengths=None, out=None, work=None, tile=64
):
    B, R, HQ, D = q.shape
    assert k.shape == v.shape and k.shape[:2] == (B, R) and k.shape[-1] == D
    HK, C = k.shape[2:4]
    assert D in (64, 128, 256) and HQ % HK == 0 and splits >= 1
    assert all(x.is_contiguous() and x.dtype == torch.bfloat16 for x in (q, k, v))
    if out is None:
        out = torch.empty_like(q)
    if lengths is None:
        lengths = torch.full((B,), C, device=q.device, dtype=torch.int32)
    if work is None:
        work = workspace(q, k, splits)
    BM = max(16, tr.next_power_of_2(R * (HQ // HK)))
    _partial[(B * HK, splits)](
        q,
        k,
        v,
        lengths,
        *work,
        C,
        R,
        HQ,
        HK,
        D,
        splits,
        tile,
        BM,
        private,
        num_warps=8 if BM * D >= 16384 else 4,
    )
    _merge[(B * HK, R * (HQ // HK))](
        *work, out, R, HQ, HK, D, splits, tr.next_power_of_2(splits), num_warps=4
    )
    return out


def reference(q, k, v, private=False, lengths=None):
    """Independent explicit per-receiver softmax; FP32 accumulation."""
    B, R, HQ, D = q.shape
    HK, C = k.shape[2:4]
    out = torch.zeros_like(q)
    for b in range(B):
        n = C if lengths is None else int(lengths[b])
        if n == 0:
            continue
        for r in range(R):
            for h in range(HQ):
                banks = [r] if private else list(range(r + 1))
                kh = torch.cat([k[b, j, h // (HQ // HK), :n] for j in banks]).float()
                vh = torch.cat([v[b, j, h // (HQ // HK), :n] for j in banks]).float()
                out[b, r, h] = (
                    (q[b, r, h].float() @ kh.T / math.sqrt(D)).softmax(-1) @ vh
                ).to(out.dtype)
    return out
