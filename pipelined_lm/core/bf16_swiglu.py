"""Fuse the native CUDA SiLU arithmetic and preserve BF16 boundaries."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def forward_kernel(G, U, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N
    g = tl.load(G + i, mask, 0).to(tl.float32)
    u = tl.load(U + i, mask, 0).to(tl.float32)
    silu = libdevice.div_rn(g, 1.0 + libdevice.exp(-g))
    tl.store(Y + i, silu * u, mask)


@triton.jit
def backward_kernel(G, U, D, DG, DU, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N
    g = tl.load(G + i, mask, 0).to(tl.float32)
    u = tl.load(U + i, mask, 0).to(tl.float32)
    d = tl.load(D + i, mask, 0).to(tl.float32)
    denom = 1.0 + libdevice.exp(-g)
    s = libdevice.div_rn(1.0, denom)
    inner = libdevice.fma_rn(g, 1.0 - s, 1.0)
    dg = ((d * u) * s) * inner
    du = d * libdevice.div_rn(g, denom)
    tl.store(DG + i, dg, mask)
    tl.store(DU + i, du, mask)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        assert (
            g.dtype == u.dtype == torch.bfloat16
            and g.is_contiguous()
            and u.is_contiguous()
        )
        ctx.save_for_backward(g, u)
        y = torch.empty_like(g)
        forward_kernel[(triton.cdiv(g.numel(), 1024),)](
            g, u, y, g.numel(), 1024, enable_fp_fusion=False
        )
        return y

    @staticmethod
    def backward(ctx, d):
        g, u = ctx.saved_tensors
        d = d.contiguous()
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        backward_kernel[(triton.cdiv(g.numel(), 1024),)](
            g, u, d, dg, du, g.numel(), 1024, enable_fp_fusion=False
        )
        return dg, du
