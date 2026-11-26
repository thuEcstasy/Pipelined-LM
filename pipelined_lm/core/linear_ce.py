"""Exact full-vocabulary CE in bounded chunks, with no head recomputation.

Backbone gradients and the tied embedding gradient remain fully differentiable
to first order. The precomputed VJP is scaled by the incoming scalar gradient.
CUDA follows BF16 linear operands/output and FP32 CE, just like autocast.
"""

import torch
import torch.nn.functional as F


def project(h, w):
    dtype = torch.bfloat16 if h.is_cuda else h.dtype
    with torch.autocast(h.device.type, enabled=False):
        return F.linear(h.to(dtype), w.to(dtype))


def chunks(h, w, y, weights, chunk, mask_id, backward):
    dtype = torch.bfloat16 if h.is_cuda else h.dtype
    reduction = torch.float64 if h.dtype == torch.float64 else torch.float32
    compute_w = w.to(dtype)
    total = torch.zeros((), device=h.device, dtype=reduction)
    dh = torch.empty_like(h) if backward else None
    dw = torch.zeros_like(w) if backward else None
    for first in range(0, len(y), chunk):
        x = h[first : first + chunk].to(dtype)
        target = y[first : first + chunk]
        scale = weights[first : first + chunk].to(reduction)
        with torch.autocast(h.device.type, enabled=False):
            logits = F.linear(x, compute_w)
        if mask_id >= 0:
            logits[:, mask_id] = float("-inf")
        lp = F.log_softmax(logits.to(reduction), -1)
        index = torch.arange(len(target), device=h.device)
        total -= torch.sum(lp[index, target] * scale)
        if backward:
            grad = lp.exp()
            grad[index, target] -= 1
            grad.mul_(scale[:, None])
            grad = grad.to(dtype)
            with torch.autocast(h.device.type, enabled=False):
                dh[first : first + len(target)] = torch.matmul(grad, compute_w).to(
                    h.dtype
                )
                dw.add_(torch.matmul(grad.transpose(0, 1), x).to(w.dtype))
    return total, dh, dw


class LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, y, weights, chunk, mask_id):
        total, dh, dw = chunks(h, w, y, weights, chunk, mask_id, True)
        ctx.save_for_backward(dh, dw)
        return total

    @staticmethod
    def backward(ctx, scale):
        dh, dw = ctx.saved_tensors
        return dh * scale, dw * scale, None, None, None, None


def linear_ce(h, w, y, weights=None, chunk=512, mask_id=-1):
    assert h.ndim == 2 and w.ndim == 2 and h.shape[1] == w.shape[1] and len(y) == len(h)
    assert chunk > 0
    if weights is None:
        weights = torch.ones(len(y), device=h.device, dtype=h.dtype)
    if not len(y):
        return h.sum() * 0 + w.sum() * 0
    if torch.is_grad_enabled():
        return LinearCE.apply(h, w, y, weights, chunk, mask_id)
    return chunks(h, w, y, weights, chunk, mask_id, False)[0]
