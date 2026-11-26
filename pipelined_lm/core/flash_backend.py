"""Original SDPA Flash global path; optional native sliding-window Flash kernel."""

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


class Window(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window):
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        nq, nk = q.shape[1], k.shape[1]
        out, lse, seed, offset, _ = torch.ops.aten._flash_attention_forward.default(
            q,
            k,
            v,
            None,
            None,
            nq,
            nk,
            0.0,
            True,
            False,
            window_size_left=window - 1,
            window_size_right=0,
        )
        ctx.save_for_backward(q, k, v, out, lse, seed, offset)
        ctx.window = window
        return out.transpose(1, 2)

    @staticmethod
    def backward(ctx, grad):
        q, k, v, out, lse, seed, offset = ctx.saved_tensors
        cu = torch.empty(0, device=q.device, dtype=torch.int32)
        dq, dk, dv = torch.ops.aten._flash_attention_backward.default(
            grad.transpose(1, 2).contiguous(),
            q,
            k,
            v,
            out,
            lse,
            None,
            None,
            q.shape[1],
            k.shape[1],
            0.0,
            True,
            seed,
            offset,
            window_size_left=ctx.window - 1,
            window_size_right=0,
        )
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), None


def global_sdpa(q, k, v, attn_mask=None, is_causal=False, **kw):
    if q.dtype in (torch.bfloat16, torch.float16) and attn_mask is None:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, **kw)
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, **kw
        )
