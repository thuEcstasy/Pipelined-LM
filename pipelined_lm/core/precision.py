"""BF16 operator boundaries with FP32 norm/nonlinear arithmetic.

Linear projections, rotary outputs, attention and KV remain BF16 under autocast.
No layer or matrix multiplication is promoted to FP32. FP64 audits retain FP64.
"""

import torch
from torch import nn
from torch.nn import functional as F


class QwenRMSNorm(nn.RMSNorm):
    def forward(self, x):
        if x.dtype == torch.float64:
            return super().forward(x)
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        # Qwen's order: normalize, restore input dtype, then apply the gain.
        return y.to(x.dtype) * self.weight


def swiglu(layer, gate, up):
    if (
        getattr(layer, "bf16_elementwise", "native") == "cuda_swiglu_v1"
        and gate.is_cuda
        and gate.dtype == up.dtype == torch.bfloat16
        and gate.is_contiguous()
        and up.is_contiguous()
    ):
        from .bf16_swiglu import SwiGLU

        return SwiGLU.apply(gate, up)
    if getattr(layer, "single_round_swiglu", False):
        dtype = torch.promote_types(gate.dtype, torch.float32)
        return (F.silu(gate.to(dtype)) * up.to(dtype)).to(gate.dtype)
    return F.silu(gate) * up


def enable(model):
    """Keep every original parameter object and state-dict key."""
    for layer in model.stack:
        for norm in (layer.n1, layer.n2, layer.q_norm, layer.k_norm):
            norm.__class__ = QwenRMSNorm
        layer.single_round_swiglu = True
        layer.round_rotary_output = True
    model.final_norm.__class__ = QwenRMSNorm
    model.bf16_epilogue = "qwen_single_round"
    return model


def optimize(model, policy):
    if policy not in ("native", "cuda_swiglu_v1"):
        raise ValueError("Unknown BF16 elementwise implementation")
    if (
        policy != "native"
        and getattr(model, "bf16_epilogue", "native") != "qwen_single_round"
    ):
        raise ValueError("The fused kernel requires single-round BF16 SwiGLU")
    for layer in model.stack:
        layer.bf16_elementwise = policy
    model.bf16_elementwise = policy
    return model
