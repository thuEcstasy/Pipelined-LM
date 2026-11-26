"""Paged KV caches decoded with FlashAttention-4, as the PLT decoder did on GB200.

A cache holds `capacity` positions per sequence in 128-token pages. A full
cache grows by one position per step; a window cache is a ring of `capacity`
positions inside a single page, valid length min(t + 1, capacity). Keys carry
RoPE already, so the order of positions inside the ring does not matter.

Everything is preallocated and the lengths are device tensors, so a decode step
using these caches can be captured as one CUDA graph.
"""

import torch

BLOCK = 128


class FlashKV:
    def __init__(self, batch, kv_heads, head_dim, capacity, dtype, device, ring=False):
        from vllm import _custom_ops as ops
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        self.ops, self.flash = ops, flash_attn_varlen_func
        self.B, self.K, self.D, self.capacity, self.ring = (
            batch,
            kv_heads,
            head_dim,
            capacity,
            ring,
        )
        if ring:
            assert capacity <= BLOCK, "a ring cache lives in one page"
        self.pages = (capacity + BLOCK - 1) // BLOCK
        self.table = torch.arange(
            batch * self.pages, device=device, dtype=torch.int32
        ).view(batch, self.pages)
        self.k = torch.zeros(
            (batch * self.pages, BLOCK, kv_heads, head_dim), dtype=dtype, device=device
        )
        self.v = torch.zeros_like(self.k)
        self.scale = torch.ones((), device=device, dtype=torch.float32)
        self.base = torch.arange(batch, device=device, dtype=torch.long) * (
            self.pages * BLOCK
        )
        self.cu = torch.arange(batch + 1, device=device, dtype=torch.int32)
        # A persistent, genuinely stride-1 length buffer. expand() of a scalar has
        # stride 0, and for batch 1 .contiguous() keeps it, which FA4 rejects.
        self.lens = torch.zeros(batch, device=device, dtype=torch.int32)

    # FA4 is a CuTe kernel that reads the raw CUDA stream handle, which a stream
    # object seen while dynamo traces does not carry. Keep these two calls out of
    # the compiled regions; they still run inside the captured CUDA graph.
    @torch.compiler.disable
    def write(self, k, v, t):
        """k, v: [B, kv_heads, head_dim] for position t (device scalar)."""
        slot = torch.remainder(t, self.capacity) if self.ring else t
        self.ops.reshape_and_cache_flash(
            k.contiguous(),
            v.contiguous(),
            self.k,
            self.v,
            self.base + slot,
            "auto",
            self.scale,
            self.scale,
        )

    def length(self, t):
        n = torch.clamp(t + 1, max=self.capacity) if self.ring else t + 1
        self.lens.copy_(n.to(torch.int32).expand(self.B))
        return self.lens

    @torch.compiler.disable
    def attend(self, q, t):
        """q: [B, heads, head_dim] -> [B, heads, head_dim]."""
        return self.flash(
            q.contiguous(),
            self.k,
            self.v,
            max_seqlen_q=1,
            cu_seqlens_q=self.cu,
            max_seqlen_k=self.capacity,
            seqused_k=self.length(t),
            causal=False,
            block_table=self.table,
            fa_version=4,
        )

    def fill_random(self, std=1.0):
        """For timing at a given context: contents do not change decode cost."""
        self.k.normal_(0, std)
        self.v.normal_(0, std)


class RefKV:
    """Same interface as FlashKV, plain PyTorch attention. Runs on CPU in FP64.

    It exists so the decoder's cache logic, which is shared by both backends,
    can be checked exactly against the training forward on CPU; on GPU the only
    thing that changes is the attention kernel.
    """

    def __init__(self, batch, kv_heads, head_dim, capacity, dtype, device, ring=False):
        self.B, self.K, self.D, self.capacity, self.ring = (
            batch,
            kv_heads,
            head_dim,
            capacity,
            ring,
        )
        self.k = torch.zeros(
            (batch, kv_heads, capacity, head_dim), dtype=dtype, device=device
        )
        self.v = torch.zeros_like(self.k)
        self.ar = torch.arange(capacity, device=device)

    def write(self, k, v, t):
        slot = (torch.remainder(t, self.capacity) if self.ring else t).view(1)
        self.k.index_copy_(2, slot, k.unsqueeze(2).to(self.k.dtype))
        self.v.index_copy_(2, slot, v.unsqueeze(2).to(self.v.dtype))

    def attend(self, q, t):
        n = torch.clamp(t + 1, max=self.capacity) if self.ring else t + 1
        mask = (self.ar < n).view(1, 1, 1, self.capacity)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(2), self.k, self.v, attn_mask=mask, enable_gqa=True
        )
        return out[:, :, 0]

    def fill_random(self, std=1.0):
        self.k.normal_(0, std)
        self.v.normal_(0, std)
