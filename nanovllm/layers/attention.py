import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


# CPU version note:
# The upstream CUDA implementation uses triton (for KV-cache store) and
# flash_attn (varlen prefill + paged decode). On CPU we reimplement both
# with plain PyTorch + scaled_dot_product_attention, keeping the exact same
# calling convention so the engine code (slot_mapping / block_tables /
# cu_seqlens) stays identical and learnable. Correspondence:
#
#   flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ..., block_table=None)
#       -> loop over sequences, slice q/k/v by cu_seqlens boundaries, SDPA each
#   flash_attn_varlen_func(..., block_table=block_tables)   # prefix/chunked prefill
#       -> gather k/v from paged cache via block_table, then same per-seq SDPA
#   flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens, block_table)
#       -> gather k/v from paged cache via block_table, single-token SDPA
#
# SDPA shapes: q/k/v are packed (N, H, D) with no batch dim; SDPA interprets that as
# (N=H, L, D) so an (Lq, Lk) mask collides with D. We therefore work in explicit 4D
# (1, H, Lq, D) and pass attn_mask as (1, 1, Lq, Lk). GQA (num_heads > num_kv_heads)
# needs enable_gqa=True so SDPA broadcasts kv over query heads.


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # upstream: triton kernel writing each (token, head, dim) row to k_cache flat slot `slot * D`.
    # CPU: same thing with one advanced-indexing write. slot == -1 (cudagraph padding) is skipped.
    N, num_heads, head_dim = key.shape
    assert slot_mapping.numel() == N
    slot_mapping = slot_mapping.to(key.device)
    valid = slot_mapping != -1
    if valid.any():
        k_cache.view(-1, num_heads, head_dim)[slot_mapping[valid]] = key[valid]
        v_cache.view(-1, num_heads, head_dim)[slot_mapping[valid]] = value[valid]


def _gather_kv_from_cache(cache: torch.Tensor, block_table: list[int] | torch.Tensor, length: int):
    # cache: (num_blocks, block_size, num_kv_heads, head_dim); block_table has >= ceil(length/block_size) ids
    # returns contiguous (length, num_kv_heads, head_dim), trimmed to `length` tokens
    block_size = cache.size(1)
    num_blocks = (length + block_size - 1) // block_size
    if isinstance(block_table, torch.Tensor):
        block_table = block_table.tolist()
    blocks = cache[block_table[:num_blocks]]
    return blocks.reshape(-1, cache.size(2), cache.size(3))[:length]


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None, scale: float):
    # q: (H, Lq, D) head-major; k/v: (KVH, Lk, D). -> returns (H, Lq, D)
    H, KVH = q.size(0), k.size(0)
    q4 = q.unsqueeze(0)     # (1, H, Lq, D)
    k4 = k.unsqueeze(0)     # (1, KVH, Lk, D)
    v4 = v.unsqueeze(0)
    mask4 = attn_mask.unsqueeze(0).unsqueeze(0) if attn_mask is not None else None    # (1, 1, Lq, Lk)
    gqa = H != KVH
    o = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask4, scale=scale, enable_gqa=gqa)
    return o.squeeze(0)     # (H, Lq, D)


def _sdpa_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, scale: float):
    # packed (N, H, D) prefill attention over sequences delimited by cu_seqlens; returns packed (N, H, D).
    # flash_attn's causal mask is aligned at the bottom-right of the q*k matrix: relative query j
    # attends relative keys [0, seqlen_k - seqlen_q + j]. For plain prefill (seqlen_q == seqlen_k)
    # this is standard causal; for chunked prefill (k holds the longer history) it keeps the diagonal aligned.
    o = torch.empty_like(q)
    for i in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        qi = q[qs:qe].transpose(0, 1)                       # (H, q_len, D)
        ki = k[ks:ke].transpose(0, 1)                       # (KVH, k_len, D)
        vi = v[ks:ke].transpose(0, 1)
        q_len, k_len = qe - qs, ke - ks
        offset = k_len - q_len                              # keys ahead of queries (0 for plain prefill)
        # bottom-right aligned causal: relative query j attends relative keys [0, offset + j]
        k_rel = torch.arange(k_len, device=q.device).unsqueeze(0)          # (1, k_len)
        q_rel = torch.arange(q_len, device=q.device).unsqueeze(1)          # (q_len, 1)
        mask = k_rel <= (q_rel + offset)                                   # (q_len, k_len)
        o[qs:qe] = _sdpa(qi, ki, vi, mask, scale).transpose(0, 1)
    return o


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is None:    # no paged cache (plain prefill / warmup)
                o = _sdpa_varlen(q, k, v, context.cu_seqlens_q, context.cu_seqlens_k, self.scale)
            else:    # prefix cache / chunked prefill: q is packed, kv lives in the paged cache
                o = torch.empty_like(q)
                for i in range(len(context.cu_seqlens_q) - 1):
                    qs, qe = int(context.cu_seqlens_q[i]), int(context.cu_seqlens_q[i + 1])
                    seqlen_k = int(context.cu_seqlens_k[i + 1]) - int(context.cu_seqlens_k[i])
                    # gather this sequence's full kv history from the paged cache via its block_table row
                    ki = _gather_kv_from_cache(k_cache, context.block_tables[i], seqlen_k).transpose(0, 1)   # (KVH, seqlen_k, D)
                    vi = _gather_kv_from_cache(v_cache, context.block_tables[i], seqlen_k).transpose(0, 1)
                    qi = q[qs:qe].transpose(0, 1)    # (H, q_len, D)
                    q_len = qe - qs
                    start = seqlen_k - q_len         # absolute position of the first new token
                    # key at absolute pos p is visible to relative query j iff p <= start + j
                    positions = torch.arange(seqlen_k, device=q.device).unsqueeze(0)     # (1, k_len)
                    q_rel = torch.arange(q_len, device=q.device).unsqueeze(1)            # (q_len, 1)
                    mask = positions <= q_rel + start
                    o[qs:qe] = _sdpa(qi, ki, vi, mask, self.scale).transpose(0, 1)
        else:    # decode: one query token per sequence, kv entirely in the paged cache
            o = torch.empty_like(q)
            for i, seq_len in enumerate(context.context_lens.tolist()):
                qi = q[i].unsqueeze(1)                           # (H, 1, D)
                ki = _gather_kv_from_cache(k_cache, context.block_tables[i], int(seq_len)).transpose(0, 1)
                vi = _gather_kv_from_cache(v_cache, context.block_tables[i], int(seq_len)).transpose(0, 1)
                o[i] = _sdpa(qi, ki, vi, None, self.scale).squeeze(1)

        # caller merges heads: (N, H, D) -> (N, H*D) via o.flatten(1, -1);
        # upstream flash_attn returns (N, H, D) as well.
        return o