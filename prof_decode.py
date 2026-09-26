"""Profile one decode step on CPU to see where time goes (bs=8, 300-token contexts)."""
import os
import time

import torch
import torch.distributed as dist

from nanovllm.utils.context import set_context
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from transformers import AutoConfig
from nanovllm.layers.attention import store_kvcache, _gather_kv_from_cache, _sdpa

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
hf_config = AutoConfig.from_pretrained(path)
store = dist.FileStore("/tmp/nanovllm-prof", 1)
dist.init_process_group("gloo", store=store, rank=0, world_size=1)
torch.set_default_dtype(hf_config.dtype)
torch.set_default_device("cpu")
model = Qwen3ForCausalLM(hf_config)
load_model(model, path)
model.eval()
torch.set_default_device("cpu")

H = hf_config.num_attention_heads
KVH = hf_config.num_key_value_heads
D = getattr(hf_config, "head_dim", 0) or hf_config.hidden_size // H    # Qwen3-0.6B: head_dim=128 != 1024/16
n_layers = hf_config.num_hidden_layers
block_size = 256
n_blocks = 1000
kv = torch.empty(2, n_layers, n_blocks, block_size, KVH, D)
layer_id = 0
for m in model.modules():
    if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
        m.k_cache = kv[0, layer_id]
        m.v_cache = kv[1, layer_id]
        layer_id += 1
print("layers wired:", layer_id)

bs = 8
ctx_lens = [300] * bs
block_tables = torch.tensor([[i * 2, i * 2 + 1] for i in range(bs)], dtype=torch.int32)
slot = block_tables[:, -1] * block_size + 300 % block_size - 1
q = torch.randn(bs, H, D)
k = torch.randn(bs, KVH, D)
v = torch.randn(bs, KVH, D)
attn0 = model.model.layers[0].self_attn.attn
qkv_proj = model.model.layers[0].self_attn.qkv_proj
ln0 = model.model.layers[0].input_layernorm
hidden = torch.randn(bs, hf_config.hidden_size)


def one_step():
    set_context(False, slot_mapping=slot, context_lens=torch.tensor(ctx_lens), block_tables=block_tables)
    t0 = time.perf_counter()
    qkv = qkv_proj(ln0(hidden))
    t1 = time.perf_counter()
    # Qwen3-0.6B: H=16, KVH=8, head_dim=128 -> qkv out = (16 + 2*8)*128 = 4096
    qq, kk, vv = qkv.split([H * D, KVH * D, KVH * D], dim=-1)
    store_kvcache(kk.view(bs, KVH, D), vv.view(bs, KVH, D), attn0.k_cache, attn0.v_cache, slot)
    t2 = time.perf_counter()
    o = torch.empty_like(q)
    for i, L in enumerate(ctx_lens):
        qi = q[i].unsqueeze(1)
        ki = _gather_kv_from_cache(attn0.k_cache, block_tables[i], L).transpose(0, 1)
        vi = _gather_kv_from_cache(attn0.v_cache, block_tables[i], L).transpose(0, 1)
        o[i] = _sdpa(qi, ki, vi, None, D**-0.5).squeeze(1)
    t3 = time.perf_counter()
    return t1 - t0, t2 - t1, t3 - t2


one_step()  # warmup
ts = [one_step() for _ in range(20)]
avg = [sum(x) / len(ts) for x in zip(*ts)]
print(f"layer0  qkv_proj: {avg[0]*1000:6.2f}ms   store_kv: {avg[1]*1000:6.3f}ms   attn gather+sdpa: {avg[2]*1000:6.2f}ms")

ids = torch.zeros(bs, dtype=torch.int64)
pos = torch.arange(300, 300 + bs)
set_context(False, slot_mapping=slot, context_lens=torch.tensor(ctx_lens), block_tables=block_tables)
for _ in range(3):
    hidden_out = model(ids, pos)
t = time.perf_counter()
for _ in range(10):
    hidden_out = model(ids, pos)
tok_fwd = (time.perf_counter() - t) / 10
t = time.perf_counter()
for _ in range(10):
    logits = model.compute_logits(hidden_out)
tok_head = (time.perf_counter() - t) / 10
print(f"full decode fwd (bs={bs}): {tok_fwd*1000:.1f}ms   lm_head (bs,{hf_config.vocab_size}): {tok_head*1000:.1f}ms")