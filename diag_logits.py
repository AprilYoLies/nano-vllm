"""Is the 66% token match a bf16 numerics gap or a real bug? Compare first-token logits."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)
messages = [{"role": "user", "content": "introduce yourself in one sentence"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# transformers reference logits
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
hf.eval()
with torch.inference_mode():
    hf_logits = hf(input_ids).logits[0, -1].float()
print("hf top5:", torch.topk(hf_logits, 5).indices.tolist())

# nanovllm logits
import torch.distributed as dist
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context
from transformers import AutoConfig

hf_config = AutoConfig.from_pretrained(path)
store = dist.FileStore("/tmp/nanovllm-diag", 1)    # no TCP, no MASTER_ADDR needed
dist.init_process_group("gloo", store=store, rank=0, world_size=1)
torch.set_default_dtype(hf_config.dtype)
torch.set_default_device("cpu")
model = Qwen3ForCausalLM(hf_config)
load_model(model, path)
model.eval()
torch.set_default_device("cpu")
with torch.inference_mode():
    n = input_ids.size(1)
    pos = torch.arange(n)
    # plain prefill: full sequence attends itself causally, no paged cache involved
    set_context(True,
                cu_seqlens_q=torch.tensor([0, n]), cu_seqlens_k=torch.tensor([0, n]),
                max_seqlen_q=n, max_seqlen_k=n,
                slot_mapping=torch.arange(n))
    # compute_logits already selects the last token per sequence -> (1, vocab)
    nano_logits = model.compute_logits(model(input_ids[0], pos))[0].float()
print("nano top5:", torch.topk(nano_logits, 5).indices.tolist())
print("max abs diff:", (hf_logits - nano_logits).abs().max().item())
print("hf argmax:", hf_logits.argmax().item(), " nano argmax:", nano_logits.argmax().item())