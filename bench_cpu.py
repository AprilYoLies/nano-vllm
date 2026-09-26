"""CPU-sized benchmark: nanovllm vs transformers batched generate.

Same prompt_token_ids, same forced output length (ignore_eos / min_new_tokens),
so the only difference is the serving strategy:
  - transformers: static batch, one forward per decode step for the whole batch
  - nanovllm:     continuous batching + paged KV cache
"""
import os
import time
from random import randint, seed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm import LLM, SamplingParams

NUM_SEQS = 8
IN_MIN, IN_MAX = 100, 300
OUT_TOKENS = 128          # forced exact output length on both sides


def main():
    seed(0)
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(IN_MIN, IN_MAX))] for _ in range(NUM_SEQS)]
    total_tokens = NUM_SEQS * OUT_TOKENS

    tokenizer = AutoTokenizer.from_pretrained(path)

    # ---------- baseline: transformers batched generate ----------
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    hf.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"                      # batched decode needs left padding
    maxlen = max(len(p) for p in prompt_token_ids)
    input_ids = torch.full((NUM_SEQS, maxlen), tokenizer.pad_token_id)
    attention_mask = torch.zeros(NUM_SEQS, maxlen, dtype=torch.long)
    for i, p in enumerate(prompt_token_ids):             # left pad
        input_ids[i, maxlen - len(p):] = torch.tensor(p)
        attention_mask[i, maxlen - len(p):] = 1
    t = time.time()
    with torch.inference_mode():
        hf.generate(
            input_ids, attention_mask=attention_mask,
            do_sample=True, temperature=0.6,
            max_new_tokens=OUT_TOKENS,
            min_new_tokens=OUT_TOKENS,                   # = ignore_eos: force exact length
        )
    hf_time = time.time() - t
    print(f"transformers : {total_tokens} tok in {hf_time:6.1f}s -> {total_tokens / hf_time:8.1f} tok/s")

    # ---------- nanovllm ----------
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=0.5)
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=OUT_TOKENS)
    llm.generate(prompt_token_ids[:1], sampling_params)  # warmup forward path
    t = time.time()
    llm.generate(prompt_token_ids, [sampling_params] * NUM_SEQS, use_tqdm=False)
    nano_time = time.time() - t
    print(f"nanovllm     : {total_tokens} tok in {nano_time:6.1f}s -> {total_tokens / nano_time:8.1f} tok/s")
    print(f"speedup      : {hf_time / nano_time:.2f}x")


if __name__ == "__main__":
    main()