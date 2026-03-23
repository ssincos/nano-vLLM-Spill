import gc
import time
import os
import torch
import numpy as np
from random import randint, seed
import json
import multiprocessing as mp

from transformers import AutoTokenizer,AutoModelForCausalLM

from myvllm.engine.llm_engine import LLMEngine as MiniLLM
from myvllm.sampling_parameters import SamplingParams as MiniSamplingParams

config = {
    'max_num_sequences': 256,
    'max_num_batched_tokens': 1024*256,
    'num_gpu_blocks': -1,
    'num_cpu_blocks': -1,
    'block_size': 256,
    'world_size': 1,
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
    'enforce_eager': True,
    'vocab_size': 151936,  # Fixed: was 151643, HF model uses 151936
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,  # Fixed: was 64, should be 128 (hidden_size / num_heads for GQA output)
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,  # Fixed: was 10000, HF uses rope_theta=1000000
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768, # should be >= max_model_length, max position index allowed in rotary embedding
    'ffn_bias': False,  # Fixed: HF Qwen3 doesn't use MLP bias
    'max_model_length': 4096,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,  # Fixed: should match tokenizer.eos_token_id
    'enable_offload': True,
    'cpu_memory_gb': 60 # it could 'auto' or a number in units of GB
    
}

MODEL_NAME = "Qwen/Qwen3-0.6B"
WARMUP_STEPS = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(Nseqs = 256, inlen = 256, outlen = 512, offload = False):
    print(f"\n--- Starting run: Nseqs={Nseqs}, inlen={inlen}, outlen={outlen}, offload={offload} ---")
    config['enable_offload'] = offload
    config['max_num_sequences'] = max(256, Nseqs)   

    # initialize llm engine
    config['enable_offload'] = offload
    llm = MiniLLM(config=config)

    # prepare prompts
    prompt_token_ids = [[randint(0, 10000) for _ in range(inlen)] for _ in range(Nseqs)]
    sampling = MiniSamplingParams(temperature=0.6, ignore_eos=True,  max_tokens=outlen)

    seed(42)

    # warmup
    warmup_prompts = [[randint(0, 10000) for _ in range(16)] for _ in range(2)]
    warmup_sampling = MiniSamplingParams(temperature=0.6, ignore_eos=True, max_tokens=10)
    for _ in range(WARMUP_STEPS):
        llm.generate(warmup_prompts, warmup_sampling)
        cuda_sync()

    print("Warmup finished, starting benchmark...")
    start = time.perf_counter()
    outputs = llm.generate(prompt_token_ids, sampling)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = sum(len(x) for x in outputs["token_ids"])
    latency = end - start

    del llm
    del prompt_token_ids
    del outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    stats = {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }

    return stats


def save_json(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def benchmark_worker(nseqs, inlen, outlen, offload):
    stats = run(nseqs, inlen, outlen, offload)
    payload = {
        "timestamp": time.time(),
        "nseqs": nseqs,
        "inlen": inlen,
        "outlen": outlen,
        "offload": offload,
        "stats": stats
    }
    save_json("benchmark_results.jsonl", payload)
    
    print("\n=== Benchmark Results ===")
    for k, v in stats.items():
         print(f"  {k}: {v:.4f}")



def main():
    print("Running Nano-vllm-Spill benchmark...")
    
    mp.set_start_method('spawn', force=True)

    test_cases = [
        {"nseqs": 256, "inlen": 256, "outlen": 512, "offload": True},
        {"nseqs": 256, "inlen": 256, "outlen": 512, "offload": False},
        {"nseqs": 512, "inlen": 1024, "outlen": 2048, "offload": True},
        {"nseqs": 512, "inlen": 1024, "outlen": 2048, "offload": False},
    ]
    
    for case in test_cases:
        print(f"Starting new test process: {case}")
        # Launch an independent process for this specific test configuration
        p = mp.Process(target=benchmark_worker, kwargs=case)
        p.start()
        # Block the main process and wait for the child process to terminate completely
        p.join() 
        
        if p.exitcode != 0:
            print(f"Warning: Test process exited abnormally with code: {p.exitcode}")
        else:
            print(f"Test process finished successfully. GPU memory perfectly clean.")
            
if __name__ == "__main__":
    main()