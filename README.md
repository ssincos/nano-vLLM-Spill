# Nano-vLLM-Spill

Extend [nano-vLLM] & [Mini-vLLM] with CPU offload — spill to CPU when GPU memory runs out.

In the original Nano-vLLM and Mini-vLLM architectures, when the GPU runs out of free blocks to allocate, the system takes a "brute-force" approach: it preempts the sequence, completely drops its KV cache, moves it back to the `waiting` queue, and relies on Recomputation later. 

In real-world production systems (like the official vLLM), the standard procedure for handling OOM (Out of Memory) is to Offload (Spill) the KV cache to CPU memory.


Note: To use the offload feature, turn on the enable_offload: True parameter in your configuration dict

### Core Updates

To implement CPU offloading, the engine's core components were extended:

* **`block_manager`**: Added `can_swapin`, `swapin`, and `swapout` functions. Introduced the management of `cpu_block` pools and implemented the state/metadata transfer mechanism between CPU and GPU.
* **`scheduler`**: Refined the overall scheduling logic by adding a pipeline to handle and transition sequences in the `swapped` state.
* **`model_runner`**: Added CPU memory capacity detection and dynamic allocation for CPU KV cache blocks. Implemented the `execute_swap` function to perform the actual asynchronous physical data transfer across the PCIe.

## Benchmark

See `benchmark.py`.

**Test Configuration:**
- Hardware: A100 (80GB)
- Model: Qwen3-0.6B

**Static Inference - Test 1:**

- Total Requests: 384 sequences
- Input Length: Randomly sampled 256 tokens
- Output Length: 640 tokens
  
|            | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| offload on           | 156,800     | 224.02    | 699.94               |
| offload off      | 245,760    | 125.99    | 1950.61               |

**Static Inference - Test 2:**

- Total Requests: 512 sequences
- Input Length: Randomly sampled 1024 tokens
- Output Length: 2048 tokens
  
|            | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| offload on           | 1,048,576     | 2575.10    | 407.20               |
| offload off      | 1,048,576     | 1586.98    | 660.74               |

**Static Inference - Test 3:**

- Total Requests: 64 sequences
- Input Length: Randomly sampled 2048 tokens
- Output Length: 4096 tokens
  
|            | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| offload on           | 163,840     | 861.12    | 190.26               |
| offload off      | 262,144     | 1206.27    | 217.32               |


## Project Structure

```
myvllm/
├── src/
│   └── myvllm/           # Core vLLM implementation
│       ├── models/       # Model implementations
│       ├── engine/       # LLM engine logic, including sequence definition for input prompts, block management for KV cache management for GPU, scheduler for iteration-based scheduling of sequences, runner for actual implementation of running prefilling and decoding, and engine for generation API interface
│       ├── layers/       # Components for model/
│       ├── utils/        # context
│       └── sampling_parameters.py
├── main.py              # Full inference demo
└── benchmark_decoding.py     # recomputation vs. offload 
```

## Requirements

- Python ≥3.11, < 3.12
- CUDA-capable GPU
- Dependencies: `transformers`, `torch`, `xxhash` (managed by uv)