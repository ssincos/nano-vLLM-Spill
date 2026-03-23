# Nano-vLLM-Spill

A custom implementation of vLLM inference engine with attention mechanism benchmarks, based on Nano-vLLM but with self-contained paged attention and flash attention implementation. 

## Quickstart

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run the main inference engine
uv run python main.py

# Run prefilling benchmark
uv run python benchmark_prefilling.py

# Run decoding benchmark
uv run python benchmark_decoding.py
```

To run multi-GPU setting, simply change world_size to n > 1 in config in main.py

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