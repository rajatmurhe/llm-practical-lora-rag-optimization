# Task 3: local SmolLM2 quantization comparison

Run `python task3_local/benchmark.py` after the two GGUF models have been pulled and after training/retrieval have exited. Uses Ollama 0.40.6 on Apple M1, 8 GB unified memory. Same upstream SmolLM2-135M-Instruct model, same GGUF publisher, Q4_K_M versus Q8_0. These are mixed quantization schemes, not uniformly four/eight bits for every tensor.

## Measurement protocol

One model at a time; unload all other resident Ollama models. One warmup per setting, then ten identical prompts repeated three times. Fixed temperature 0, seed 42, 2048-token context, maximum 96 generated tokens, four CPU threads, GPU offload requested. No training or retrieval runs concurrently. Raw API results record output and prompt token counts, nanosecond decode/prefill/load/total durations, and wall time. Cached prefill may vary, so the main rate is decode-only: sum(eval_count) / sum(eval_duration in seconds), including returned completion tokens. Median per-request decode rate is also saved. Prompt lengths and early EOS differ; outputs are not forced to the same length.

Memory: `/api/ps` records Ollama's loaded size and `size_vram`, and psutil polls the sum of Ollama runner RSS every 50 ms during measured requests. RSS can include memory-mapped files, and sampled peaks can miss shorter spikes. On M1, GPU and CPU use shared physical RAM; the reported GPU allocation is not dedicated VRAM. RSS and size_vram overlap and must not be added. No claim of total machine peak or CUDA memory measurement is made.

The ten prompts and deterministic pass rules are stored before measurement. Quality scores use the first repeat, with all three outputs retained to assess repeatability. This is a small diagnostic pass count, not a standard language benchmark. See the report and `results/analysis.md` for output-level observations, not merely aggregate scores.

## KV cache and quantization

For each token, transformer attention computes a key and a value in each layer. During autoregressive generation, keeping past keys/values lets the next token attend to history without recomputing that history. This KV cache grows with active sequences and their context lengths, KV heads, layers and numerical precision. It is separate from model weights; quantizing weights alone does not shrink an FP16 KV cache.

Weight quantization stores an approximate value using fewer bits plus scale/metadata, reducing weight storage and often memory bandwidth. It can lose precision and requires dequantization or specialized kernels. Consequently Q4 need not be faster for a tiny model with overhead dominating. The cache can also be quantized independently with a separate quality tradeoff.

## Serving 20 concurrent users on one 24 GB GPU (estimate)

Assume a 7B GQA model with 32 layers, 8 KV heads, head dimension 128, FP16 cache, 4-bit weights, an appropriate serving engine, short prompts, and a hard maximum of 2048 total input-plus-output tokens per user. This is **not** the 135M benchmark model. One KV token = 2 (K,V) * 32 * 8 * 128 * 2 bytes = 131072 bytes. Twenty full contexts require 20*2048*131072 = 5.37 GB. A multi-head model with 32 KV heads would require 21.47 GB instead, so GQA is an essential assumption.

Budget decimal GB: quantized weights and metadata 4.5 + cache 5.37 + CUDA graphs/activations/workspaces 4 + allocator/headroom 3 = 16.87 GB, below a conservative 24 GB capacity. Start with continuous batching/paged KV allocation, max 20 active sequences, bounded prompt prefill batches, token limits and admission control. Share weights, not users' private KV states. Queue overflow and evict completed caches. A 4096-token limit doubles cache to 10.74 GB and raises this budget to 22.24 GB, leaving little margin; do not silently increase it.

Concurrency is not a throughput guarantee. If a representative load test measures 100 output tokens/s aggregate at 20-way concurrency, users average about 5 tokens/s; a 150-token response takes roughly 30 seconds after prefill/queueing. This is a hypothetical service target, not a measurement. Benchmark time-to-first-token, p95 latency and aggregate decode on the actual GPU; lower active concurrency or use the measured 135M model if latency misses the target. Quantization support and kernel efficiency depend on the GPU/engine.

Quantization order is fixed (Q4 first, Q8 second), without thermal/power control. This fanless laptop may drift with temperature. Differences should be treated as local diagnostic measurements, not controlled causal estimates of the quantization effect.
