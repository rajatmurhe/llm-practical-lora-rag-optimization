# Task 1: support-ticket intent classification with LoRA

Run `python task1_lora/run.py` from the repository root. This trains `HuggingFaceTB/SmolLM2-135M-Instruct` with PEFT on MPS if available, otherwise CUDA or CPU. Uses float32, which fits the local 8 GB unified-memory machine. No claim of an actual QLoRA run is made.

## Protocol

Four classes: billing, account, technical, delivery. The prompt defines these and maps them to A/B/C/D. Evaluation chooses the largest next-token logit among those four single-token labels, identically for the base and adapted model. This is constrained classification accuracy, not a claim about free-generation format compliance.

320 synthetic training examples: 40 hand-authored intent cores, eight templates each. 68 test examples: 56 examples from two unseen templates using the first seven intent cores in each class, plus 12 hand-authored multi-intent/negation challenges. Thus no exact sentence overlap exists, but intent cores overlap by design. This is a phrasing/composition holdout, not an intent-disjoint or external dataset. Test labels are balanced (17 per class). No hyperparameter search or checkpoint selection uses the test set; training is fixed at three epochs. Dataset generation has seed 42.

## Settings and rationale

- Rank 8, alpha 16: small adapter with scale alpha/r = 2, sufficient capacity for four labels while limiting training cost.
- Targets `q_proj`, `v_proj`: attention-only adaptation minimizes trainable weights and memory; expanding to MLP and output projections is a future experiment.
- Learning rate 3e-4, AdamW (PyTorch defaults, including weight_decay=0.01), effective batch 8 (microbatch 1, accumulation 8), three epochs, gradient norm cap 1.0, no scheduler or warmup.
- Adapter dropout 0.05 to reduce template overfitting.
- Only the answer token and EOS contribute to loss. Prompt tokens and padding are masked.
- All choices were fixed before measured evaluation. They are practical starting choices, not proven optimal.

See `results/base.json` and `results/finetuned.json` for every prediction; `results/settings.json` for parameter counts, losses, runtime and revision; the report and `results/analysis.md` discuss actual failures.

## 7B memory working (estimates, decimal GB)

Assume mixed-precision AdamW, no offload/sharding, microbatch 1, sequence 1024, checkpointing. Reserve an illustrative 4 GB for activations plus 2 GB for workspaces/allocator. Activation memory is workload-dependent; these reserves are assumptions, not universal constants.

Full tuning: 2 bytes weights + 2 gradients + 4 FP32 master weights + 8 Adam moments = 16 bytes/parameter. 7e9 * 16 = 112 GB persistent; about 118 GB including reserves. Some optimizers omit master copies or use different gradient dtypes, changing this total.

LoRA: for a Llama-style 32-layer model with hidden 4096 and MLP 11008, rank 16 on q/k/v/o/gate/up/down gives A = 32 * 16 * [4*(4096+4096) + 3*(4096+11008)] = 39,976,960 adapter parameters. Frozen FP16 base = 7e9*2 = 14 GB; trainable adapter states A*16 = 0.640 GB. Total = 14.64 GB persistent, about 20.64 GB including reserves. Frozen weights require no optimizer moments or stored gradients, but backprop still needs activations.

QLoRA: frozen 4-bit base = 7e9*0.5 = 3.50 GB. Illustrative quantization metadata/mixed unquantized tensor allowance = 0.05 bytes/base parameter = 0.35 GB. Adapter states = 0.640 GB. Persistent = 4.49 GB; about 10.49 GB including reserves (practical budget roughly 12-16 GB). Dequantization workspace, actual layer shapes, sequence length and implementation can change this substantially.

`python scripts/memory_estimate.py` reproduces the arithmetic. The adapter shape in this 7B illustration differs from the rank-8 q/v-only adapter actually trained here.

The initial padded-batch implementation was stopped for memory pressure before completing an epoch. The final run uses unpadded microbatches of one, accumulation eight, disables KV caching in training, and projects only the last three hidden positions for the two answer targets. This computes the same answer-only causal objective while avoiding full-vocabulary logits at every prompt position. Base/final evaluation projects only the last position. The model was restarted from the unchanged base and seed; no incomplete-run score is reported.
