---
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
library_name: peft
license: apache-2.0
tags:
- lora
- text-classification
---

# Support-ticket LoRA adapter

Trained for four-way billing/account/technical/delivery classification. Base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

Rank 8, alpha 16, q_proj/v_proj, dropout 0.05, AdamW 3e-4, three epochs, 320 synthetic examples. Float32 MPS training, microbatch 1 and accumulation 8. Held-out constrained-label accuracy: base 15/68, adapter 64/68.

Load the exact base with Transformers, then use `PeftModel.from_pretrained(base, adapter_directory)`. Use the same prompt and label-selection protocol as `../run.py`; this adapter was not evaluated as a general chat model. The tokenizer and chat template are included.

See `../README.md`, `../results/settings.json` and `../results/failures.json` for the protocol and limitations. Base-model Apache 2.0 license is included in `BASE_MODEL_LICENSE.txt`. Repository code/dataset licensing is described in the root README.
