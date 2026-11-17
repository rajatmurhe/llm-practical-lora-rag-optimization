# Small-model practical: LoRA, retrieval, local inference

Three executed experiments on an Apple M1 MacBook Air (8 GB unified memory). The four-page report is `output/pdf/report.pdf`. Raw outputs and reproducible scripts live beside each task. No LangChain, LlamaIndex, paid API, or remote GPU is used.

## Measured results

| Experiment | Result |
|---|---|
| LoRA classification, 68 held-out cases | Base 22.1%; adapted 94.1% |
| RAG passage recall@5, 15 questions | 96 tokens: 53.3%; 256 tokens: 66.7% |
| Local decode, Q4_K_M / Q8_0 | 201.2 / 186.5 tokens per second |
| Peak runner RSS, Q4_K_M / Q8_0 | 238.2 / 301.4 MB |

Quantized model quality passes are 1/10 for both under the declared mixed content/format rules; Q8 nevertheless improves arithmetic and the Mars answer. See full output-level analysis. The final RAG generator is Llama 3.2 3B; the failed 135M generator output is retained.

## Reproduce

Python 3.11; install Ollama separately and start its local service (`ollama serve`) if it is not running. Model downloads require internet access. Allow several GB of RAM and about 2 GB of free disk space. Do not run the benchmark alongside training.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull all-minilm
ollama pull llama3.2:3b
ollama pull hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M
ollama pull hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0
HF_HUB_DISABLE_XET=1 python task1_lora/run.py
python task2_rag/prepare.py
python task2_rag/run.py
python task2_rag/run.py --ask 'Which ship recovered the Apollo 11 astronauts?'
python task3_local/benchmark.py
python scripts/memory_estimate.py
python scripts/audit_results.py
python scripts/local_analysis.py
python scripts/build_report.py
python scripts/verify.py
```

The benchmark unloads resident Ollama models between settings. On other systems, use CPU or an appropriate GPU backend; local timings will differ. Task 1 uses float32 LoRA (not QLoRA) for MPS compatibility. The requested hypothetical 7B QLoRA calculation is separate.

## Evidence and scope

- `task1_lora/data/`: fixed synthetic training and held-out tests; `results/`: both prediction files, settings and losses; `adapter/`: trained PEFT weights.
- `task2_rag/data/`: 359-page NASA report, extraction, SHA256 provenance, 15 questions with exact source passages; `results/`: every top-5 retrieval and reusable NumPy indexes.
- `task3_local/prompts.json`: same ten prompts and declared scoring rules; `results/`: three repetitions per quantization with outputs, nanosecond timing, model digests, allocations and sampled runner RSS.
- `output/pdf/report.pdf`: at most four pages. Task folders contain fuller explanations and limitations.

Synthetic template holdout measures a narrow task, not real-world support accuracy. Retrieval is evaluated against one annotated passage per question, so other answer-bearing passages may count as misses. Local quality checks are a small diagnostic suite, not a general model ranking. Estimates for 7B training and 20-user serving are calculations, not measurements.

## Sources

- Base model and Apache 2.0 license: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- GGUF variants: https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF
- NASA Apollo 11 Mission Report, November 1969: https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf
- PEFT LoRA: https://huggingface.co/docs/peft/developer_guides/lora
- QLoRA paper: https://arxiv.org/abs/2305.14314
- Ollama timing API: https://docs.ollama.com/api/chat
- Ollama resident model memory: https://docs.ollama.com/api/ps

Original experiment code and authored datasets are MIT licensed. Upstream model weights retain their upstream licenses. NASA material retains its source attribution; NASA insignia are not licensed by this repository.
