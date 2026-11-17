# Task 2: Apollo 11 retrieval from scratch

359-page public NASA Apollo 11 Mission Report (November 1969). `prepare.py` downloads the PDF, normalizes extracted whitespace, and writes fifteen manually authored questions with one exact source passage and its offsets per question. PDF page indices are one-based; they differ from printed chapter page numbers. No retrieval results were used to write the questions.

`run.py` creates page-bounded overlapping chunks using MiniLM's WordPiece tokenizer. Sizes are 96 and 256 tokens, with 25% overlap (24 and 64 tokens). It invokes local `all-minilm` through Ollama's embedding API, normalizes vectors, and retrieves five chunks by NumPy cosine similarity. There is no RAG framework. A 512-token embedding runtime context is explicitly set; truncation is disabled, so oversized inputs fail rather than silently dropping evidence. Character offsets keep gold labels stable across chunk sizes.

## Metric

Question hit-rate/recall@5: a success requires one of the first five chunks to have the same PDF page and overlap the annotated source passage by at least min(80, passage length) characters. This is a documented proxy for passage relevance, not semantic answer correctness. It may count partial evidence; source-page hit rate is also saved as a looser diagnostic. Only one source passage is annotated, so a correct duplicate elsewhere can count as a miss. There are 15 questions; one question changes recall by 6.67 percentage points. No significance claim is made.

All fifteen questions, source passages, five retrieved chunks, scores and overlap counts are saved for both sizes. See `results/analysis.md` for the measured comparison and actual misses. A QA command passes retrieved text to local Llama 3.2 3B with instructions to cite sources and abstain when unsupported:

```bash
python task2_rag/run.py --ask 'Which ship recovered the Apollo 11 astronauts?'
```

This produces an actual generated answer and the underlying passages in `results/qa_example.json`. Retrieval accuracy is evaluated separately from generated answer quality. A 135M generator can hallucinate even with correct context; inspect its claims against the saved sources.

## What BM25 and re-ranking add

Embedding search finds meaning even when a question and passage use different wording. BM25 adds an exact-word signal: rare words, identifiers, error numbers and technical abbreviations can matter more than broad semantic resemblance. Combine independent BM25 and embedding ranks (for example reciprocal rank fusion); do not add their raw, differently scaled scores blindly.

A re-ranker reads the question and each candidate passage together, giving a more precise relevance score than independently computed embeddings. Retrieve perhaps 30-50 candidates cheaply, then re-rank to five. It costs additional model computation and cannot rescue evidence absent from its candidate pool. Neither BM25 nor a re-ranker is implemented in this embedding-only baseline.

Extraction preserves noisy OCR and broken words intentionally. Improving OCR, joining line-break hyphenation, retaining table rows, and indexing section titles would be useful next steps. Whole-document tokenization warnings concern preprocessing only; the encoder receives bounded chunks.

The initial SmolLM2 Q8 generator correctly named USS Hornet but hallucinated an April 17 recovery date and omitted page citations. Its unedited output is retained as `results/qa_small_model_failure.json`. The final QA demonstration uses the already installed local Llama 3.2 3B model (`ollama pull llama3.2:3b` if needed); retrieval remains unchanged. This generator change is outside the two-setting inference benchmark.
