# Retrieval results

| WordPiece chunk size | Overlap | Chunks | Passage recall@5 | Source-page recall@5 |
|---|---:|---:|---:|---:|
|96|24|2012|8/15 = 53.33%|11/15 = 73.33%|
|256|64|834|10/15 = 66.67%|11/15 = 73.33%|

256 tokens worked better on this small fixed suite: it preserved broader context, including the nitrogen-leak cause and pressure values, and the recovery ship passage. It also lost two questions that 96-token chunks retrieved (lunar sea and why EVA preceded sleep), consistent with dilution of a narrow semantic signal in larger chunks. The equal page recall shows this is primarily a passage-coverage gain. The overlap-based proxy also favors broader spans, so do not attribute the entire difference to better semantic ranking. Fifteen queries are too few for a strong general claim.

## Two actual failures at 256 tokens

1. Q3, ocean landing moved 215 miles. Gold: PDF page 13, inclement weather. Top hit: recovery-map figure, PDF page 274 (score 0.508); other top-five pages are 49, 115, 55, 16. The retriever matches landing geometry more strongly than the reason for relocation, and OCR breaks the phrase into `In-clement wea ther`. Better OCR/dehyphenation plus hybrid BM25/embedding retrieval on `215` and reranking question-passage pairs could recover the causal passage.
2. Q13, frequency of astronaut-generated seismic signals. Gold: PDF page 215, 7.2-7.3 Hz. Top hit: seismometer deployment diagram, PDF page 339 (score 0.469); other hits are 214, 216, 216, 217. Related apparatus and adjacent seismic discussion outrank the numeric evidence. Index coherent subsections with headings, retrieve a larger pool and re-rank for a frequency-bearing answer. OCR repair helps the broken words and numeric units.

Q11 also illustrates a labeling limitation: sections on recorder/voice anomalies are retrieved on pages 294-296, while the single gold passage is on page 160. A reviewer should check whether these contain equivalent evidence before treating the miss as an actual QA failure. Future labels should enumerate all valid passages.

## Per-question audit

Use `data/questions.json` for all exact source passages. `retrieval_96.json` and `retrieval_256.json` contain every ranked chunk, score, page, overlap and hit. Source passages and offsets are frozen by `prepare.py`; changing chunk size does not change labels.
