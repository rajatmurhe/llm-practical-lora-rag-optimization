# Local inference results

Q4_K_M is listed first in pass pairs. Fixed order and uncontrolled thermal conditions limit causal interpretation.

|Setting|Aggregate decode tok/s|Per-repeat aggregate tok/s|Peak runner RSS MB|Ollama allocation MB|First-repeat passes|
|---|---:|---|---:|---:|---:|
|Q4_K_M|201.16|200.42, 198.88, 204.27|238.24|167.12|1/10|
|Q8_0|186.46|187.20, 185.38, 186.82|301.42|206.48|1/10|

Q4_K_M: 10/10 prompts produced identical output across three repetitions.
Q8_0: 10/10 prompts produced identical output across three repetitions.

## Ten-prompt observations

- Capital: Paris (pass / pass): Both name Paris but add prose; capital rubric permits extra text.
- 17 + 25: 42 (fail / fail): Q4 says 32; Q8 gets 42 but fails the number-only rule.
- JSON: Ada, age 36 (fail / fail): Both generate Python instead of the requested JSON object.
- Reverse planet: tenalp (fail / fail): Neither reverses the letters; Q4 degenerates into repetition.
- Extract email address (fail / fail): Q4 writes an email; Q8 refuses to extract the supplied address.
- Sentiment: positive (fail / fail): Both identify positive sentiment but fail the one-word rule.
- Translate cat: chat (fail / fail): Both fabricate French words and explanations.
- Repeat blue three times (fail / fail): Both give unrelated sentence rewrites instead of three blues.
- Red Planet: Mars (fail / fail): Q4 is tautological; Q8 names Mars but fails the name-only rule.
- Summarize storm and delays (fail / fail): Both retain storm/bridge/rerouting but omit commuters being late.

Both quantizations pass 1/10 under the predeclared rules. This aggregate hides content improvements: Q8 computes 42 and recalls Mars, but adds forbidden extra text. Both also get positive sentiment right while violating its one-word instruction. The capital rubric checks factual presence only and permits extra words; the other short-answer rules are stricter. Do not interpret this mixed diagnostic pass count as a general capability score.

Q4 uses less model memory; observed speed differences combine quantization, different completion lengths, and possible thermal drift. A 135M model has weak instruction-following and factual/reasoning capacity even at Q8. These measurements do not isolate every difference as a consequence of quantization.

A pilot RSS collector returned zero because this Ollama version names its worker llama-server (with a hyphen). The collector was corrected and the full serial benchmark was rerun; only that rerun is reported. Peak RSS is sampled, includes mappings, and overlaps unified GPU memory.
