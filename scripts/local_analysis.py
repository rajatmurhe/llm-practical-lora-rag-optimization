"""Summarize measured timings and the ten inspected prompt outputs."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
models={q:read(f'task3_local/results/{q}.json') for q in ['Q4_K_M','Q8_0']}
tasks=['Capital: Paris','17 + 25: 42','JSON: Ada, age 36','Reverse planet: tenalp','Extract email address','Sentiment: positive','Translate cat: chat','Repeat blue three times','Red Planet: Mars','Summarize storm and delays']
observations=['Both name Paris but add prose; capital rubric permits extra text.','Q4 says 32; Q8 gets 42 but fails the number-only rule.','Both generate Python instead of the requested JSON object.','Neither reverses the letters; Q4 degenerates into repetition.','Q4 writes an email; Q8 refuses to extract the supplied address.','Both identify positive sentiment but fail the one-word rule.','Both fabricate French words and explanations.','Both give unrelated sentence rewrites instead of three blues.','Q4 is tautological; Q8 names Mars but fails the name-only rule.','Both retain storm/bridge/rerouting but omit commuters being late.']
quality=[]
for i,(task,observation) in enumerate(zip(tasks,observations),1):
 rows={q:next(r for r in x['rows'] if r['repeat']==0 and r['id']==i) for q,x in models.items()}
 quality.append({'id':i,'task':task,'passes':' / '.join('pass' if rows[q]['pass'] else 'fail' for q in models),'observation':observation,'outputs':{q:r['response'] for q,r in rows.items()}})
(ROOT/'task3_local/results/quality.json').write_text(json.dumps(quality,indent=2))
lines=['# Local inference results','','Q4_K_M is listed first in pass pairs. Fixed order and uncontrolled thermal conditions limit causal interpretation.','', '|Setting|Aggregate decode tok/s|Per-repeat aggregate tok/s|Peak runner RSS MB|Ollama allocation MB|First-repeat passes|','|---|---:|---|---:|---:|---:|']
stabilities=[]
for q,x in models.items():
 rates=[]
 for repeat in range(3):
  rows=[r for r in x['rows'] if r['repeat']==repeat];rates.append(sum(r['eval_count'] for r in rows)/(sum(r['eval_duration'] for r in rows)/1e9))
 mem=next(m for m in x['resident']['models'] if m['name']==x['model'])
 lines.append(f"|{q}|{x['tokens_per_second']:.2f}|{', '.join(f'{r:.2f}' for r in rates)}|{x['runner_peak_rss_bytes']/1e6:.2f}|{mem['size_vram']/1e6:.2f}|{x['quality_passes_first_repeat']}/10|")
 stable=sum(len({r['response'] for r in x['rows'] if r['id']==i})==1 for i in range(1,11));stabilities.append(f"{q}: {stable}/10 prompts produced identical output across three repetitions.")
lines+=['']+stabilities+['','## Ten-prompt observations','']
for row in quality:lines.append(f"- {row['task']} ({row['passes']}): {row['observation']}")
lines+=['','Both quantizations pass 1/10 under the predeclared rules. This aggregate hides content improvements: Q8 computes 42 and recalls Mars, but adds forbidden extra text. Both also get positive sentiment right while violating its one-word instruction. The capital rubric checks factual presence only and permits extra words; the other short-answer rules are stricter. Do not interpret this mixed diagnostic pass count as a general capability score.','', 'Q4 uses less model memory; observed speed differences combine quantization, different completion lengths, and possible thermal drift. A 135M model has weak instruction-following and factual/reasoning capacity even at Q8. These measurements do not isolate every difference as a consequence of quantization.','', 'A pilot RSS collector returned zero because this Ollama version names its worker llama-server (with a hyphen). The collector was corrected and the full serial benchmark was rerun; only that rerun is reported. Peak RSS is sampled, includes mappings, and overlaps unified GPU memory.']
(ROOT/'task3_local/results/analysis.md').write_text('\n'.join(lines)+'\n')
