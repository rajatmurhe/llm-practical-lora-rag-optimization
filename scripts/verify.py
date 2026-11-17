"""Integrity checks on the measured submission (no expensive model re-runs)."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
train=[json.loads(s) for s in (ROOT/'task1_lora/data/train.jsonl').read_text().splitlines()]
test=[json.loads(s) for s in (ROOT/'task1_lora/data/test.jsonl').read_text().splitlines()]
assert len(train)==320 and len(test)==68
assert not {r['text'] for r in train}&{r['text'] for r in test}
assert all(sum(x['label']==label for x in test)==17 for label in ['billing','account','technical','delivery'])
for tag in ['base','finetuned']:
 x=read(f'task1_lora/results/{tag}.json');assert x['n']==len(test)==len(x['predictions'])
 assert {(r['text'],r['label']) for r in x['predictions']}=={(r['text'],r['label']) for r in test}
 assert all(r['correct']==(r['prediction']==r['label']) for r in x['predictions'])
 assert abs(x['accuracy']-sum(r['correct'] for r in x['predictions'])/len(test))<1e-10
assert (ROOT/'task1_lora/adapter/adapter_model.safetensors').stat().st_size>1000
assert len(read('task1_lora/results/failures.json'))>=3
prov=read('task2_rag/data/provenance.json');assert prov['pages']>=30
assert prov['sha256']==hashlib.sha256((ROOT/'task2_rag/data/apollo11.pdf').read_bytes()).hexdigest()
qs=read('task2_rag/data/questions.json');pages=read('task2_rag/data/pages.json');assert len(qs)==15
for q in qs:assert pages[q['page']-1]['text'][q['start']:q['end']].strip()==q['source_passage']
for size in [96,256]:
 x=read(f'task2_rag/results/retrieval_{size}.json');vec=np.load(ROOT/f'task2_rag/results/vectors_{size}.npy')
 chunks=read(f'task2_rag/results/chunks_{size}.json');assert len(chunks)==vec.shape[0]==x['chunks']
 np.testing.assert_allclose(np.linalg.norm(vec,axis=1),1,atol=1e-5)
 assert abs(x['recall_at_5']-sum(q['hit'] for q in x['questions'])/15)<1e-10
 for q in x['questions']:
  assert len(q['retrieved'])==5
  assert q['hit']==any(c['gold_overlap_chars']>=min(80,q['end']-q['start']) for c in q['retrieved'])
  assert all(q['retrieved'][i]['score']>=q['retrieved'][i+1]['score'] for i in range(4))
qmod=__import__('importlib.util',fromlist=['util']); spec=qmod.spec_from_file_location('benchmark',ROOT/'task3_local/benchmark.py');module=qmod.module_from_spec(spec);spec.loader.exec_module(module)
for quant in ['Q4_K_M','Q8_0']:
 x=read(f'task3_local/results/{quant}.json');rows=x['rows'];assert len(rows)==30
 assert {(r['repeat'],r['id']) for r in rows}=={(i,j) for i in range(3) for j in range(1,11)}
 assert all(r['pass']==module.score(module.PROMPTS[r['id']-1],r['response']) for r in rows)
 expected=sum(r['eval_count'] for r in rows)/(sum(r['eval_duration'] for r in rows)/1e9)
 assert abs(expected-x['tokens_per_second'])<1e-8
 assert x['runner_peak_rss_bytes']>0 and x['quantization']==quant
 assert x['quality_passes_first_repeat']==sum(r['pass'] for r in rows if r['repeat']==0)
qa=read('task2_rag/results/qa_example.json');assert len(qa['sources'])==5 and qa['answer'].strip()
pdf=PdfReader(ROOT/'output/pdf/report.pdf');assert len(pdf.pages)==4, f'Report has {len(pdf.pages)} pages'
assert all(len(p.extract_text())>400 for p in pdf.pages)
print('PASS: splits, predictions, sources, vector norms, metrics, 60 timing records, QA output, adapter, four-page report')
