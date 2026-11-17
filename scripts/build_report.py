"""Generate the four-page submission from actual experiment artifacts."""
import json,html,re
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def esc(s):return html.escape(str(s)).replace('\n','<br/>')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodySmall',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=6,textColor=colors.HexColor('#243247')))
styles.add(ParagraphStyle(name='Tiny',fontName='Helvetica',fontSize=7.5,leading=10,spaceAfter=4))
styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=21,leading=25,textColor=colors.HexColor('#142d4e'),spaceAfter=10))
styles['Heading2'].fontSize=12;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=8;styles['Heading2'].spaceAfter=6
styles.add(ParagraphStyle(name='TinyCompact',fontName='Helvetica',fontSize=7,leading=8.5,spaceAfter=0))
story=[]
def p(text,style='BodySmall'):story.append(Paragraph(text,styles[style]))
def h(text):p(text,'Heading2')
def table(rows,widths,compact=False):
 cells=[[Paragraph(esc(v),styles['TinyCompact' if compact else 'Tiny']) for v in row] for row in rows]
 t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.7,colors.HexColor('#5e7997')),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#d9e0e8')),('TOPPADDING',(0,0),(-1,-1),3 if compact else 5),('BOTTOMPADDING',(0,0),(-1,-1),3 if compact else 5),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,7))
base=read('task1_lora/results/base.json');fine=read('task1_lora/results/finetuned.json');cfg=read('task1_lora/results/settings.json');rag=read('task2_rag/results/summary.json');local={q:read(f'task3_local/results/{q}.json') for q in ['Q4_K_M','Q8_0']}
p('Small-model practical','TitleCustom');p('LoRA / retrieval / local optimisation | 17 November 2026 | Rajat Murhe','Tiny')
p('<link href="https://github.com/rajatmurhe/llm-practical-lora-rag-optimization" color="#245a8d">github.com/rajatmurhe/llm-practical-lora-rag-optimization</link>','Tiny')
p('Measured on an Apple M1 MacBook Air, 8 GB unified memory. All three experiments ran locally. SmolLM2-135M-Instruct is below the 1.5B parameter limit. Code, checkpoints, data, per-example outputs and timing evidence accompany this report.')
h('1. LoRA: route support tickets to four classes')
p('Billing, account, technical and delivery; prompt defines labels A/B/C/D. Accuracy chooses the highest next-token logit among those four labels for both models. This evaluates constrained classification, not free-text output compliance.')
table([['Model','Correct / held-out examples','Accuracy'],['Base SmolLM2-135M',f"{sum(r['correct'] for r in base['predictions'])} / {base['n']}",f"{base['accuracy']:.1%}"],['Same model + LoRA',f"{sum(r['correct'] for r in fine['predictions'])} / {fine['n']}",f"{fine['accuracy']:.1%}"]],[190,190,110])
p('Protocol: 320 synthetic train examples (40 intent cores x eight templates); 68 balanced tests (56 new-template examples plus 12 multi-intent/negation cases). No exact sentence overlap, but intent cores overlap. This narrow phrasing/composition holdout does not establish performance on real tickets. Seed 42; fixed three-epoch run, no test-driven tuning or checkpoint selection.')
p(f"Settings: rank 8, alpha 16 (scale 2), targets q_proj/v_proj; dropout 0.05; AdamW learning rate 3e-4, effective batch 8 (microbatch 1, accumulation 8), three epochs, gradient cap 1.0. Answer-token/EOS loss only; prompt/padding masked. Float32 on {cfg['device']}; {cfg['trainable_parameters']:,} trainable parameters. Epoch losses: {', '.join(f'{x:.3f}' for x in cfg['epoch_losses'])}.")
p('Why: rank 8 and q/v-only adaptation keep capacity and memory modest for four classes; alpha/r=2 gives a practical update scale. 3e-4 makes a short adapter run feasible; three passes and dropout limit template memorisation. These are starting choices, not an optimality claim. QLoRA was not used in this measured run.')
h('Three remaining errors')
failures=read('task1_lora/results/failures.json')
for i,r in enumerate(failures[:3],1):p(f"<b>{i}.</b> {esc(r['text'])}<br/>Gold: <b>{esc(r['label'])}</b>; predicted: <b>{esc(r['prediction'])}</b>. {esc(r['reason'])}")
p('Evidence: task1_lora/results/{base,finetuned,settings,failures}.json; trained adapter and exact splits are included. Failure causes are hypotheses, not causal ablations.','Tiny')
story.append(PageBreak())
p('2. Retrieval from scratch','TitleCustom')
p('Corpus: NASA Apollo 11 Mission Report, 359 PDF pages. MiniLM embeddings via local Ollama; WordPiece, page-bounded chunks, 25% overlap; L2-normalised float32 vectors; NumPy dot-product top five. No LangChain/LlamaIndex. Truncation disabled; embedding context 512 tokens.')
table([['Chunk / overlap','Chunks','Passage recall@5','Page recall@5'],['96 / 24 tokens',rag['96']['chunks'],'8/15 = 53.3%','11/15 = 73.3%'],['256 / 64 tokens',rag['256']['chunks'],'10/15 = 66.7%','11/15 = 73.3%']],[135,65,145,145])
p('A hit requires a retrieved chunk on the gold PDF page with at least min(80, gold passage length) overlapping characters. One annotated passage per query; other valid passages may count as misses. Larger chunks preserved more evidence but lost two narrow matches. Equal page recall and the overlap-based metric suggest a coverage gain, not a strong ranking claim. With 15 queries, one hit changes recall by 6.7 points.')
h('15 authored test questions and source evidence')
qs=read('task2_rag/data/questions.json')
answers=['Armstrong: commander; Collins: CM pilot; Aldrin: LM pilot','Sea of Tranquillity','Inclement weather','USS Hornet','Range zero: 13:32 GMT, 16 July 1969','Not overly tired; adjusting to 1/6 gravity','10 August 1969','Noise, light and cold','Greater attitude/translation manoeuvring','5 hours 4 minutes','No timing; weak voice; 400-Hz hum/noise','Cabin leak below specification','7.2-7.3 Hz','Contamination downstream of injector pre-valve','2307 to 1883 psia (424 psia drop)']
rows=[['Question (full wording/source passages in repository)','Gold PDF page / answer']]
short=['Crew members and roles?','Which lunar sea?','Why move the landing point 215 miles?','Which recovery ship?','What reference time?','Why explore before sleep?','When released from quarantine?','What prevented sleep?','Why more propellant in manual descent?','How long was television operating?','What recorder playback problems?','Why lower oxygen use?','Astronaut seismic signal frequency?','Likely nitrogen leak cause?','Nitrogen pressure change?']
for q,a,t in zip(qs,answers,short): rows.append([f"{q['id']}. {t}",f"{q['page']}: {a}"])
table(rows,[260,230],compact=True)
p('Every correct source passage is preserved verbatim with character offsets in task2_rag/data/questions.json; top-five passages and scores are in results/retrieval_{96,256}.json.','Tiny')
p('<b>Two failures (256):</b> Q3 retrieves recovery map p274 instead of weather p13: repair broken OCR and add lexical matching on 215 plus causal reranking. Q13 retrieves seismometer diagram p339 and adjacent pages instead of 7.2-7.3 Hz on p215: retain section context, retrieve more candidates, then rerank for numeric evidence.','Tiny')
p('<b>BM25</b> adds exact-word/rare-identifier matching that embeddings can blur. Fuse ranks with semantic search. A <b>re-ranker</b> reads question and candidate together to refine relevance; it costs extra inference and cannot rescue evidence outside its candidate pool. Neither extension is in this baseline. The final Llama 3.2 3B QA demo correctly answers USS Hornet with citation to p13. The retained 135M pilot invented a recovery date, showing that good retrieval does not guarantee faithful generation.','Tiny')
story.append(PageBreak())
p('3. Local model and quantisation','TitleCustom')
p('Same SmolLM2-135M-Instruct GGUF publisher (bartowski), Ollama 0.40.6. Serial Q4_K_M and Q8_0; one warmup then the same ten prompts x three repetitions each. Temperature 0, seed 42, context 2048, output cap 96, four CPU threads, GPU offload requested. Training and retrieval finished first.')
rows=[['Setting','Decode tok/s','Peak runner RSS','Ollama GPU allocation','Quality pass count']]
for q,r in local.items():
 resident=next(x for x in r['resident']['models'] if x['name']==r['model'])
 rows.append([q,f"{r['tokens_per_second']:.1f}",f"{r['runner_peak_rss_bytes']/1e6:.1f} MB",f"{resident['size_vram']/1e6:.1f} MB",f"{r['quality_passes_first_repeat']}/10"])
table(rows,[85,85,100,120,100])
p('Rate = total generated tokens / total API decode seconds; excludes prefill, loading and network overhead. Memory is sampled runner RSS at 50 ms and Ollama size_vram. Apple Silicon uses unified RAM; these overlap and must not be added. RSS includes mappings and may miss brief peaks. These numbers are not total-machine memory or dedicated VRAM.')
h('Same ten prompts: observed output quality')
quality=read('task3_local/results/quality.json')
rows=[['Prompt / expected','Q4 / Q8 pass','Observation']]
for row in quality:rows.append([row['task'],row['passes'],row['observation']])
table(rows,[145,65,280],compact=True)
p('Rules and complete outputs are in prompts.json and results/Q*.json. Pass count uses the first repeat; rules check narrow task correctness and formatting. Repeats measure timing variability, not independent examples. The suite is too small to establish general equivalence or superiority.')
h('KV cache and why quantisation saves memory')
p('Attention computes keys and values for every token at every layer. Saving past keys/values avoids recomputing them on each generation step. The KV cache grows with context length and concurrent users, and depends on layers, KV heads, head size and precision. It is separate from model weights.')
p('Quantisation stores approximate weights using fewer bits plus scale metadata. It reduces weight storage and memory traffic, with possible quality loss and kernel/dequantisation overhead. Q4_K_M/Q8_0 use mixed tensor precisions. Weight quantisation alone does not reduce an FP16 KV cache; cache quantisation is a separate option.')
p('See task3_local/results/analysis.md for output comparisons and repeatability. Timings are specific to this small model and machine; they do not predict a 7B model on a 24 GB GPU.','Tiny')
story.append(PageBreak())
p('4. Memory working and serving plan','TitleCustom')
h('7B training: explicit assumptions and arithmetic')
p('Estimates in decimal GB, not measured. Mixed-precision AdamW, no sharding/offload; illustrative microbatch 1, sequence 1024 and checkpointing. Reserve 4 GB activations plus 2 GB workspace/allocator for each method. These reserves depend strongly on implementation and workload.')
p('<b>Full:</b> 2 B weights + 2 B gradients + 4 B FP32 master weights + 8 B Adam moments = 16 B/parameter. 7 x 10^9 x 16 = <b>112 GB</b> persistent; +6 = <b>118 GB</b>. Different optimisers/master-copy policies alter this figure.')
p('<b>LoRA:</b> frozen FP16 base = 7 x 10^9 x 2 = 14 GB. For 32 layers, hidden 4096, MLP 11008, rank 16 on q/k/v/o/gate/up/down:<br/>A = 32 x 16 x [4 x (4096+4096) + 3 x (4096+11008)] = 39,976,960 parameters.<br/>Trainable adapter states = A x 16 = 0.640 GB. Persistent <b>14.64 GB</b>; +6 = <b>20.64 GB</b>. No gradients/moments for frozen base; activations are still needed.')
p('<b>QLoRA:</b> 4-bit frozen base = 7 x 10^9 x 0.5 = 3.50 GB. Add an assumed 0.05 B/base parameter for scales/mixed tensors = 0.35 GB; adapters 0.640 GB. Persistent <b>4.49 GB</b>; +6 = <b>10.49 GB</b>. Plan roughly 12-16 GB with margin. Dequantisation, sequence length and non-quantised tensors vary. This rank-16 example differs from the actual rank-8 q/v run.')
h('20 users on one 24 GB GPU: a capacity plan')
p('Assume a 7B GQA model, 32 layers, 8 KV heads, head dimension 128; 4-bit weights and FP16 KV; at most 2048 total prompt+response tokens per active user. Suitable quantisation kernels and continuous batching are required. This is a hypothetical deployment, not a GPU benchmark.')
p('KV bytes per token = 2 (K,V) x 32 x 8 x 128 x 2 = 131,072 B.<br/>20 x 2048 x 131,072 = <b>5.37 GB</b> KV. Budget: 4.5 GB weights/metadata + 5.37 cache + 4 runtime/graphs/workspace + 3 headroom = <b>16.87 GB</b>. A 4096-token cap doubles cache: total 22.24 GB. A 32-KV-head model would need 21.47 GB for cache alone, so GQA is essential.')
p('Use continuous batching with paged KV allocation, max 20 active sequences, bounded prefill batches, hard token limits and admission control; queue overflow and release completed caches. Users share weights, with separate private KV states. Capacity does not guarantee speed: if a load test reaches 100 output tok/s aggregate, 20 users average 5 tok/s; 150 output tokens take about 30 s after queue/prefill. This rate is a target assumption. Load-test aggregate throughput, time to first token and p95 latency on the actual GPU; lower concurrency or model size if targets fail.')
h('Reproduction and sources')
p('Install requirements.txt and Ollama. Run task1_lora/run.py, task2_rag/prepare.py and run.py, then task3_local/benchmark.py serially. scripts/memory_estimate.py reproduces calculations; scripts/build_report.py rebuilds this PDF from raw artifacts; scripts/verify.py checks split sizes, metrics and report length. See root README for exact commands.','Tiny')
for s in ['Base model: huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct (Apache 2.0).','GGUF variants: huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF.','NASA corpus: nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf (Nov 1969).','PEFT LoRA: huggingface.co/docs/peft/developer_guides/lora; QLoRA: arxiv.org/abs/2305.14314.','Ollama API timings/memory: docs.ollama.com/api/chat and docs.ollama.com/api/ps.'] :p(esc(s),'Tiny')
def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#d9e0e8'));canvas.line(42,37,553,37);canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#506680'));canvas.drawString(42,24,'Small-model practical | Measured results and explicit estimates');canvas.drawRightString(553,24,str(doc.page))
out=ROOT/'output/pdf/report.pdf';out.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(out),title='Small-model practical: LoRA, RAG and local inference',author='Rajat Murhe',pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=35,bottomMargin=47)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
