"""Embedding + NumPy retrieval, no orchestration framework. Local Ollama is model runtime only."""
import json,re,argparse
from pathlib import Path
import numpy as np
import requests
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
API='http://localhost:11434/api'
def post(route,body):
 r=requests.post(API+'/'+route,json=body,timeout=300); r.raise_for_status(); return r.json()
def embed(texts):
 out=[]
 for i in range(0,len(texts),32):
  x=post('embed',{'model':'all-minilm','input':texts[i:i+32],'truncate':False,'options':{'num_ctx':512},'keep_alive':'5m'})
  out.extend(x['embeddings'])
  if i%320==0: print('embedded',i,'/',len(texts),flush=True)
 x=np.asarray(out,dtype=np.float32); return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def build(size):
 pages=json.loads((ROOT/'data/pages.json').read_text())
 tok=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
 chunks=[]
 for page in pages:
  text=page['text']; offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
  for start in range(0,len(offsets),size-size//4):
   block=offsets[start:start+size]
   if not block: continue
   a,b=block[0][0],block[-1][1]
   chunks.append({'id':len(chunks),'page':page['page'],'start':a,'end':b,'text':text[a:b]})
   if start+size>=len(offsets): break
 vectors=embed([c['text'] for c in chunks]); np.save(ROOT/f'results/vectors_{size}.npy',vectors)
 (ROOT/f'results/chunks_{size}.json').write_text(json.dumps(chunks,indent=2)); return chunks,vectors

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--ask');parser.add_argument('--size',type=int,default=256);parser.add_argument('--generator',default='llama3.2:3b'); args=parser.parse_args()
 if args.ask:
  chunks=json.loads((ROOT/f'results/chunks_{args.size}.json').read_text()); vec=np.load(ROOT/f'results/vectors_{args.size}.npy')
  q=embed([args.ask])[0]; top=np.argsort(-(vec@q))[:5]
  context='\n\n'.join(f'[PDF page {chunks[i]["page"]}] {chunks[i]["text"]}' for i in top)
  answer=post('chat',{'model':args.generator,'messages':[{'role':'system','content':'Answer only using the supplied passages. Cite PDF page numbers. If the answer is absent, say you do not know.'},{'role':'user','content':context+'\n\nQuestion: '+args.ask}],'stream':False,'options':{'temperature':0,'num_ctx':4096,'num_predict':160}})
  result={'generator':args.generator,'question':args.ask,'sources':[chunks[i] for i in top],'answer':answer['message']['content']}
  print(json.dumps(result,indent=2)); (ROOT/'results/qa_example.json').write_text(json.dumps(result,indent=2));return
 questions=json.loads((ROOT/'data/questions.json').read_text()); qvec=embed([q['question'] for q in questions]); np.save(ROOT/'results/query_vectors.npy',qvec); summary={}
 for size in [96,256]:
  chunks,vec=build(size); rows=[]
  for q,v in zip(questions,qvec):
   scores=vec@v; top=np.argsort(-scores)[:5]
   found=[]
   for i in top:
    c=chunks[i]; overlap=max(0,min(c['end'],q['end'])-max(c['start'],q['start'])) if c['page']==q['page'] else 0
    found.append(dict(c,score=float(scores[i]),gold_overlap_chars=overlap))
   # Frozen relevance definition: >=80 chars of annotated passage, or entire short passage.
   threshold=min(80,q['end']-q['start'])
   hit=any(c['gold_overlap_chars']>=threshold for c in found)
   rows.append(dict(q,hit=hit,page_hit=any(c['page']==q['page'] for c in found),retrieved=found))
  result={'chunk_tokens':size,'overlap_tokens':size//4,'chunks':len(chunks),'recall_at_5':sum(r['hit'] for r in rows)/len(rows),'source_page_recall_at_5':sum(r['page_hit'] for r in rows)/len(rows),'questions':rows,'relevance':'same PDF page and >= min(80, gold passage length) characters intersect annotated gold passage; question hit rate@5, one annotated source passage per query'}
  (ROOT/f'results/retrieval_{size}.json').write_text(json.dumps(result,indent=2));summary[size]={k:v for k,v in result.items() if k!='questions'}
  print(size,summary[size],flush=True)
 (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
 info=post('show',{'model':'all-minilm'})
 tags=requests.get(API+'/tags',timeout=30).json()['models']
 info['local_manifest_digest']=next(m['digest'] for m in tags if m['name']=='all-minilm:latest')
 (ROOT/'results/embedding_model.json').write_text(json.dumps(info,indent=2))
if __name__=='__main__':main()
