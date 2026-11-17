"""Serial, warm-run Ollama benchmark. Run after training/retrieval have exited."""
import json,time,threading,statistics,platform
from pathlib import Path
import requests,psutil
ROOT=Path(__file__).resolve().parent
MODELS={q:f'hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:{q}' for q in ['Q4_K_M','Q8_0']}
PROMPTS=[
 {'id':1,'prompt':'What is the capital of France? Answer with only the city name.','expected':'Paris','rule':'case-insensitive substring paris'},
 {'id':2,'prompt':'What is 17 + 25? Answer with only the number.','expected':'42','rule':'exact 42 after stripping whitespace and final period'},
 {'id':3,'prompt':'Return only valid JSON with name Ada and age 36.','expected':{'name':'Ada','age':36},'rule':'JSON parses and equals expected object'},
 {'id':4,'prompt':'Reverse the letters in planet. Reply with only the reversed word.','expected':'tenalp','rule':'exact word, case insensitive'},
 {'id':5,'prompt':'Extract the email address: Please contact Mira at mira@example.com tomorrow. Reply with only the email.','expected':'mira@example.com','rule':'exact email after whitespace stripping'},
 {'id':6,'prompt':'Classify the sentiment as positive or negative. Text: I love this wonderful product. Reply with one word.','expected':'positive','rule':'exact label, case insensitive'},
 {'id':7,'prompt':'Translate the English word cat into French. Reply with only the French word.','expected':'chat','rule':'exact word, case insensitive'},
 {'id':8,'prompt':'Write the word blue exactly three times, separated by spaces. Nothing else.','expected':'blue blue blue','rule':'exact string, case insensitive'},
 {'id':9,'prompt':'Which planet is known as the Red Planet? Reply with only its name.','expected':'Mars','rule':'exact word, case insensitive'},
 {'id':10,'prompt':'Summarize in one sentence: A storm closed the bridge. Buses used a longer route. Commuters arrived late.','expected':'storm closes bridge, buses reroute, commuters late','rule':'one sentence contains storm, bridge, bus, and late/delay'}]
def post(route,body):
 r=requests.post('http://localhost:11434/api/'+route,json=body,timeout=300);r.raise_for_status();return r.json()
def score(p,text):
 s=text.strip().rstrip('.').lower()
 if p['id']==1:return 'paris' in s
 if p['id']==3:
  try:return json.loads(text)==p['expected']
  except ValueError:return False
 if p['id']==10:
  return all(w in s for w in ['storm','bridge','bus']) and ('late' in s or 'delay' in s) and len([x for x in text.strip().split('.') if x.strip()])==1
 return s==str(p['expected']).lower()
def runner_rss():
 total=0
 for p in psutil.process_iter(['name','cmdline','memory_info']):
  try:
   cmd=' '.join(p.info['cmdline'] or [])
   name=(p.info['name'] or '').lower()
   if name in ('llama-server','ollama_llama_server') or (name=='ollama' and 'runner' in cmd):total+=p.info['memory_info'].rss
  except (psutil.NoSuchProcess,psutil.AccessDenied): pass
 return total

def main():
 (ROOT/'prompts.json').write_text(json.dumps(PROMPTS,indent=2)); results={}
 for quant,model in MODELS.items():
  # Explicitly unload every resident Ollama model to isolate memory.
  for m in requests.get('http://localhost:11434/api/ps').json()['models']:
   post('generate',{'model':m['name'],'keep_alive':0})
  options={'temperature':0,'seed':42,'num_ctx':2048,'num_predict':96,'num_thread':4,'num_gpu':99}
  post('chat',{'model':model,'messages':[{'role':'user','content':'Say hello.'}],'stream':False,'options':options,'keep_alive':'10m'})
  resident=requests.get('http://localhost:11434/api/ps').json(); show=post('show',{'model':model}); peak=[runner_rss()]; stop=threading.Event()
  def sample():
   while not stop.wait(0.05):peak.append(runner_rss())
  thread=threading.Thread(target=sample);thread.start(); rows=[]
  try:
   for repeat in range(3):
    for p in PROMPTS:
     start=time.perf_counter()
     out=post('chat',{'model':model,'messages':[{'role':'user','content':p['prompt']}],'stream':False,'options':options,'keep_alive':'10m'})
     rows.append({'id':p['id'],'repeat':repeat,'prompt':p['prompt'],'response':out['message']['content'],'pass':score(p,out['message']['content']),'wall_seconds':time.perf_counter()-start,'eval_count':out['eval_count'],'eval_duration':out['eval_duration'],'prompt_eval_count':out.get('prompt_eval_count'),'prompt_eval_duration':out.get('prompt_eval_duration'),'load_duration':out.get('load_duration'),'total_duration':out.get('total_duration')})
     print(quant,repeat,p['id'],round(out['eval_count']/(out['eval_duration']/1e9),2),repr(out['message']['content']),flush=True)
  finally:stop.set();thread.join()
  result={'model':model,'quantization':show['details']['quantization_level'],'options':options,'repetitions':3,'tokens_per_second':sum(r['eval_count'] for r in rows)/(sum(r['eval_duration'] for r in rows)/1e9),'per_request_tps_median':statistics.median(r['eval_count']/(r['eval_duration']/1e9) for r in rows),'runner_peak_rss_bytes':max(peak),'resident':resident,'model_details':show['details'],'quality_passes_first_repeat':sum(r['pass'] for r in rows if r['repeat']==0),'rows':rows,'ollama_version':requests.get('http://localhost:11434/api/version').json(),'platform':platform.platform(),'measurement':'decode tokens/sec excludes prefill and load; 50ms polling sum of Ollama runner RSS; size_vram from /api/ps is unified-memory allocation on Apple Silicon, not dedicated VRAM'}
  results[quant]=result;(ROOT/f'results/{quant}.json').write_text(json.dumps(result,indent=2))
  post('generate',{'model':model,'keep_alive':0})
 (ROOT/'results/summary.json').write_text(json.dumps({q:{k:v for k,v in r.items() if k not in ['rows','resident']} for q,r in results.items()},indent=2))
if __name__=='__main__':main()
