"""Answer-only LoRA training and held-out constrained-label evaluation."""
import os, json, random, time, platform
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model
ROOT=Path(__file__).resolve().parent
MODEL='HuggingFaceTB/SmolLM2-135M-Instruct'
REVISION='12fd25f77366fa6b3b4b768ec3050bf629380bac'
LABELS=['billing','account','technical','delivery']
SYSTEM='Classify support tickets. Reply with one letter: A=billing (charges/refunds), B=account (login/profile), C=technical (software/device faults), D=delivery (shipping/tracking). Choose the main requested action.'
# Hand-authored intent cores; held-out phrasing is disjoint from training templates.
CORES={
'billing':['reverse the duplicate charge','refund my payment','explain the extra fee','correct the invoice','cancel the recurring subscription charge','return the money for my cancelled order','fix the tax on my bill','check why my card was charged twice','apply my missing discount','change the payment method'],
'account':['reset my password','unlock my account','change my login email','recover access to my profile','enable two factor authentication','delete my account','update my username','remove an old phone number from my profile','restore my suspended login','help me sign in'],
 'technical':['fix the app crash','repair the frozen screen','resolve the server error','fix the broken search function','stop the app from closing unexpectedly','repair the failing file upload','fix the blank dashboard','resolve the software installation error','repair the connection timeout','fix the video playback bug'],
'delivery':['locate my missing parcel','update the shipping address','track the delayed shipment','reschedule the courier visit','find the package marked delivered','explain why the parcel has not arrived','arrange redelivery of my package','check the estimated arrival date','contact the courier about my shipment','change the delivery time']}
TRAIN=['Please {x}.','Can you {x}?','I need you to {x}.','Support, please {x} today.','Could someone {x} for me?','My request is to {x}.','I would like to {x}.','Urgent: {x}.']
TEST=['The only help I want right now is: {x}.','After a frustrating morning, I am asking you to {x}.']
def make_data():
    rng=random.Random(42)
    train=[{'text':t.format(x=x),'label':label} for label,cores in CORES.items() for x in cores for t in TRAIN]
    test=[{'text':t.format(x=x),'label':label,'type':'paraphrase'} for label,cores in CORES.items() for x in cores[:7] for t in TEST]
    hard=[
      ('The app crashed after checkout, but please refund the duplicate payment.','billing'),
      ('My parcel arrived; now delete my shopping profile.','account'),
      ('I can sign in, but file uploads fail with error 500.','technical'),
      ('Payment went through; please locate the parcel the courier lost.','delivery'),
      ('Do not reset my password; reverse the extra charge.','billing'),
      ('Do not refund anything; recover my locked profile.','account'),
      ('The courier tracking page crashes. Repair the page, not the shipment.','technical'),
      ('My account works. Change where the courier should bring the package.','delivery'),
      ('I was billed for a replacement password. I need my money back.','billing'),
      ('The invoice shows my old email. Change my sign-in email, not my invoice.','account'),
      ('The delivery date widget freezes. Debug the widget.','technical'),
      ('The software is fine; the physical box is missing. Find it.','delivery')]
    test += [{'text':x,'label':y,'type':'compositional'} for x,y in hard]
    rng.shuffle(train); rng.shuffle(test)
    assert len(test)>=50 and not {x['text'] for x in train}&{x['text'] for x in test}
    for name,rows in [('train',train),('test',test)]:
      (ROOT/'data'/f'{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    return train,test

def main():
    set_seed(42); torch.set_num_threads(4)
    train,test=make_data()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION); tok.pad_token=tok.eos_token
    device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION, dtype=torch.float32).to(device)
    def prompt(text): return tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCD']; assert all(len(i)==1 for i in ids)
    choice_ids=torch.tensor([i[0] for i in ids],device=device)
    def evaluate(tag):
      model.eval(); rows=[]
      with torch.no_grad():
        for i in range(0,len(test),8):
          batch=test[i:i+8]; tok.padding_side='left'
          inp=tok([prompt(r['text']) for r in batch],padding=True,return_tensors='pt').to(device)
          logits=model(**inp,logits_to_keep=1).logits[:,-1,:][:,choice_ids]
          pred=logits.argmax(-1).tolist()
          for row,p in zip(batch,pred): rows.append(dict(row,prediction=LABELS[p],correct=LABELS[p]==row['label']))
      accuracy=sum(r['correct'] for r in rows)/len(rows)
      result={'accuracy':accuracy,'n':len(rows),'predictions':rows}
      (ROOT/'results'/f'{tag}.json').write_text(json.dumps(result,indent=2)); print(tag,accuracy,flush=True)
      return result
    base=evaluate('base')
    model.config.use_cache=False
    if device=='mps': torch.mps.empty_cache()
    config=LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],lora_dropout=0.05,bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,config); model.print_trainable_parameters()
    encoded=[]
    for row in train:
      p=tok.encode(prompt(row['text']),add_special_tokens=False)
      answer=ids[LABELS.index(row['label'])]+[tok.eos_token_id]
      encoded.append((p+answer,[-100]*len(p)+answer))
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4)
    started=time.time(); losses=[]
    for epoch in range(3):
      random.shuffle(encoded); model.train(); total=0
      opt.zero_grad()
      for i,(tokens,targets) in enumerate(encoded):
        # Unpadded microbatch of one. Last two targets are label and EOS.
        # Retain three output positions: the first two predict those targets.
        inputs=torch.tensor([tokens],device=device)
        logits=model(input_ids=inputs,logits_to_keep=3).logits[:,:-1,:]
        labels=torch.tensor([targets[-2:]],device=device)
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
        (loss/8).backward(); total+=loss.item()
        if (i+1)%8==0 or i+1==len(encoded):
          torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        if i%80==0: print('epoch',epoch+1,'microstep',i,'loss',round(loss.item(),4),flush=True)
      losses.append(total/len(encoded))
    training_seconds=time.time()-started
    model.save_pretrained(ROOT/'adapter'); tok.save_pretrained(ROOT/'adapter')
    fine=evaluate('finetuned')
    meta={'model':MODEL,'model_revision':getattr(model.config,'_commit_hash',None),'device':device,'dtype':'float32','seed':42,'train_n':len(train),'test_n':len(test),'epochs':3,'batch_size':8,'microbatch_size':1,'gradient_accumulation_steps':8,'learning_rate':3e-4,'lora_rank':8,'lora_alpha':16,'targets':['q_proj','v_proj'],'dropout':0.05,'training_seconds':training_seconds,'epoch_losses':losses,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'platform':platform.platform(),'metric':'accuracy of highest-logit label among A/B/C/D; constrained classification, not free-text generation'}
    (ROOT/'results'/'settings.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__': main()
