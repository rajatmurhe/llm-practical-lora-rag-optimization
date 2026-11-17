"""Write human-readable tables from saved model outputs, without model calls."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def main():
 base=read('task1_lora/results/base.json');fine=read('task1_lora/results/finetuned.json')
 failures=[]
 for r in fine['predictions']:
  if r['correct']:continue
  if r['type']=='compositional':reason='The input combines competing class cues or negation; single-intent training does not teach reliable requested-action selection.'
  else:reason='The phrasing is held out but the intent core is familiar; the small attention-only adapter still confuses this semantic class. More varied examples and a validation split could test whether capacity or coverage limits it.'
  reasons={
   'Do not reset my password; reverse the extra charge.':'The negated password-reset phrase appears to dominate the actual charge-reversal request. Training has no negation examples, so keyword association is a plausible explanation.',
   'The software is fine; the physical box is missing. Find it.':'The software cue pulls toward technical despite being described as fine. Physical box is an unfamiliar paraphrase of parcel, adding vocabulary shift to the competing intent cue.',
   'I was billed for a replacement password. I need my money back.':'The password entity appears to outweigh the requested financial remedy. The model must separate what was purchased from the action requested, a composition absent from training.',
   'The delivery date widget freezes. Debug the widget.':'Delivery describes the widget, while debugging is the requested action. Single-intent templates do not teach this modifier-versus-action distinction.'}
  reason=reasons.get(r['text'],reason)
  failures.append(dict(r,reason=reason))
 (ROOT/'task1_lora/results/failures.json').write_text(json.dumps(failures,indent=2))
 lines=['# Measured classification results','',f"Base accuracy: {base['accuracy']:.4%} ({sum(r['correct'] for r in base['predictions'])}/{base['n']}).",f"LoRA accuracy: {fine['accuracy']:.4%} ({sum(r['correct'] for r in fine['predictions'])}/{fine['n']}).",'','## By subset','']
 for kind in ['paraphrase','compositional']:
  line=[]
  for tag,result in [('base',base),('LoRA',fine)]:
   rows=[r for r in result['predictions'] if r['type']==kind];line.append(f"{tag} {sum(r['correct'] for r in rows)}/{len(rows)}")
  lines.append(f"- {kind}: "+'; '.join(line))
 lines+=['','## Remaining mistakes','']
 for r in failures:lines.append(f"- {r['text']} Gold: **{r['label']}**; predicted: **{r['prediction']}**. {r['reason']}")
 lines+=['','The first three errors in saved test order are used in the report. Causes are hypotheses; no ablation tests were run. Test examples were not used to tune hyperparameters.']
 (ROOT/'task1_lora/results/analysis.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
