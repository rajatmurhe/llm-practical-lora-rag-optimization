# Measured classification results

Base accuracy: 22.0588% (15/68).
LoRA accuracy: 94.1176% (64/68).

## By subset

- paraphrase: base 12/56; LoRA 56/56
- compositional: base 3/12; LoRA 8/12

## Remaining mistakes

- Do not reset my password; reverse the extra charge. Gold: **billing**; predicted: **account**. The negated password-reset phrase appears to dominate the actual charge-reversal request. Training has no negation examples, so keyword association is a plausible explanation.
- The software is fine; the physical box is missing. Find it. Gold: **delivery**; predicted: **technical**. The software cue pulls toward technical despite being described as fine. Physical box is an unfamiliar paraphrase of parcel, adding vocabulary shift to the competing intent cue.
- I was billed for a replacement password. I need my money back. Gold: **billing**; predicted: **account**. The password entity appears to outweigh the requested financial remedy. The model must separate what was purchased from the action requested, a composition absent from training.
- The delivery date widget freezes. Debug the widget. Gold: **technical**; predicted: **delivery**. Delivery describes the widget, while debugging is the requested action. Single-intent templates do not teach this modifier-versus-action distinction.

The first three errors in saved test order are used in the report. Causes are hypotheses; no ablation tests were run. Test examples were not used to tune hyperparameters.
