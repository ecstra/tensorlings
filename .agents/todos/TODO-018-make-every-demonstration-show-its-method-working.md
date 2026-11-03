---
name: TODO-018-make-every-demonstration-show-its-method-working
kind: todo
description: audit every lesson for a method that underdelivers behind a caveat, then research and redesign each setup until the method works
updated: 2026-11-03
links: []
status: open
by: owner
area: course
born: 2026-10-30
---

## What

The owner's word on 2026-10-30: "no more 'Honest caveats and the lessons say so' cuz what fucking lessons says the honest caveats? Remove the caveats and plan accordingly. Research online if needed." Every lesson where a demonstrated method fails to show its benefit, with a caveat explaining it away, gets its setup researched and redesigned until the method works, measured across seeds. Failures that are themselves the lesson stay, such as leaks, reward hacking, overfitting and planted bugs.

Known cases from the build reports include SimCLR losing to supervised training in probe3, LoRA well short of full fine tuning in lora3, plain DPO collapsing in post2, PPO never settling in ppo3, the KV cache slowing the small GPT on MPS, int8 running slower, the forecaster's small model losing to least squares and its third leak staying under the ceiling, and combine3's zero start losing when the input carries signal.

## Steps

| # | Step | Points at | By | Status |
|---|---|---|---|---|
| 1 | Audit Sections 1 to 4 for demonstrations that underdeliver | | owner | done |
| 2 | Audit Sections 5 to 7 | | owner | done |
| 3 | Audit Sections 8 to 10 | | owner | done |
| 4 | Audit Sections 11 and 12 | | owner | done |
| 5 | Audit Section 13 once it is built | | owner | done |
| 6 | Research and redesign each finding, rerun it across seeds, and rewrite its lesson without the caveat | | owner | pending |
| 7 | Recheck every changed exercise, its wrong answers and any checkpoint it trains | | owner | pending |

## Done when

No lesson shows a method underdelivering behind a caveat, every redesigned demonstration is measured across seeds with the module's own code, and the dev check and every wrong answer pass.
