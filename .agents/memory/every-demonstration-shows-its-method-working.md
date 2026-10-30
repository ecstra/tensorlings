---
name: every-demonstration-shows-its-method-working
kind: memory
description: the owner wants no lesson where a method underdelivers and a caveat explains it away, so setups are researched and redesigned until the method works
updated: 2026-10-30
links: []
type: feedback
---

A lesson's demonstration must show its method doing what its papers say it does. On 2026-10-30 the owner said, after reading a run of reports that a method lost and "the lesson says so plainly": "no more 'Honest caveats and the lessons say so' cuz what fucking lessons says the honest caveats? Remove the caveats and plan accordingly. Research online if needed."

**Why:** a lesson that shows SimCLR losing to supervised training, or LoRA falling short of full fine tuning, and then explains it away, teaches the learner nothing about the method. The owner also said early on "Dont be biased. Be ground truthful", so no number is ever faked or cherry picked.

**How to apply:** when a demonstration underdelivers, change the setup and not the claim. Research how the method's own papers and good small scale reproductions show it, and pick the task, size, data, training length and settings where its real effect appears. Then measure it across seeds and write what the numbers show. A failure stays in a lesson only when the failure itself is what the exercise teaches, such as a data leak inflating a score, reward hacking, overfitting, a planted bug, or a check that catches a broken module.
