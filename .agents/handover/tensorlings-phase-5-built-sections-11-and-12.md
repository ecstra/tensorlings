---
name: tensorlings-phase-5-built-sections-11-and-12
kind: doc
description: phase 5 adds 47 exercises, Sections 11 and 12 with the post training and fast GPT projects, and brings phase 1a up to the later standards, on feat/tensorlings-phase-5
updated: 2026-10-30
links: []
---

## Where it stands

Phase 5 is built and checked on `feat/tensorlings-phase-5`, which stacks on the phase 4 branch and the ones under it. Nothing is merged, since merges wait for the owner's word. The course now has 314 exercises, Sections 1 to 12 of [the detailed curriculum](../docs/research/tensorlings-detailed-curriculum.md).

- Section 11: 76_finetuning to 82_dpo, quiz14, and the post training project, post1 to post3. LoRA SFT teaches the course GPT pig latin, DPO then cuts its one slip, and a reward model scores both.
- Section 12: 84_cost to 92_data, quiz15, and the fast GPT project, fast1 to fast3. The course GPT's writing is timed, then a KV cache and int8 weights are added and measured.
- Phase 1a's seven exercises were reviewed again at the owner's ask and brought up to the later standards. The owner's old answers, progress and unlocked solutions were cleared, and the owner's folder was set up from nothing.

New pieces: the Persuasion text in `DATASETS["gutenberg"]` for fine tuning, `run.py dev check` and `dev/wrong_answers.py` refusing exercise names that don't exist, and a top level line that runs too long failing only the names it sets, as the design doc now says.

## Verified against written

- `run.py dev check` reported all 314 exercises healthy with every reference checkpoint in place.
- `dev/wrong_answers.py` holds 1,835 wrong answers and every one fails.
- Six beginner reader reviews by Opus sub agents read every phase 5 exercise, and one more read phase 1a again. Their 108 findings were all fixed. Among them were a reward model that blocked gradients and still passed, three device gaps that would crash on the GPU, a quiz that demanded `.item()` before any lesson taught it, and lesson claims the numbers did not hold, such as LoRA coming close to full fine tuning.
- Every timing table was rerun on a quiet machine, at a load average of about 2.5. The KV cache writes about 2.7 times faster on the CPU and is slower on this Mac's GPU for the small course GPT. int8 weights are 3.5 times smaller and slower to run here.
- The post training projects trained for real on MPS, about 30, 20 and 10 seconds each. Plain DPO wrecked the task, from 76% right down to 1% to 4%, and DPO with Llama 3's NLL term reached 89%.
- The screen driver drew post2's lesson and result at 110 by 40.

## Next step

Phase 6, Section 13 of the detailed curriculum, on a branch from this one's tip, per the plan build-tensorlings. After it comes the final deep review of the whole course, which the plan now holds as its own step.

## Open questions for the owner

- When to merge the stacked branches into main, and whether to add a remote to push to.

## Traps

- The reference checkpoints sit outside `checkpoints/`, so the owner trains their own. Until then the dev check reports these 19 as failing on the missing file: cifar2, cifar3, adder2, gpt3, gpt4, probe2, digits2, digits3, finetune1, lora1, sft2, reward1, dpo2, post2, post3, quant3, fast1, fast2 and fast3. The reference checkpoints, for copying back to check, are adder2, cifar2, digits2, gpt3, post1 to post3 and probe1.
- zsh does not split an unquoted `$VAR`, so a list of names in one variable reaches the dev tools as one name. The tools now refuse it, where they once reported every exercise healthy.
- Timing tables are worthless while builders run. Measure them alone, as this phase's quiet rerun did.
- CPU numbers can move with the thread count, so measure functions behind exact tables pin it, as `pinned_threads` in dpo.py does.
- `torch.multinomial` on MPS needs a generator made on that device.
