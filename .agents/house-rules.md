---
name: house-rules-agent
kind: doc
description: the branch contract and code review protocol, the agent's operating version
updated: 2026-09-29
links: [spec/reviews.md, spec/bugs.md, spec/todos.md, README.md]
---

# House Rules

The development rules of every project that carries this directory. A
project may keep a short human cut at `docs/house-rules.md` and the hard
bans in a root CONTRIBUTING.md, both pointing here. The project's own
AGENTS.md section 0 names its exact gates, tooling, and PR rules.

## The branch contract

1. **One branch, one intent.** A feature, a fix, a restructure: one
   coherent intent per branch, named `feat/<slug>`, `fix/<slug>`, or
   `sweep/<slug>`. Size is measured by intent, never by commit count: a
   long branch serving one intent is fine, a second intent born mid
   branch forks its own branch the day it appears, it never rides.
2. **Sweeps carry the small stuff.** Small bugs and small features group
   into a sweep branch by surface, one screen or one subsystem per sweep.
   A sweep never contains a breaking change.
3. **Branch from fresh main.** Merge whole, never cherry pick out of an
   unreviewed branch. A merged branch is deleted, its review ledger is
   the record.
4. **Main is reviewed code only.** Gates first, then review, then
   merge. No step skips, and nothing merges before the review has
   closed. When the PR opens and whether as a draft is the project's
   rule in AGENTS.md section 0, never a question the agent asks.

## The gates

Before any review: every suite, analyzer, and build the project's
AGENTS.md section 0 names, green, and `brain doctor` clean. The review
judges code that already works, reviewers never burn effort on red
suites or a brain with broken links. When the review starts the working
tree is clean as well, since a review reads committed work only.

## The review protocol

1. **The timing is the owner's, the running is the agent's.** The owner
   says when a branch gets reviewed and the agent never decides that on
   its own. On the word, the agent runs the review itself with the
   kit's `branch-review` skill
   ([skills/branch-review/SKILL.md](skills/branch-review/SKILL.md))
   at the level the branch earns, low or medium for a small or
   mechanical branch and high or max where the surface is wide or the
   logic is subtle, and never low for a branch that breaks or changes
   an interface. The level sets how many angles the finding pass
   runs and never how hard the model thinks, which is the session's
   effort and the owner's to set. A tool's own review command, paid or
   cloud run ones included, is the owner's alone, so the agent neither
   reaches for one nor asks for it by name. Codex `/review --base main`
   (GPT) is a second pair of eyes the owner can add. All of it works on
   the local branch, whether or not a PR is open yet.
2. **Findings first, whole.** Finder sub agents read the branch all at
   once, each over a slice of it or over the whole as its angle reads,
   and further sub agents merge what they found into findings. Every
   finding is filed, the rare and the small included, and the agent
   reports them all before touching anything. No fixing during
   reporting, no silent triage.
3. **The ledger.** All findings land in
   `.agents/reviews/<yyyy-mm>-<branch-slug>/ledger.md` per
   [spec/reviews.md](spec/reviews.md), which holds the grades and the
   cap on batches. Each is graded breaks, bug, debt or nit and filed
   into a batch by theme, correctness first and wording last. Every
   finding leaves the open list as fixed, as refuted where the claim is
   wrong, as left as it is (wontfix) where the claim is true and the
   behaviour on purpose, or as deferred, and one that is not fixed
   carries its reason, so no future review re litigates it.
4. **The rounds of fixes.** The batches that have the owner's word are
   fixed together in rounds, on the path the
   [skill sheet](skills/branch-review/SKILL.md) gives. In a round one
   fixer sub agent per group of files fixes the cause of each of its
   findings, and verifier sub agents check each fix against the code.
   Once the gates are green each group is one commit, and hunter sub
   agents read what the round changed for defects the fixes brought
   in, which the next round fixes, no backlog. The rounds end at the
   first hunt that finds nothing. The last pass follows that hunt, and
   nothing hunts it. It brings the docs up to the changed code and
   holds the new lines to the house style. The skill sheet decides
   which cases are the branch's, and any other that behaves the same
   on the base is an older defect, filed in `.agents/bugs/` the moment
   it is met. Sub agents run in parallel. With sub agents only a fixer
   edits, and only its group's files, and the commits and the ledger
   are the main session's. An agent without sub agents takes each
   step itself, in the same order.
5. **The close of the batches.** After the last pass the batches are
   marked verified and done in the ledger, and one report to the owner
   follows. A finding the fixes did not settle, or one that cannot be
   fixed inside the branch, is named there, and its batch waits for
   the owner's word instead of closing. Such a finding leaves for
   `.agents/bugs/` on that word alone, as its own file, linked both
   ways, and earns a future branch sparingly.
6. **The owner's ack gates each batch.** A batch starts only when they
   say so, in number order, and a word that covers several starts them
   together. The ack is that word to start a batch, recorded on the
   batch it starts, and verification is what closes a batch. A
   standing word, `brain ack on` with an end by time or by task, is the
   ack for every gate of the review until that end, each batch and the
   closing pass, a batch the owner acked ahead starting on that ack
   instead, and each report says which of the two the step ran on. The
   close follows the last batch with no word. The PR and the merge are
   not the review's gates and stay on the owner's explicit ask, per the
   git rules below. A standing word with no end is not taken, the agent
   proposes one and waits. Nothing starts a review but the owner's own
   word.
7. **The closing pass.** Above low, the branch gets one more pass over
   itself, whole, as it stands after the last batch, by fresh finders,
   to catch what the fixes introduced and what the review missed. What
   they find is one more batch, fixed on the owner's word in rounds
   like any other. Sweeps skip this.

## Git rules

- Commits are allowed and expected: one commit per change, so every
  change is undoable and addressable by hash.
- Single author, no co-author, no AI attribution anywhere: not in
  commits, not in PR titles or bodies, no generated-with footers.
- Commit messages in the owner's voice: added:/fixed:/improved:/updated:/
  removed:, one short sentence, two clauses at most.
- Push after each commit.
- PRs only on the owner's explicit ask in the current message, through
  the project's PR tooling, base main, plain unless section 0 says
  otherwise. Merges only on the owner's explicit ask, never before the
  review has closed.
- Never commit to main directly.
- Worktrees are disabled: never create one, and never use a worktree
  isolation mode for sub-agents. If the owner ever explicitly asks, the
  one sanctioned home is `.agents/.worktrees/` (gitignored), nowhere
  else, torn down when the task ends.

## The trackers

Defects go to [bugs/](spec/bugs.md), work goes to [todos/](spec/todos.md),
long horizon work to [plans/](spec/plans.md), a goal with a steps table
the tool keeps, and the agent classifies on intake and uses todos
extensively (multi step work gets filed, not carried in a session's
head). Owner written todos get picked up, cleaned, and filed with their
wording preserved, and on completion the agent tells the owner which
of their own notes to tick. Work of more than one step starts with
`brain task on` and moves with `brain task done`, or of itself when
the entry a step points at closes, a refiled one handing the step on
to the entry filed in its place, per SYSTEM.md 8.3, so every hook line
says what the agent is on.
