---
name: branch-review
description: The kit's code review of one branch, run on the owner's word. Finder sub agents read the branch and every finding goes into a ledger through the brain tool. Fixer sub agents then fix the findings in rounds, each checked by verifiers and followed by a hunt, and a last pass and, above low, a closing pass end the review. Use it when the owner asks for a review, and never start one on your own.
argument-hint: [low|medium|high|max] [<base>...<head>]
---

# The review

The review protocol of [house-rules.md](../../house-rules.md), run by
the agent, step by step. The level names how wide the finding pass goes
and never how hard the model thinks, which is the session's effort and
the owner's to set. Sub agents do the work wherever the agent has them,
and the main session coordinates them. An agent without them follows
the section Without sub agents.

This sheet decides the plain path of a review. What it does not decide,
the main session decides by SYSTEM.md and by what serves the owner, and
the report says what it decided.

## Words

- Round: a round of fixes over the batches that have the owner's word,
  its fixers, its verifiers, the gates, the commits and the hunt.
- Pass: one exchange inside a group, the fixer's work and the
  verifier's vote on it. A finding gets three passes at most.
- Group: the findings one fixer owns, with the files it may edit, a
  code file with its test file, or a doc, or several small docs.
  Findings whose fixes need each other's files make one group. A group
  is never split, and two fixers never edit one file at the same time.

## What holds throughout

- The owner said "review". The agent never decides a branch is ready,
  and no standing word starts one.
- Every finding lands, and none is dropped for being rare or small. It
  lands as fixed, refuted, left as it is, which the ledger calls
  wontfix, or deferred, and [spec/reviews.md](../../spec/reviews.md)
  says what the ledger records of each. Refuted, where the claim is
  wrong, and left as it is, where it is true and the behaviour on
  purpose, stand only on the verifier's agreement, and the report
  names the reason. Deferred means the finding cannot be fixed inside
  the branch. It leaves for `.agents/bugs/` on the owner's word alone,
  through `brain new bug` with `--link` to the ledger, and the
  finding's reason names the bug.
- A finding is the branch's own. A case that sits outside the functions
  the range changed and the code the change made reachable, and behaves
  the same on the base, is an older defect, and every other case is the
  branch's. An older case is filed as a bug at once and named in the
  report, and it never counts against a finding. A whole finding that
  proves older is filed the same way and set deferred with that bug as
  its reason, which needs no word from the owner, since nothing of the
  branch is put off.
- Every gate of the review that waits for the owner's word, house rule
  6 names them, first asks `brain ack`. While the owner's standing word
  holds, the gate passes, the batch takes `--ack auto`, or starts on
  the owner's dated ack where it already carries one, and the report
  says which of the two the step ran on and, for the standing word,
  what is left of it. Off, the gate waits for the word in chat, unless
  the batch already carries the owner's ack, which starts it. The PR
  and the merge are not the review's gates and wait for the owner's
  ask either way. An owner who says "auto ack" with no end is asked
  for one, with a proposal such as this review's ledger as the task,
  which once the ledger is open is `brain ack on --until-done <its key>`
  and ends when it closes, or a time tonight, and nothing is recorded
  until they answer.
- The sub agents are the four definitions, finder, fixer, verifier and
  hunter, under `.claude/agents/` in Claude Code, each on Opus at the
  session's effort. A merger and a joiner are finders, and a model
  check is a verifier's. A docs writer and a sweep are fixers. The
  finder, the verifier and the hunter have Read, Bash and Write, the
  Write for their answer file alone, and the fixer has Edit besides.
  None has the web, so where a finding or a model turns on how another
  product behaves, the main session fetches that product's page and
  hands it over as a file. Claude Code reads the definitions when a
  session starts, and a session that has none spawns a general purpose
  sub agent on Opus with the definition's text at the top of its
  prompt. Another agent uses the strongest model it has.
- The prompts, with the block shape each sub agent writes, are in
  [references/prompts.md](references/prompts.md). A sub agent writes
  its blocks to the answer file its prompt names and answers with one
  line per block, then the lines its task puts after the blocks, word
  for word. Every run has an answer file of its own. Earlier blocks
  reach a sub agent as the paths of the files that hold them, and a
  follow up carries the answer files the sub agent needs and the ids it
  answers for. The answer files live in a scratch folder of the
  review's own, the session's scratchpad in Claude Code, outside the
  repository, which SYSTEM.md 10.3 does not govern.
- As many sub agents run at once as the harness allows, and the next
  starts as one returns. The main session's own writes, git and the
  brain tool, go one at a time, since none holds a lock. Every path,
  range, reason and title it puts in a command goes in single quotes, a
  single quote in it written as `'\''`, and a summary or a claim goes in
  on standard input inside a quoted heredoc. Every commit is pushed.
- A sub agent that returns with no answer file runs again once, fresh,
  with the same prompt and an answer file of its own, a fixer's with
  one line more that says the working tree holds its earlier changes.
  What follows a second failure is the main session's to decide.
- Only a fixer edits, and only the files of its group. No sub agent
  commits or writes the ledger. A fixer runs the tests of its files and
  a verifier never runs the project's test suite. In each pass a
  verifier may run one snippet, and each fixed finding's test once with
  the fix taken out, on a copy of the tree in the scratch folder, never
  a worktree. It judges the tests of the findings of its own share, and
  the group's first verifier those no finding names. A test is loosened
  when it accepts what it refused before or loses a case that still
  applies, and a loosened test is a NEW PROBLEM.
- The main session reads code to write a model and to read a red gate,
  and leaves every other read to a sub agent.
- A model with its limits goes to the fixer and the verifiers of the
  group that owns its files, in its round and in every later one, to
  the hunters of later rounds, to the closing finders and to the docs
  writer who states the limits. A case a model or the docs name as a
  limit is no NEW PROBLEM and no candidate.
- Every report follows [references/report.md](references/report.md),
  and the rule behind it is SYSTEM.md section 7.6.

## Before the review

1. Before. The project's gates, every suite, analyzer and build that
   AGENTS.md section 0 names, are green, and `brain doctor` is clean.
   The working tree is clean too, since a review reads committed work.
   Where any of them is not, the review stops before the ledger opens
   and the agent tells the owner which.

   The range is `main...HEAD` unless the owner gave one, and
   `<base>...<head>` stands for it in every command here. The scope is
   every file the command below lists, less the review's own paper,
   which step 3 names, and every command that takes the scope carries
   the same pathspecs. The one below keeps out `.agents/reviews`, where
   only a review writes. Once the ledger is open, each other path a
   paper commit named that this first scope did not list is kept out by
   a pathspec `':!<path>'` of its own, so the branch's own edit to a bug
   or an index stays in.

   ```bash
   git diff --name-only --no-renames '<base>...<head>' -- ':!.agents/reviews'
   ```

   In a project the files the kit writes whole come off the list, since
   a finding there is the kit's to fix and the next upgrade writes them
   back. They are its skills under `.agents/skills/`, `SYSTEM.md`, the
   brain's own `README.md` and `house-rules.md`, the specs,
   `.agents/VERSION`, the definitions under `.claude/agents/`, the skill
   links under `.claude/skills/`, the chat style `plain-and-short.md`
   under `.claude/output-styles/`, `.codex/` and `.gemini/`,
   `CLAUDE.md`, `GEMINI.md` and `CONTRIBUTING.md`.

2. Slices. The main session cuts the scope into slices of about 300
   changed lines from git's count per file, which the command below
   prints. A test file goes with its code file, and docs go together. A
   file of more than 400 changed lines is cut into parts by its hunks,
   which `git diff -U0 '<base>...<head>' -- '<file>'` lists, each part a
   range of lines of the file as HEAD has it.
   [references/angles.md](references/angles.md) says which angles read
   by slice and which read the whole branch.

   ```bash
   git diff --numstat --no-renames '<base>...<head>' -- ':!.agents/reviews'
   ```

## The finding pass

3. The ledger opens before anything is found, with the level and the
   summary. The open puts the review on the current work chain, as the
   [command reference](../agent-brain/references/commands.md) says. The
   review's paper is committed at once, and again each time the brain
   tool wrote it. The paper is what the tool wrote for the review, its
   ledger, the bugs it filed, the entry whose step points at the review
   and the indexes, and a paper commit names those paths one by one.

   ```bash
   brain ledger open 'feat/billing' --level high --summary - <<'EOF'
   what the branch did, one line
   EOF
   git add -- '.agents/INDEX.md' '.agents/reviews/INDEX.md' '.agents/reviews/2026-09-feat-billing/ledger.md'
   git commit -m 'added: the review of feat/billing' -- '.agents/INDEX.md' '.agents/reviews/INDEX.md' '.agents/reviews/2026-09-feat-billing/ledger.md'
   git push
   ```

4. The finders run at once, each with the finder's prompt and its angle
   from [references/angles.md](references/angles.md), one per slice for
   an angle that reads by slice and one over the range for an angle
   that reads the whole branch. A finder reports up to 20 candidates.
   One whose answer ends with (more remain) runs again, fresh, with its
   earlier answer files as found already, three runs at most, and the
   report names one that still had more.
5. At level max a second look follows, once per slice, when every run of
   the finders is back, with every candidate that names the slice's
   files as found already. The other levels have none.
6. The merge starts once every run of the finding pass is back. The main
   session cuts the groups from the files the candidates name, a file
   outside the scope among them, and spawns one merger per group, all at
   once, with the merge prompt. A merger makes one finding of the
   candidates one fix settles, and drops none. It grades each finding by
   what the defect does and puts it under one of the five themes the
   merge prompt names, correctness first and style and wording last.
   Each finding names the group that owns it and what it needs of
   another group, and two groups a finding needs both of become one. A
   joiner then names the findings of different groups that are one, and
   a joined finding is filed once, from the block the joiner keeps, at
   the grade the joiner gives it.

   The main session files the list, a batch per theme that holds any, in
   the order of the themes, within the cap of five batches that
   [spec/reviews.md](../../spec/reviews.md) sets. Each claim opens with
   the path and the line of its File line, and one call files the claims
   of a batch that share a grade, the grades in their order, breaks
   first. A file in the scratch folder maps each F-nn to its group and
   to the answer file and the P-n of each block that holds it.

   ```bash
   brain ledger batch 2026-09-feat-billing --theme correctness
   brain ledger add 2026-09-feat-billing - --severity bug --batch 1 <<'EOF'
   billing/invoice.py:412 the total rounds each line where the sum should be rounded
   billing/invoice.py:388 a credit note is counted twice
   EOF
   ```

   The report follows with the findings shape, findings first and
   whole. A finding pass that files nothing ends the review at once,
   with no batch and no closing pass, as step 16 says.
7. The batches start on the owner's word, in number order, and the word
   is recorded on each batch it covers, which starts it. The order is
   this sheet's rule, and the tool does not hold it.

   ```bash
   brain ledger batch 2026-09-feat-billing 1 --status fixing --ack 2026-09-23
   ```

## The rounds

8. The round. One fixer per group, all at once, with the fixer's
   prompt. A style finding, one on how a code file is written by
   SYSTEM.md section 2, waits for the sweep, and a doc finding on code
   a group of the round owns waits for the docs writer, since both
   follow the code and would be done again each time it moved. Every
   other finding is fixed in the round by the group of its file. The
   prose in a code file belongs to the group that owns the file, and the
   fixer that changes a behaviour changes the prose that describes it.

   A group that is one mechanism, such as a parser or a merge, gets its
   model first. The main session writes it, and writes it again whole
   with the cases of one check by a fresh verifier on the model check
   prompt. The model says what the code has to do, each rule with its
   purpose, and names as limits what the code cannot keep.
9. A fixer names the cause, lists the sibling cases in the branch's
   code, writes a failing test where a test can drive the claim and
   says how it checked where none can, fixes the cause and runs the
   tests of its files. It answers FIXED, REFUTED, LEFT or BLOCKED per
   finding and names every file it made or took away. A change it needs
   in a file it does not own it names under Outside, and the main
   session hands that to the fixer that owns the file, or to a fresh
   fixer and verifier where no group owns it. A doc its fix makes stale
   it names for the docs writer.
10. A group's verifiers start when its fixer has returned, with the
    verifier's prompt, about fifteen findings each. What is not GONE,
    REFUTED or LEFT goes back to the fixer, and each verifier votes
    again on what it sent back. A fixer takes a follow up, one for an
    Outside change too, only once it and every verifier of its group
    are back, so no verifier reads a file while it changes.

    A finding answered BLOCKED for want of a mechanism gets its group's
    model written, or written again whole, and so does a group that is
    one mechanism before its third pass. A finding that still stands
    after its third pass, whatever its fixer answered, goes to the owner
    in the report with what was tried, and its batch waits for their
    word. Its fixer takes its test and its change out of the working
    tree before the gates run.
11. When every finding of the round has its last verdict, the gates run
    on the working tree. The main session reads a red gate and hands it
    to the group whose change it follows, and once that group's
    verifier has voted on the fixer's change the gates run again. On
    green each group whose files changed is one commit of its own files
    by path, those its fixer made or took away among them, and its hash
    goes on its fixed findings in one call, as
    [spec/reviews.md](../../spec/reviews.md) says. A refuted finding
    and a finding left as it is, which is wontfix, take their reason.

    ```bash
    brain ledger set 2026-09-feat-billing F-03 F-04 status=fixed fix=abc1234
    brain ledger set 2026-09-feat-billing F-05 status=refuted reason='line 212 already guards it'
    brain ledger set 2026-09-feat-billing F-09 status=wontfix reason='the guard is on purpose'
    ```

12. The hunt. Fresh hunters read the code the round changed, with the
    hunter's prompt, one per slice, cut as step 2 says from
    `git diff --numstat --no-renames '<the commit before the round>..HEAD'`
    with the scope's pathspecs. They look for defects the fixes brought
    in and nothing else, with the finder's cap and its reruns. What they
    find is merged, joined and filed as step 6 says, with
    `--found-by hunter`, into the batch of the finding whose fix brought
    it, and the next round takes it with fresh fixers and verifiers. The
    rounds repeat until a hunt returns nothing.

## The last pass

13. The last pass follows the hunt that returned nothing. It is where a
    doc follows the code in a review, which is how the review keeps
    SYSTEM.md 8.1. The docs writers bring the docs up to the code the
    rounds changed, take the doc findings that waited and the docs a
    fixer or a verifier named, and state the limits of the models. A
    sweep holds the lines the rounds wrote, and the style findings of
    the batches of the word, to SYSTEM.md section 2, and changes no
    behaviour. Each has its verifier, the sweep's with a prompt of its
    own. Once both are quiet the gates run, and each is one commit.
    Nothing hunts the last pass.

    Records, which are decisions, handovers, memory, bugs and closed
    ledgers, stay as they were written, and in a project so do the files
    the kit writes whole. The docs writers own every other doc of the
    project, the review's own paper aside.

## The close

14. The batches of the word are set verified and done, the findings
    that waited carrying the last pass's commits. A batch whose finding
    went to the owner waits for their word. One report follows, in the
    round shape. Where a closing pass is owed it asks the owner's word
    for it, and where none is owed it is the closing report of step 16.

    ```bash
    brain ledger batch 2026-09-feat-billing 1 --status verified
    brain ledger batch 2026-09-feat-billing 1 --status done
    ```

15. The closing pass. Above low, and never on a sweep branch, as house
    rule 7 says, a review whose finding pass filed something owes one,
    which starts on the owner's word once every batch is done. Fresh
    finders read the whole branch as it stands, with the closing
    finder's prompt and the angles of the level, the scope and the
    slices taken anew, and no second look follows. Each runs once with
    the ledger's findings as found already, and the report names one
    that ended with (more remain). What they find is merged, joined and
    filed as step 6 says, into one batch themed `closing pass`, which
    runs on the owner's word like any batch, through to its own last
    pass. A closing pass that finds nothing still adds its batch, empty,
    and sets it done with no ack, since that row is how the tool knows
    the pass ran.

    ```bash
    brain ledger batch 2026-09-feat-billing --theme 'closing pass'
    brain ledger batch 2026-09-feat-billing 2 --status done
    ```

16. The ledger closes after the last batch, and after the closing pass
    where one was owed, with no word, and the paper is committed. The
    closing report states the time the review took, from the author
    date of the commit that opened the ledger, which a rebase keeps, to
    the close, and says the branch is ready to merge on the owner's ask.
    No report follows it, and the PR and the merge wait for that ask
    whether or not a standing word holds.

    ```bash
    git log --reverse --format=%ad --date=iso -- '.agents/reviews/2026-09-feat-billing/ledger.md' | head -1
    brain ledger close 2026-09-feat-billing
    ```

## Without sub agents

The agent does every step itself, in the same order. A second read of
each finding's answer and of each test the round touched takes the
place of the verifier, and one read of the round's diff takes the place
of the hunt. Refuted and left as it is stand on that second read. The
hard line of SYSTEM.md 9.5 that only a fixer edits holds with sub
agents, so here the agent edits itself.

## Limits

- The hunt is narrow on purpose, and nothing hunts the last pass.
- The commits of a round before its last one are not proven alone.
- A review reads committed work only.
- The scratch folder does not outlive its session everywhere, and a
  later session starts from the ledger's claims.
- A group is never split, so many findings on one file take one fixer,
  and one rerun, as long as they take.
- The tool does not hold the order of the batches, nor when a batch
  takes the closing theme.
- An angle that reads the whole branch has no bound on what it reads,
  so on a very large branch it may end with (more remain) three times,
  and the report says so.
- What this sheet does not decide is the main session's call.
