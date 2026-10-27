---
name: agent-brain
description: The one way to file, change, count or check anything in the project's .agents/ brain. Use it to file a plan, a todo, a bug, a decision, a doc or a memory, to set the work the agent is on and mark its steps, to run a review ledger, to retitle a file, or to check the brain. Never write a brain file by hand.
---

# The brain tool

One command does everything under `.agents/`. It is plain Python with no
dependencies, and it runs from anywhere inside the project:

```bash
python3 .agents/skills/agent-brain/scripts/brain.py status
```

The examples below shorten that to `brain`. Every flag is listed in
[references/commands.md](references/commands.md).

## What the tool holds to

- No brain file is written by hand. `brain new` picks the id, writes the
  header block, puts the file in the right folder and rebuilds the index.
  The header block of an existing file changes through `brain set` and
  `brain refile`, and the rows of a steps table through `brain task`
  and `brain steps`. The prose below and around them is yours to edit,
  and doctor reads the file after. A hook holds all of this in every
  agent that runs hooks, so a new file, an index, a rename, a header
  change or a step row changed by hand is refused. A delete is not,
  and doctor runs after it.
- Ids only go up. A gap left by a moved or deleted file stays a gap.
- The indexes are generated. `brain index` rebuilds them, doctor rebuilds
  them too, and nobody edits an `INDEX.md`.
- A link in frontmatter is a path relative to `.agents/`, so a file opens
  without a search. `brain refile` carries every link along.
- `brain doctor` is a gate. It exits 0 only when the brain is clean, so
  it runs green before any review.
- The tool's code is never read by hand, save in the kit's own checkout
  where it is the work. SYSTEM.md section 8 carries the rule, and this
  sheet with the reference is what to read instead.

## When a session starts

The hook prints `brain status` when a session opens. If it did not, run
it. Every number in it is counted from the files at that moment, so
trust it over anything remembered.

```bash
brain status
brain list bug --status open
brain list todo --status open --area auth
brain show BUG-004
brain path BUG-004
brain links DEC-012
```

`show --meta` prints the frontmatter only. `--json` on `status`, `list`
and `links` gives the same answer for a script, less the clock and the
standing word that `status` opens with.

## Filing something

The prose goes in on standard input inside a quoted heredoc, which the
shell passes through untouched. The tool writes the structure around it
and refuses anything that breaks the spec, such as a missing heading or
a value outside the allowed set.

```bash
brain new bug --title "Token refresh loops after expiry" \
  --description "sign in loops once the refresh token expires" \
  --severity breaks --area auth <<'EOF'
## Problem

After the refresh token expires the sign in screen loops.

## Evidence

The auth log shows refresh, 401, refresh, 401.
EOF
```

The same shape files a todo (`--area`, `--by`), a decision (`--area`),
a memory (`--type project|feedback|reference`), a doc (`--category`), a
handover, or a review. Each kind's required headings are in its spec
under `.agents/spec/`, and the tool names the one that is missing.

A plan is a goal with the steps that reach it, and the steps are a
table the tool keeps, on a plan always and on a todo or a bug whose
work has steps. `brain new` turns a numbered Steps list in the prose,
each line free to open with the id or the name of the todo, bug or
review it points at, into that table. `brain steps <key> add` appends
steps as the work teaches them, `brain steps <key> step <n> done`
marks a step that points at nothing, or one whose pointer reaches
nothing the brain can read, a pointed step follows its entry on its
own, and `brain steps <key>` shows the table with its marks. A plan or
a todo goes done, and a bug fixed, only once every step is done or
skipped, while dropped, wontfix and refiled close as they are, and a
closed entry's table is its record, marked no more.

## Changing something

```bash
brain set BUG-004 status=branched
brain set TODO-012 status=done done=2026-09-16
brain refile BUG-004 --title "Token refresh loops on expiry"
```

`set` holds to the lifecycle rules. A bug goes `fixed` only once its Fix
section is written, a todo goes `done` only with a date and a filled
Done when, and a decision goes `superseded` only when it names what
replaced it.

`refile --kind bug` turns a todo into a bug, or the reverse, with the
new body on standard input. Both files stay and link to each other, the
old one marked `refiled` and naming the new one in `became`, so the id
trail holds and every step that pointed at the old one follows it,
save one whose own entry the new one's steps point back at, which
stays on the old one as skipped while the new one is open. It is
refused when a step of the new body would point back, at any remove,
at an open entry that points at the old one.

## A review

```bash
brain ledger open feat/billing --level high --summary - <<'EOF'
The billing branch, six commits.
EOF
brain ledger batch 2026-09-feat-billing --theme correctness
brain ledger add 2026-09-feat-billing - --severity breaks --batch 1 <<'EOF'
the guard is inverted
checkout fails for a saved card
EOF
brain ledger batch 2026-09-feat-billing 1 --status fixing --ack 2026-09-16
brain ledger set 2026-09-feat-billing F-01 F-02 status=fixed fix=abc1234
brain ledger batch 2026-09-feat-billing 1 --status verified
brain ledger batch 2026-09-feat-billing 1 --status done
brain ledger report 2026-09-feat-billing
brain ledger batch 2026-09-feat-billing --theme "closing pass"
brain ledger batch 2026-09-feat-billing 2 --status done
brain ledger close 2026-09-feat-billing
```

Findings and batches number themselves, with no more batches than
[spec/reviews.md](../../spec/reviews.md) allows, and `set` takes every
finding a group's commit fixed in one call, the findings first and the
changes after them. A batch that holds findings runs only on the
owner's ack, the word that starts it, given as a date or as `auto`
while the owner's standing word holds. Verification closes a batch,
`report` says where the review stands and what comes next, the closing
pass leaves one more batch of its own, outside the cap and empty when
it found nothing, and `close` refuses while any finding is still open
or the report names a step before the close. The standing word is
`brain ack on` with an end, by time or by task, `brain ack off` takes
it back, `brain ack` says where it stands, and the reference says what
a second `brain ack on` does and what each refuses.

```bash
brain ack on --for 8h --until-done 2026-10-feat-search
brain ledger batch 2026-10-feat-search 1 --status fixing --ack auto
brain ack off
```

## The current work

Work of more than one step starts with `brain task on` before the
first step, and SYSTEM.md section 8.3 is the rule for what may go
under what and when a link drops off, `--replace` being the owner's
call. The step the work stands on is the first pending row of the entry's
steps table, `brain task done` marks it done and moves on, `brain task
skip` marks it skipped, `brain task step <n>` reopens one, and
`brain task off` drops the innermost link. The reference says what
each line looks like and what each refuses.

```bash
brain task on kit-phase-7
brain task on TODO-015
brain task done
brain task off
```

## Checking the brain

```bash
brain doctor
brain doctor --fix
```

Doctor reports each defect on its own line, marked `fixable` or `flag`.
`--fix` repairs the mechanical ones, a stale index, a name that
disagrees with its filename, a link written as a name or repeated in
one header, a pointed step's mark that drifted from its entry or its
pointer left on a refiled one or written by name, a numbered list
under a plan's or a todo's Steps and a plan with no Steps section,
which get the table, a chat style copy that drifted from the Claude
Code one, a missing or dead skill link, and stamps every file it
touches. Whatever is left needs a decision, so read it and act on it.
Silencing a finding is never the answer.

## Bringing a project to the kit

```bash
python3 <kit>/.agents/skills/agent-brain/scripts/brain.py port
python3 <kit>/.agents/skills/agent-brain/scripts/brain.py upgrade
```

Run from the project's root, with `<kit>` a checkout of agent-setup.
`port` installs the kit into a project with no brain and lists what is
left to file through `brain new`. `upgrade` brings an existing brain up
to the kit, the old `.agent/` layout included, migrations first. Both
keep the project's section 0, its own hooks and everything it wrote,
and end with `doctor --fix`. The full contract is in
[commands.md](references/commands.md).
