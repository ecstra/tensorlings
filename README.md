# Aikyat Agent Setup

A small kit you drop into a project so a coding agent (Claude Code,
Codex, Gemini CLI, anything that reads `AGENTS.md`) knows how to behave
there and has a place to keep what it learns. It is one folder called
`.agents/`, holding the rules and everything the project knows, plus a
few short files at the root that point every agent into it. All plain
files, committed to git, all but what its ignore file keeps out. That
is the whole trick. Because they live in the repo, a fresh session on
any machine picks up exactly where the last one stopped.

## Why you would want it

Without something like this, every agent session starts from zero. The
rules live in someone's head and the project knowledge lives in old chat
logs, so the agent redoes the same discovery every morning. With the kit
in place, both are in the repo and the agent reads them before it
touches anything. You get the same behavior from every session and a
written record of what the project knows.

## What you get

| File | In plain terms |
|---|---|
| `AGENTS.md` | The front door every agent reads. It points at the rulebook, and it holds your project's own rules when you have some. |
| `CLAUDE.md` and `GEMINI.md` | A line or two each, pulling in `AGENTS.md` and, for Gemini, its chat style, because Claude Code and Gemini CLI look for their own filename. |
| `.agents/SYSTEM.md` | The rulebook. How to write code, how to talk to you, where docs go, how a branch gets reviewed, how git is used. |
| `.agents/` | The project's memory. Docs, plans, todos, bugs, decisions, review ledgers, handovers, and small facts the agent learns. Ships empty apart from the rules that say how each folder is used, and the tool that keeps them. |
| `.agents/skills/agent-brain/` | The brain tool. Files, changes, counts and checks everything in `.agents/`, so an id, a header block or an index line is never typed by hand. It also installs the kit into a project and brings it up to a newer release. Plain Python, no dependencies. |
| `.agents/skills/branch-review/` | The code review, run on your word. Finder sub agents in parallel, each over a slice of the branch or the whole of it as its angle reads, the findings filed in a ledger, then fixer sub agents in parallel, one per group of files, each group checked by verifiers and each round swept by hunters until clean, and a report that says where things stand first. |
| `.agents/VERSION` | The kit release your project carries, so an upgrade knows where it starts from. |
| `.claude/`, `.codex/`, `.gemini/` | The hooks each agent runs, so the brain is checked without anyone asking and the clock, your standing word and the work the agent is on stay in front of it, the links Claude Code needs to find the skills, the review's sub agent definitions, and the Plain & Short chat style, once per agent in the form that agent reads. |
| `CONTRIBUTING.md` | The hard rules for humans and agents on one page. |
| `CODEOWNERS` | Names who owns the repo on the forge. Put your own handle in. |
| `.gitignore` | Keeps your `.env`, your local Claude Code settings and Python's cache out of git, along with what the brain's conventions list as never entering the repo. |

Everything is generic on purpose. Anything specific to your project goes
in a section called `## 0. Project Rules` at the top of `AGENTS.md`, and
into the files your project grows inside `.agents/`. The rest of the kit
stays as it ships, apart from your handle in `CODEOWNERS`.

## Installation

The kit installs itself. Get a copy of it next to your project, then run
its port from your project's root, and the tool puts every file where
it belongs. Three ways to get the copy, easiest first.

### Ask your agent

If your machine can already reach the forge (you have cloned from
`forge.aikyat.com` before, or `tea` is logged in), open your agent in
your project and paste this prompt.

```text
Clone https://forge.aikyat.com/aikyat/agent-setup into a folder beside
this repo, then run its port from our root:
python3 <that folder>/.agents/skills/agent-brain/scripts/brain.py port
It refuses while a file it would replace has uncommitted changes, so
commit first. Copy the kit's CODEOWNERS in with my handle in place of
@ecstra. If this project already has test, build, or lint commands, add
a "## 0. Project Rules" section at the top of AGENTS.md naming them as
the gates. Read what the port printed, and tell me what you did and
what it listed as left to file.
```

### Terminal

Clone the kit beside your project, then run its port from your
project's root.

```bash
git clone https://forge.aikyat.com/aikyat/agent-setup ../agent-setup
python3 ../agent-setup/.agents/skills/agent-brain/scripts/brain.py port
cp ../agent-setup/CODEOWNERS .
```

Adjust the `../agent-setup` path if you cloned somewhere else.

### Download

Use the Code button at the top of this page and download the ZIP.
Unzip it beside your project, then run the port from your project's
root exactly as the terminal path does, with the unzipped folder's
path in place of `../agent-setup`.

### What the port does

It works on the git repository around the folder you run it in, never
on your home folder, and it refuses while a file it would replace holds
uncommitted work, so commit first. Then it does all of this in one run.

- The kit's files land: the rulebook and its specs under `.agents/`,
  the front doors and `CONTRIBUTING.md` at the root, the hooks, the
  review's sub agent definitions and the chat style in the agent
  folders, and the version stamp.
- Your own rules are kept. A `## 0. Project Rules` section you already
  had in `AGENTS.md`, `CLAUDE.md` or `GEMINI.md` goes to the top of the
  new `AGENTS.md`. If you had no such section but a rules file of your
  own under one of those names, the first of them goes there whole, for
  your agent to trim against the rulebook, and any other stays in git
  where the report says.
- Your own hooks stay. The kit's hooks are added beside them in each
  agent's config, and any other setting of yours is left as it was.
  The kit's own two settings for Claude Code, the sub agent model and
  how many tool calls may run at once, are added the same way, and a
  value you already set for either stays yours.
- Your `.gitignore` keeps every line it had. The kit's lines go in a
  block at the top, so your own lines sit below them and win.
- The tool's doctor runs once, builds the indexes and makes the skill
  links Claude Code needs.
- The report names each root file it replaced along with where git
  keeps the old one, and ends with what it found that belongs in the
  brain, for your agent to file. Keep that output, nothing prints it
  again.

A second port refuses, since the brain is there. An upgrade straight
after it writes nothing.

> **Two things every path leaves to you.**
>
> 1. `CODEOWNERS`: copy the kit's in and replace `@ecstra` with your
>    own handle. The port never touches ownership.
> 2. `## 0. Project Rules`: optional. It goes at the top of `AGENTS.md`,
>    directly under the title and above the comment line, and holds what
>    the general rules cannot know about your project. The gates (the
>    exact test suites, analyzers, builds, and validators that must be
>    green before a review), the tooling (how PRs are opened, how the
>    app runs), anything else the general sections do not cover. A
>    project with no gates yet has nothing to put there, so skip it and
>    add it when the first gate exists.

Commit, and start your first session. The agent reads `.agents/INDEX.md`
on its own, you do not need to point at anything.

From then on `.agents/` fills up as a side effect of the work. The agent
writes docs as it builds things, files todos and bugs as they come up,
records decisions when they are made, and keeps small facts it learns.
You will never sit down to "maintain" it.

## Porting an existing project

> **Do not follow these steps by hand, the prompt is below.** Paste it
> into your agent and it runs the whole flow. The steps here describe
> what the agent will do, so you know what to expect and what to check.

Most projects that adopt the kit already have a pile of docs, an old
`CLAUDE.md`, a TODO list somewhere, and notes nobody has opened in
weeks. None of that is lost. All of it gets filed into `.agents/`, and
the agent does the filing, because there is a lot of it and every file
needs the right header block and index line.

Porting builds on the installation above, so do that first, any of the
three ways. Nothing of yours is thrown away by the port. The first old
rules file of yours is carried into section 0 for the agent to trim,
every root file the kit replaces is named in the report with where git
keeps it, and your own hooks and ignore lines stay where they were. The
port ends by listing what it found that belongs in the brain, and that
list is where the agent starts, so paste the port's output into the
prompt below if the agent did not run the port itself.

### What the agent will do

1. **Sync with the new rulebook first**: read the new `AGENTS.md` and
   the rules it points at, then work by them from the first move. The
   port itself obeys the kit's rules, so prose is edited with the proper
   edit tools, no scripts, the structure inside `.agents/` is the tool's,
   and the house style holds throughout.
2. **Read everything**: the whole project, the code, every doc and note
   it can find, then the specs in `.agents/spec/` so it knows the target
   format. It shows you a plan before it moves anything.
3. **File each thing where it belongs**: on its own branch, one file at
   a time, through the brain tool, `brain new` with the prose on
   standard input, per the map below. The tool writes the header block,
   the id and the index line, so none of them is typed, and the hook
   refuses a brain file written any other way. Your wording is kept in
   todos and nothing gets shortened on the way.
4. **Trim the old rules, never keep them whole**: what the port carried
   into `## 0. Project Rules` is read against the new rulebook. Whatever
   the general sections already say is dropped, and only what is truly
   project specific stays at the top of the new `AGENTS.md`. Your old
   `CONTRIBUTING.md` is read the same way, its extra rules into section 0.
5. **Do all of it**: a project can hold hundreds or thousands of files,
   old and new, and the port covers every one, never a sample. If the
   volume calls for it, the agent lists everything in a ledger first and
   works through it in batches, ticking each file off, so nothing is
   skipped and a later session can pick up where it stopped.
6. **Clean up**: once a doc lives in `.agents/`, the old copy is
   deleted. The one exception is anything a person wrote by hand, which
   stays where it is, untouched. The agent tells the two apart by the
   writing itself, since a human's notes do not read like generated
   docs, and it lists what it left alone so you can decide. Then it runs
   the tool's doctor with `--fix` and leaves the brain clean.
7. **Hand you the branch**: you say when it is ready for review, and
   the agent runs the review the way the house rules describe, findings
   in a ledger, fixed in rounds on your ack of the batches. Then it
   opens the PR and merges, both on your ask. When the PR opens and
   whether as a draft is your project's own rule in section 0, so the
   agent never asks. After that the root holds `README.md`, `AGENTS.md`,
   `CLAUDE.md`, `GEMINI.md`, `CONTRIBUTING.md`, and `CODEOWNERS`, plus
   your `docs/` folder if you keep one. What a person wrote by hand
   stays as it is, and anything new in there links into `.agents/docs/`
   rather than repeating it.

### Where things go

| What you have today | Where it goes |
|---|---|
| Your own `AGENTS.md`, `CLAUDE.md`, or `GEMINI.md` | Carried into `## 0. Project Rules` by the port, then read against the new rulebook. Anything the general rules already cover is dropped, only the truly project specific parts stay. |
| A `docs/` folder, wiki pages, architecture or design notes | `.agents/docs/`, sorted into a category folder (`foundation/`, `systems/`, `ops/`, `research/`), one line each in `.agents/docs/INDEX.md`. What a person wrote by hand stays where it is, and a `docs/` folder you keep links into the brain rather than repeating it. |
| ADRs or a decisions log | `.agents/decisions/`, one file per decision. |
| A TODO list, roadmap, or backlog | `.agents/todos/`, one file per item, your wording kept as is, and a roadmap with its steps as `.agents/plans/`, one file per goal. |
| Known bugs, issue notes | `.agents/bugs/`, one file per bug. |
| Gotchas, tribal knowledge, "remember that" notes | `.agents/memory/`, one fact per file. |
| Your `CONTRIBUTING.md` | Replaced by the kit's, the old one kept by git. Extra rules of yours go into `## 0. Project Rules`. |
| Your `README.md` | Stays at the root, trimmed to what the project is and how to run it, with links onward. Detail moves into `.agents/docs/`. |
| Changelogs, status reports, meeting notes | Not agent knowledge. Left where they are or dropped. |

### The prompt

Paste this into your agent once the kit is in. It covers the whole flow
above.

```text
We are adopting the agent kit that now sits in this repo. Do the port
on a new branch. Before anything else, read the new AGENTS.md and the
rules it points at, and work by them from here on: prose edited with
the proper edit tools, no scripts or heredocs, the header blocks and
indexes inside .agents/ left to the brain tool, house style throughout.
Then read the whole project: the code, every doc and note, and every
TODO list, and the list the port printed of what is left to file, which
I will paste if you did not run the port. Then read .agents/INDEX.md,
everything in .agents/spec/, and the agent-brain skill so you know the
target format and the tool that writes it. Show me a plan of what goes
where before you move anything. Then file it all into .agents/ (docs,
decisions, todos, bugs, memory, everything) one file at a time through
brain new with the prose on standard input, never by hand, keeping my
wording in todos. Section 0 at the top of AGENTS.md holds what the port
carried over from our old rules: read it against the new rulebook, drop
what the general sections already cover, and keep only the truly
project specific parts, and read every other old rules file the port
named (CLAUDE.md, GEMINI.md, CONTRIBUTING.md) from git the same way,
their extra rules into section 0.
This may be hundreds or thousands of files. Do every one, never a
sample. If the volume needs it, list everything in a ledger first and
work in batches, ticking each file off. When it is all filed, delete
the old copies of the docs you moved, but leave anything a human wrote
by hand where it is and list those for me. Run brain doctor --fix and
leave it clean. Then tell me what you did and where, and wait for me to
call the review.
```

## Keeping the kit current

When this repo puts out a new release, bring your copy of the kit up to
date, or clone one beside your project, and run its upgrade from your
project's root.

```bash
git -C ../agent-setup pull || git clone https://forge.aikyat.com/aikyat/agent-setup ../agent-setup
python3 ../agent-setup/.agents/skills/agent-brain/scripts/brain.py upgrade
```

The upgrade replaces what is the kit's and leaves what is yours.

- The kit's: the rulebook, the front doors around your section 0,
  `CONTRIBUTING.md`, the house rules, the specs, the tool and its
  skills, the hooks, the review's sub agent definitions, the chat
  style, and the version stamp. All new over old. A skill folder or a
  definition under a kit name that you made before the kit shipped it
  is refused, not replaced, until you move it aside.
- Yours: `## 0. Project Rules`, which stays above the comment line under
  the title of `AGENTS.md`, every file your project filed into
  `.agents/`, your own hooks and settings in the agent folders, and every
  line of your `.gitignore`.
- Rebuilt: the indexes, by the doctor run at the end, never edited.

It works on the nearest folder at or above you that carries a brain, refuses a
kit older than the one you are on, refuses while a file it would replace
holds uncommitted work, and names each root file it replaces with where
git keeps the old one. Run it on its own branch and read the diff. Or
hand your agent this.

```text
The agent kit at https://forge.aikyat.com/aikyat/agent-setup has a new
release. Pull the clone beside this repo, or clone it there, then run
its upgrade from our root:
python3 <that folder>/.agents/skills/agent-brain/scripts/brain.py upgrade
Read what it printed, fix whatever doctor still names, through brain
set for a header field and by hand for prose, and tell me what changed.
```

### Coming from the older layout

A project that took the kit before the rulebook moved has a `.agent/`
folder, a root `CLAUDE.md` carrying all the rules, and an `AGENTS.md`
that copies it. The same `upgrade` command carries it over. The folder
moves to `.agents/` through git, so the diff shows renames. Every file
git sees that named the old folder is pointed at the new one, and a file
git ignores that names it stops the run, by name, so nothing is left
half done. Your `## 0. Project Rules` comes out of `CLAUDE.md` into the
new `AGENTS.md`, the kit's files land, the ignore lines follow, and
doctor rebuilds every index from the files.

## The brain tool

Everything inside `.agents/` goes through `brain`, the script under
`.agents/skills/agent-brain/scripts/`. Your agent runs it, you rarely
need to, but the commands are short enough to know.

| Command | What it does |
|---|---|
| `brain status` | The clock, where your standing word stands, on, held or off, the work the agent is on, then the brain in numbers, what each tracker holds and whether doctor is clean. The session start hook prints this into the agent's context, and two more hooks keep the clock and the work in front of the agent on every prompt and after tool calls, since a model does not know the time on its own and forgets what it was doing. |
| `brain task on <entry>` | The work the agent is on, one chain at a time from a plan down to the step, which the agent sets before multi step work and moves with `brain task done`, a mark in the entry's own steps table, or which moves of itself when the entry a step points at closes, a refiled one handing the step on to the entry filed in its place. The agent cannot start a second piece of work while one stands, short of `--replace` on your word, and a link falls off the chain when its entry closes. |
| `brain doctor` and `brain doctor --fix` | Checks every file against its spec, and `--fix` repairs the mechanical slips, a stale index, a link written as a name, a chat style copy that drifted. |
| `brain new`, `brain set`, `brain refile` | Files a plan, a todo, a bug, a decision, a doc or a memory with the prose on standard input, changes a header, retitles or refiles. The tool writes the id, the header block and the index line. |
| `brain ledger` | Opens, fills and closes a code review ledger, and `brain ledger report` says where a review stands, counted from the file. |
| `brain ack on --for 8h` | Your standing word to proceed, for a time or until a task is done, so the agent passes the review's gates that normally wait for you, the house rules name them, and each report says which ack the step ran on. It never starts a review, and the PR and the merge still wait for your ask. `brain ack off` takes it back. |
| `brain port` and `brain upgrade` | The two commands above, run from a kit checkout against your project. |

The full contract, every flag and every refusal, is in
[`commands.md`](.agents/skills/agent-brain/references/commands.md), and
[`SKILL.md`](.agents/skills/agent-brain/SKILL.md) is what the agent reads
to use it.

## What `.agents/` looks like

Here is how it ships, empty and ready.

```text
.agents/
├── SYSTEM.md             the rulebook every agent works by
├── VERSION               the kit release this project carries
├── INDEX.md              the spine, built by the tool, its lines set by the conventions
├── README.md             what the directory is and how an agent boots from it
├── house-rules.md        the branch contract and code review protocol
├── skills/agent-brain/   the brain tool, as a skill every agent can read
├── skills/branch-review/ the code review, run on your word
├── spec/                 the format rules, one per content folder
│   ├── conventions.md    the shared rules, header block, naming, links, indexes
│   ├── docs.md
│   ├── memory.md
│   ├── bugs.md
│   ├── todos.md
│   ├── decisions.md
│   ├── plans.md
│   ├── reviews.md
│   └── handover.md
├── docs/INDEX.md         the knowledge library
├── memory/INDEX.md       facts learned along the way
├── plans/INDEX.md        the plans, each a goal with its steps
├── todos/INDEX.md        the work tracker
├── bugs/INDEX.md         the defect tracker
├── decisions/INDEX.md    decisions taken and why
└── reviews/INDEX.md      code review ledgers
```

And here is a filled in example, so you can see the shape once files
land.

```text
.agents/
├── SYSTEM.md
├── INDEX.md
├── README.md
├── house-rules.md
├── spec/
├── docs/
│   ├── INDEX.md
│   ├── foundation/
│   │   ├── architecture.md
│   │   └── stack.md
│   ├── systems/
│   │   ├── auth.md
│   │   └── billing.md
│   └── ops/
│       └── deploy.md
├── memory/
│   ├── INDEX.md
│   └── stripe-webhooks-retry-twice.md
├── plans/
│   ├── INDEX.md
│   └── billing-launch.md
├── todos/
│   ├── INDEX.md
│   ├── TODO-001-export-to-csv.md
│   └── TODO-002-dark-mode.md
├── bugs/
│   ├── INDEX.md
│   └── BUG-001-login-loop-on-expired-token.md
├── decisions/
│   ├── INDEX.md
│   └── DEC-001-postgres-over-sqlite.md
├── reviews/
│   ├── INDEX.md
│   └── 2026-09-feat-billing/
│       └── ledger.md
└── handover/
    └── billing-cutover.md
```

The trackers, the docs library, and memory each carry an `INDEX.md`
listing what is inside, and `.agents/INDEX.md` at the top names each
of those folders with its counts. `spec/` and `handover/` carry none,
the spine covers both.
`handover/` stays empty until a session hands work over, so a fresh
clone has no such folder until then.

## Where to read next

- [`.agents/SYSTEM.md`](.agents/SYSTEM.md): the full rulebook.
- [`.agents/README.md`](.agents/README.md): how a cold agent boots.
- [`.agents/house-rules.md`](.agents/house-rules.md): the branch contract and review protocol.
- [`.agents/spec/conventions.md`](.agents/spec/conventions.md): the shared rules every file in `.agents/` obeys.
- [`.agents/skills/agent-brain/references/commands.md`](.agents/skills/agent-brain/references/commands.md): every command of the tool, with its flags and refusals.
- [`.agents/skills/branch-review/SKILL.md`](.agents/skills/branch-review/SKILL.md): the review as the agent runs it, step by step.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the absolutes on one page.
