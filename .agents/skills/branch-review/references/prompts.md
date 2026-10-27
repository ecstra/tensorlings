# The prompts

Each prompt below, with the lines its section puts into it, is a sub
agent's whole task, and the line above it says when it is sent. The
steps that send them are the skill sheet's, [SKILL.md](../SKILL.md).
The main session fills in the parts in angle brackets, save those in a
block or a line the sub agent writes, which the sub agent fills.

A finding in a prompt is one line, its F-nn and then the answer file
and the P-n of each block that holds it, so a joined finding names each
of its blocks. Where the scratch folder is gone, the line is the F-nn
and its claim from the ledger. Where a finding or a model turns on how
another product behaves, the main session fetches the page of that
product's docs that decides it and adds after the findings the line
"The page of <the product>'s docs on it is in <the file>."

Every prompt ends with these lines.

```text
Write your blocks, or the single line (none) where you have none, to
<the answer file> with the Write tool before you answer, and after them
every line this task puts after the blocks. Your answer is one line per
block, its id and its Type, Result or Verdict, or (none), then every
line this task puts after the blocks, word for word. Change no file of
the project your task does not give you, and never run its whole test
suite. Put each path and range you hand the shell in single quotes.

A case you meet outside the functions the range changed and the code
the change made reachable, which behaves the same on <base>, is older
than the branch, and every other case is the branch's. An older case
is no candidate and never counts against a finding. Name it after your
blocks, one line each, as
Older: <file>:<line> <the defect and the input that shows it>
```

The prompt of a fixer, a docs writer or a sweep has these lines before
them.

```text
<for a second run, "An earlier run of yours worked on this task, and its changes, the files it made among them, are in the working tree, uncommitted.">

Each finding lands with one Result.
- FIXED, once the claim no longer holds and every case passes. A
  finding a change of the review already settles, yours or another
  group's, is FIXED too, and its Cause names that change.
- REFUTED, where the claim was never true.
- LEFT, where the claim is true and what it names is on purpose.
- BLOCKED, where the fix needs a mechanism no model of yours names,
  such as a parser, or a change in a file you do not own, which your
  Outside names. A finding older than the branch as a whole is BLOCKED
  too, beside its Older line.

Write one block per finding in this shape:
### <F-nn>
- Result: <FIXED or REFUTED or LEFT or BLOCKED>
- Cause: <for FIXED the cause in one sentence, for REFUTED or LEFT the line or the doc that shows it, quoted, for BLOCKED what the fix needs>
- Cases: <for FIXED the sibling cases the fix covers, otherwise none>
- Test: <for FIXED the test's file and name and that it failed before the fix, or what you checked where no test can drive the claim, otherwise none>
- Outside: <the file you do not own and the change it needs, or none>
After the blocks write one line `Outside: <the id>` per block whose
Outside is not none, then one line per doc outside your files that
your change left stale, as
Doc: <the doc> <what it says that no longer holds>
then one line `Files:` with every file of the project your work added
or took away against HEAD, or none. Take away no file outside your
list. Do not commit, stash, switch branch or make a worktree, and do
not write the ledger.
```

## The finder

Sent to each finder of the finding pass, and again, fresh, to one
whose answer ended with (more remain).

```text
The review's range is <base>...<head> in <project root>, and its files
are these, one per line:
<the scope's files>

Your slice, the part of the range you judge, one per line:
<the slice's files, a part of a big file as "<file> lines <from> to <to>" as HEAD numbers them, or every file of the scope for an angle that reads the whole branch>
The rest of the range is other finders', and you read it only to
understand your own.

Your angle:
<the angle's text from angles.md>
<for a closing finder, the line of its section>
<where earlier answers exist, "Found already, and reported, are the candidates in these files" and their paths, one per line>

Report every candidate you can name a scenario for, the rare and the
small included, and none that is found already. One you half believe
still goes on the list, since the fixer checks every claim before it
changes anything. Where more than <the cap> deserve a scenario, write
the ones that matter most and end your answer with the single line
(more remain).

Write one block per candidate, numbered C-1 onward, in this shape:
### C-1
- File: <the path from the project root>:<line>
- Type: <breaks or bug or debt or nit>
- Claim: <the defect in one sentence>
- Scenario: <the concrete input or state and the wrong result>
```

## The second look

Sent at level max, once per slice when every run of the finders is
back, as the finder prompt with the second look angle, and with every
candidate that names a file of the slice as found already.

## The merge

Sent to one finder per group of files once every run of the finding
pass, of a hunt or of the closing pass is back.

```text
You merge the candidates the <finders or hunters> of a code review
returned for one group of files, in <project root>, range
<base>...<head>. They are in <the file>, each block headed by the
answer file of its run and its C-n. Make one list of findings of them,
judge no claim true or false, and drop none.

Your group's files, one per line, and a file off this list is another
group's:
<the files>

1. Two candidates are one finding when one fix of one cause settles
   both, and a further case of that cause joins the finding's scenario
   as a sibling case. When in doubt, keep them apart. Every candidate
   lands in exactly one finding.
2. The type is graded by what the defect does, never by the rule it
   breaks, by the four grades of `.agents/spec/reviews.md`.
3. The group is the file that holds the cause, which is where the fix
   goes and rarely where the defect shows. A test file goes with its
   code file.
4. Needs names a change the fix cannot do without in a file of another
   group. A doc the fix leaves stale is no need, since the last pass
   brings the docs up to the code.
5. The theme is one of five, which in their order are correctness,
   steps that cannot be followed, docs against the code or each other,
   tests and facts kept twice, and style and wording. A hunter's
   finding has no theme, and names the finding whose fix brought it in
   as its candidates do.

Write one block per finding, numbered P-1 onward in the order of theme
and then of type from breaks to nit, in this shape:
### P-1
- File: <the path from the project root>:<line>
- Type: <breaks or bug or debt or nit>
- Theme: <the theme, or none for a hunter's finding>
- Brought by: <for a hunter's finding the F-nn of the finding whose fix brought it in, otherwise none>
- Group: <the file that holds the cause>
- Claim: <the defect in one sentence on one line, named by its cause>
- Scenario: <the input or state and the wrong result, the sibling cases after it, all on one line>
- Needs: <the file of another group and the change the fix needs there, or none>
- From: <each candidate merged, as its run's answer file and its C-n>
```

## The joiner

Sent to one finder when the mergers are back.

```text
The mergers of a code review in <project root>, range <base>...<head>,
each merged the candidates of one group of files into findings. Their
lists are in these answer files, one per line:
<the mergers' answer files>

Name the findings of different lists that are one, since one fix of
one cause settles them. When in doubt, keep them apart, and judge no
claim true or false. A joined finding is filed once, as the finding of
the group whose file holds the cause, with the others as its sibling
cases.

Write one block per join, numbered J-1 onward, in this shape:
### J-1
- Keeps: <the finding that is filed, as its merger's answer file and its P-n>
- Joins: <each other finding it takes in, as its merger's answer file and its P-n>
- Type: <the grade of the joined finding, by what the defect does>
```

## The model check

Sent once to a fresh verifier when a model is written, which is then
written again with its cases.

```text
A group of review findings in <project root>, range <base>...<head>,
is one mechanism, and its fix will be built to the model in
<the model's file>. Find the cases the model decides wrongly.

The findings, one per line:
<the findings>
<where verdicts stood against an earlier fix, "The verdicts that stood are in these answer files" and their paths>

Read the model whole, then the code the findings name as HEAD has it
and what the project's docs say it supports. Hold each scenario of the
findings, its sibling cases with it, against the rules, then look for
more cases of the kind. A case is yours when a rule refuses an input or
a layout the docs say is supported and no finding asks it to, when a
rule stated for every kind holds for one only, when two rules decide it
differently, or when no rule decides it. A case the model names as a
limit is yours only where it is a scenario of the findings.

Write one block per case, numbered M-1 onward, in this shape:
### M-1
- Verdict: <WRONG where a rule decides it wrongly, CLASH where two rules decide it differently, OPEN where no rule decides it>
- Case: <the input or the layout, and the state it meets>
- Model: <what the model does with it, quoting the rule>
- Right: <what it has to do, and the doc or the line of code that says so>
```

## The fixer

Sent to one fresh fixer per group of a round, and to a fresh fixer for
a change in a file no group owns.

```text
You fix one group of review findings in <project root>, on the branch
<branch>, range <base>...<head>. Your findings, one per line:
<the findings>
<for a change in a file no group owns, "The fixer of <F-nn> asked for it in <its answer file>.">

The files you own, one per line, and a new file you make is yours too.
You edit no other:
<the files>
Other fixers edit other files of this project at the same time. A
change you need in a file you do not own goes into your answer, never
into the file.
<for a group built to a model, "The fix is built to the model in <the model's file>, which you read whole first. Never work around a limit it names.">

Read the blocks of your findings whole, and `.agents/SYSTEM.md`
sections 1, 2, 3 and 5 and rule 10.3, with sections 6, 7.5 and 8.1
where your files are docs, and section 0 of AGENTS.md where the project
has one. You write by them. Then, for each finding:
1. Name the cause, the reason the code does this, which is rarely the
   line where it shows.
2. List the sibling cases, every other input or place the same cause
   reaches, an older case aside. One in a file you do not own goes
   under Outside.
3. Where a test can drive the claim, write one test over the scenario
   and its sibling cases, as rows of a table the test files have where
   one covers the function, and see each case fail on the code as HEAD
   has it. Where your changes already sit in the tree, see that on a
   copy of the tree at a new path of this run's own in
   <the scratch folder>, never a worktree, and remove the copy after.
4. Fix the cause, so every case passes. Never special case the
   scenario's own input, and never loosen a test, which is to make it
   accept what it refused before or drop a case that still applies. A
   docstring or a help string of your files that describes what you
   change, you change in the same fix, and a comment that does so too.
5. Run the tests of your files, a test file you made among them, with
   <the command that runs a test file, one line per interpreter the gates name, or "none, since your files are docs">
   A test that fails for a file you do not own, which another fixer may
   be changing, you never chase.
```

## The verifier

Sent to a group's verifiers, each over its share, when its fixer or its
docs writer returns.

```text
One group of review findings was fixed in <project root>, range
<base>...<head>, and is not yet committed. The group's files, one per
line:
<the files>
The fixer's answer is in <its answer file>, and a test file of the
group runs with <the command the fixer was given>. The findings you
vote on, one per line:
<the findings of your share>
<for the group's first verifier, "You judge too every test the group's diff adds or changes that no finding names.">
<for a group built to a model, "The fix was built to the model in <the model's file>, which names its limits.">
<for a docs writer's group, "You vote too on its block Rounds, which answers for the docs the rounds' code made stale, <the commit before the first round>..HEAD, and for these files, the answer files whose Doc lines it took and the models whose limits it stated" and their paths>

Read the group's changes against HEAD, each new file whole, and
`.agents/SYSTEM.md` 3.1 and 5.2, with 6.3 and 7.5 where the files are
docs. Vote on each finding the fixer answered FIXED, REFUTED or LEFT,
and leave a BLOCKED one out. For a fixed finding, ask whether the
scenario is gone with its sibling cases, and whether its test fails
with the fix taken out, or where no test can drive the claim, whether
the check its Test line names holds. For a refuted one, ask whether
the line the fixer quoted shows that the claim never held, and for one
left as it is, whether it shows the behaviour is on purpose.

A fix is a NEW PROBLEM where it hides its symptom by a move 5.2 names,
splices raw input into a query or a shell command against 3.1, breaks
what the project's docs say it supports, or writes a line of a doc
that breaks 6.3 or 7.5. A change of behaviour the finding asked for is
none, and nor is a case a model names as a limit. A test has to fail
with the fix taken out, and its name has to say what its body checks.
One that is loosened, which accepts what it refused before or has lost
a case that still applies, is a NEW PROBLEM, and so is one that falls
short, named by the test.

In each pass you may run one snippet to see a scenario, and each fixed
finding's test once with the fix taken out, which you do on a copy of
the tree at a new path of this run's own in <the scratch folder>, never
a worktree, and remove before you answer. Where a doc outside the group
describes behaviour a finding changed, name it after your blocks, one
line each, as
Doc: <the doc> <what it says that no longer holds>

Write one block per finding in this shape:
### <F-nn or Rounds>
- Verdict: <for FIXED GONE or STILL THERE or NEW PROBLEM, for REFUTED REFUTED or REAL, for LEFT LEFT or REAL>
- Why: <for GONE how the files now handle the scenario and its sibling cases, for STILL THERE the input or state that still triggers it, for NEW PROBLEM what the fix broke as file:line with its scenario or the test and what it lacks, for REFUTED or LEFT what the quoted line decides, for REAL the input or state that shows the claim, each quoting the line that decides it>
```

## The follow ups

A fixer or a verifier with more to do is resumed with a follow up. It
carries the files the sub agent needs and the ids it answers for, each
an F-nn or Rounds, or Gate for the change a red gate asks of a fixer,
and each heads a block in the shape of the sub agent's prompt. A file
is an answer file or a model, and a red gate's output goes into a file
in the scratch folder first. The sub agent answers those ids alone, in
an answer file of its own.

## The hunter

Sent after each round's commits, and never after the last pass, to one
fresh hunter per slice of what the round changed, and again, fresh, to
one whose answer ended with (more remain).

```text
You hunt after a round of review fixes in <project root>, range
<base>...<head>. The round landed as <the hashes>, so its changes are
<the commit before the round>..HEAD, and the findings it fixed are in
the ledger at <the ledger's path>, each beside its commit.
<where the rounds so far built to a model, "The models they built to, each with its limits, are in these files" and their paths>

Your slice of the round's changes, one per line:
<the slice's files, a part of a big file as "<file> lines <from> to <to>" as HEAD numbers them>
The rest of the round's changes are other hunters', and you read them
only to understand your own.

Look for one thing only, a defect the round's fixes brought into the
code, which the files before the round did not have. For each function
the round changed, read its callers too, wherever they sit, since a fix
in one place breaks another. A case a model or the project's docs name
as a limit is no candidate. How code is written, which is SYSTEM.md
section 2, is the last pass's, and so is a doc that has not yet
followed the code.
<where earlier answers exist, "Found already, and reported, are the candidates in these files" and their paths>

Report every candidate you can name a scenario for, the rare and the
small included, and none that is found already. One you half believe
still goes on the list. Where more than <the finder's cap> deserve a
scenario, write the ones that matter most and end your answer with the
single line (more remain).

Write one block per candidate, numbered C-1 onward, in this shape:
### C-1
- File: <the path from the project root>:<line>
- Type: <breaks or bug or debt or nit>
- Claim: <the defect in one sentence>
- Scenario: <the concrete input or state and the wrong result>
- Brought by: <the F-nn of the finding whose fix brought it in, which the ledger's commits and git log on the line show>
```

## The docs writer

Sent in the last pass to one fresh fixer per group of docs.

```text
You write the docs of one group after the rounds of a code review in
<project root>, on the branch <branch>, range <base>...<head>. The
rounds' changes are <the commit before the first round>..HEAD.

The docs you own, one per line, and you edit no other file:
<the docs>
Other writers and the sweeps edit other files at the same time. A
change you need in code goes into your answer, never into the file.
<where there are any, "Your findings, one per line" and the findings>
<where a fixer or a verifier named a doc of yours, "The Doc lines in these answer files name docs of yours" and their paths>
<where a model's limits belong in a doc of yours, "The models in these files, whose limits you state in the doc that describes the mechanism" and their paths>

Read `.agents/SYSTEM.md` sections 6, 7.5 and 8.1 and rule 10.3, and
section 0 of AGENTS.md where the project has one. You write by them.
Bring each doc of yours that describes code the rounds changed up to
that code. Settle your findings and the Doc lines too, and state the
limits you were handed. Write again what the code made false, and add
what a doc should state and does not, such as a flag or a refusal.
Change only what those reach, and never promise what the code does not
do. A paragraph you change is wrapped again whole at the width the file
uses.

Besides the blocks of your findings, write one block for the rest of
your work, in this shape:
### Rounds
- Result: <FIXED where you changed a doc, REFUTED where each already said what the code does>
- Cause: <each doc you changed and what it now says, in short, and each Doc line you were handed with what you did>
- Cases: <each limit you were handed and where you stated it, or none>
- Test: <what you checked each doc against>
- Outside: <the code file and the change it needs, or none>
```

## The sweep

Sent in the last pass to one fresh fixer per group of code files.

```text
You make the style sweep of one group of code files after the rounds
of a code review in <project root>, on the branch <branch>, range
<base>...<head>. The rounds' changes are
<the commit before the first round>..HEAD.

The files you own, one per line, and you edit no other:
<the files>
Other sweeps and the docs writers edit other files at the same time.
<where there are any, "Your findings, one per line" and the findings>

Read `.agents/SYSTEM.md` section 2 and rule 10.3, and section 0 of
AGENTS.md where the project has one. Hold to section 2 every line the
rounds wrote in your files, and each line your findings name, found by
its claim where the rounds moved it. Touch another line only where the
fix of one of those needs it, as a rename does. Change no behaviour and
no assertion. A line that cannot keep its rule without a change of
behaviour is LEFT. Run the tests of your files after the sweep with
<the command that runs a test file, one line per interpreter the gates name, or "none, since no test reaches your files">
and take back a change that fails one.

Where the rounds wrote lines in your files, write besides the blocks
of your findings one block for those lines, in this shape:
### Rounds
- Result: <FIXED where you changed a line, LEFT where you only left some, REFUTED where every line already kept its rule>
- Cause: <the kinds of line you changed, with a count for each, or none>
- Cases: <each line you left, as file:line and what keeps it, or none>
- Test: <the tests you ran and that they passed, or none>
- Outside: <the file you do not own and the change it needs, or none>
```

## The sweep's verifier

Sent to a sweep's verifiers when it returns.

```text
One group of code files had its style sweep in <project root>, range
<base>...<head>, over the rounds' changes,
<the commit before the first round>..HEAD, and it is not yet
committed. The group's files, one per line:
<the files>
The sweep's answer is in <its answer file>. You vote on these
findings, one per line, and on Rounds where the sweep wrote it:
<the findings of your share>

Read the sweep's changes against HEAD, each new file whole, and the
lines of the rounds in the group's files, and hold them to
`.agents/SYSTEM.md` section 2 and to section 0 of AGENTS.md where the
project has one. The sweep may change no behaviour. A changed line
that breaks a rule or changes behaviour, the values each call gets and
their order or the tests the runner collects, is a NEW PROBLEM on its
block, and so is a change outside what the sweep holds that no fix of
it needs. A REFUTED finding stands where the line the sweep quoted
shows that the claim never held, and a REFUTED Rounds where its lines
kept the rule already. A LEFT block stands where behaviour needs the
line as it is. A line of the rounds the sweep left that breaks a rule
with nothing that needs it leaves Rounds STILL THERE.

Write one block per id you vote on, in this shape:
### <F-nn or Rounds>
- Verdict: <for FIXED GONE or STILL THERE or NEW PROBLEM, for REFUTED REFUTED or REAL, for LEFT LEFT or REAL>
- Why: <the line and the rule that decide it, quoted>
```

## The closing finder

Sent where a closing pass is owed, as the finder prompt over the
branch as it stands, each finder run once, with this line after the
angle.

```text
The branch was reviewed, and the ledger at <the ledger's path> holds
every finding with its status and its reason where it has one. Each is
found already, one not fixed settled for its reason, save a fixed one
whose scenario or a sibling case still fails at HEAD, which you report.
Report under your angle what the fixes brought in or the review missed.
<where the rounds built to a model, "The models they built to are in these files, and a case one names as a limit is no candidate" and their paths>
```
