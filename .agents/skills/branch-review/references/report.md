# The reports

Every report follows SYSTEM.md section 7.6. The state lines say in
plain words what `brain ledger report` prints, never from memory, a
finding by what it is with its id beside it. Never a tip line.

Of the review's own run a report says only what five lines say, the
one exception 7.6 names, since the owner measures the review by them.
They are the hunts line, the cut short line, the decided line, the
time line and the line that says the review ran without sub agents.

While the owner's standing word holds, a report's last line, where it
starts a batch or the closing pass on the owner's word, says instead
that the step started on the standing word and what is left of it,
"Batches 1 to 3 started on your standing word, 2 h left." A batch that
already carried the owner's dated ack started on that ack, and the
line says so instead, "Batch 2 started on your ack of 2026-09-22."
The same holds with the standing word off. When `brain ledger report`
says the next batch has the word, the owner having acked it ahead,
the last line says it starts on that ack rather than asking for one,
"Batch 3 starts on your ack of 2026-09-23." A finding put to the owner
waits for their own word all the same, and the close and the merge
lines stay as written, since the close follows the last batch with no
word and the PR and the merge wait for the owner's ask either way.

## After a finding pass

The shape after the finding pass, and after the closing pass.

```text
<what brain ledger report prints, in plain words>

Batch 1, <theme>
- <file>:<line>, <what is wrong, in plain words>
Batch 2, <theme>
- ...

- Cut short: <the finder, its angle and its slice or the whole range>, candidates past its cap remained after its last run.
- Decided: <the case the skill sheet leaves open>, <what was decided>.
Batches <n> to <m> start on your word, in number order. (or: Batch <n> starts on your word.)
```

A labelled line appears only where there is one, a line each, here
and in the next shape. A pass that filed nothing lists no batch, and
its report is the closing report.

## After the rounds

The shape once the last pass of the batches the word covered is done.

```text
<what brain ledger report prints, in plain words>

Fixed in batch 1, <theme>
- <file>, <what changed, in plain words>
Fixed in batch 2, <theme>
- ...
The hunts found <n> more. (or: The hunts found nothing.)

- Refuted: <the claim, in plain words> (F-nn), <why it is wrong>.
- Left as it is: <the finding, in plain words> (F-nn), <the reason>.
- Yours to call: <the finding, in plain words> (F-nn), <what was tried, or why it cannot be fixed here>.
- Sent to the bugs: <the finding, in plain words> (F-nn), filed as <the bug, in plain words> (BUG-nnn), on your word.
- Older than the branch: <the defect, in plain words> (BUG-nnn), filed as a bug.
- Cut short: <the hunter and its slice>, candidates past its cap remained after its last run.
- Decided: <the case the skill sheet leaves open>, <what was decided>.
Batches <n> to <m> start on your word, in number order. (or: Batch <n> starts on your word.) (or: The closing pass starts on your word.)
```

The last line asks for what the [skill sheet](../SKILL.md) puts next,
a batch or the closing pass, and says so where a batch waits for the
owner's word on a finding, "Batch 3 waits for your word on the finding
above." Where nothing comes next, the report is the closing report.

## The closing report

The report the review ends on, which no report follows. It keeps the
shape of the step it ends on, with these two lines in place of its
last line, the time as the [skill sheet](../SKILL.md) counts it.

```text
The review took <the time, such as 1 h 40 min>.
The ledger is closed, and the branch is ready to merge on your ask.
```

## Without sub agents

The same shapes, with one line right after the state lines saying that
this review ran without sub agents, every slice and angle in one
context.
