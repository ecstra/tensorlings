# Contributing

The rules of the road, human or agent. The operating version is
`.agents/house-rules.md`, and `.agents/SYSTEM.md` carries agent
behavior.

## The absolutes

- **No commit lands on main directly.** All work rides a branch: one
  branch, one intent.
- **No merge before the branch's code review has closed.** The owner
  calls the timing and the agent runs it, its ledger lives under
  `.agents/reviews/`, and every finding is fixed, or refuted, wontfixed
  or deferred with a written reason, before anything merges.
- **No merge without the owner's explicit ask.** Same for opening the PR
  itself. When it opens and whether as a draft is this project's own
  rule in AGENTS.md section 0.
- **Single author, no co-authors, no AI attribution anywhere.** Not in
  commits, not in PR titles or bodies, no generated-with footers.
- **Gates before review.** Suites, analyzer, builds, validators, and
  `brain doctor` pass and the working tree is clean before a review
  starts. Reviewers judge committed, working code and a sound brain.
- **Secrets stay out of git.** They live in the gitignored root `.env`
  or in the project's own secret store, never in code, docs, `.agents/`,
  or git history.
- **No git worktrees.** Disabled outright for agents. The one exception
  is the owner explicitly asking, confined to `.agents/.worktrees/` and
  torn down after.

## The short loop

Branch from fresh main, work in small undoable commits and keep the
gates green. The owner calls the review and the agent runs it, then
merges on their ask. The PR follows section 0, on their ask. The
branch dies after merge, its ledger is the record.
