# AGENTS.md

## 0. Project Rules

1. All work goes straight onto main. No branches and no code reviews, by the owner's word on 2026-11-10. This replaces section 9 of SYSTEM.md and the line in 10.1 that keeps commits off main.
2. Course text, the lessons, TODOs, hints and check messages, is written in clear, natural English that a programmer understands on the first read. The writing rules in `.agents/docs/research/tensorlings-course-design.md` say how.

<!-- The kit's words start here. Section 0 goes above this line. -->

The rules for working in this repo are in @.agents/SYSTEM.md and every
agent works by them. Claude Code and Gemini CLI pull that file into
context on their own. Codex gets it from its session start hook. Anything
else that reads this file as plain text opens it first.

Rules that belong to this project alone go in a `## 0. Project Rules`
section directly under the title above, ahead of the comment line.
Everything from that line on is the kit's, and the upgrade keeps what
sits above it.
