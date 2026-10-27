# The angles

Each angle's text below is what the finder prompt of
[prompts.md](prompts.md) carries as the angle. The level picks the set.

| Level | Angles |
|---|---|
| low | scan |
| medium | scan, removed, callers, rules |
| high | scan, removed, callers, pitfalls, rules, tests, docs |
| max | scan, removed, callers, pitfalls, wrappers, rules, tests, docs, reuse, simplification, efficiency, altitude |

Low fits a small mechanical branch. Medium is the floor for anything
that touches logic or changes an interface. High is the usual level for
a feature branch. Max is for a wide surface or subtle logic. The last
brief below is the second look's, and the [skill sheet](../SKILL.md)
says when it runs. Which levels owe the closing pass is house rule 7.

An angle that judges the changed lines themselves reads by slice, so
each finder has little enough to read in one run. One that follows a
change out of its own lines to where it lands reads the whole branch,
since a slice would cut the path it follows.

| Reads | Angles |
|---|---|
| by slice | scan, pitfalls, rules, tests, reuse, simplification, efficiency, altitude |
| the whole branch | removed, callers, wrappers, docs |

## scan, the line by line read

Read every hunk of the diff, line by line, then the enclosing function
of each hunk, since a bug on an unchanged line of a touched function is
in scope. For every line ask what input, state, timing or platform makes
it wrong. Look for an inverted or wrong condition, an off by one, a null
or missing value dereferenced, a falsy zero read as missing, a wrong
variable from copy and paste, an error swallowed, a regex with an
unescaped metacharacter, a missing await or return. In a doc ask of
every sentence what makes it false against the code or another doc,
and of every step whether it can be followed as written.

## removed, the behaviour the diff took away

For every line the diff deletes or replaces, name the invariant or the
behaviour it enforced, then find where the new code enforces it. Where
you cannot, that is a candidate, whether a dropped guard, a lost error
path, a narrowed validation or a deleted test that covered a real case.

## callers, the change seen from outside

For each function the diff changes, find every caller with grep through
Bash and check whether the change breaks the call, through a new
precondition, a changed return shape or type, a new exception or a
changed order of effects. Check the callees too, since a second change
in the same branch can make a call unsafe.

## pitfalls, the language's own traps

Scan for the classic traps of the diff's language and framework that
the diff introduces. In Python that is a mutable default argument, a
late binding closure, a bare except, a path compared as a string, bytes
read as text without an encoding, a dict changed while iterated. In
JavaScript a falsy zero, loose equality, a closure over a loop variable.
In Go a write to a nil map, a range variable captured. In shell an
unquoted variable, a pipeline whose failure is lost. In any language a
float compared for equality, a timezone or DST drift, a lock held over
a call that can block, a query or a shell command built by splicing.

## wrappers, the type that stands in for another

When the diff adds or changes a type that wraps another, a cache, a
proxy, a decorator or an adapter, check that every method routes to the
wrapped instance and never back through a global, a registry or the
session, which re enters the wrapper or recurses. Check that the wrapper
forwards every method the callers use.

## rules, the house rules the diff breaks

Read `.agents/SYSTEM.md` sections 1 to 3, section 5, section 6,
section 7.5 and section 8.2, and the section 0 of AGENTS.md when the
project has one. Then check every changed line against them. In code
that is an exception swallowed to quiet it, a hack over a symptom, a
test that cannot fail, a comment that restates the code, a name that
hides its meaning, a missing type annotation, a positional call where
the language has keyword arguments, a compressed block with no room to
breathe, a secret in source, a dependency pulled in for what the
standard library covers. In prose it is an em dash, a semicolon, a
colon mid sentence, a triplet by habit, a machine's local path. Flag a
rule only where you can quote the rule and the line that breaks it,
and put both in the candidate.

## tests, what the tests would let through

For every behaviour the diff adds or changes, find the test that fails
without it. Where there is none, that is a candidate. Where there is one,
check that it would fail, since a test that passes whatever the code
does, an assertion loosened to fit a wrong output, a test that exercises
the mock and not the code, or a setup and a teardown that do not match
each let the behaviour through. Never run the whole suite, the gates
did.

## docs, the claims the diff falsifies

Find every document that describes what the diff changed, the root
README, the docs under `.agents/docs/`, a skill's own references, a
docstring, a help string. Check each claim there against the code and
the other docs as they now stand. A claim the diff made false is a
candidate, and so is a new behaviour the docs should state and do not,
such as a flag, a refusal or an exit code.

## reuse, the helper that exists already

Look for new code that does what a helper in the codebase already
does, found by grep through Bash in the shared and the adjacent
modules, and name the helper. The scenario states the cost.

## simplification, the simpler form

Look for complexity the diff adds with no observable effect, redundant
or derivable state, copy and paste with one variation, deep nesting,
dead code left behind, and name the simpler form. The scenario states
the cost.

## efficiency, the cheaper form

Look for work the diff wastes, a read or a computation done twice or
independent work run in sequence, and for blocking work on a hot path,
and name the cheaper form. The scenario states the cost.

## altitude, the fix at the wrong depth

Check that each change fixes its cause at the right depth and not a
symptom above it. A special case layered on shared code is the sign, and
the candidate names the simpler general change.

## the second look, the gaps after everything else

Read the slice's diff and the enclosing functions for defects no
candidate already found names, such as moved or extracted code that
dropped a guard, a default evaluated once, a lock scope that shrank, a
predicate with a side effect, a config default flipped.
