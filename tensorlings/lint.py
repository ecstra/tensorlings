"""The lesson lint, part of the maintainer check."""

from __future__ import annotations

import ast
import io
import re
import tokenize

from rich.console import Console
from rich.markdown import Markdown

from .course import Exercise
from .engine import assigned_names
from .text import CODE_THEME

# Terms a beginner cannot be expected to know. Each must be introduced by a
# lesson, through Exercise.introduces, before any lesson or hint uses it.
JARGON: tuple[str, ...] = (
    "model",
    "training",
    "tensor",
    "shape",
    "dimension",
    "index",
    "slice",
    "token",
    "label",
    "padding",
    "elementwise",
    "batch",
    "broadcasting",
    "reduction",
    "weight",
    "weighted sum",
    "dot product",
    "matrix",
    "matrix multiplication",
    "vector",
    "scalar",
    "gradient",
    "derivative",
    "loss",
    "layer",
    "neuron",
    "activation",
    "embedding",
    "softmax",
    "parameter",
    "bias",
    "logit",
    "optimizer",
    "epoch",
    "backpropagation",
    "autograd",
    "attention",
    "transformer",
    "convolution",
    "chain rule",
    "computation graph",
    "topological order",
    "feature",
    "regression",
    "prediction",
    "learning rate",
    "gradient descent",
    "least squares",
    "minibatch",
    "probability distribution",
    "likelihood",
    "cross entropy",
    "one hot",
    "overfitting",
    "validation",
    "weight decay",
    "variance",
    "fan in",
    "dropout",
    "policy",
    "episode",
    "information coefficient",
    "walk forward",
)

FILLER: tuple[str, ...] = (
    "simply",
    "just",
    "basically",
    "obviously",
    "clearly",
    "easily",
    "essentially",
    "of course",
    "note that",
    "it is important",
    "delve",
    "leverage",
    "utilize",
    "crucial",
    "robust",
    "seamless",
    "unlock",
    "pivotal",
)

LONGEST_SENTENCE = 25
TALLEST_LESSON = 80
LESSON_WIDTH = 96

# A paragraph that opens with a bold label, like "**Why it exists.**", reads as a filled in template.
BOLD_LABEL = re.compile(r"^\*\*[^*\n]+[.:]\*\*", re.MULTILINE)

# A solution shorter than this many code tokens is too generic to count as copied, like "x @ y".
SHORTEST_ANSWER = 6

# A solution line shorter than this many characters, like "return x" or "pass", gives nothing away in a hint.
SHORTEST_LINE = 20

CODE_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d*)?|\S")


def prose_sentences(text: str) -> list[str]:
    """The sentences of a lesson, with code blocks and inline code taken out."""
    without_blocks = re.sub(
        pattern=r"```.*?```",
        repl="",
        string=text,
        flags=re.DOTALL,
    )
    without_code = re.sub(
        pattern=r"`[^`]*`",
        repl="CODE",
        string=without_blocks,
    ).replace("**", "")
    sentences: list[str] = []

    for block in re.split(r"\n\s*\n", without_code):
        lines = [line.strip() for line in block.splitlines() if line.strip()]

        if lines and lines[0].startswith("|"):
            sentences.extend(line.replace("|", " ") for line in lines if not set(line) <= set("|-: "))
            continue

        if lines and lines[0].startswith("- "):
            sentences.extend(line[2:] for line in lines)
            continue

        sentences.extend(re.split(r"(?<=[.!?])\s+", " ".join(lines)))

    return [sentence for sentence in sentences if sentence.strip()]


def mentions(
    text: str,
    term: str,
) -> bool:
    return re.search(rf"\b{re.escape(term)}(s|es)?\b", text, flags=re.IGNORECASE) is not None


def code_tokens(code: str) -> list[str]:
    """Code as tokens, so spacing and comments don't hide a copy."""
    tokens: list[str] = []

    for line in code.splitlines():
        tokens.extend(CODE_TOKEN.findall(line.split("#")[0]))

    return tokens


def solution_bodies(solution: str) -> dict[str, list[str]]:
    """Each solution function's body as code tokens, with the return keyword left out,
    the value of each top level name the solution sets, and each method's body named like Value.__mul__."""
    bodies: dict[str, list[str]] = {}

    for node in ast.parse(solution).body:
        if isinstance(node, ast.Assign):
            for name in assigned_names(node):
                bodies[name] = code_tokens(ast.get_source_segment(solution, node.value))

            continue

        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    bodies[f"{node.name}.{method.name}"] = body_tokens(
                        function=method,
                        solution=solution,
                    )

            continue

        if isinstance(node, ast.FunctionDef):
            bodies[node.name] = body_tokens(
                function=node,
                solution=solution,
            )

    return bodies


def body_tokens(
    function: ast.FunctionDef,
    solution: str,
) -> list[str]:
    tokens: list[str] = []

    for statement in function.body:
        if isinstance(statement, ast.Return) and statement.value is not None:
            statement = statement.value

        tokens.extend(code_tokens(ast.get_source_segment(solution, statement)))

    return tokens


def shown_code(text: str) -> str:
    """Every code block and inline code span of a text, as tokens, with a bar between the pieces."""
    pieces = re.findall(r"```.*?```|`[^`\n]+`", text, flags=re.DOTALL)
    tokens: list[str] = []

    for piece in pieces:
        code = piece.strip("`").removeprefix("python").removeprefix("text")
        tokens.extend(code_tokens(code))
        tokens.append("|")

    return " " + " ".join(tokens) + " "


def written_lines(exercise: Exercise) -> dict[str, str]:
    """Each solution line the learner has to write, keyed to its code as tokens, with a leading return and a
    trailing comma left out. Short lines and lines the starting file already shows, as code or in a TODO, are left out."""
    comment_columns: dict[int, int] = {
        token.start[0]: token.start[1]
        for token in tokenize.generate_tokens(io.StringIO(exercise.solution).readline)
        if token.type == tokenize.COMMENT
    }
    # A TODO that wraps opens each of its lines with a #, so the marks come off and the TODO reads as one run.
    starting_text = " ".join(line.strip().lstrip("#") for line in exercise.source.splitlines())
    shown = f" {' '.join(CODE_TOKEN.findall(starting_text))} "
    lines: dict[str, str] = {}

    for number, line in enumerate(exercise.solution.split("\n"), start=1):
        code = " ".join(line[: comment_columns.get(number)].split()).removesuffix(",")

        if len(code) < SHORTEST_LINE:
            continue

        written = f" {' '.join(CODE_TOKEN.findall(code.removeprefix('return ')))} "

        if written not in shown:
            lines[code] = written

    return lines


def lint_exercise(
    exercise: Exercise,
    introduced_before: set[str],
    console: Console,
) -> list[str]:
    findings: list[str] = []
    known = introduced_before | set(exercise.introduces)
    # Hints are plain text, so code in them like images.shape is taken out before the prose checks.
    sentences = prose_sentences(exercise.lesson) + [
        sentence for hint in exercise.hints.values() for sentence in prose_sentences(re.sub(r"\w+\.\w+", "CODE", hint))
    ]
    readable = " ".join(sentences)

    for mark in ("—", "–", ";"):
        if mark in readable:
            findings.append(f"uses '{mark}' in prose")

    for sentence in sentences:
        word_count = len(sentence.split())

        if word_count > LONGEST_SENTENCE:
            findings.append(f"a {word_count} word sentence, '{sentence[:70]}...'")

    for word in FILLER:
        if mentions(readable, word):
            findings.append(f"uses the filler word '{word}'")

    # The learner reads the TODOs in their editor, so a term there has to be explained by then too.
    comments = " ".join(
        line.strip().lstrip("#")
        for line in exercise.source.splitlines()
        if line.strip().startswith("#")
    )

    for term in JARGON:
        if term not in known and mentions(readable, term):
            findings.append(f"uses '{term}' before a lesson introduces it")

        if term not in known and mentions(comments, term):
            findings.append(f"the starting file uses '{term}' before a lesson or TODO explains it")

    for label in BOLD_LABEL.findall(exercise.lesson):
        findings.append(f"opens a paragraph with the bold label {label}")

    # The lesson teaches the tools on its own examples. It and the hints never show a task's line as written.
    shown = shown_code(exercise.lesson) + " | " + " ".join(code_tokens(" ".join(exercise.hints.values()))) + " "

    for function, body in solution_bodies(exercise.solution).items():
        if function not in exercise.hints:
            continue

        if len(body) >= SHORTEST_ANSWER and f" {' '.join(body)} " in shown:
            findings.append(f"shows the answer to {function}")

    lines_to_write = written_lines(exercise)

    for function, hint in exercise.hints.items():
        hint_code = f" {' '.join(CODE_TOKEN.findall(hint))} "

        for line, line_code in lines_to_write.items():
            if line_code in hint_code:
                findings.append(f"the hint for {function} gives the solution line '{line}' word for word")

    height = len(
        console.render_lines(
            Markdown(exercise.lesson, code_theme=CODE_THEME),
            console.options.update_width(LESSON_WIDTH),
            pad=False,
        ),
    )

    if height > TALLEST_LESSON:
        findings.append(f"is {height} lines tall at width {LESSON_WIDTH}, more than {TALLEST_LESSON}")

    # One line of a readme left short of the others' indent stops the dedent, and Markdown then shows every
    # other line, heading and links included, as one code block.
    if exercise.chapter.readme[:1].isspace():
        findings.append(f"its chapter's readme starts with spaces, so {exercise.chapter.folder}/README.md shows as a code block")

    return findings
