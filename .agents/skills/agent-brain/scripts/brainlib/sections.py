"""The `## Heading` blocks a body carries.

Reading only. Nothing here rewrites prose, it reports what is there so a
command can refuse to file something the spec says is incomplete.
"""

import re
from typing import Optional

from . import frontmatter, schema

HEADING = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)

# The two characters the house style bans outright, in prose and in a
# header value alike, since a value flows into an index's prose.
EM_DASH = "—"
EN_DASH = "–"

# Code in a body, fenced or inline. It keeps its own punctuation and
# its own links, so the checks and the rewrites step over it. A fence
# opens and closes at the start of a line, the way markdown reads it, so
# three backticks in the middle of a sentence are just backticks and a
# fence with no closer fences nothing.
FENCE = r"^[ \t]{0,3}```[^\n]*\n.*?^[ \t]{0,3}```[ \t]*$"
INLINE = r"`[^`\n]*`"
CODE = re.compile(f"{FENCE}|{INLINE}", re.DOTALL | re.MULTILINE)


def prose_of(
    text: str,
) -> str:
    """The text with its code taken out."""
    return CODE.sub("", text)


def style_slips(
    text: str,
) -> list[str]:
    """What the house style bans in the text, each said the way doctor says it."""
    found: list[str] = []
    if EM_DASH in text or EN_DASH in text:
        found.append("carries a dash the house style bans")

    if ";" in text:
        found.append("carries a semicolon")

    return found


def code_spans(
    text: str,
) -> list[tuple[int, int]]:
    """Where the code sits in the text, as start and end offsets."""
    return [(match.start(), match.end()) for match in CODE.finditer(text)]


def in_code(
    spans: list[tuple[int, int]],
    position: int,
) -> bool:
    """Whether an offset falls inside one of the code spans."""
    return any(start <= position < end for start, end in spans)


def heading_after(
    body: str,
    start: int,
    depth: int = 6,
    spans: Optional[list[tuple[int, int]]] = None,
) -> int:
    """The offset of the next heading outside code at or after `start`, else the body's end.

    `depth` is the deepest heading level that counts, so 2 finds the
    next `#` or `##` line and steps over the `###` lines under it. A
    caller that walks many headings of one body passes the code spans
    it already has, so the body is scanned for code once.
    """
    if spans is None:
        spans = code_spans(body)

    heading = re.compile(rf"^#{{1,{depth}}}[ \t]", re.MULTILINE)

    for match in heading.finditer(string=body, pos=start):
        if not in_code(spans=spans, position=match.start()):
            return match.start()

    return len(body)


def heading_matches(
    body: str,
) -> list[re.Match]:
    """Every `## Heading` line outside code, in order."""
    spans = code_spans(body)

    return [match for match in HEADING.finditer(body) if not in_code(spans=spans, position=match.start())]


def headings(
    body: str,
) -> list[str]:
    return [match.group(1) for match in heading_matches(body)]


def has(
    body: str,
    title: str,
) -> bool:
    return title.lower() in [found.lower() for found in headings(body)]


def span(
    body: str,
    title: str,
) -> Optional[tuple[int, int]]:
    """Where the text under one heading starts and ends, up to the next heading of any level.

    The title is matched without regard to case, the one rule for every
    reader of a section.
    """
    for match in heading_matches(body):
        if match.group(1).lower() != title.lower():
            continue

        return match.end(), heading_after(body=body, start=match.end())

    return None


def block(
    body: str,
    title: str,
) -> Optional[str]:
    """The text under one heading, up to the next heading of any level."""
    found = span(body=body, title=title)
    if found is None:
        return None

    return body[found[0] : found[1]]


def is_filled(
    body: str,
    title: str,
) -> bool:
    text = block(body=body, title=title)
    if text is None:
        return False

    return bool(text.strip())


def missing(
    body: str,
    kind: schema.Kind,
) -> list[str]:
    """The required headings this body does not carry."""
    return [
        section.title
        for section in kind.sections
        if section.required and not has(body=body, title=section.title)
    ]


def lifecycle_gaps(
    kind: schema.Kind,
    fields: dict[str, str],
    body: str,
) -> list[str]:
    """What a file in its status is required to carry and does not, per the kind's rules.

    The writer asks this before a status change lands, doctor asks it of
    every file as it stands, so the two never disagree on a rule.
    """
    status = fields.get("status", "")

    return [rule.says for rule in kind.rules if rule.status == status and rule_broken(rule=rule, fields=fields, body=body)]


def rule_broken(
    rule: schema.Rule,
    fields: dict[str, str],
    body: str,
) -> bool:
    """Whether the file falls short of the rule in any of the ways it names, said once."""
    if rule.field and not fields.get(rule.field, "").strip():
        return True

    if rule.section and not is_filled(body=body, title=rule.section):
        return True

    return bool(rule.link_kind) and not links_into(fields=fields, kind=schema.KINDS[rule.link_kind])


def links_into(
    fields: dict[str, str],
    kind: schema.Kind,
) -> bool:
    """Whether the header links to an entry of the kind, judged by the shape of the path.

    A path in the kind's folder that is not its index and, for a kind
    with ids, carries the kind's own id in its name. Whether the file is
    there is the link check's business.
    """
    for link in frontmatter.parse_list(fields.get("links", "")):
        parts = link.split("/")
        if len(parts) < 2 or parts[0] != kind.folder or parts[-1] == schema.INDEX:
            continue

        match = schema.FILENAME_WITH_ID.match(parts[-1])
        if kind.has_ids and (match is None or match.group(1) != kind.id_prefix):
            continue

        return True

    return False


def skeleton(
    kind: schema.Kind,
    title: str,
) -> str:
    """The empty body a new file starts with, headings and nothing else."""
    if not kind.sections:
        return f"\n# {title}\n"

    blocks = [
        f"## {section.title}\n" for section in kind.sections if section.required
    ]

    return "\n" + "\n".join(blocks)
