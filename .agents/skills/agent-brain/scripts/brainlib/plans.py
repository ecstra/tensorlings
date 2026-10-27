"""A plan, a goal with the steps that reach it, filed with its table from the start.

The prose is the owner's and the agent's, the why and the done when.
The steps table is the tool's, per `steps`, and a plan always carries
one, empty when the plan is a goal alone and the steps are still to
find out.
"""

from typing import Optional

from . import schema, steps, store, writes


class PlanError(Exception):
    """A plan the tool cannot file as asked, with the reason in the message."""


def create(
    brain: store.Brain,
    title: str,
    description: str,
    body: str,
    values: Optional[dict[str, str]] = None,
    category: str = "",
    links: Optional[list[str]] = None,
    stamp: str = "",
) -> store.Entry:
    """File a plan from its prose, the steps it lists under Steps turned into the table, or an empty table when it lists none, with the fields, the category and the links any other kind takes."""
    fields = dict(values or {})
    fields["by"] = fields.get("by") or "owner"
    if fields["by"] not in steps.BY:
        raise PlanError(f"a plan is set by {' or '.join(steps.BY)}, not '{fields['by']}'")

    text = body if body.strip() else ""
    if not text:
        raise PlanError("a plan needs its prose on standard input, the Why and the Done when at least")

    kind = schema.kind_named("plan")
    if steps.span(text) is None:
        text = steps.with_table(body=text, rows=[], kind=kind)

    return writes.create(
        brain=brain,
        kind=kind,
        title=title,
        description=description,
        values=fields,
        body=text,
        category=category,
        links=links or [],
        stamp=stamp,
    )
