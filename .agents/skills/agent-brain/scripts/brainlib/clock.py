"""The clock the hooks put in front of the agent.

A model does not know the time and forgets what it was told, so the
hooks print this line at the start of a session, on every prompt and
after tool calls, and `brain ack` and `brain status` print it too,
since both answer with where the owner's standing word stands, which
turns on the time.
"""

from datetime import datetime
from typing import Optional


def now() -> datetime:
    """The local time, aware of its zone."""
    return datetime.now().astimezone()


def line(
    at: Optional[datetime] = None,
) -> str:
    """One line naming the day, the date, the time and the zone, the way every hook prints it."""
    at = at or now()

    return f"now: {at:%A} {at:%Y-%m-%d %H:%M} {at:%Z}"
