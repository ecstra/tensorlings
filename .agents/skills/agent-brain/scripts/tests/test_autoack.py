"""The owner's standing word to proceed has an end it never runs past, and the tool reads it truthfully."""

import contextlib
import os
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional
from unittest import mock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from brainlib import autoack, clock, writes
from tests import fixtures

NOON = datetime(2026, 9, 23, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def named_zone(
    name: str,
) -> Optional[ZoneInfo]:
    """The zone, or None where the machine carries no zone data, so the tests that need it skip rather than the module failing to load."""
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return None


LONDON = named_zone("Europe/London")


@contextlib.contextmanager
def machine_zone(
    name: str,
) -> Iterator[None]:
    """The machine's own zone set to the named one for the block, and put back after, or the test skipped where the C library carries no data for it and falls back to UTC."""
    before = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        if (time.timezone, time.daylight) == (0, 0):
            raise unittest.SkipTest(f"the machine carries no zone data for {name}")

        yield
    finally:
        if before is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = before
        time.tzset()


class TurningOn(fixtures.BrainCase):
    def test_a_word_without_an_end_is_refused(self) -> None:
        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, at=NOON)

        self.assertEqual(str(caught.exception), "auto ack needs an end, a time with --for or --until, a task with --until-done, or both")
        self.assertIsNone(autoack.current(brain=self.brain, at=NOON))

    def test_a_time_ends_it_and_the_earliest_of_two_ends_wins(self) -> None:
        grant = autoack.turn_on(brain=self.brain, hours=8, until_text="15:30", at=NOON)

        self.assertEqual(grant.until, NOON.replace(hour=15, minute=30))
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 3 h 30 min left, until 15:30 today")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON + timedelta(seconds=30)), "auto ack on, 3 h 30 min left, until 15:30 today")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON + timedelta(hours=3, minutes=29)), "auto ack on, 1 min left, until 15:30 today")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON + timedelta(hours=3, minutes=29, seconds=30)), "auto ack on, 1 min left, until 15:30 today")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON + timedelta(hours=4)), "auto ack off")

        grant = autoack.turn_on(brain=self.brain, hours=1, until_text="22:00", at=NOON)

        self.assertEqual(grant.until, NOON.replace(hour=13, minute=0))
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 1 h left, until 13:00 today")

    def test_a_bare_time_already_past_means_tomorrow(self) -> None:
        grant = autoack.turn_on(brain=self.brain, until_text="09:00", at=NOON)

        self.assertEqual(grant.until, (NOON + timedelta(days=1)).replace(hour=9, minute=0))
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 21 h left, until 09:00 tomorrow")

    def test_a_moment_reads_as_tomorrow_a_date_or_a_date_and_time(self) -> None:
        for text, until in (
            ("tomorrow 22:00", (NOON + timedelta(days=1)).replace(hour=22, minute=0)),
            ("2026-09-25", NOON.replace(day=25, hour=23, minute=59, second=59, microsecond=999999)),
            ("2026-09-25 08:15", NOON.replace(day=25, hour=8, minute=15)),
        ):
            with self.subTest(text=text):
                self.assertEqual(autoack.parse_until(text=text, at=NOON), until)

        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")
        autoack.turn_on(brain=self.brain, until_text="2026-09-25 08:15", at=NOON)
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 44 h 15 min left, until 2026-09-25 08:15")

        autoack.turn_on(brain=self.brain, until_text="2026-09-25", at=NOON)
        self.assertIsNotNone(autoack.current(brain=self.brain, at=NOON.replace(day=25, hour=23, minute=59, second=30)))
        self.assertIsNone(autoack.current(brain=self.brain, at=NOON.replace(day=26, hour=0, minute=0)))
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 60 h left, until 2026-09-25 23:59")

    def test_a_duration_reads_in_minutes_hours_or_days(self) -> None:
        self.assertEqual(autoack.parse_duration("90m"), 1.5)
        self.assertEqual(autoack.parse_duration("8h"), 8.0)
        self.assertEqual(autoack.parse_duration("2d"), 48.0)

        for text in ("soon", "8", "-2h", "0h"):
            with self.subTest(text=text):
                with self.assertRaises(autoack.AckError):
                    autoack.turn_on(brain=self.brain, hours=autoack.parse_duration(text), at=NOON)

    def test_a_moment_in_the_past_or_a_bad_one_is_refused(self) -> None:
        for text in ("2026-09-22 10:00", "noon", "25:00"):
            with self.subTest(text=text):
                with self.assertRaises(autoack.AckError):
                    autoack.turn_on(brain=self.brain, until_text=text, at=NOON)

    def test_a_task_ends_it_when_the_entry_leaves_open_and_the_word_is_taken_back(self) -> None:
        grant = autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)

        self.assertEqual(grant.task, "TODO-001")
        self.assertIsNone(grant.until)
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on until TODO-001 is done")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON + timedelta(days=400)), "auto ack on until TODO-001 is done")

        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("status: open", "status: done"))
        self.brain.refresh()

        self.assertIsNone(autoack.current(brain=self.brain, at=NOON))
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")
        self.assertFalse(autoack.path_of(brain=self.brain).exists())

        fixtures.write(path=path, text=text)
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")

    def test_an_unreadable_task_holds_the_word_off_and_says_so(self) -> None:
        autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        text = path.read_text(encoding="utf-8")

        fixtures.write(path=path, text="not a header\n" + text)
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack held, TODO-001 cannot be read right now")
        self.assertTrue(autoack.path_of(brain=self.brain).exists())
        with self.assertRaises(autoack.AckError) as caught:
            autoack.stamp(brain=self.brain, at=NOON)
        self.assertEqual(str(caught.exception), "auto ack is held, TODO-001 cannot be read right now, the owner's word starts a batch")

        fixtures.write(path=path, text=text)
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on until TODO-001 is done")

    def test_a_doubled_id_takes_the_word_back_and_is_refused_as_a_task(self) -> None:
        autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        text = path.read_text(encoding="utf-8")
        twin = self.root / "todos" / "TODO-001-export-to-tsv.md"
        fixtures.write(path=twin, text=text.replace("TODO-001-export-to-csv", "TODO-001-export-to-tsv"))
        self.brain.refresh()

        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")
        self.assertFalse(autoack.path_of(brain=self.brain).exists())

        for key in ("todos/TODO-001-export-to-csv.md", "TODO-001"):
            with self.subTest(key=key):
                with self.assertRaises(autoack.AckError) as caught:
                    autoack.turn_on(brain=self.brain, task=key, at=NOON)

                self.assertEqual(str(caught.exception), "TODO-001 names more than one entry, so auto ack cannot end with it until the doctor clears the clash")

        fixtures.write(
            path=self.root / "reviews" / "2026-10-fix-login" / "ledger.md",
            text=fixtures.LEDGER.format(name="2026-10-fix-login", description="the login review"),
        )
        self.brain.refresh()
        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, task="ledger", at=NOON)

        self.assertIn("use the path", str(caught.exception))

        fixtures.write(
            path=self.root / "reviews" / "2026-09-feat-billing-2" / "ledger.md",
            text=fixtures.LEDGER.format(name="2026-09-feat-billing", description="the billing review, doubled"),
        )
        self.brain.refresh()
        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, task="reviews/2026-09-feat-billing/ledger.md", at=NOON)

        self.assertEqual(str(caught.exception), "2026-09-feat-billing names more than one entry, so auto ack cannot end with it until the doctor clears the clash")

        nameless = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        fixtures.write(path=nameless, text=nameless.read_text(encoding="utf-8").replace("name: 2026-10-fix-login\n", ""))
        self.brain.refresh()
        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, task="reviews/2026-10-fix-login/ledger.md", at=NOON)

        self.assertEqual(str(caught.exception), "reviews/2026-10-fix-login/ledger.md carries no name, so auto ack cannot end with it until it is repaired")

    def test_a_todo_or_a_bug_with_no_id_is_refused_since_the_word_follows_the_id(self) -> None:
        source = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=self.root / "todos" / "tidy-up.md", text=source.read_text(encoding="utf-8").replace("name: TODO-001-export-to-csv\n", "name: tidy-up\n"))
        self.brain.refresh()

        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, task="tidy-up", at=NOON)

        self.assertEqual(str(caught.exception), "todos/tidy-up.md carries no id, so auto ack cannot end with it until it is repaired")
        self.assertFalse(autoack.path_of(brain=self.brain).exists())

    def test_an_end_past_what_every_machine_can_place_is_refused(self) -> None:
        for kwargs in ({"until_text": "3000-01-01"}, {"until_text": "9999-12-30"}, {"hours": 3000000 * 24}, {"hours": 99999999999 * 24}):
            with self.subTest(**kwargs):
                with self.assertRaises(autoack.AckError) as caught:
                    autoack.turn_on(brain=self.brain, at=NOON, **kwargs)

                self.assertEqual(str(caught.exception), "auto ack would end past the year 2999, give a nearer end")

        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, at=NOON, until_text="3000-01-01 08:00")

        self.assertEqual(str(caught.exception), "auto ack would end past the year 2999, give a nearer end")
        self.assertFalse(autoack.path_of(brain=self.brain).exists())

        with mock.patch.object(autoack, "on_the_wall", side_effect=OSError(22, "Invalid argument")):
            with self.assertRaises(autoack.AckError) as caught:
                autoack.turn_on(brain=self.brain, at=NOON, hours=400000 * 24)

        self.assertEqual(str(caught.exception), "auto ack would end past the year 2999, give a nearer end")

    @unittest.skipUnless(hasattr(time, "tzset"), "needs a machine zone the test can set")
    def test_a_moment_this_machine_cannot_place_is_refused_as_the_past_or_the_far_end_it_is(self) -> None:
        class WindowsClock(datetime):
            def astimezone(self, tz=None):  # noqa: ANN001
                if self.year < 1970 or self.year > 3000:
                    raise OSError(22, "Invalid argument")

                return super().astimezone(tz)

        with machine_zone("Asia/Kolkata"), mock.patch.object(autoack, "datetime", WindowsClock):
            for text, said in (
                ("3001-01-01 10:00", "auto ack would end past the year 2999, give a nearer end"),
                ("1969-06-01 10:00", "auto ack would end before it starts, give a time still to come"),
            ):
                with self.subTest(text=text):
                    with self.assertRaises(autoack.AckError) as caught:
                        autoack.turn_on(brain=self.brain, at=NOON, until_text=text)

                    self.assertEqual(str(caught.exception), said)

        self.assertFalse(autoack.path_of(brain=self.brain).exists())

    def test_a_time_of_zero_or_less_is_refused_in_its_own_words(self) -> None:
        for hours in (0, -2, -0.5):
            with self.subTest(hours=hours):
                with self.assertRaises(autoack.AckError) as caught:
                    autoack.turn_on(brain=self.brain, at=NOON, hours=hours)

                self.assertEqual(str(caught.exception), "auto ack lasts for a time above zero")

        self.assertFalse(autoack.path_of(brain=self.brain).exists())

    def test_the_last_day_of_2999_is_still_an_end_on_the_words_own_clock(self) -> None:
        autoack.turn_on(brain=self.brain, at=NOON, until_text="2999-12-31")

        self.assertTrue(autoack.describe(brain=self.brain, at=NOON).startswith("auto ack on, "))
        self.assertTrue(autoack.describe(brain=self.brain, at=NOON).endswith(" left, until 2999-12-31 23:59"))

        behind = NOON.astimezone(timezone(timedelta(hours=-5)))
        autoack.turn_on(brain=self.brain, at=behind, until_text="2999-12-31")
        self.assertTrue(autoack.describe(brain=self.brain, at=behind).endswith(" left, until 2999-12-31 23:59"))

    def test_a_task_gone_from_the_brain_takes_the_word_back_for_good(self) -> None:
        autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        text = path.read_text(encoding="utf-8")

        path.unlink()
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")
        self.assertFalse(autoack.path_of(brain=self.brain).exists())

        fixtures.write(path=self.root / "todos" / "TODO-001-dark-mode.md", text=text.replace("TODO-001-export-to-csv", "TODO-001-dark-mode"))
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")

    def test_a_review_ledger_missing_from_the_checkout_holds_the_word_until_it_is_back(self) -> None:
        autoack.turn_on(brain=self.brain, task="2026-09-feat-billing", at=NOON)
        folder = self.root / "reviews" / "2026-09-feat-billing"
        aside = self.root.parent / "aside"

        folder.rename(aside)
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack held, 2026-09-feat-billing is not in this checkout, brain ack off clears it")
        with self.assertRaises(autoack.AckError) as caught:
            autoack.stamp(brain=self.brain, at=NOON)
        self.assertEqual(str(caught.exception), "auto ack is held, 2026-09-feat-billing is not in this checkout, brain ack off clears it, the owner's word starts a batch")
        self.assertTrue(autoack.path_of(brain=self.brain).exists())

        aside.rename(folder)
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on until 2026-09-feat-billing is done")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot list")
    def test_a_folder_the_brain_cannot_list_keeps_the_word_in_place(self) -> None:
        for task, folder in (
            ("TODO-001", self.root / "todos"),
            ("2026-09-feat-billing", self.root / "reviews" / "2026-09-feat-billing"),
        ):
            with self.subTest(task=task):
                autoack.turn_on(brain=self.brain, task=task, at=NOON)
                mode = folder.stat().st_mode

                os.chmod(folder, 0)
                try:
                    self.brain.refresh()
                    self.assertEqual(autoack.describe(brain=self.brain, at=NOON), f"auto ack held, {task} cannot be read right now")
                    self.assertTrue(autoack.path_of(brain=self.brain).exists())
                finally:
                    os.chmod(folder, mode)

                self.brain.refresh()
                self.assertEqual(autoack.describe(brain=self.brain, at=NOON), f"auto ack on until {task} is done")

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0, "needs a folder the process cannot list")
    def test_a_task_in_a_subfolder_the_brain_cannot_list_keeps_the_word_in_place(self) -> None:
        archive = self.root / "todos" / "archive"
        archive.mkdir()
        (self.root / "todos" / "TODO-001-export-to-csv.md").rename(archive / "TODO-001-export-to-csv.md")
        self.brain.refresh()
        autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)

        os.chmod(archive, 0)
        try:
            self.brain.refresh()
            self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack held, TODO-001 cannot be read right now")
            self.assertTrue(autoack.path_of(brain=self.brain).exists())
        finally:
            os.chmod(archive, 0o755)

        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on until TODO-001 is done")

    def test_a_word_is_dropped_only_while_the_file_still_holds_it(self) -> None:
        old = autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)
        new = autoack.turn_on(brain=self.brain, hours=8, at=NOON)

        autoack.forget(brain=self.brain, grant=old)
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 8 h left, until 20:00 today")

        autoack.forget(brain=self.brain, grant=new)
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")

    def test_a_word_its_task_ended_stays_off_when_the_file_cannot_be_dropped(self) -> None:
        autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("status: open", "status: done"))
        self.brain.refresh()

        with mock.patch.object(Path, "unlink", side_effect=PermissionError("read only")):
            self.assertIsNone(autoack.current(brain=self.brain, at=NOON))
            self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off, .ack could not be cleared since the brain's folder cannot be written, and the word would stand again if TODO-001 were reopened")

        self.assertTrue(autoack.path_of(brain=self.brain).exists())
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")
        self.assertFalse(autoack.path_of(brain=self.brain).exists())

    def test_a_bug_on_a_branch_still_holds_the_word(self) -> None:
        autoack.turn_on(brain=self.brain, task="BUG-002", at=NOON)
        path = self.root / "bugs" / "BUG-002-login-loop.md"
        text = path.read_text(encoding="utf-8")

        fixtures.write(path=path, text=text.replace("status: open", "status: branched"))
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on until BUG-002 is done")

        fixtures.write(path=path, text=text.replace("status: open", "status: fixed"))
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")

    def test_a_retitled_task_keeps_the_word(self) -> None:
        autoack.turn_on(brain=self.brain, task="TODO-001", at=NOON)

        writes.retitle(brain=self.brain, key="TODO-001", title="export to xlsx")
        self.brain.refresh()

        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on until TODO-001 is done")

    def test_only_an_open_todo_bug_or_review_ends_the_word(self) -> None:
        for key, reason in (
            ("DEC-001", "auto ack ends with a todo, a bug or a review, and DEC-001-postgres-over-sqlite is a decision"),
            ("login", "auto ack ends with a todo, a bug or a review, and login is a doc"),
            ("TODO-003", "TODO-003-dark-mode is already done, so auto ack cannot end with it"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(autoack.AckError) as caught:
                    autoack.turn_on(brain=self.brain, task=key, at=NOON)

                self.assertEqual(str(caught.exception), reason)
                self.assertFalse(autoack.path_of(brain=self.brain).exists())

        grant = autoack.turn_on(brain=self.brain, task="2026-09-feat-billing", at=NOON)
        self.assertEqual(grant.task, "2026-09-feat-billing")

    def test_a_task_the_brain_does_not_know_is_refused(self) -> None:
        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, task="TODO-404", at=NOON)

        self.assertEqual(str(caught.exception), "nothing in the brain answers to 'TODO-404', so auto ack cannot end with it")

    def test_a_task_and_a_time_together_end_on_whichever_comes_first(self) -> None:
        autoack.turn_on(brain=self.brain, hours=2, task="BUG-002", at=NOON)

        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 2 h left, until 14:00 today, or until BUG-002 is done")
        self.assertIsNone(autoack.current(brain=self.brain, at=NOON + timedelta(hours=2)))

        autoack.turn_on(brain=self.brain, hours=2, task="BUG-002", at=NOON)
        path = self.root / "bugs" / "BUG-002-login-loop.md"
        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("status: open", "status: fixed"))
        self.brain.refresh()

        self.assertIsNone(autoack.current(brain=self.brain, at=NOON + timedelta(hours=1)))

        autoack.turn_on(brain=self.brain, hours=400 * 24, task="TODO-001", at=NOON)

        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 9600 h left, until 2027-10-28 12:00, or until TODO-001 is done")

    @unittest.skipUnless(hasattr(time, "tzset"), "needs a machine zone the test can set")
    def test_a_word_given_on_this_machine_follows_the_machines_own_zone_rules(self) -> None:
        with machine_zone("Europe/London"):
            evening = datetime(2026, 10, 24, 20, 0).astimezone()
            self.assertTrue(autoack.on_this_machine(evening))
            self.assertEqual(f"{evening:%z}", "+0100")

            until = autoack.parse_until(text="tomorrow 22:00", at=evening)
            self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2026-10-25 22:00 +0000")
            self.assertEqual(autoack.when(at=evening, moment=until), "22:00 tomorrow")

            later = datetime(2026, 10, 25, 10, 0).astimezone()
            self.assertEqual(autoack.when(at=later, moment=until), "22:00 today")

    @unittest.skipUnless(hasattr(time, "tzset"), "needs a machine zone the test can set")
    def test_a_zone_the_machine_has_no_data_for_skips_the_test_rather_than_failing_it(self) -> None:
        before = os.environ.get("TZ")

        with self.assertRaises(unittest.SkipTest):
            with machine_zone("Nowhere/Nowhere"):
                self.fail("the block ran on a zone the machine could not take")

        self.assertEqual(os.environ.get("TZ"), before)

    @unittest.skipUnless(hasattr(time, "tzset"), "needs a machine zone the test can set")
    def test_a_word_given_in_the_repeated_hour_still_follows_the_machines_rules(self) -> None:
        with machine_zone("America/New_York"):
            second_pass = datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-5)))
            self.assertTrue(autoack.on_this_machine(second_pass))

            until = autoack.parse_until(text="2027-04-01 22:00", at=second_pass)
            self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2027-04-01 22:00 -0400")

            until = autoack.parse_until(text="01:45", at=second_pass)
            self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2026-11-01 01:45 -0500")

            first_pass = datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-4)))
            until = autoack.parse_until(text="01:45", at=first_pass)
            self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2026-11-01 01:45 -0400")

    @unittest.skipUnless(hasattr(time, "tzset"), "needs a machine zone the test can set")
    @unittest.skipIf(LONDON is None, "needs the machine's zone data")
    def test_a_moment_in_a_named_zone_keeps_that_zones_rules_whatever_the_machine_shows(self) -> None:
        with machine_zone("Africa/Lagos"):
            evening = datetime(2026, 10, 24, 20, 0, tzinfo=LONDON)
            self.assertEqual(f"{evening.astimezone():%z}", "+0100")
            self.assertFalse(autoack.on_this_machine(evening))

            until = autoack.parse_until(text="tomorrow 22:00", at=evening)
            self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2026-10-25 22:00 +0000")

    def test_a_time_today_that_has_passed_is_refused_rather_than_rolled_over(self) -> None:
        self.assertEqual(autoack.parse_until(text="today 15:00", at=NOON), NOON.replace(hour=15, minute=0))

        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, until_text="today 09:00", at=NOON)

        self.assertEqual(str(caught.exception), "auto ack would end before it starts, give a time still to come")

    def test_a_review_whose_ledger_cannot_be_read_holds_the_word(self) -> None:
        autoack.turn_on(brain=self.brain, task="2026-09-feat-billing", at=NOON)
        path = self.root / "reviews" / "2026-09-feat-billing" / "ledger.md"
        text = path.read_text(encoding="utf-8")

        fixtures.write(path=path, text="<<<<<<< HEAD\n" + text)
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack held, 2026-09-feat-billing cannot be read right now")
        self.assertTrue(autoack.path_of(brain=self.brain).exists())

        fixtures.write(path=path, text=text)
        self.brain.refresh()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on until 2026-09-feat-billing is done")

    def test_a_held_word_is_off_once_its_time_has_run_out(self) -> None:
        autoack.turn_on(brain=self.brain, hours=1, task="TODO-001", at=NOON)
        path = self.root / "todos" / "TODO-001-export-to-csv.md"
        fixtures.write(path=path, text="not a header\n" + path.read_text(encoding="utf-8"))
        self.brain.refresh()

        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack held, TODO-001 cannot be read right now")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON + timedelta(hours=2)), "auto ack off")
        with self.assertRaises(autoack.AckError) as caught:
            autoack.stamp(brain=self.brain, at=NOON + timedelta(hours=2))
        self.assertEqual(str(caught.exception), "auto ack is off, the owner's word starts a batch")

    @unittest.skipIf(LONDON is None, "needs the machine's zone data")
    def test_the_end_follows_the_zone_across_a_daylight_saving_change(self) -> None:
        evening = datetime(2026, 10, 24, 20, 0, tzinfo=LONDON)

        until = autoack.parse_until(text="tomorrow 22:00", at=evening)
        self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2026-10-25 22:00 +0000")

        until = autoack.parse_until(text="2026-11-02 09:00", at=evening)
        self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2026-11-02 09:00 +0000")

        second_pass = datetime(2026, 10, 25, 1, 30, fold=1, tzinfo=LONDON)
        until = autoack.parse_until(text="2026-10-25 01:45", at=second_pass)
        self.assertEqual(f"{until:%Y-%m-%d %H:%M %z}", "2026-10-25 01:45 +0000")

        autoack.turn_on(brain=self.brain, hours=48, at=evening)
        later = datetime(2026, 10, 26, 10, 0, tzinfo=LONDON)
        self.assertEqual(autoack.describe(brain=self.brain, at=later), "auto ack on, 9 h left, until 19:00 today")


class TurningOff(fixtures.BrainCase):
    def test_off_takes_the_word_back_and_says_whether_there_was_one(self) -> None:
        self.assertFalse(autoack.turn_off(brain=self.brain))

        autoack.turn_on(brain=self.brain, hours=1, at=NOON)

        self.assertTrue(autoack.turn_off(brain=self.brain))
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")
        self.assertFalse(autoack.path_of(brain=self.brain).exists())

    def test_off_says_there_was_nothing_when_a_hook_took_the_word_first(self) -> None:
        autoack.turn_on(brain=self.brain, hours=1, at=NOON)

        with mock.patch.object(Path, "unlink", side_effect=FileNotFoundError("gone already")):
            self.assertFalse(autoack.turn_off(brain=self.brain))

    def test_a_file_that_does_not_read_as_a_word_is_no_word(self) -> None:
        for text in (
            "not json\n",
            '{"since": "2026-09-23T12:00", "until": "2026-09-23T22:00", "task": null}\n',
            '{"since": "2026-09-23T12:00+05:30", "until": null, "task": null}\n',
            '{"since": "2026-09-23T12:00+05:30", "until": "2026-09-23T22:00+05:30", "task": ["x"]}\n',
            '{"since": "2026-09-23T12:00+05:30", "until": "9999-12-31T23:00:00-05:00", "task": null}\n',
            '{"since": "2026-09-23T12:00+05:30", "until": "3000-01-01T12:00:00+00:00", "task": null}\n',
        ):
            with self.subTest(text=text):
                fixtures.write(path=autoack.path_of(brain=self.brain), text=text)

                self.assertIsNone(autoack.current(brain=self.brain, at=NOON))
                self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off, .ack does not read as a word and brain ack off clears it")
                with self.assertRaises(autoack.AckError) as caught:
                    autoack.stamp(brain=self.brain, at=NOON)
                self.assertEqual(str(caught.exception), "auto ack is off, .ack does not read as a word and brain ack off clears it, the owner's word starts a batch")

        fixtures.write(path=autoack.path_of(brain=self.brain), text='{"since": "2026-09-23T12:00+05:30", "until": "3000-01-01T11:59:59.999999+00:00", "task": null}\n')
        self.assertEqual(autoack.read(brain=self.brain), autoack.Grant(since=NOON, until=autoack.LATEST))

        self.assertTrue(autoack.turn_off(brain=self.brain))
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off")

        fixtures.write(path=autoack.path_of(brain=self.brain), text="[" * 100000 + "\n")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off, .ack does not read as a word and brain ack off clears it")
        self.assertTrue(autoack.turn_off(brain=self.brain))

    @unittest.skipUnless(os.name == "posix", "needs a symlink the process can make")
    def test_a_folder_or_a_dangling_link_where_the_word_goes_is_said_so(self) -> None:
        path = autoack.path_of(brain=self.brain)
        os.symlink(self.root / "nowhere.json", path)
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off, .ack does not read as a word and brain ack off clears it")
        self.assertTrue(autoack.turn_off(brain=self.brain))
        self.assertFalse(path.is_symlink())

        os.mkfifo(path)
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off, .ack does not read as a word and brain ack off clears it")
        self.assertTrue(autoack.turn_off(brain=self.brain))
        self.assertFalse(path.exists())

        os.symlink(self.root / "todos", path)
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off, .ack does not read as a word and brain ack off clears it")
        self.assertTrue(autoack.turn_off(brain=self.brain))
        self.assertFalse(path.is_symlink())
        self.assertTrue((self.root / "todos" / "TODO-001-export-to-csv.md").is_file())

        path.mkdir()
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack off, .ack is a folder the tool never made, move it away by hand")
        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_off(brain=self.brain)
        self.assertEqual(str(caught.exception), ".ack is a folder the tool never made, move it away by hand")
        with self.assertRaises(autoack.AckError) as caught:
            autoack.turn_on(brain=self.brain, hours=1, at=NOON)
        self.assertEqual(str(caught.exception), ".ack is a folder the tool never made, move it away by hand")
        path.rmdir()

    @unittest.skipUnless(os.name == "posix", "needs a symlink the process can make")
    def test_the_word_is_written_whole_and_never_through_a_link(self) -> None:
        elsewhere = self.root.parent / "elsewhere.json"
        fixtures.write(path=elsewhere, text="theirs\n")
        os.symlink(elsewhere, autoack.path_of(brain=self.brain))

        autoack.turn_on(brain=self.brain, hours=1, at=NOON)

        self.assertFalse(autoack.path_of(brain=self.brain).is_symlink())
        self.assertEqual(elsewhere.read_text(encoding="utf-8"), "theirs\n")
        self.assertEqual(autoack.describe(brain=self.brain, at=NOON), "auto ack on, 1 h left, until 13:00 today")


class Stamping(fixtures.BrainCase):
    def test_the_stamp_names_auto_and_the_day_while_the_word_stands(self) -> None:
        with self.assertRaises(autoack.AckError) as caught:
            autoack.stamp(brain=self.brain, at=NOON)
        self.assertEqual(str(caught.exception), "auto ack is off, the owner's word starts a batch")

        autoack.turn_on(brain=self.brain, hours=1, at=NOON)

        self.assertEqual(autoack.stamp(brain=self.brain, at=NOON), "auto 2026-09-23")

    def test_the_stamp_names_the_day_the_batch_starts_and_not_the_day_the_word_was_given(self) -> None:
        autoack.turn_on(brain=self.brain, hours=48, at=NOON)

        self.assertEqual(autoack.stamp(brain=self.brain, at=NOON + timedelta(days=1)), "auto 2026-09-24")
        self.assertEqual(autoack.stamp(brain=self.brain, at=NOON.replace(day=24, hour=1)), "auto 2026-09-24")


class TheClock(unittest.TestCase):
    def test_the_line_names_the_day_the_date_the_time_and_the_zone(self) -> None:
        self.assertEqual(clock.line(at=NOON), "now: Wednesday 2026-09-23 12:00 UTC+05:30")
        self.assertTrue(clock.line().startswith("now: "))


if __name__ == "__main__":
    unittest.main()
