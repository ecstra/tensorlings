"""A ledger's batches and findings are records the tool keeps, not prose."""

import unittest
from unittest import mock

from brainlib import autoack, ledger, sections
from tests import fixtures


class LedgerCase(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        self.entry = ledger.open_ledger(
            brain=self.brain,
            branch="fix/login",
            summary="The login branch, three commits.",
            stamp="2026-10-03",
        )
        self.key = self.entry.name

    def body(self) -> str:
        return self.brain.find(self.key).document.body


class Opening(LedgerCase):
    def test_it_is_named_for_the_month_and_the_branch(self) -> None:
        self.assertEqual(self.entry.relative, "reviews/2026-10-fix-login/ledger.md")
        self.assertEqual(self.entry.field("branch"), "fix/login")
        self.assertEqual(self.entry.field("status"), "open")

    def test_it_carries_the_three_sections_and_an_empty_table(self) -> None:
        self.assertIn("## Summary\n\nThe login branch, three commits.", self.body())
        self.assertIn("| # | Theme | Status | Ack |", self.body())
        self.assertTrue(self.body().rstrip().endswith("## Findings"))


class Batches(LedgerCase):
    def test_batches_number_themselves(self) -> None:
        first = ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        second = ledger.add_batch(brain=self.brain, key=self.key, theme="the docs rule")

        self.assertEqual((first, second), (1, 2))
        self.assertEqual(
            [row["theme"] for row in ledger.batches(self.brain.find(self.key))],
            ["correctness", "the docs rule"],
        )

    def test_a_gap_in_the_numbers_stays_and_the_next_number_goes_up(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_batch(brain=self.brain, key=self.key, theme="the docs rule")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug")
        ledger.add_finding(brain=self.brain, key=self.key, claim="two", severity="bug")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |\n", "").replace("### F-01: one\n\n- severity: bug\n- found-by: review\n- batch: \n- status: open\n- fix:\n", ""))
        self.brain.refresh()

        self.assertEqual(ledger.add_batch(brain=self.brain, key=self.key, theme="hooks"), 3)
        self.assertEqual(ledger.add_finding(brain=self.brain, key=self.key, claim="three", severity="nit"), "F-03")

    def test_the_cap_holds_wherever_a_batch_gets_or_changes_its_theme_and_the_closing_pass_is_outside_it(self) -> None:
        for number in range(1, ledger.MAX_BATCHES + 1):
            ledger.add_batch(
                brain=self.brain,
                key=self.key,
                theme=f"theme {number}",
            )

        closing_number: int = ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="Closing Pass",
        )

        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        cap_refusal = "a review has {} batches at most, file the finding into the batch whose theme fits"
        rename_refusal = f"a review has {{}} batches at most besides the {ledger.CLOSING_THEME}, so batch {closing_number} keeps its theme"
        for batch_to_rename, theme, cap, said in (
            (None, "one theme too many", ledger.MAX_BATCHES, cap_refusal.format(ledger.MAX_BATCHES)),
            (None, "one theme too many", ledger.MAX_BATCHES - 1, cap_refusal.format(ledger.MAX_BATCHES - 1)),
            (closing_number, "one theme too many", ledger.MAX_BATCHES, rename_refusal.format(ledger.MAX_BATCHES)),
        ):
            with self.subTest(batch_to_rename=batch_to_rename, theme=theme, cap=cap):
                before = path.read_text(encoding="utf-8")
                with mock.patch.object(
                    target=ledger,
                    attribute="MAX_BATCHES",
                    new=cap,
                ):
                    with self.assertRaises(ledger.LedgerError) as caught:
                        if batch_to_rename:
                            ledger.set_batch(
                                brain=self.brain,
                                key=self.key,
                                number=batch_to_rename,
                                theme=theme,
                            )
                        else:
                            ledger.add_batch(
                                brain=self.brain,
                                key=self.key,
                                theme=theme,
                            )

                self.assertEqual(str(caught.exception), said)
                self.assertEqual(path.read_text(encoding="utf-8"), before)

        for number, theme, cap in (
            (3, "the hooks", ledger.MAX_BATCHES),
            (3, "the docs", ledger.MAX_BATCHES - 1),
            (closing_number, ledger.CLOSING_THEME.upper(), ledger.MAX_BATCHES),
        ):
            with self.subTest(number=number, theme=theme, cap=cap):
                with mock.patch.object(
                    target=ledger,
                    attribute="MAX_BATCHES",
                    new=cap,
                ):
                    ledger.set_batch(
                        brain=self.brain,
                        key=self.key,
                        number=number,
                        theme=theme,
                    )

                self.assertEqual(ledger.batches(entry=self.brain.find(key=self.key))[number - 1]["theme"], theme)

        fixtures.write(
            path=path,
            text=path.read_text(encoding="utf-8").replace("| 2 | theme 2 | pending | |\n", ""),
        )
        self.brain.refresh()

        self.assertEqual(
            ledger.add_batch(
                brain=self.brain,
                key=self.key,
                theme="one theme more",
            ),
            closing_number + 1,
        )
        self.assertEqual(len(ledger.batches(entry=self.brain.find(key=self.key))), ledger.MAX_BATCHES + 1)

    def test_a_batch_takes_a_status_and_an_ack(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified", ack="2026-10-04")

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertEqual((row["status"], row["ack"]), ("verified", "2026-10-04"))

    def test_a_batch_with_findings_starts_on_the_owners_ack_and_verification_closes_it(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing")

        self.assertEqual(str(caught.exception), "batch 1 holds findings and runs on the owner's ack, give --ack")

        autoack.turn_on(brain=self.brain, hours=2)
        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing")

        self.assertEqual(str(caught.exception), "batch 1 holds findings and runs on the standing word, give --ack auto")
        autoack.turn_off(brain=self.brain)

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified")
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertEqual((row["status"], row["ack"]), ("done", "2026-10-04"))

    def test_any_auto_ack_is_the_standing_word_and_only_starts_a_batch(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)

        for ack in ("Auto", "auto 2026-09-01", "AUTO", "auto-2026-09-01", "autoack"):
            with self.subTest(ack=ack):
                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack=ack)

                self.assertEqual(str(caught.exception), "auto ack is off, the owner's word starts a batch")

        autoack.turn_on(brain=self.brain, hours=2)
        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1, ack="auto")

        self.assertEqual(str(caught.exception), "auto ack is the word at a batch's start, give it with --status fixing")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="Auto")
        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertRegex(row["ack"], r"^auto \d{4}-\d{2}-\d{2}$")
        autoack.turn_off(brain=self.brain)

    def test_a_bad_ledger_or_batch_is_reported_as_that_before_the_word_is_asked_for(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key="2026-10-typo", number=1, status="fixing", ack="auto")
        self.assertEqual(str(caught.exception), "no review ledger answers to '2026-10-typo'")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=9, status="fixing", ack="auto")
        self.assertEqual(str(caught.exception), "2026-10-fix-login has no batch 9")

    def test_a_stamp_written_ahead_is_checked_again_when_the_batch_starts(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | pending | auto 2026-09-01 |"))
        self.brain.refresh()

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing")

        self.assertEqual(str(caught.exception), "auto ack is off, the owner's word starts a batch")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")

        self.assertEqual(str(caught.exception), "auto ack is off, the owner's word starts a batch")

    def test_the_standing_word_lands_on_a_batch_already_started_with_any_status(self) -> None:
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="correctness",
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")
        ledger.add_finding(brain=self.brain, key=self.key, claim="late", severity="nit", batch=1)
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        autoack.turn_on(brain=self.brain, hours=2)

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified", ack="auto")

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertEqual(row["status"], "verified")
        self.assertRegex(row["ack"], r"^auto \d{4}-\d{2}-\d{2}$")
        autoack.turn_off(brain=self.brain)

    def test_a_stamp_a_batch_started_on_stays_past_its_start_while_the_word_stands(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | fixing | auto 2026-09-01 |"))
        self.brain.refresh()
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        autoack.turn_on(brain=self.brain, hours=2)

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified", ack="auto")

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertEqual((row["status"], row["ack"]), ("verified", "auto 2026-09-01"))
        autoack.turn_off(brain=self.brain)

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done", ack="auto")

        self.assertEqual(str(caught.exception), "auto ack is off, the owner's word starts a batch")
        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["status"], "verified")

    def test_the_standing_word_never_writes_over_the_owners_ack_and_says_so(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        autoack.turn_on(brain=self.brain, hours=2)

        for status in ("", "fixing"):
            with self.subTest(status=status):
                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_batch(brain=self.brain, key=self.key, number=1, status=status, ack="auto")

                self.assertEqual(str(caught.exception), "batch 1 already carries the owner's ack 2026-10-04")

        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["ack"], "2026-10-04")
        autoack.turn_off(brain=self.brain)

    def test_a_stale_stamp_on_a_pending_batch_gives_way_to_todays_at_the_start(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | pending | auto 2026-09-01 |"))
        self.brain.refresh()
        autoack.turn_on(brain=self.brain, hours=2)

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="auto")

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertRegex(row["ack"], r"^auto \d{4}-\d{2}-\d{2}$")
        self.assertNotEqual(row["ack"], "auto 2026-09-01")
        autoack.turn_off(brain=self.brain)

    def test_an_empty_batch_with_a_stale_stamp_still_closes_without_a_word(self) -> None:
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="correctness",
        )
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(
            path=path,
            text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | pending | auto 2026-09-01 |"),
        )
        self.brain.refresh()

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")

        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["status"], "done")

    def test_a_stale_stamp_gives_way_to_todays_when_the_batch_starts_without_an_ack(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | pending | auto 2026-09-01 |"))
        self.brain.refresh()
        autoack.turn_on(brain=self.brain, hours=2)

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing")

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertRegex(row["ack"], r"^auto \d{4}-\d{2}-\d{2}$")
        self.assertNotEqual(row["ack"], "auto 2026-09-01")
        autoack.turn_off(brain=self.brain)

    def test_a_batch_that_started_on_the_word_keeps_its_stamp_when_it_reopens_after_the_end(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        autoack.turn_on(brain=self.brain, hours=2)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="auto")
        stamp = ledger.batches(self.brain.find(self.key))[0]["ack"]
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified")
        autoack.turn_off(brain=self.brain)

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing")

        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["ack"], stamp)

    def test_a_batch_is_verified_or_done_once_every_finding_has_landed(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)

        for status in ("verified", "done"):
            with self.subTest(status=status):
                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_batch(brain=self.brain, key=self.key, number=1, status=status, ack="2026-10-04")
                self.assertEqual(str(caught.exception), f"F-01 still open, batch 1 is {status} once every finding has landed")

        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["status"], "pending")

    def test_a_batch_starts_while_an_earlier_one_that_holds_findings_is_still_pending(self) -> None:
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="correctness",
        )
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="the docs",
        )
        ledger.add_finding(
            brain=self.brain,
            key=self.key,
            claim="one",
            severity="bug",
            batch=1,
        )
        ledger.add_finding(
            brain=self.brain,
            key=self.key,
            claim="two",
            severity="bug",
            batch=2,
        )

        ledger.set_batch(
            brain=self.brain,
            key=self.key,
            number=2,
            status="fixing",
            ack="2026-10-04",
        )

        self.assertEqual([row["status"] for row in ledger.batches(entry=self.brain.find(key=self.key))], ["pending", "fixing"])

    def test_a_theme_change_leaves_a_done_batch_without_an_ack_alone(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | done | |"))
        self.brain.refresh()

        ledger.set_batch(brain=self.brain, key=self.key, number=1, theme="correctness, round two")

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertEqual((row["theme"], row["status"], row["ack"]), ("correctness, round two", "done", ""))

    def test_an_empty_batch_closes_without_an_ack(self) -> None:
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="correctness",
        )

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")

        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["status"], "done")

    def test_a_finding_goes_only_into_a_batch_the_table_has(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=3)

        self.assertEqual(str(caught.exception), "2026-10-fix-login has no batch 3")

        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug")

        with self.assertRaises(ledger.LedgerError):
            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=["F-01"],
                changes={"batch": "2"},
            )

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"batch": "1"},
        )
        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["batch"], "1")

    def test_a_theme_or_an_ack_with_a_pipe_or_a_line_break_is_refused(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")

        for theme, said in (("hooks | docs", "the theme cannot hold a pipe, the batch table's columns are pipes"), ("two\nlines", "the theme is one line"), ("  ", "a batch needs a theme")):
            with self.subTest(theme=theme):
                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.add_batch(brain=self.brain, key=self.key, theme=theme)
                self.assertEqual(str(caught.exception), said)

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1, ack="yes | no")
        self.assertEqual(str(caught.exception), "the ack cannot hold a pipe, the batch table's columns are pipes")

        self.assertEqual(len(ledger.batches(self.brain.find(self.key))), 1)

    def test_a_batch_can_be_renamed_and_a_change_with_nothing_in_it_is_refused(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        before = (self.root / "reviews" / "2026-10-fix-login" / "ledger.md").read_text(encoding="utf-8")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_batch(brain=self.brain, key=self.key, number=1)
        self.assertEqual(str(caught.exception), "nothing to set, give a status, an ack or a theme")
        self.assertEqual((self.root / "reviews" / "2026-10-fix-login" / "ledger.md").read_text(encoding="utf-8"), before)

        ledger.set_batch(brain=self.brain, key=self.key, number=1, theme="the hooks")

        self.assertIn("| 1 | the hooks | pending | |\n", self.body())

    def test_a_status_outside_the_spec_is_refused(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")

        with self.assertRaises(ledger.LedgerError):
            ledger.set_batch(brain=self.brain, key=self.key, number=1, status="finished")

    def test_a_batch_that_does_not_exist_is_refused(self) -> None:
        with self.assertRaises(ledger.LedgerError):
            ledger.set_batch(brain=self.brain, key=self.key, number=3, status="done")


class Findings(LedgerCase):
    def test_several_claims_file_in_one_write_or_not_at_all(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")

        with mock.patch.object(ledger, "save", wraps=ledger.save) as saves:
            filed = ledger.add_findings(brain=self.brain, key=self.key, claims=["one", "two", "three"], severity="bug", batch=1)

        self.assertEqual(filed, ["F-01", "F-02", "F-03"])
        self.assertEqual(saves.call_count, 1)
        self.assertIn("### F-01: one\n\n- severity: bug\n- found-by: review\n- batch: 1\n- status: open\n- fix:\n\n### F-02: two\n", self.body())

        with self.assertRaises(ledger.LedgerError):
            ledger.add_findings(brain=self.brain, key=self.key, claims=["four", "  "], severity="bug")

        self.assertEqual(len(ledger.findings(self.brain.find(self.key))), 3)

    def test_several_findings_change_in_one_write_or_not_at_all(self) -> None:
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="correctness",
        )
        ledger.add_findings(
            brain=self.brain,
            key=self.key,
            claims=["one", "two", "three"],
            severity="bug",
            batch=1,
        )

        with mock.patch.object(
            target=ledger,
            attribute="save",
            wraps=ledger.save,
        ) as saves:
            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=["F-01", "F-03"],
                changes={"status": "fixed", "fix": "abc1234"},
            )

        self.assertEqual(saves.call_count, 1)

        records: list[dict] = ledger.findings(entry=self.brain.find(key=self.key))
        self.assertEqual([(record["status"], record["fix"]) for record in records], [("fixed", "abc1234"), ("open", ""), ("fixed", "abc1234")])

        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        before = path.read_text(encoding="utf-8")
        for identifiers, changes, said in (
            (["F-02", "F-07"], {"status": "wontfix", "reason": "not worth it"}, "2026-10-fix-login has no finding F-07"),
            (["F-02", "F-01"], {"fix": ""}, "a fixed finding names its commit, fix=<hash>"),
            (["F-02", "F-03", "F-02"], {"status": "wontfix", "reason": "not worth it"}, "F-02 is named twice, name each finding once"),
            ([], {"status": "wontfix", "reason": "not worth it"}, "nothing to set, name a finding like F-03"),
            (["F-02"], {}, "nothing to set, give a change like status=fixed"),
        ):
            with self.subTest(identifiers=identifiers, changes=changes):
                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_findings(
                        brain=self.brain,
                        key=self.key,
                        identifiers=identifiers,
                        changes=changes,
                    )

                self.assertEqual(str(caught.exception), said)
                self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_findings_number_themselves_with_two_digits(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_batch(brain=self.brain, key=self.key, theme="the docs")
        first = ledger.add_finding(brain=self.brain, key=self.key, claim="the guard is inverted", severity="breaks")
        second = ledger.add_finding(brain=self.brain, key=self.key, claim="a stale comment", severity="nit", found_by="hunter", batch=2)

        self.assertEqual((first, second), ("F-01", "F-02"))
        records = ledger.findings(self.brain.find(self.key))
        self.assertEqual(records[1]["claim"], "a stale comment")
        self.assertEqual(records[1]["found-by"], "hunter")
        self.assertEqual(records[1]["batch"], "2")
        self.assertEqual(records[0]["status"], "open")

    def test_a_finding_goes_fixed_with_its_commit(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="the guard is inverted", severity="breaks")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234, 2026-10-04"},
        )

        record = ledger.findings(self.brain.find(self.key))[0]
        self.assertEqual(record["status"], "fixed")
        self.assertEqual(record["fix"], "abc1234, 2026-10-04")

    def test_a_value_with_backslashes_lands_as_written(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="the path is escaped", severity="bug")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "wontfix", "fix": r"C:\new\1 \g<0> is fine", "reason": "not worth it"},
        )

        record = ledger.findings(self.brain.find(self.key))[0]
        self.assertEqual(record["fix"], r"C:\new\1 \g<0> is fine")

    def test_a_bullet_quoted_in_code_is_never_the_one_rewritten(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        quoted = "\nSeen as:\n\n```\n- status: fixed\n```\n"
        fixtures.write(path=path, text=text.replace("- status: open\n- fix:\n", "- fix:\n" + quoted))
        self.brain.refresh()

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )

        self.assertIn("- batch: \n- fix:\n- status: wontfix\n- reason: not worth it\n" + quoted, self.body())
        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["status"], "wontfix")

    def test_a_repeated_bullet_is_read_and_rewritten_on_its_first_line(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("- status: open\n- fix:\n", "- status: open\n- fix: abc1234\n- fix:\n"))
        self.brain.refresh()

        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["fix"], "abc1234")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "def5678"},
        )

        self.assertIn("- status: fixed\n- fix: def5678\n- fix:\n", self.body())
        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["fix"], "def5678")

    def test_a_fixed_finding_that_lost_its_fix_line_takes_the_commit_back(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("- status: open\n- fix:\n", "- status: fixed\n"))
        self.brain.refresh()

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=["F-01"],
                changes={"severity": "nit"},
            )
        self.assertEqual(str(caught.exception), "a fixed finding names its commit, fix=<hash>")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"fix": "abc1234"},
        )

        self.assertIn("- status: fixed\n- fix: abc1234\n", self.body())
        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["fix"], "abc1234")

    def test_a_field_the_spec_does_not_name_is_refused(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=["F-01"],
                changes={"owner": "me"},
            )

        self.assertEqual(str(caught.exception), "a finding has no 'owner' field, its fields are severity, found-by, batch, status, fix, reason")

    def test_a_fixed_finding_keeps_its_commit_whatever_change_comes(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=["F-01"],
                changes={"fix": ""},
            )

        self.assertEqual(str(caught.exception), "a fixed finding names its commit, fix=<hash>")
        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["fix"], "abc1234")

    def test_the_fix_leaves_with_the_fixed_status_so_a_finding_fixed_again_names_its_commit_anew(self) -> None:
        ledger.add_finding(
            brain=self.brain,
            key=self.key,
            claim="x",
            severity="bug",
        )
        later_statuses = [status for status in ledger.FINDING_STATUSES if status != "fixed"]

        for status in later_statuses:
            with self.subTest(status=status):
                ledger.set_findings(
                    brain=self.brain,
                    key=self.key,
                    identifiers=["F-01"],
                    changes={"status": "fixed", "fix": "abc1234"},
                )

                changes = {"status": status}
                if status in ledger.NEEDS_REASON:
                    changes["reason"] = f"the reason it is {status}"

                ledger.set_findings(
                    brain=self.brain,
                    key=self.key,
                    identifiers=["F-01"],
                    changes=changes,
                )

                self.assertNotIn("fix", ledger.findings(entry=self.brain.find(key=self.key))[0])

                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_findings(
                        brain=self.brain,
                        key=self.key,
                        identifiers=["F-01"],
                        changes={"status": "fixed"},
                    )

                self.assertEqual(str(caught.exception), "a fixed finding names its commit, fix=<hash>")

    def test_a_fix_given_with_the_status_stands_on_every_finding_the_call_names_whatever_status_each_had(self) -> None:
        later_statuses = [status for status in ledger.FINDING_STATUSES if status != "fixed"]

        for status in later_statuses:
            with self.subTest(status=status):
                fixed_id, open_id = ledger.add_findings(
                    brain=self.brain,
                    key=self.key,
                    claims=["was fixed", "was open"],
                    severity="bug",
                )
                ledger.set_findings(
                    brain=self.brain,
                    key=self.key,
                    identifiers=[fixed_id],
                    changes={"status": "fixed", "fix": "abc1234"},
                )

                changes = {"status": status, "fix": "def5678"}
                if status in ledger.NEEDS_REASON:
                    changes["reason"] = f"the reason it is {status}"

                ledger.set_findings(
                    brain=self.brain,
                    key=self.key,
                    identifiers=[fixed_id, open_id],
                    changes=changes,
                )

                named_records = [record for record in ledger.findings(entry=self.brain.find(key=self.key)) if record["id"] in (fixed_id, open_id)]
                landed_fields = [(record["status"], record.get("fix"), record.get("reason")) for record in named_records]
                self.assertEqual(landed_fields, [(status, "def5678", changes.get("reason"))] * 2)

    def test_a_finding_set_fixed_names_its_commit_in_the_same_call_whatever_fix_an_earlier_call_left(self) -> None:
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        fixed_id = ledger.add_finding(
            brain=self.brain,
            key=self.key,
            claim="was fixed",
            severity="bug",
        )
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=[fixed_id],
            changes={"status": "fixed", "fix": "abc1234"},
        )

        carrying_ids: list[str] = []
        for status in [status for status in ledger.FINDING_STATUSES if status != "fixed"]:
            carrying_id = ledger.add_finding(
                brain=self.brain,
                key=self.key,
                claim=f"was {status}",
                severity="bug",
            )
            changes = {"status": status, "fix": "def5678"}
            if status in ledger.NEEDS_REASON:
                changes["reason"] = f"the reason it is {status}"

            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=[carrying_id],
                changes=changes,
            )
            carrying_ids.append(carrying_id)

        lost_id = ledger.add_finding(
            brain=self.brain,
            key=self.key,
            claim="lost its status line",
            severity="bug",
        )
        text = path.read_text(encoding="utf-8")
        fixtures.write(
            path=path,
            text=text.replace(
                f"### {lost_id}: lost its status line\n\n- severity: bug\n- found-by: review\n- batch: \n- status: open\n- fix:\n",
                f"### {lost_id}: lost its status line\n\n- severity: bug\n- found-by: review\n- batch: \n- fix: def5678\n",
            ),
        )
        self.brain.refresh()
        carrying_ids.append(lost_id)

        before = path.read_text(encoding="utf-8")
        for carrying_id in carrying_ids:
            for identifiers in ([carrying_id], [fixed_id, carrying_id]):
                with self.subTest(identifiers=identifiers):
                    with self.assertRaises(ledger.LedgerError) as caught:
                        ledger.set_findings(
                            brain=self.brain,
                            key=self.key,
                            identifiers=identifiers,
                            changes={"status": "fixed"},
                        )

                    self.assertEqual(str(caught.exception), "a fixed finding names its commit, fix=<hash>")
                    self.assertEqual(path.read_text(encoding="utf-8"), before)

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=[fixed_id],
            changes={"status": "fixed"},
        )

        fixed_record = [record for record in ledger.findings(entry=self.brain.find(key=self.key)) if record["id"] == fixed_id][0]
        self.assertEqual((fixed_record["status"], fixed_record["fix"]), ("fixed", "abc1234"))

    def test_a_value_in_backticks_reads_back_whole(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="the guard is inverted", severity="breaks")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "`abc1234`"},
        )

        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["fix"], "`abc1234`")

    def test_a_value_or_a_claim_on_two_lines_is_refused(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="the guard is inverted", severity="breaks")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=["F-01"],
                changes={"fix": "abc1234\n- status: fixed"},
            )
        self.assertEqual(str(caught.exception), "the fix is one line")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.add_finding(brain=self.brain, key=self.key, claim="one\ntwo", severity="nit")
        self.assertEqual(str(caught.exception), "the claim is one line")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.add_finding(brain=self.brain, key=self.key, claim="   ", severity="nit")
        self.assertEqual(str(caught.exception), "a finding needs a claim")
        self.assertEqual(len(ledger.findings(self.brain.find(self.key))), 1)

        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["status"], "open")

    def test_fixed_without_a_commit_is_refused(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")

        with self.assertRaises(ledger.LedgerError):
            ledger.set_findings(
                brain=self.brain,
                key=self.key,
                identifiers=["F-01"],
                changes={"status": "fixed"},
            )

    def test_a_bad_severity_is_refused(self) -> None:
        with self.assertRaises(ledger.LedgerError):
            ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="huge")

    def test_a_batch_below_one_is_refused_on_a_new_finding_too(self) -> None:
        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug", batch=-1)

        self.assertEqual(str(caught.exception), "batch is a number from 1, or blank, not '-1'")
        self.assertEqual(ledger.findings(self.brain.find(self.key)), [])

    def test_every_field_is_checked_on_a_change_too(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")

        for changes, said in (
            ({"severity": "enormous"}, "severity is breaks or bug or debt or nit, not 'enormous'"),
            ({"found-by": "nobody"}, "found-by is review or hunter, not 'nobody'"),
            ({"batch": "two"}, "batch is a number from 1, or blank, not 'two'"),
            ({"batch": "0"}, "batch is a number from 1, or blank, not '0'"),
            ({"status": "fixed", "fix": "  "}, "a fixed finding names its commit, fix=<hash>"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_findings(
                        brain=self.brain,
                        key=self.key,
                        identifiers=["F-01"],
                        changes=changes,
                    )
                self.assertEqual(str(caught.exception), said)

        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_batch(brain=self.brain, key=self.key, theme="the docs")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"severity": "nit", "batch": "2", "found-by": "hunter"},
        )

        record = ledger.findings(self.brain.find(self.key))[0]
        self.assertEqual((record["severity"], record["batch"], record["found-by"]), ("nit", "2", "hunter"))

    def test_a_heading_or_a_bullet_inside_code_does_not_end_or_change_a_finding(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="the guard is inverted", severity="breaks")
        ledger.add_finding(brain=self.brain, key=self.key, claim="a stale comment", severity="nit")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        evidence = "\nSeen as:\n\n```\n# a comment in the shell\n- status: fixed\n### F-09: not a finding\n```\n"
        fixtures.write(path=path, text=text.replace("- fix:\n\n### F-02", "- fix:\n" + evidence + "\n### F-02"))
        self.brain.refresh()

        records = ledger.findings(self.brain.find(self.key))
        self.assertEqual([record["id"] for record in records], ["F-01", "F-02"])
        self.assertEqual(records[0]["status"], "open")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )

        after = self.body()
        self.assertIn("- status: wontfix\n- fix:\n- reason: not worth it\n" + evidence, after)
        self.assertEqual(ledger.findings(self.brain.find(self.key))[1]["status"], "open")

    def test_a_finding_lands_inside_the_findings_section_when_another_follows(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.rstrip("\n") + "\n\n## Notes\n\nKept by hand.\n")
        self.brain.refresh()

        ledger.add_finding(brain=self.brain, key=self.key, claim="y", severity="nit")

        body = self.body()
        self.assertIn("### F-02: y\n\n- severity: nit\n- found-by: review\n- batch: \n- status: open\n- fix:\n\n## Notes\n\nKept by hand.\n", body)
        self.assertLess(body.index("### F-02"), body.index("## Notes"))

    def test_a_fence_inside_a_claim_or_a_value_hides_no_finding(self) -> None:
        first = ledger.add_finding(brain=self.brain, key=self.key, claim="a ``` fence in the docs is unclosed", severity="bug")
        second = ledger.add_finding(brain=self.brain, key=self.key, claim="the guard is inverted", severity="breaks")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=[first],
            changes={"fix": "see the ``` block"},
        )
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=[second],
            changes={"fix": "the ``` block too"},
        )
        third = ledger.add_finding(brain=self.brain, key=self.key, claim="another ``` fence, in the README", severity="nit")

        self.assertEqual((first, second, third), ("F-01", "F-02", "F-03"))
        self.assertEqual([record["id"] for record in ledger.findings(self.brain.find(self.key))], ["F-01", "F-02", "F-03"])

    def test_a_row_shaped_line_outside_the_batches_section_is_not_a_batch(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("## Summary\n\n", "## Summary\n\n| 7 | not a batch | done | |\n\n") + "\n| 8 | nor this | done | |\n")
        self.brain.refresh()

        self.assertEqual([row["number"] for row in ledger.batches(self.brain.find(self.key))], [1])
        self.assertEqual(ledger.add_batch(brain=self.brain, key=self.key, theme="the docs rule"), 2)

        with self.assertRaises(ledger.LedgerError):
            ledger.set_batch(brain=self.brain, key=self.key, number=7, status="done")

    def test_the_prose_around_the_records_is_untouched(self) -> None:
        before = self.body()

        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug", batch=1)

        after = self.body()
        self.assertTrue(after.startswith(before.split("## Batches")[0]))
        self.assertIn("The login branch, three commits.", after)


class Reading(LedgerCase):
    def test_the_findings_are_read_with_one_scan_of_the_body_for_code(self) -> None:
        for claim in ("one", "two", "three", "four"):
            ledger.add_finding(brain=self.brain, key=self.key, claim=claim, severity="nit")
        entry = self.brain.find(self.key)

        with mock.patch.object(sections, "code_spans", wraps=sections.code_spans) as scans:
            found = ledger.findings(entry)

        self.assertEqual([item["id"] for item in found], ["F-01", "F-02", "F-03", "F-04"])
        self.assertEqual(scans.call_count, 1 + len(found))


class Reporting(LedgerCase):
    def next_step(self) -> str:
        last = ledger.report(brain=self.brain, key=self.key).splitlines()[-1]

        return last[len("next: ") :]

    def test_the_state_is_counted_from_the_file(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_batch(brain=self.brain, key=self.key, theme="docs")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.add_finding(brain=self.brain, key=self.key, claim="two", severity="nit", batch=2)
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done", ack="2026-10-04")

        self.assertEqual(
            ledger.report(brain=self.brain, key=self.key),
            "2026-10-fix-login, open, 2 batches, 2 findings, 1 fixed, 1 open\n"
            "batch 1, correctness: done, ack 2026-10-04, 1 finding, 1 fixed\n"
            "batch 2, docs: pending, 1 finding, 1 open\n"
            "next: batch 2 on the owner's word",
        )

    def test_the_next_step_follows_a_batch_through_its_states(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug")
        self.assertEqual(self.next_step(), "put F-01 in a batch")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"batch": "1"},
        )
        self.assertEqual(self.next_step(), "batch 1 on the owner's word")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        self.assertEqual(self.next_step(), "fix F-01 in batch 1")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        self.assertEqual(self.next_step(), "verify batch 1")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified")
        self.assertEqual(self.next_step(), "mark batch 1 done")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")
        self.assertEqual(self.next_step(), "the closing pass")

        ledger.add_batch(brain=self.brain, key=self.key, theme="closing pass")
        ledger.add_finding(brain=self.brain, key=self.key, claim="late", severity="nit", batch=2)
        self.assertEqual(self.next_step(), "batch 2 on the owner's word")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-02"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=2, status="done", ack="2026-10-05")
        self.assertEqual(self.next_step(), "close the ledger")

        ledger.close(brain=self.brain, key=self.key)
        self.assertEqual(self.next_step(), "nothing, the ledger is closed")

    def test_an_empty_ledger_reports_itself(self) -> None:
        self.assertEqual(
            ledger.report(brain=self.brain, key=self.key),
            "2026-10-fix-login, open, 0 batches, 0 findings\nnext: close the ledger",
        )

    def test_the_level_is_recorded_and_low_or_a_sweep_owes_no_closing_pass(self) -> None:
        self.assertEqual(self.entry.field("level"), "high")

        for branch, level, ending in (
            ("fix/login-low", "low", "next: close the ledger"),
            ("sweep/login", "high", "next: close the ledger"),
            ("fix/login-medium", "medium", "next: the closing pass"),
            ("fix/login-max", "max", "next: the closing pass"),
        ):
            with self.subTest(branch=branch):
                entry = ledger.open_ledger(brain=self.brain, branch=branch, summary="One.", level=level, stamp="2026-10-03")
                ledger.add_batch(brain=self.brain, key=entry.name, theme="correctness")
                ledger.set_batch(brain=self.brain, key=entry.name, number=1, status="done")

                self.assertTrue(ledger.report(brain=self.brain, key=entry.name).endswith(ending))

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.open_ledger(brain=self.brain, branch="fix/x", summary="One.", level="ultra")

        self.assertEqual(str(caught.exception), "a review runs at low or medium or high or max, not 'ultra'")

    def test_a_late_finding_reopens_its_batch_and_a_wontfix_needs_no_batch(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done", ack="2026-10-04")
        ledger.add_finding(brain=self.brain, key=self.key, claim="late", severity="bug", found_by="hunter", batch=1)

        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertEqual((row["status"], row["ack"]), ("fixing", "2026-10-04"))
        self.assertEqual(self.next_step(), "fix F-02 in batch 1")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-02"],
            changes={"status": "fixed", "fix": "def5678"},
        )
        ledger.add_finding(brain=self.brain, key=self.key, claim="not worth it", severity="nit")
        self.assertEqual(self.next_step(), "put F-03 in a batch")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-03"],
            changes={"status": "fixed", "fix": "cab5678"},
        )
        self.assertEqual(self.next_step(), "put F-03 in a batch")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-03"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )
        self.assertEqual(self.next_step(), "verify batch 1")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done", ack="2026-10-05")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "open"},
        )
        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["status"], "fixing")
        self.assertEqual(self.next_step(), "fix F-01 in batch 1")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done", ack="2026-10-06")
        ledger.add_finding(brain=self.brain, key=self.key, claim="later still", severity="nit", found_by="hunter")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-04"],
            changes={"batch": "1", "status": "fixed", "fix": "fed9876"},
        )
        self.assertEqual(ledger.batches(self.brain.find(self.key))[0]["status"], "fixing")
        self.assertEqual(self.next_step(), "verify batch 1")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done", ack="2026-10-07")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-04"],
            changes={"batch": "1"},
        )
        row = ledger.batches(self.brain.find(self.key))[0]
        self.assertEqual((row["status"], row["ack"]), ("done", "2026-10-07"))

    def test_one_call_reopens_every_batch_a_finding_it_names_sits_in(self) -> None:
        for theme in ("correctness", "the docs"):
            ledger.add_batch(
                brain=self.brain,
                key=self.key,
                theme=theme,
            )

        for number in (1, 2):
            ledger.add_finding(
                brain=self.brain,
                key=self.key,
                claim=f"claim {number}",
                severity="bug",
                batch=number,
            )

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01", "F-02"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        for number in (1, 2):
            ledger.set_batch(
                brain=self.brain,
                key=self.key,
                number=number,
                status="done",
                ack="2026-10-04",
            )

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01", "F-02"],
            changes={"status": "open"},
        )

        rows: list[dict] = ledger.batches(entry=self.brain.find(key=self.key))
        self.assertEqual([(row["status"], row["ack"]) for row in rows], [("fixing", "2026-10-04"), ("fixing", "2026-10-04")])

    def test_a_ledger_with_no_level_recorded_is_read_at_the_default_level(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("level: high\n", ""))
        self.brain.refresh()

        self.assertIsNone(self.brain.find(self.key).field("level"))
        self.assertEqual(self.next_step(), "the closing pass")

    def test_the_closing_pass_is_known_however_its_theme_is_cased(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")
        ledger.add_batch(brain=self.brain, key=self.key, theme="Closing Pass")
        ledger.set_batch(brain=self.brain, key=self.key, number=2, status="done")

        self.assertEqual(self.next_step(), "close the ledger")

    def test_every_pending_batch_waits_for_its_own_ack_whatever_came_before(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_batch(brain=self.brain, key=self.key, theme="nothing here")
        ledger.add_batch(brain=self.brain, key=self.key, theme="docs")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")
        ledger.set_batch(brain=self.brain, key=self.key, number=2, status="done")
        ledger.add_finding(brain=self.brain, key=self.key, claim="two", severity="nit", batch=3)

        self.assertEqual(self.next_step(), "batch 3 on the owner's word")

        ledger.set_batch(brain=self.brain, key=self.key, number=3, ack="2026-10-05")
        self.assertEqual(self.next_step(), "start batch 3, it has the word")

        ledger.set_batch(brain=self.brain, key=self.key, number=3, status="fixing")

        self.assertEqual(self.next_step(), "fix F-02 in batch 3")

    def test_a_pending_batch_starts_on_the_standing_word_while_one_holds(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        autoack.turn_on(brain=self.brain, hours=2)

        self.assertEqual(self.next_step(), "start batch 1 on the standing word")

        autoack.turn_off(brain=self.brain)
        self.assertEqual(self.next_step(), "batch 1 on the owner's word")

    def test_a_stamp_written_ahead_is_no_word_once_the_word_is_off(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | pending | auto 2026-09-01 |"))
        self.brain.refresh()

        self.assertEqual(self.next_step(), "batch 1 on the owner's word")

        autoack.turn_on(brain=self.brain, hours=2)
        self.assertEqual(self.next_step(), "start batch 1, it has the word")
        autoack.turn_off(brain=self.brain)

    def test_a_finding_in_a_batch_the_table_lost_is_stray(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("- batch: 1\n", "- batch: 5\n"))
        self.brain.refresh()

        self.assertEqual(self.next_step(), "put F-01 in a batch")

    def test_a_verified_batch_with_its_ack_recorded_only_needs_marking_done(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified", ack="2026-10-04")

        self.assertEqual(self.next_step(), "mark batch 1 done")

        ledger.set_batch(
            brain=self.brain,
            key=self.key,
            number=1,
            status="done",
        )
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="closing pass",
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=2, status="verified")

        self.assertEqual(self.next_step(), "mark batch 2 done, it holds nothing")

    def test_every_status_that_needs_a_reason_says_why(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="nit")

        for status in ledger.NEEDS_REASON:
            with self.subTest(status=status):
                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_findings(
                        brain=self.brain,
                        key=self.key,
                        identifiers=["F-01"],
                        changes={"status": status},
                    )
                self.assertEqual(str(caught.exception), f"a {status} finding says why, reason=<one line>")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "wontfix", "reason": "the guard is on purpose"},
        )

        record = ledger.findings(self.brain.find(self.key))[0]
        self.assertEqual((record["status"], record["reason"]), ("wontfix", "the guard is on purpose"))
        self.assertIn("- fix:\n- reason: the guard is on purpose\n", self.body())

    def test_the_reason_comes_with_the_status_and_leaves_with_it(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="nit")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("- status: open\n", "- status: wontfix\n"))
        self.brain.refresh()

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"batch": "1"},
        )
        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["batch"], "1")

        for status in ledger.NEEDS_REASON:
            with self.subTest(status=status):
                reason = f"the reason it is {status}"
                ledger.set_findings(
                    brain=self.brain,
                    key=self.key,
                    identifiers=["F-01"],
                    changes={"status": status, "reason": reason},
                )

                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.set_findings(
                        brain=self.brain,
                        key=self.key,
                        identifiers=["F-01"],
                        changes={"reason": ""},
                    )

                self.assertEqual(str(caught.exception), f"a {status} finding keeps its reason, give a new one or a new status")
                record: dict = ledger.findings(entry=self.brain.find(key=self.key))[0]
                self.assertEqual((record["status"], record["reason"]), (status, reason))

                ledger.set_findings(
                    brain=self.brain,
                    key=self.key,
                    identifiers=["F-01"],
                    changes={"status": "open"},
                )

                self.assertNotIn("reason", ledger.findings(entry=self.brain.find(key=self.key))[0])
                self.assertIn("- status: open\n- fix:\n\n", self.body() + "\n")

    def test_a_refuted_finding_lands_without_a_batch_and_counts_under_its_own_name(self) -> None:
        ledger.add_finding(
            brain=self.brain,
            key=self.key,
            claim="one",
            severity="bug",
        )

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "refuted", "reason": "line 212 already guards it, and a verifier agreed"},
        )

        self.assertEqual(ledger.tally(items=ledger.findings(entry=self.brain.find(key=self.key))), "1 finding, 1 refuted")
        self.assertEqual(self.next_step(), "close the ledger")

        ledger.close(
            brain=self.brain,
            key=self.key,
        )
        self.assertIn("Closed with 1 finding, 1 refuted.", self.body())

    def test_a_wontfix_in_a_fixing_batch_counts_as_handled(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.add_finding(brain=self.brain, key=self.key, claim="two", severity="nit", batch=1)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        self.assertEqual(self.next_step(), "fix F-02 in batch 1")

        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-02"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )

        self.assertEqual(self.next_step(), "verify batch 1")

    def test_a_batch_that_ran_without_an_ack_on_record_asks_for_it_first(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| 1 | correctness | pending | |", "| 1 | correctness | done | |"))
        self.brain.refresh()

        self.assertEqual(self.next_step(), "record the owner's ack of batch 1, it holds findings and has none")

    def test_a_batch_with_findings_and_no_ack_takes_the_standing_word_while_it_holds(self) -> None:
        ledger.add_batch(
            brain=self.brain,
            key=self.key,
            theme="correctness",
        )
        ledger.set_batch(
            brain=self.brain,
            key=self.key,
            number=1,
            status="done",
        )
        ledger.add_finding(
            brain=self.brain,
            key=self.key,
            claim="late",
            severity="nit",
            batch=1,
        )
        autoack.turn_on(brain=self.brain, hours=2)

        self.assertEqual(self.next_step(), "record the standing word on batch 1, it holds findings and has none")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, ack="auto")
        self.assertEqual(self.next_step(), "fix F-01 in batch 1")

        autoack.turn_off(brain=self.brain)
        self.assertEqual(self.next_step(), "fix F-01 in batch 1")

    def test_a_batch_cell_written_with_a_leading_zero_is_still_that_batch(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("- batch: 1\n", "- batch: 01\n"))
        self.brain.refresh()

        self.assertEqual(
            ledger.report(brain=self.brain, key=self.key),
            "2026-10-fix-login, open, 1 batch, 1 finding, 1 open\n"
            "batch 1, correctness: fixing, ack 2026-10-04, 1 finding, 1 open\n"
            "next: fix F-01 in batch 1",
        )

        with self.assertRaises(ledger.LedgerError):
            ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")

        fixtures.write(path=path, text=path.read_text(encoding="utf-8").replace("- batch: 01\n", "- batch: ²\n"))
        self.brain.refresh()

        self.assertEqual(self.next_step(), "put F-01 in a batch")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"severity": "nit"},
        )
        self.assertEqual(ledger.findings(self.brain.find(self.key))[0]["severity"], "nit")

    def test_a_blank_status_counts_as_open_everywhere(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="one", severity="bug", batch=1)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("- status: open\n", "- status:\n"))
        self.brain.refresh()

        self.assertEqual(
            ledger.report(brain=self.brain, key=self.key),
            "2026-10-fix-login, open, 1 batch, 1 finding, 1 open\n"
            "batch 1, correctness: fixing, ack 2026-10-04, 1 finding, 1 open\n"
            "next: fix F-01 in batch 1",
        )


class Closing(LedgerCase):
    def test_it_refuses_while_a_finding_is_open(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.close(brain=self.brain, key=self.key)

        self.assertIn("F-01 still open", str(caught.exception))

    def test_a_finding_without_a_status_line_counts_as_open(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        ledger.add_finding(brain=self.brain, key=self.key, claim="y", severity="bug")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-02"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("- status: open\n", "", 1))
        self.brain.refresh()

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.close(brain=self.brain, key=self.key)

        self.assertIn("F-01 still open", str(caught.exception))
        self.assertEqual(ledger.tally(ledger.findings(self.brain.find(self.key))), "2 findings, 1 open, 1 wontfix")

    def test_it_refuses_while_a_batch_is_not_done_or_the_closing_pass_is_owed(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug", batch=1)
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="fixing", ack="2026-10-04")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "fixed", "fix": "abc1234"},
        )
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="verified")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.close(brain=self.brain, key=self.key)
        self.assertEqual(str(caught.exception), "the ledger closes after mark batch 1 done")

        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.close(brain=self.brain, key=self.key)
        self.assertEqual(str(caught.exception), "the ledger closes after the closing pass")

        ledger.add_batch(brain=self.brain, key=self.key, theme="closing pass")
        ledger.set_batch(brain=self.brain, key=self.key, number=2, status="done")
        ledger.close(brain=self.brain, key=self.key)

        self.assertEqual(self.brain.find(self.key).field("status"), "closed")

    def test_it_flips_the_status_and_writes_the_count(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )

        ledger.close(brain=self.brain, key=self.key, stamp="2026-10-05")

        entry = self.brain.find(self.key)
        self.assertEqual(entry.field("status"), "closed")
        self.assertEqual(entry.field("updated"), "2026-10-05")
        self.assertIn("Closed with 1 finding, 1 wontfix.", entry.document.body)

        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.close(brain=self.brain, key=self.key)
        self.assertEqual(str(caught.exception), "2026-10-fix-login is already closed")
        self.assertEqual(self.body().count("Closed with"), 1)

    def test_a_heading_in_another_case_is_the_same_section_everywhere(self) -> None:
        ledger.add_finding(brain=self.brain, key=self.key, claim="x", severity="bug")
        ledger.set_findings(
            brain=self.brain,
            key=self.key,
            identifiers=["F-01"],
            changes={"status": "wontfix", "reason": "not worth it"},
        )
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("## Summary", "## SUMMARY").replace("## Batches", "## batches"))
        self.brain.refresh()

        self.assertEqual(
            ledger.add_batch(
                brain=self.brain,
                key=self.key,
                theme="correctness",
            ),
            1,
        )
        ledger.set_batch(
            brain=self.brain,
            key=self.key,
            number=1,
            status="done",
        )

        self.assertEqual(
            ledger.add_batch(
                brain=self.brain,
                key=self.key,
                theme="closing pass",
            ),
            2,
        )
        ledger.set_batch(
            brain=self.brain,
            key=self.key,
            number=2,
            status="done",
        )

        ledger.close(
            brain=self.brain,
            key=self.key,
            stamp="2026-10-05",
        )

        body = self.body()
        self.assertIn("## SUMMARY\n\nThe login branch, three commits.\n\nClosed with 1 finding, 1 wontfix.\n\n## batches", body)

    def test_a_table_header_quoted_in_code_is_not_the_table(self) -> None:
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        quoted = "The table looks like:\n\n```\n| # | Theme | Status | Ack |\n|---|---|---|---|\n```\n\n"
        fixtures.write(path=path, text=text.replace("## Batches\n\n", "## Batches\n\n" + quoted))
        self.brain.refresh()

        first = ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        second = ledger.add_batch(brain=self.brain, key=self.key, theme="the docs")

        self.assertEqual((first, second), (1, 2))
        self.assertIn(quoted + "| # | Theme | Status | Ack |\n|---|---|---|---|\n| 1 | correctness | pending | |\n| 2 | the docs | pending | |\n", self.body())
        self.assertEqual([row["number"] for row in ledger.batches(self.brain.find(self.key))], [1, 2])

    def test_a_row_apart_from_the_table_is_refused_not_read_or_ignored(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        ledger.add_batch(brain=self.brain, key=self.key, theme="the docs")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        apart = "a batch row sits apart from the table, put it directly under the header"
        not_whole = "a line of the batch table is not a whole row, a row is | number | theme | status | ack |"
        damaged_ways = (
            (text.replace("|---|---|---|---|\n| 1 |", "|---|---|---|---|\n\n| 1 |"), apart),
            (text.replace("| 1 | correctness | pending | |\n", "| 1 | correctness |\n"), not_whole),
            (text.replace("| 2 | the docs | pending | |\n", "| 2 | the docs | pending |\n"), not_whole),
            (text.replace("| 2 | the docs | pending | |\n", " | 2 | the docs | pending | |\n"), not_whole),
            (text.replace("| 2 | the docs | pending | |\n", "| 2 | the docs | pending | |\n\nWho owns what:\n\n| # | Owner | Since | Note |\n|---|---|---|---|\n| 4 | ana | may | - |\n"), apart),
            (text.replace("| # | Theme |", "| 7 | stray | done | |\n| # | Theme |"), apart),
        )

        for damaged, said in damaged_ways:
            with self.subTest(damaged=damaged):
                fixtures.write(path=path, text=damaged)
                self.brain.refresh()

                with self.assertRaises(ledger.LedgerError) as caught:
                    ledger.add_batch(brain=self.brain, key=self.key, theme="hooks")

                self.assertEqual(str(caught.exception), said)
                self.assertEqual(path.read_text(encoding="utf-8"), damaged)

    def test_a_row_is_one_line(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")

        self.assertIsNone(ledger.BATCH_ROW.match("| 2 | the docs |\n| 3 | hooks | pending | |"))
        self.assertEqual([row["number"] for row in ledger.batches(self.brain.find(self.key))], [1])

    def test_a_row_is_added_whole_when_the_table_ends_the_file(self) -> None:
        ledger.add_batch(brain=self.brain, key=self.key, theme="correctness")
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text[: text.index("\n\n## Findings")])
        self.brain.refresh()

        ledger.add_batch(brain=self.brain, key=self.key, theme="the docs")

        self.assertTrue(self.body().endswith("| 1 | correctness | pending | |\n| 2 | the docs | pending | |"))
        ledger.set_batch(brain=self.brain, key=self.key, number=1, status="done")
        self.assertEqual([row["status"] for row in ledger.batches(self.brain.find(self.key))], ["done", "pending"])

    def test_a_table_header_in_another_case_is_still_the_table(self) -> None:
        path = self.root / "reviews" / "2026-10-fix-login" / "ledger.md"
        text = path.read_text(encoding="utf-8")
        fixtures.write(path=path, text=text.replace("| # | Theme | Status | Ack |", "| # | theme | status | ack |"))
        self.brain.refresh()

        self.assertEqual(ledger.add_batch(brain=self.brain, key=self.key, theme="correctness"), 1)
        self.assertIn("|---|---|---|---|\n| 1 | correctness | pending | |\n", self.body())

    def test_something_that_is_not_a_ledger_is_refused(self) -> None:
        with self.assertRaises(ledger.LedgerError):
            ledger.add_finding(brain=self.brain, key="BUG-002", claim="x", severity="bug")


if __name__ == "__main__":
    unittest.main()
