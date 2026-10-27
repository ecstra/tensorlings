"""The chat style is written once and the other two agents get exact copies."""

import shutil
import unittest
from pathlib import Path
from unittest import mock

from brainlib import doctor, index, store, style
from tests import fixtures

STYLE = """---
name: Plain & Short
description: Answer first, then stop.
keep-coding-instructions: true
---

# Plain & Short

Answer first, then stop.
"""


class StyleCase(fixtures.BrainCase):
    def setUp(self) -> None:
        super().setUp()
        self.project = self.root.parent
        fixtures.write(path=self.project / style.SOURCE, text=STYLE)
        for relative in style.COPIES:
            (self.project / relative.parent).mkdir(exist_ok=True)


class Copying(StyleCase):
    def test_a_project_without_an_agent_folder_owes_no_copy_and_gets_no_folder(self) -> None:
        shutil.rmtree(self.project / ".codex")

        self.assertEqual(style.expected(self.brain), [Path(".gemini") / "plain-and-short.md"])
        self.assertEqual(style.stale(self.brain), [Path(".gemini") / "plain-and-short.md"])
        self.assertEqual(style.write(self.brain), [Path(".gemini") / "plain-and-short.md"])
        self.assertFalse((self.project / ".codex").exists())

    def test_a_missing_source_is_no_finding_when_no_copy_is_owed(self) -> None:
        shutil.rmtree(self.project / ".codex")
        shutil.rmtree(self.project / ".gemini")
        (self.project / style.SOURCE).unlink()

        self.assertEqual([finding for finding in doctor.run(self.brain) if finding.path == str(style.SOURCE)], [])

        (self.project / ".codex").mkdir()

        self.assertEqual(
            [finding.message for finding in doctor.run(self.brain) if finding.path == str(style.SOURCE)],
            [f"{style.SOURCE} is missing, so there is no style to copy"],
        )

    def test_the_copies_are_the_source_body_without_its_frontmatter(self) -> None:
        style.write(self.brain)

        for relative in style.COPIES:
            with self.subTest(copy=str(relative)):
                self.assertEqual(
                    (self.project / relative).read_text(encoding="utf-8"),
                    "# Plain & Short\n\nAnswer first, then stop.\n",
                )

    def test_a_missing_copy_is_stale_and_a_current_one_is_not(self) -> None:
        self.assertEqual(style.stale(self.brain), list(style.COPIES))

        style.write(self.brain)

        self.assertEqual(style.stale(self.brain), [])

    def test_a_changed_source_makes_every_copy_stale_again(self) -> None:
        style.write(self.brain)
        fixtures.write(path=self.project / style.SOURCE, text=STYLE.replace("then stop.", "then stop. Really."))

        self.assertEqual(len(style.stale(self.brain)), 2)

    def test_a_copy_with_windows_line_endings_is_the_same_text(self) -> None:
        style.write(self.brain)
        target = self.project / style.COPIES[0]
        target.write_bytes(target.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8"))

        self.assertEqual(style.stale(self.brain), [])

    def test_a_symlinked_copy_becomes_a_real_file_and_the_other_end_is_untouched(self) -> None:
        elsewhere = self.project / "elsewhere.md"
        elsewhere.write_text("theirs\n", encoding="utf-8")
        target = self.project / style.COPIES[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(elsewhere)

        style.write(self.brain)

        self.assertFalse(target.is_symlink())
        self.assertEqual(elsewhere.read_text(encoding="utf-8"), "theirs\n")
        self.assertEqual(style.stale(self.brain), [])

    def test_a_copy_the_process_cannot_open_is_stale_and_gets_rewritten(self) -> None:
        style.write(self.brain)
        target = self.project / style.COPIES[0]
        target.chmod(0)

        self.assertEqual(style.stale(self.brain), [style.COPIES[0]])

        style.write(self.brain)

        self.assertEqual(style.stale(self.brain), [])
        self.assertEqual(target.read_text(encoding="utf-8"), "# Plain & Short\n\nAnswer first, then stop.\n")

    def test_a_copy_folder_the_process_cannot_search_reads_as_stale_on_every_floor(self) -> None:
        style.write(self.brain)
        folder = self.project / style.COPIES[0].parent
        folder.chmod(0)

        try:
            self.assertEqual(style.stale(self.brain), [style.COPIES[0]])
        finally:
            folder.chmod(0o700)

    def test_a_copy_that_is_not_utf8_is_stale(self) -> None:
        style.write(self.brain)
        (self.project / style.COPIES[0]).write_bytes(b"# Plain & Short\n\ncaf\xe9\n")

        self.assertEqual(style.stale(self.brain), [style.COPIES[0]])

    def test_a_missing_source_is_an_error_that_names_it(self) -> None:
        (self.project / style.SOURCE).unlink()

        with self.assertRaises(style.StyleError) as caught:
            style.stale(self.brain)

        self.assertIn(".claude/output-styles/plain-and-short.md", str(caught.exception))


class ThroughDoctor(StyleCase):
    def test_doctor_flags_a_stale_copy_as_fixable_and_fixes_it(self) -> None:
        index.write(brain=self.brain, stamp="2026-09-16")

        before = [finding for finding in doctor.run(self.brain) if finding.check == "layout"]
        self.assertTrue(any("chat style copy" in finding.message and finding.fixable for finding in before))

        doctor.run(brain=self.brain, fix=True)

        self.assertEqual(style.stale(self.brain), [])

    def test_each_stale_copy_repairs_itself_once(self) -> None:
        index.write(brain=self.brain, stamp="2026-09-16")

        with mock.patch("brainlib.style.write_one", wraps=style.write_one) as writing:
            doctor.run(brain=self.brain, fix=True)

        self.assertEqual(writing.call_count, len(style.COPIES))
        self.assertEqual(style.stale(self.brain), [])

    def test_the_kit_ships_its_copies_current(self) -> None:
        kit = store.Brain(root=Path(__file__).resolve().parents[4])

        self.assertEqual(style.stale(kit), [])


if __name__ == "__main__":
    unittest.main()
