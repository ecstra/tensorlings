"""The frontmatter reader must hand every body back untouched."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from brainlib import frontmatter

BRAIN = Path(__file__).resolve().parents[4]

SAMPLE = """---
name: BUG-001-login-loop
kind: bug
description: the sign in screen loops, colons: and commas, all of it
updated: 2026-09-16
links: [spec-bugs, TODO-004-session-timeout]
status: open
---

# Problem

The token refresh loops.

---

```text
--- this fence is body, not frontmatter ---
```

Trailing spaces live here:
"""


class ParseKeepsTheBody(unittest.TestCase):
    def test_round_trip_is_byte_for_byte(self) -> None:
        self.assertEqual(frontmatter.render(frontmatter.parse(SAMPLE)), SAMPLE)

    def test_round_trip_without_a_closing_newline(self) -> None:
        text = SAMPLE.rstrip("\n")

        self.assertEqual(frontmatter.render(frontmatter.parse(text)), text)

    def test_body_starts_after_the_closing_fence(self) -> None:
        document = frontmatter.parse(SAMPLE)

        self.assertTrue(document.body.startswith("\n# Problem"))

    def test_a_rule_in_the_body_is_not_the_fence(self) -> None:
        document = frontmatter.parse(SAMPLE)

        self.assertIn("\n---\n", document.body)
        self.assertEqual(len(document.lines), 6)

    def test_every_brain_file_round_trips(self) -> None:
        for path in sorted(BRAIN.rglob("*.md")):
            if path.name == "SYSTEM.md" or "skills" in path.relative_to(BRAIN).parts:
                continue

            text = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(BRAIN))):
                self.assertEqual(
                    frontmatter.render(frontmatter.parse(text)),
                    text,
                )


class ReadingFields(unittest.TestCase):
    def test_a_value_may_hold_colons_and_commas(self) -> None:
        document = frontmatter.parse(SAMPLE)

        self.assertEqual(
            document.get("description"),
            "the sign in screen loops, colons: and commas, all of it",
        )

    def test_a_key_is_lowercase_with_digits_and_hyphens_and_a_space_after_the_colon(self) -> None:
        self.assertEqual(frontmatter.split_field("superseded-by: DEC-002"), ("superseded-by", "DEC-002"))
        self.assertEqual(frontmatter.split_field("description:"), ("description", ""))
        self.assertEqual(frontmatter.split_field("a2: x"), ("a2", "x"))

        for line in ("Note: see below", "https://example.com", ": x", "name\t: x", "- a: 1", " name: x"):
            with self.subTest(line=line):
                self.assertEqual(frontmatter.split_field(line), (None, None))

    def test_a_missing_field_reads_as_none(self) -> None:
        document = frontmatter.parse(SAMPLE)

        self.assertIsNone(document.get("severity"))

    def test_a_bracketed_value_reads_as_a_list(self) -> None:
        document = frontmatter.parse(SAMPLE)

        self.assertEqual(
            document.get_list("links"),
            ["spec-bugs", "TODO-004-session-timeout"],
        )

    def test_an_empty_list_reads_as_no_items(self) -> None:
        document = frontmatter.parse("---\nlinks: []\n---\nbody\n")

        self.assertEqual(document.get_list("links"), [])


class WritingFields(unittest.TestCase):
    def test_setting_a_field_leaves_every_other_line_alone(self) -> None:
        document = frontmatter.parse(SAMPLE)
        document.set(
            key="status",
            value="fixed",
        )

        written = frontmatter.render(document)

        self.assertIn("status: fixed", written)
        self.assertEqual(
            written.replace("status: fixed", "status: open"),
            SAMPLE,
        )

    def test_setting_an_unknown_field_appends_it(self) -> None:
        document = frontmatter.parse(SAMPLE)
        document.set(
            key="severity",
            value="breaks",
        )

        self.assertEqual(document.lines[-1], "severity: breaks")

    def test_a_value_with_a_line_break_is_refused_before_it_is_set(self) -> None:
        document = frontmatter.parse(SAMPLE)

        with self.assertRaises(frontmatter.FrontmatterError) as caught:
            document.set(key="description", value="first\nsecond: no")

        self.assertEqual(str(caught.exception), "description is one line, and this value has a line break in it")
        self.assertEqual(frontmatter.render(document), SAMPLE)

    def test_a_value_is_trimmed_on_the_way_in(self) -> None:
        document = frontmatter.parse(SAMPLE)
        document.set(key="description", value="  padded  ")

        self.assertIn("\ndescription: padded\n", frontmatter.render(document))

    def test_a_document_the_parser_would_refuse_is_never_rendered(self) -> None:
        document = frontmatter.Document(lines=["name: x", "loose\nline"], body="\nbody\n")

        with self.assertRaises(frontmatter.FrontmatterError):
            frontmatter.render(document)

    def test_setting_a_list_renders_it_bracketed(self) -> None:
        document = frontmatter.parse(SAMPLE)
        document.set_list(
            key="links",
            values=["a", "b"],
        )

        self.assertIn("links: [a, b]", frontmatter.render(document))


class Writing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "note.md"
        self.path.write_text(SAMPLE, encoding="utf-8")
        self.document = frontmatter.parse(SAMPLE)
        self.document.set(key="status", value="fixed")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def leftovers(self) -> list[str]:
        return sorted(child.name for child in Path(self.tmp.name).iterdir() if child.name != "note.md")

    def test_a_write_that_fails_part_way_leaves_the_old_file_as_it_was(self) -> None:
        with mock.patch("brainlib.frontmatter.os.replace", side_effect=OSError(5, "Input/output error")):
            with self.assertRaises(OSError):
                frontmatter.write(path=str(self.path), document=self.document)

        self.assertEqual(self.path.read_text(encoding="utf-8"), SAMPLE)
        self.assertEqual(self.leftovers(), [])

    def test_a_file_the_process_may_not_write_is_refused_before_anything_is_touched(self) -> None:
        self.path.chmod(0o444)

        try:
            with self.assertRaises(PermissionError) as caught:
                frontmatter.write(path=str(self.path), document=self.document)
        finally:
            self.path.chmod(0o644)

        self.assertEqual(caught.exception.filename, str(self.path))
        self.assertEqual(self.path.read_text(encoding="utf-8"), SAMPLE)
        self.assertEqual(self.leftovers(), [])

    def test_a_new_file_gets_the_mode_the_umask_gives_not_the_temporary_files(self) -> None:
        fresh = Path(self.tmp.name) / "fresh.md"
        before = os.umask(0o022)

        try:
            frontmatter.write(path=str(fresh), document=self.document)
        finally:
            os.umask(before)

        self.assertEqual(fresh.stat().st_mode & 0o777, 0o644)

    def test_the_text_reaches_the_disk_before_the_move_and_the_folder_after(self) -> None:
        order: list[str] = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(handle: int) -> None:
            order.append("fsync")
            real_fsync(handle)

        def replace(source: str, target: str) -> None:
            order.append("replace")
            real_replace(source, target)

        with mock.patch("brainlib.frontmatter.os.fsync", side_effect=fsync), mock.patch("brainlib.frontmatter.os.replace", side_effect=replace):
            frontmatter.write(path=str(self.path), document=self.document)

        self.assertEqual(order, ["fsync", "replace", "fsync"])

    def test_a_folder_that_cannot_be_flushed_does_not_fail_the_write(self) -> None:
        real_open = os.open

        def refuse_the_folder(path: str, flags: int, *rest: int) -> int:
            if path == self.tmp.name:
                raise OSError(13, "Permission denied")
            return real_open(path, flags, *rest)

        with mock.patch("brainlib.frontmatter.os.open", side_effect=refuse_the_folder):
            frontmatter.write(path=str(self.path), document=self.document)

        self.assertIn("status: fixed", self.path.read_text(encoding="utf-8"))
        self.assertEqual(self.leftovers(), [])

    def test_a_write_keeps_the_files_mode(self) -> None:
        self.path.chmod(0o640)

        frontmatter.write(path=str(self.path), document=self.document)

        self.assertEqual(self.path.stat().st_mode & 0o777, 0o640)
        self.assertIn("status: fixed", self.path.read_text(encoding="utf-8"))


class BadInput(unittest.TestCase):
    def test_a_file_without_a_fence_is_refused(self) -> None:
        with self.assertRaises(frontmatter.FrontmatterError):
            frontmatter.parse("# Just a heading\n")

    def test_an_unclosed_fence_is_refused(self) -> None:
        with self.assertRaises(frontmatter.FrontmatterError):
            frontmatter.parse("---\nname: x\nbody with no closing fence\n")

    def test_a_block_list_is_refused_with_the_bracket_form_named(self) -> None:
        with self.assertRaises(frontmatter.FrontmatterError) as caught:
            frontmatter.parse("---\nname: x\nlinks:\n  - bugs/BUG-001-x.md\n  - docs/y.md\n---\n\nbody\n")

        self.assertEqual(str(caught.exception), "line 4 of the header is a block list item, the kit writes a list on its key's line as [a, b]")

    def test_an_indented_line_is_refused(self) -> None:
        with self.assertRaises(frontmatter.FrontmatterError) as caught:
            frontmatter.parse("---\nname: x\n  description: y\n---\n\nbody\n")

        self.assertEqual(str(caught.exception), "line 3 of the header is indented, the kit writes one key per line at the margin")

    def test_a_blank_line_or_a_line_that_is_not_a_field_is_refused(self) -> None:
        cases = (
            ("---\nname: x\n\nkind: doc\n---\n\nbody\n", "line 3 of the header is blank, the kit writes one key per line with nothing between"),
            ("---\nname: x\n   \nkind: doc\n---\n\nbody\n", "line 3 of the header is blank, the kit writes one key per line with nothing between"),
            ("---\nname: x\n# a comment\n---\n\nbody\n", "line 3 of the header is not key: value"),
            ("---\nname: x\n: see below\n---\n\nbody\n", "line 3 of the header is not key: value"),
            ("---\nname: x\nhttps://example.com\n---\n\nbody\n", "line 3 of the header is not key: value"),
            ("---\nname: x\nMy Key: y\n---\n\nbody\n", "line 3 of the header is not key: value"),
            ("---\nname: x\n- a: 1\n---\n\nbody\n", "line 3 of the header is a block list item, the kit writes a list on its key's line as [a, b]"),
        )

        for text, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(frontmatter.FrontmatterError) as caught:
                    frontmatter.parse(text)

                self.assertEqual(str(caught.exception), message)

    def test_a_key_that_appears_twice_is_refused(self) -> None:
        with self.assertRaises(frontmatter.FrontmatterError) as caught:
            frontmatter.parse("---\nname: x\nstatus: open\nstatus: fixed\n---\n\nbody\n")

        self.assertEqual(str(caught.exception), "line 4 of the header repeats status, one key per line and each key once")

    def test_a_file_that_is_not_utf8_is_refused_the_same_way(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latin.md"
            path.write_bytes(b"---\nname: caf\xe9\n---\n\nbody\n")

            with self.assertRaises(frontmatter.FrontmatterError) as caught:
                frontmatter.read(str(path))

        self.assertIn("not UTF-8", str(caught.exception))

    def test_a_dangling_symlink_or_a_folder_is_refused_the_same_way(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dangling = Path(tmp) / "gone.md"
            dangling.symlink_to("nowhere.md")
            folder = Path(tmp) / "folder.md"
            folder.mkdir()

            for path in (dangling, folder):
                with self.subTest(path=path.name):
                    with self.assertRaises(frontmatter.FrontmatterError) as caught:
                        frontmatter.read(str(path))

                    self.assertIn("the file cannot be read: ", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
