"""The schema must stay in step with the prose specs it came from.

Every check here reads `.agents/spec/` on disk, so a rule reworded in the
prose without the code following it fails the suite rather than drifting
quietly.
"""

import unittest
from pathlib import Path

from brainlib import schema

BRAIN = Path(__file__).resolve().parents[4]


def spec_text(
    kind: schema.Kind,
) -> str:
    return (BRAIN / "spec" / f"{kind.folder}.md").read_text(encoding="utf-8")


class SpecsAndCodeAgree(unittest.TestCase):
    def test_every_kind_has_a_spec_file(self) -> None:
        for kind in schema.KINDS.values():
            with self.subTest(kind=kind.name):
                self.assertTrue((BRAIN / "spec" / f"{kind.folder}.md").exists())

    def test_every_field_is_named_in_its_spec(self) -> None:
        for kind in schema.KINDS.values():
            text = spec_text(kind)

            for item in kind.fields:
                with self.subTest(kind=kind.name, field=item.name):
                    self.assertIn(f"{item.name}:", text)

    def test_every_allowed_value_is_named_in_its_spec(self) -> None:
        for kind in schema.KINDS.values():
            text = spec_text(kind)

            for item in kind.fields:
                for value in item.values:
                    with self.subTest(kind=kind.name, value=value):
                        self.assertIn(value, text)

    def test_every_section_is_named_in_its_spec(self) -> None:
        for kind in schema.KINDS.values():
            text = spec_text(kind)

            for section in kind.sections:
                with self.subTest(kind=kind.name, section=section.title):
                    self.assertIn(section.title, text)

    def test_the_common_fields_are_the_ones_conventions_lists(self) -> None:
        text = (BRAIN / "spec" / "conventions.md").read_text(encoding="utf-8")

        for item in schema.COMMON_FIELDS:
            with self.subTest(field=item.name):
                self.assertIn(f"{item.name}:", text)

    def test_every_id_prefix_is_named_in_conventions(self) -> None:
        text = (BRAIN / "spec" / "conventions.md").read_text(encoding="utf-8")

        for kind in schema.KINDS.values():
            if not kind.has_ids:
                continue

            with self.subTest(kind=kind.name):
                self.assertIn(f"{kind.id_prefix}-###", text)


class KindTable(unittest.TestCase):
    def test_the_indexed_folders_are_read_off_the_kinds(self) -> None:
        self.assertEqual(schema.INDEXED_FOLDERS, ("docs", "memory", "bugs", "todos", "decisions", "plans", "reviews"))
        self.assertFalse(schema.KINDS["handover"].indexed)

    def test_every_column_a_kind_lists_is_one_of_its_fields(self) -> None:
        for kind in schema.KINDS.values():
            names = {field.name for field in schema.fields_for(kind)}
            with self.subTest(kind=kind.name):
                self.assertTrue(set(kind.columns) <= names, kind.columns)

    def test_the_bug_severities_are_the_one_table(self) -> None:
        severity = next(field for field in schema.KINDS["bug"].fields if field.name == "severity")

        self.assertIs(severity.values, schema.SEVERITIES)


class FoldersOnDisk(unittest.TestCase):
    def test_every_indexed_folder_carries_an_index(self) -> None:
        for folder in schema.INDEXED_FOLDERS:
            with self.subTest(folder=folder):
                self.assertTrue((BRAIN / folder / "INDEX.md").exists())

    def test_spec_and_handover_carry_no_index(self) -> None:
        self.assertFalse((BRAIN / "spec" / "INDEX.md").exists())
        self.assertFalse((BRAIN / "handover" / "INDEX.md").exists())

    def test_the_rulebook_carries_no_frontmatter(self) -> None:
        for name in schema.NO_FRONTMATTER:
            text = (BRAIN / name).read_text(encoding="utf-8")

            with self.subTest(name=name):
                self.assertFalse(text.startswith("---\n"))


class NamingRules(unittest.TestCase):
    def test_an_id_is_zero_padded_to_three_digits(self) -> None:
        self.assertEqual(
            schema.format_id(prefix="BUG", number=7),
            "BUG-007",
        )

    def test_an_id_past_three_digits_keeps_every_digit(self) -> None:
        self.assertEqual(
            schema.format_id(prefix="TODO", number=1042),
            "TODO-1042",
        )

    def test_a_title_becomes_a_hyphenated_slug(self) -> None:
        self.assertEqual(
            schema.slugify("Login loops on an expired token!"),
            "login-loops-on-an-expired-token",
        )

    def test_a_slug_holds_no_capitals_or_spaces(self) -> None:
        self.assertTrue(schema.is_slug("export-to-csv"))
        self.assertFalse(schema.is_slug("Export To CSV"))
        self.assertFalse(schema.is_slug("export--to--csv"))

    def test_a_date_is_iso_or_nothing(self) -> None:
        self.assertTrue(schema.is_date("2026-09-16"))
        self.assertTrue(schema.is_date("2026-12-31"))
        self.assertFalse(schema.is_date("16 September 2026"))

    def test_a_date_has_a_real_month_and_day(self) -> None:
        for wrong in ("2026-13-45", "2026-00-10", "2026-09-00", "2026-09-32"):
            with self.subTest(date=wrong):
                self.assertFalse(schema.is_date(wrong))

    def test_a_dotted_path_is_never_the_kits(self) -> None:
        self.assertTrue(schema.is_kit_path("spec/todos.md"))
        self.assertTrue(schema.is_kit_path("README.md"))

        for dotted in ("spec/.draft.md", ".README.md", ".hidden/INDEX.md", "docs/.putback.x/INDEX.md"):
            with self.subTest(path=dotted):
                self.assertTrue(schema.is_hidden(dotted))
                self.assertFalse(schema.is_kit_path(dotted))

    def test_a_review_folder_is_a_real_month_then_a_branch_slug(self) -> None:
        self.assertTrue(schema.is_review_folder("2026-09-feat-billing"))
        self.assertTrue(schema.is_review_folder("2026-12-x"))

        for wrong in ("2026-13-feat-billing", "2026-00-feat-billing", "billing", "2026-09-Feat-Billing", "2026-9-feat"):
            with self.subTest(folder=wrong):
                self.assertFalse(schema.is_review_folder(wrong))

    def test_an_unknown_kind_names_the_ones_that_exist(self) -> None:
        with self.assertRaises(schema.UnknownKind) as caught:
            schema.kind_named("ticket")

        self.assertEqual(str(caught.exception), "unknown kind 'ticket', expected one of bug, decision, doc, handover, memory, plan, review, todo")


if __name__ == "__main__":
    unittest.main()
