"""The skill file meets the Agent Skills spec every agent reads it by."""

import re
import unittest
from pathlib import Path

from brainlib import frontmatter

SKILL_DIR = Path(__file__).resolve().parents[2]


class SkillFile(unittest.TestCase):
    def setUp(self) -> None:
        self.document = frontmatter.read(str(SKILL_DIR / "SKILL.md"))

    def test_the_name_matches_the_directory(self) -> None:
        self.assertEqual(self.document.get("name"), SKILL_DIR.name)
        self.assertTrue(re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", self.document.get("name") or ""))

    def test_the_description_fits_the_limit_and_says_when(self) -> None:
        description = self.document.get("description") or ""

        self.assertTrue(0 < len(description) <= 400)
        self.assertIn("Use it to", description)

    def test_the_body_stays_short_enough_to_load_whole(self) -> None:
        self.assertLess(self.document.body.count("\n"), 500)

    def test_every_referenced_file_exists(self) -> None:
        for target in re.findall(r"\]\(([^)]+\.md)\)", self.document.body):
            with self.subTest(target=target):
                self.assertTrue((SKILL_DIR / target).is_file())


if __name__ == "__main__":
    unittest.main()
