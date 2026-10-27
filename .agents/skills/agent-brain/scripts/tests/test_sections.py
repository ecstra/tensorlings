"""Headings are read from prose, never from code."""

import unittest

from brainlib import sections

BODY = (
    "## Symptom\n\n"
    "The shell prints this:\n\n"
    "```\n"
    "# not a heading\n"
    "## Evidence\n"
    "```\n\n"
    "Then `## inline` is quoted too.\n\n"
    "## Evidence\n\n"
    "A real section.\n"
)


class Reading(unittest.TestCase):
    def test_a_heading_inside_code_is_not_a_heading(self) -> None:
        self.assertEqual(sections.headings(BODY), ["Symptom", "Evidence"])

    def test_a_block_runs_past_the_code_it_holds(self) -> None:
        symptom = sections.block(body=BODY, title="symptom")

        self.assertIn("# not a heading", symptom)
        self.assertIn("quoted too", symptom)
        self.assertNotIn("A real section", symptom)
        self.assertEqual(sections.block(body=BODY, title="Evidence"), "\n\nA real section.\n")

    def test_a_fence_opens_only_at_the_start_of_a_line(self) -> None:
        body = (
            "## Symptom\n\n"
            "The ``` fence in the docs is unclosed.\n\n"
            "## Evidence\n\n"
            "Another ``` fence, in the README.\n\n"
            "## Fix\n\n"
            "```\n"
            "## not a heading\n"
            "```\n"
        )

        self.assertEqual(sections.headings(body), ["Symptom", "Evidence", "Fix"])
        self.assertIn((body.index("```\n## not"), len(body) - 1), sections.code_spans(body))
        self.assertEqual(sections.block(body=body, title="Evidence"), "\n\nAnother ``` fence, in the README.\n\n")

    def test_a_fence_with_no_closer_fences_nothing(self) -> None:
        body = "## Symptom\n\n```\n## Evidence\n\nstill prose\n"

        self.assertEqual(sections.headings(body), ["Symptom", "Evidence"])

    def test_the_next_heading_is_the_first_one_outside_code(self) -> None:
        self.assertEqual(BODY[sections.heading_after(body=BODY, start=0) :].startswith("## Symptom"), True)
        self.assertEqual(BODY[sections.heading_after(body=BODY, start=1) :].startswith("## Evidence\n\nA real"), True)
        self.assertEqual(sections.heading_after(body=BODY, start=len(BODY) - 5), len(BODY))

    def test_a_depth_steps_over_the_deeper_headings(self) -> None:
        body = "## Findings\n\n### F-01: x\n\n- status: open\n\n### F-02: y\n\n## Notes\n"

        self.assertEqual(body[sections.heading_after(body=body, start=1) :], "### F-01: x\n\n- status: open\n\n### F-02: y\n\n## Notes\n")
        self.assertEqual(body[sections.heading_after(body=body, start=1, depth=2) :], "## Notes\n")


if __name__ == "__main__":
    unittest.main()
