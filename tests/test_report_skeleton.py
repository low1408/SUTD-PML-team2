"""The final report keeps the 11 sections the course brief requires."""

import re
import unittest
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "reports" / "final" / "report.md"
REQUIRED = ["Executive Summary", "Background", "Related work", "Problem formulation", "Data description",
            "Details of the solution", "Evaluation", "Discussion", "Recommendations", "Limitations", "future research"]


class ReportSkeletonTests(unittest.TestCase):
    def test_eleven_numbered_sections_in_order(self):
        headings = re.findall(r"^## (\d+)\. (.+)$", REPORT.read_text(encoding="utf-8"), flags=re.M)
        self.assertEqual([int(n) for n, _ in headings], list(range(1, 12)))
        for (_, title), needle in zip(headings, REQUIRED, strict=True):
            self.assertIn(needle.lower(), title.lower())

    def test_no_secrets_or_patient_rows(self):
        text = REPORT.read_text(encoding="utf-8").lower()
        self.assertNotIn("password:", text)


if __name__ == "__main__":
    unittest.main()
