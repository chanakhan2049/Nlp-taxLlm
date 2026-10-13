import json
import tempfile
import unittest
from pathlib import Path

import evaluate


class EvaluateTests(unittest.TestCase):
    def test_default_evaluation_set_has_twenty_unique_cases(self):
        cases = evaluate.load_cases(evaluate.DEFAULT_CASES)

        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case["id"] for case in cases}), 20)

    def test_expected_term_can_match_title_or_content(self):
        case = {"expected_source_terms": ["ภ.ง.ด.91"]}
        chunks = [{"title": "แบบ ภ.ง.ด.91", "content": "รายละเอียด"}]

        self.assertTrue(evaluate.retrieval_passes(case, chunks))

    def test_out_of_scope_case_passes_only_without_chunks(self):
        case = {"expect_no_answer": True, "expected_source_terms": []}

        self.assertTrue(evaluate.retrieval_passes(case, []))
        self.assertFalse(evaluate.retrieval_passes(case, [{"content": "irrelevant"}]))

    def test_empty_evaluation_file_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps([]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-empty JSON list"):
                evaluate.load_cases(path)


if __name__ == "__main__":
    unittest.main()
