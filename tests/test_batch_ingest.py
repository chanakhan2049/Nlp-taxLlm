import tempfile
import unittest
from pathlib import Path

from batch_ingest import get_document_type, get_title
from create_embeddings import read_document


class MarkdownIngestionTests(unittest.TestCase):
    def test_markdown_is_read_as_text(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "คำแนะนำ.md"
            path.write_text("# หัวข้อ\n\nเนื้อหาภาษี", encoding="utf-8")

            pages = read_document(path)

        self.assertEqual(pages, [{"page": 1, "text": "# หัวข้อ\n\nเนื้อหาภาษี"}])

    def test_text_guidance_uses_filename_as_title(self):
        path = Path("การขอคืนเงินภาษีอากร.txt")

        self.assertEqual(get_document_type(path), "tax_guidance")
        self.assertEqual(
            get_title(path, 2568),
            "การขอคืนเงินภาษีอากร "
            "(เอกสารเผยแพร่ปี 2568; การขอคืนเงินภาษีอากร.txt)",
        )


if __name__ == "__main__":
    unittest.main()
