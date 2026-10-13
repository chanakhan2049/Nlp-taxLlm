import unittest

from refresh_embeddings import embedding_input


class RefreshEmbeddingsTests(unittest.TestCase):
    def test_embedding_input_preserves_both_year_meanings(self):
        row = {
            "content": "เนื้อหา",
            "page_start": 2,
            "documents": {
                "title": "ล.ย.03",
                "document_year": 2560,
                "tax_year": 2568,
            },
        }

        text = embedding_input(row)

        self.assertIn("ปีที่เผยแพร่เอกสาร: 2560", text)
        self.assertIn("ใช้สำหรับปีภาษี: 2568", text)
        self.assertIn("หน้า: 2", text)


if __name__ == "__main__":
    unittest.main()
