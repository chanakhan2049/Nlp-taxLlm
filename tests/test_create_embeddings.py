import unittest
from unittest.mock import MagicMock

from create_embeddings import has_corrupt_pdf_text, search_similar_chunks


class ExtractionQualityTests(unittest.TestCase):
    def test_normal_thai_text_is_not_flagged(self):
        self.assertFalse(has_corrupt_pdf_text("หนังสือรับรองการหักลดหย่อนภาษี"))

    def test_custom_font_mojibake_is_flagged(self):
        broken = "®´­º°¦´¦° ¸É°¥¼n ¼o¤¸Á·Åo \x85 \x85 \x85"

        self.assertTrue(has_corrupt_pdf_text(broken))


class RetrievalTests(unittest.TestCase):
    def test_rejects_corporate_tax_question_without_querying_database(self):
        client = MagicMock()
        model = MagicMock()

        result = search_similar_chunks(
            client, model, "บริษัทจำกัดคำนวณภาษีอย่างไร"
        )

        self.assertEqual(result, [])
        client.rpc.assert_not_called()

    def test_lexical_overlap_reranks_vector_candidates(self):
        client = MagicMock()
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.1]
        client.rpc.return_value.execute.return_value.data = [
            {"title": "ทั่วไป", "content": "รายการลดหย่อนอื่น", "similarity": 0.70},
            {
                "title": "ประกันสังคม",
                "content": "เงินสมทบกองทุนประกันสังคม",
                "similarity": 0.60,
            },
        ]

        result = search_similar_chunks(
            client, model, "เงินสมทบกองทุนประกันสังคมลดหย่อนได้หรือไม่", k=1
        )

        self.assertEqual(result[0]["title"], "ประกันสังคม")
        self.assertEqual(client.rpc.call_args.args[1]["match_count"], 50)

    def test_old_document_intent_prefers_historical_version(self):
        client = MagicMock()
        model = MagicMock()
        model.encode.return_value.tolist.return_value = [0.1]
        client.rpc.return_value.execute.return_value.data = [
            {
                "title": "แบบ ล.ย.03 ฉบับใหม่",
                "content": "แบบ ล.ย.03",
                "similarity": 0.65,
                "document_year": 2568,
                "tax_year": 2568,
            },
            {
                "title": "แบบ ล.ย.03 ฉบับเก่า",
                "content": "แบบ ล.ย.03",
                "similarity": 0.50,
                "document_year": 2560,
                "tax_year": 2568,
            },
        ]

        result = search_similar_chunks(
            client, model, "แบบ ล.ย.03 ฉบับเก่าเผยแพร่ปีใด", k=1
        )

        self.assertEqual(result[0]["document_year"], 2560)


if __name__ == "__main__":
    unittest.main()
