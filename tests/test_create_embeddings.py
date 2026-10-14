import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf

from create_embeddings import (
    CorruptDocumentError,
    clean_text,
    has_corrupt_pdf_text,
    read_document,
    search_similar_chunks,
    upload_document,
)


class ExtractionQualityTests(unittest.TestCase):
    def test_normal_thai_text_is_not_flagged(self):
        self.assertFalse(has_corrupt_pdf_text("หนังสือรับรองการหักลดหย่อนภาษี"))

    def test_custom_font_mojibake_is_flagged(self):
        broken = "®´­º°¦´¦° ¸É°¥¼n ¼o¤¸Á·Åo \x85 \x85 \x85"

        self.assertTrue(has_corrupt_pdf_text(broken))

    def test_replacement_characters_are_flagged(self):
        self.assertTrue(has_corrupt_pdf_text("ภาษี\ufffdเงินได้\ufffdบุคคลธรรมดา"))

    def test_clean_text_normalizes_unicode_and_pdf_whitespace(self):
        dirty = "  ภาษี\u00a0เงินได้\u200b  \r\n\r\n\r\nบุคคลธรรมดา  "

        self.assertEqual(clean_text(dirty), "ภาษี เงินได้\n\nบุคคลธรรมดา")

    def test_empty_pdf_is_rejected_with_ocr_hint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / "empty.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(source)
            document.close()

            with patch("create_embeddings.ocr_pdf_page", return_value=""):
                with self.assertRaisesRegex(CorruptDocumentError, "no text"):
                    read_document(source)

    def test_corrupt_pdf_uses_tesseract_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / "corrupt.pdf"
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), "placeholder")
            document.save(source)
            document.close()

            with (
                patch("create_embeddings.has_corrupt_pdf_text", return_value=True),
                patch(
                    "create_embeddings.ocr_pdf_page",
                    return_value="ข้อความจาก Tesseract",
                ) as ocr,
            ):
                pages = read_document(source)

            self.assertEqual(pages[0]["text"], "ข้อความจาก Tesseract")
            self.assertEqual(pages[0]["extraction_method"], "tesseract_ocr")
            ocr.assert_called_once()


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


class UploadTests(unittest.TestCase):
    def test_reingestion_upserts_before_deleting_only_stale_chunks(self):
        client = MagicMock()
        document_upsert = MagicMock()
        document_select = MagicMock()
        chunk_select = MagicMock()
        chunk_upsert = MagicMock()
        chunk_delete = MagicMock()
        client.table.side_effect = [
            document_upsert,
            document_select,
            chunk_select,
            chunk_upsert,
            chunk_delete,
        ]
        document_select.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
            "id": 32
        }
        chunk_select.select.return_value.eq.return_value.execute.return_value.data = [
            {"chunk_uid": "unchanged"},
            {"chunk_uid": "stale"},
        ]
        chunks = [
            {
                "chunk_uid": "unchanged",
                "chunk_index": 0,
                "content": "เนื้อหา",
                "page_start": 1,
                "page_end": 1,
                "tax_year": 2568,
                "metadata": {},
                "embedding": [0.1],
            }
        ]

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source = Path(directory) / "source.txt"
            source.write_text("เนื้อหา", encoding="utf-8")
            upload_document(
                client,
                source,
                "เอกสาร",
                2568,
                "tax_guidance",
                chunks,
            )

        uploaded_rows = chunk_upsert.upsert.call_args.args[0]
        self.assertEqual(uploaded_rows[0]["document_id"], 32)
        self.assertEqual(uploaded_rows[0]["chunk_uid"], "unchanged")
        chunk_delete.delete.return_value.in_.assert_called_once_with(
            "chunk_uid", ["stale"]
        )
        self.assertEqual(
            [call.args[0] for call in client.table.call_args_list],
            [
                "documents",
                "documents",
                "document_chunks",
                "document_chunks",
                "document_chunks",
            ],
        )


if __name__ == "__main__":
    unittest.main()
