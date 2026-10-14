"""Core functions for the Thai tax RAG pipeline.

Pipeline: read documents -> create chunks -> create embeddings -> Supabase.
"""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client


MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300
RETRIEVAL_CANDIDATES = 50
OCR_DPI = 300
OCR_LANGUAGES = "tha+eng"
TESSDATA_DIR = Path(__file__).with_name("tessdata")
OCR_TEMP_DIR = Path(__file__).with_name(".ocr-temp")
OUT_OF_SCOPE_MARKERS = (
    "บริษัทจำกัด",
    "ภาษีเงินได้นิติบุคคล",
    "ห้างหุ้นส่วนจำกัด",
    "ภ.พ.30",
    "ภ.พ.36",
)
INVISIBLE_CHARACTERS = str.maketrans(
    "",
    "",
    "\u200b\u200c\u200d\u2060\ufeff",
)
SPACE_CHARACTERS = str.maketrans(
    {
        "\u00a0": " ",  # non-breaking space
        "\u202f": " ",  # narrow non-breaking space
        "\u3000": " ",  # ideographic space
    }
)


class CorruptDocumentError(ValueError):
    """Raised when extracted PDF text is unusable without OCR."""


def has_corrupt_pdf_text(text):
    """Detect common broken custom-font extraction that requires OCR."""
    if not text:
        return False
    control_count = sum("\x80" <= character <= "\x9f" for character in text)
    mojibake_markers = sum(text.count(marker) for marker in ("", "", "", ""))
    replacement_count = text.count("\ufffd")
    return (
        control_count >= 3
        or mojibake_markers >= 8
        or replacement_count >= 2
    )


def clean_text(text):
    """Normalize extracted Unicode text without changing tax values or wording."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(SPACE_CHARACTERS).translate(INVISIBLE_CHARACTERS)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_tesseract():
    """Locate Tesseract from configuration, PATH, or common Windows installs."""
    configured = os.getenv("TESSERACT_CMD")
    candidates = [
        configured,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    raise CorruptDocumentError(
        "Tesseract OCR is required but was not found. Install Tesseract 5 "
        "or set TESSERACT_CMD in .env."
    )


def ocr_pdf_page(page):
    """Render one PDF page and extract Thai/English text with Tesseract."""
    if not all(
        (TESSDATA_DIR / f"{language}.traineddata").is_file()
        for language in OCR_LANGUAGES.split("+")
    ):
        raise CorruptDocumentError(
            f"Missing Tesseract language data in {TESSDATA_DIR}."
        )

    pixmap = page.get_pixmap(dpi=OCR_DPI, alpha=False)
    OCR_TEMP_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="page-", dir=OCR_TEMP_DIR) as directory:
        image_path = Path(directory) / "page.png"
        pixmap.save(image_path)
        result = subprocess.run(
            [
                find_tesseract(),
                str(image_path),
                "stdout",
                "--tessdata-dir",
                str(TESSDATA_DIR),
                "-l",
                OCR_LANGUAGES,
                "--psm",
                "3",
                "-c",
                "preserve_interword_spaces=1",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    if result.returncode:
        detail = result.stderr.strip() or "unknown Tesseract error"
        raise CorruptDocumentError(f"Tesseract OCR failed: {detail}")
    return clean_text(result.stdout)


def read_document(file_path):
    """Read a PDF page by page, or read a TXT file as one page."""
    file_path = Path(file_path)

    if file_path.suffix.lower() in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8-sig")
        return [{"page": 1, "text": clean_text(text), "extraction_method": "text"}]

    pages = []
    raw_pages = []
    image_only_pages = []
    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks", sort=True)
            raw_text = "\n".join(block[4] for block in blocks)
            raw_pages.append(raw_text)
            if not raw_text.strip() and page.get_images(full=True):
                image_only_pages.append(page_number)
            pages.append(
                {
                    "page": page_number,
                    "text": clean_text(raw_text),
                    "extraction_method": "embedded_text",
                }
            )

        combined_raw_text = "\n".join(raw_pages)
        if has_corrupt_pdf_text(combined_raw_text) or not any(
            page["text"] for page in pages
        ):
            pages_to_ocr = set(range(1, len(pages) + 1))
        else:
            pages_to_ocr = set(image_only_pages)

        for page_number in sorted(pages_to_ocr):
            text = ocr_pdf_page(pdf[page_number - 1])
            if not text:
                raise CorruptDocumentError(
                    f"Tesseract found no text on page {page_number} of "
                    f"{file_path.name}."
                )
            pages[page_number - 1]["text"] = text
            pages[page_number - 1]["extraction_method"] = "tesseract_ocr"

    if not any(page["text"] for page in pages):
        raise CorruptDocumentError(
            f"No extractable text in {file_path.name}, including after OCR."
        )
    return pages


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            search_from = start + chunk_size // 2
            section = text[search_from:end]
            for separator in ("\n\n", "\n", ". ", " "):
                position = section.rfind(separator)
                if position >= 0:
                    end = search_from + position + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap

    return chunks


def create_chunks(
    file_path,
    title,
    tax_year,
    document_type,
    document_year=None,
):
    """Create chunks and attach metadata used for citation and filtering."""
    chunks = []
    document_year = document_year or tax_year

    for page in read_document(file_path):
        for page_chunk_index, content in enumerate(split_text(page["text"])):
            chunk_number = len(chunks)
            text_for_embedding = (
                f"เอกสาร: {title}\nปีที่เผยแพร่เอกสาร: {document_year}\n"
                f"ใช้สำหรับปีภาษี: {tax_year}\n"
                f"หน้า: {page['page']}\n{content}"
            )
            chunk_id = hashlib.sha256(
                (
                    f"{title}|{document_year}|{tax_year}|{page['page']}|"
                    f"{page_chunk_index}|{content}"
                ).encode("utf-8")
            ).hexdigest()

            chunks.append(
                {
                    "chunk_uid": chunk_id,
                    "chunk_index": chunk_number,
                    "content": content,
                    "text_for_embedding": text_for_embedding,
                    "page_start": page["page"],
                    "page_end": page["page"],
                    "tax_year": tax_year,
                    "metadata": {
                        "title": title,
                        "document_type": document_type,
                        "document_year": document_year,
                        "applicable_tax_year": tax_year,
                        "extraction_method": page["extraction_method"],
                    },
                }
            )

    return chunks


def add_embeddings(chunks, model):
    """Create normalized 1,024-dimensional BGE-M3 embeddings."""
    texts = [chunk["text_for_embedding"] for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector.tolist()
    return chunks


def _character_overlap(question, chunk):
    """Return a Thai-friendly lexical score without requiring a tokenizer."""
    normalize = lambda value: re.sub(r"[^\wก-๙]", "", value.lower())
    query = normalize(question)
    text = normalize(f"{chunk.get('title', '')} {chunk.get('content', '')}")
    if len(query) < 3 or len(text) < 3:
        return 0.0
    query_grams = {query[index : index + 3] for index in range(len(query) - 2)}
    text_grams = {text[index : index + 3] for index in range(len(text) - 2)}
    return len(query_grams & text_grams) / len(query_grams)


def _distinctive_term_overlap(question, chunk):
    """Boost explicit form names and Latin fund abbreviations in a question."""
    terms = re.findall(
        r"[A-Za-z]{2,}|[ก-๙](?:\s*\.\s*[ก-๙])+\s*\.\s*\d+", question
    )
    if not terms:
        return 0.0
    text = f"{chunk.get('title', '')} {chunk.get('content', '')}".lower()
    normalize = lambda value: re.sub(r"\s+", "", value.lower())
    normalized_text = normalize(text)
    return sum(normalize(term) in normalized_text for term in terms) / len(terms)


def _year_intent_bonus(question, chunk):
    """Prefer the requested historical or current version of a document."""
    document_year = chunk.get("document_year")
    tax_year = chunk.get("tax_year")
    if not document_year or not tax_year:
        return 0.0
    if any(term in question for term in ("ฉบับเก่า", "เอกสารเก่า", "เดิม")):
        return 0.35 if document_year < tax_year else 0.0
    if any(term in question for term in ("ฉบับล่าสุด", "เอกสารล่าสุด", "ปัจจุบัน")):
        return 0.20 if document_year == tax_year else 0.0
    return 0.0


def connect_supabase():
    """Connect using SUPABASE_URL and SUPABASE_SERVICE_KEY from .env."""
    load_dotenv()
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )


def upload_document(
    client,
    file_path,
    title,
    tax_year,
    document_type,
    chunks,
    document_year=None,
):
    """Upload one document and its chunks. Existing rows are updated."""
    file_path = Path(file_path)
    document_year = document_year or tax_year
    source_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    document = {
        "title": title,
        "tax_year": tax_year,
        "document_year": document_year,
        "document_type": document_type,
        "source_file": file_path.name,
        "source_hash": source_hash,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": 1024,
    }

    client.table("documents").upsert(document, on_conflict="source_hash").execute()
    result = (
        client.table("documents")
        .select("id")
        .eq("source_hash", source_hash)
        .single()
        .execute()
    )
    document_id = result.data["id"]

    # Keep unchanged chunk rows (and their numeric IDs) stable across re-ingestion.
    # Stale rows are removed only after the current rows have been upserted, so a
    # failed upload cannot erase the document's previously working chunks.
    existing_rows = (
        client.table("document_chunks")
        .select("chunk_uid")
        .eq("document_id", document_id)
        .execute()
        .data
    )
    existing_uids = {row["chunk_uid"] for row in existing_rows}

    rows = []
    for chunk in chunks:
        rows.append(
            {
                "document_id": document_id,
                "chunk_uid": chunk["chunk_uid"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "tax_year": chunk["tax_year"],
                "metadata": chunk["metadata"],
                "embedding": chunk["embedding"],
            }
        )

    for start in range(0, len(rows), 50):
        client.table("document_chunks").upsert(
            rows[start : start + 50], on_conflict="chunk_uid"
        ).execute()

    current_uids = {row["chunk_uid"] for row in rows}
    stale_uids = sorted(existing_uids - current_uids)
    for start in range(0, len(stale_uids), 50):
        client.table("document_chunks").delete().in_(
            "chunk_uid", stale_uids[start : start + 50]
        ).execute()


def search_similar_chunks(client, model, question, tax_year=2568, k=5):
    """Retrieve and hybrid-rerank chunks for the personal-income-tax corpus."""
    if any(marker in question for marker in OUT_OF_SCOPE_MARKERS):
        return []

    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    candidate_count = max(k, RETRIEVAL_CANDIDATES)
    result = client.rpc(
        "match_tax_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.30,
            "match_count": candidate_count,
            "filter_tax_year": tax_year,
        },
    ).execute()
    candidates = result.data
    candidates.sort(
        key=lambda chunk: (
            chunk["similarity"]
            + 0.35 * _character_overlap(question, chunk)
            + 0.50 * _distinctive_term_overlap(question, chunk)
            + _year_intent_bonus(question, chunk)
        ),
        reverse=True,
    )
    return candidates[:k]


def load_embedding_model():
    load_dotenv()
    return SentenceTransformer(MODEL_NAME)
