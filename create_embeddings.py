"""Create vector embeddings from a PDF and optionally upload them to Supabase.

Example:
    python create_embeddings.py ^
      --pdf "data/Ins90_2568.pdf" ^
      --title "วิธีกรอกแบบ ภ.ง.ด.90 ปีภาษี 2568" ^
      --tax-year 2568 ^
      --source-url "https://www.rd.go.th/..." ^
      --output "output/Ins90_2568_embeddings.jsonl"

Add --upload-supabase after creating the tables in supabase_schema.sql.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a PDF, create chunks and embeddings, and save JSONL."
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the PDF")
    parser.add_argument("--title", required=True, help="Human-readable document title")
    parser.add_argument("--tax-year", type=int, required=True, help="Thai tax year, e.g. 2568")
    parser.add_argument("--source-url", default="", help="Official source URL")
    parser.add_argument(
        "--document-type",
        default="tax_reference",
        help="Metadata such as tax_form_instruction, faq, or announcement",
    )
    parser.add_argument("--output", type=Path, help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--upload-supabase",
        action="store_true",
        help="Upload the document and chunks after creating the JSONL file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing JSONL output file",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is missing. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    document = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []
    possible_scans: list[int] = []

    try:
        for page_number, page in enumerate(document, start=1):
            # Blocks with sorting preserve reading order better than plain extraction.
            blocks = page.get_text("blocks", sort=True)
            text = "\n".join(str(block[4]) for block in blocks if len(block) > 4)
            text = clean_text(text)
            if len(text) < 50:
                possible_scans.append(page_number)
            pages.append({"page_number": page_number, "text": text})
    finally:
        document.close()

    return pages, possible_scans


def preferred_break(text: str, lower: int, upper: int) -> int:
    """Find a readable split near upper, while always making progress."""
    candidate = text[lower:upper]
    separators = ("\n\n", "\n", ". ", "。", " ")
    for separator in separators:
        position = candidate.rfind(separator)
        if position >= 0:
            return lower + position + len(separator)
    return upper


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1")

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        if hard_end == text_length:
            end = hard_end
        else:
            lower = start + max(chunk_size // 2, 1)
            end = preferred_break(text, lower, hard_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(
    pages: list[dict[str, Any]],
    *,
    title: str,
    tax_year: int,
    source_url: str,
    document_type: str,
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        if not page["text"]:
            continue

        # Prefixing critical metadata helps retrieval even when metadata filters are not used.
        prefix = f"เอกสาร: {title}\nปีภาษี: {tax_year}\nหน้า: {page['page_number']}\n"
        page_chunks = split_text(page["text"], chunk_size, overlap)

        for page_chunk_index, content in enumerate(page_chunks):
            embedded_content = f"{prefix}{content}"
            chunk_uid = hashlib.sha256(
                f"{title}|{tax_year}|{page['page_number']}|{page_chunk_index}|{content}".encode(
                    "utf-8"
                )
            ).hexdigest()
            chunks.append(
                {
                    "chunk_uid": chunk_uid,
                    "chunk_index": chunk_index,
                    "content": content,
                    "embedded_content": embedded_content,
                    "page_start": page["page_number"],
                    "page_end": page["page_number"],
                    "tax_year": tax_year,
                    "metadata": {
                        "title": title,
                        "source_url": source_url,
                        "document_type": document_type,
                        "page_chunk_index": page_chunk_index,
                    },
                }
            )
            chunk_index += 1

    return chunks


def add_embeddings(
    chunks: list[dict[str, Any]], model_name: str, batch_size: int
) -> tuple[list[dict[str, Any]], int]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is missing. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    inputs = [chunk["embedded_content"] for chunk in chunks]
    vectors = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dimension = int(vectors.shape[1])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding_model"] = model_name
        chunk["embedding_dimension"] = dimension
        chunk["embedding"] = vector.tolist()

    return chunks, dimension


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(
    output_path: Path,
    chunks: list[dict[str, Any]],
    document_metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for chunk in chunks:
            row = {"document": document_metadata, **chunk}
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_to_supabase(
    document_metadata: dict[str, Any], chunks: list[dict[str, Any]], batch_size: int = 50
) -> None:
    try:
        from dotenv import load_dotenv
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "Supabase dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env before upload."
        )

    client = create_client(supabase_url, supabase_key)
    client.table("documents").upsert(
        document_metadata, on_conflict="source_hash"
    ).execute()
    document_result = (
        client.table("documents")
        .select("id")
        .eq("source_hash", document_metadata["source_hash"])
        .single()
        .execute()
    )
    document_id = document_result.data["id"]

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
                "metadata": {
                    **chunk["metadata"],
                    "embedding_model": chunk["embedding_model"],
                    "embedding_dimension": chunk["embedding_dimension"],
                },
                "embedding": chunk["embedding"],
            }
        )

    for start in range(0, len(rows), batch_size):
        client.table("document_chunks").upsert(
            rows[start : start + batch_size], on_conflict="chunk_uid"
        ).execute()
        print(f"Uploaded {min(start + batch_size, len(rows))}/{len(rows)} chunks")


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Input is not a PDF: {pdf_path}", file=sys.stderr)
        return 2

    output_path = args.output
    if output_path is None:
        output_path = Path("output") / f"{pdf_path.stem}_embeddings.jsonl"
    output_path = output_path.expanduser().resolve()

    print(f"Extracting: {pdf_path}")
    pages, possible_scans = extract_pdf_pages(pdf_path)
    if possible_scans:
        preview = ", ".join(map(str, possible_scans[:20]))
        suffix = " ..." if len(possible_scans) > 20 else ""
        print(
            "Warning: pages with little extractable text may require OCR: "
            f"{preview}{suffix}"
        )

    chunks = create_chunks(
        pages,
        title=args.title,
        tax_year=args.tax_year,
        source_url=args.source_url,
        document_type=args.document_type,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    if not chunks:
        print("No text chunks were created. The PDF may require OCR.", file=sys.stderr)
        return 3

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    chunks, dimension = add_embeddings(chunks, args.model, args.batch_size)

    document_metadata = {
        "title": args.title,
        "source_url": args.source_url,
        "tax_year": args.tax_year,
        "document_type": args.document_type,
        "source_file": pdf_path.name,
        "source_hash": file_sha256(pdf_path),
        "embedding_model": args.model,
        "embedding_dimension": dimension,
    }
    write_jsonl(output_path, chunks, document_metadata, overwrite=args.overwrite)
    print(f"Saved embeddings: {output_path}")
    print(f"Embedding dimension: {dimension}")

    if args.upload_supabase:
        if dimension != 1024:
            raise RuntimeError(
                "supabase_schema.sql expects 1024 dimensions. Update the schema before "
                f"uploading embeddings with {dimension} dimensions."
            )
        upload_to_supabase(document_metadata, chunks)
        print("Supabase upload completed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
