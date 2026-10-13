"""Core functions for the Thai tax RAG pipeline.

Pipeline: read documents -> create chunks -> create embeddings -> Supabase.
"""

import hashlib
import os
import re
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client


MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 300


def clean_text(text):
    """Remove unnecessary whitespace from extracted text."""
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_document(file_path):
    """Read a PDF page by page, or read a TXT file as one page."""
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".txt":
        text = file_path.read_text(encoding="utf-8-sig")
        return [{"page": 1, "text": clean_text(text)}]

    pages = []
    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks", sort=True)
            text = "\n".join(block[4] for block in blocks)
            text = clean_text(text)
            pages.append({"page": page_number, "text": text})
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


def create_chunks(file_path, title, tax_year, document_type):
    """Create chunks and attach metadata used for citation and filtering."""
    chunks = []

    for page in read_document(file_path):
        for page_chunk_index, content in enumerate(split_text(page["text"])):
            chunk_number = len(chunks)
            text_for_embedding = (
                f"เอกสาร: {title}\nปีภาษี: {tax_year}\n"
                f"หน้า: {page['page']}\n{content}"
            )
            chunk_id = hashlib.sha256(
                f"{title}|{tax_year}|{page['page']}|{page_chunk_index}|{content}".encode("utf-8")
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


def connect_supabase():
    """Connect using SUPABASE_URL and SUPABASE_SERVICE_KEY from .env."""
    load_dotenv()
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )


def upload_document(client, file_path, title, tax_year, document_type, chunks):
    """Upload one document and its chunks. Existing rows are updated."""
    file_path = Path(file_path)
    source_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    document = {
        "title": title,
        "source_url": "",
        "tax_year": tax_year,
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


def search_similar_chunks(client, model, question, tax_year=2568, k=5):
    """Retrieve the k most similar chunks from Supabase."""
    query_vector = model.encode(question, normalize_embeddings=True).tolist()
    result = client.rpc(
        "match_tax_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.30,
            "match_count": k,
            "filter_tax_year": tax_year,
        },
    ).execute()
    return result.data


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)
