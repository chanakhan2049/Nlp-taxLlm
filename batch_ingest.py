"""Create embeddings for every PDF/TXT in a directory and upload to Supabase."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from create_embeddings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    clean_text,
    create_chunks,
    extract_pdf_pages,
    file_sha256,
    upload_to_supabase,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-ingest tax PDF and TXT files.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--default-tax-year", type=int, default=2568)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--upload-supabase", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_tax_year(path: Path, default: int) -> int:
    # Filenames from the Revenue Department commonly end with DDMMYY (B.E.).
    matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", path.stem)
    if matches:
        short_year = int(matches[-1][-2:])
        if 50 <= short_year <= 99:
            return 2500 + short_year
    return default


def infer_document_type(path: Path) -> str:
    name = path.stem.lower()
    if path.suffix.lower() == ".txt":
        return "revenue_code"
    if name.startswith("ins"):
        return "tax_form_instruction"
    if "pit" in name:
        return "tax_form"
    if "foreign" in name:
        return "foreign_income_attachment"
    if "child" in name:
        return "child_deduction_attachment"
    if name.startswith("ly") or "loryor" in name:
        return "deduction_certificate"
    return "tax_form_attachment"


def title_for(path: Path, tax_year: int) -> str:
    known = {
        "Ins90": "วิธีกรอกแบบ ภ.ง.ด.90",
        "Ins91": "วิธีกรอกแบบ ภ.ง.ด.91",
        "Ins94": "วิธีกรอกแบบ ภ.ง.ด.94",
        "PIT90": "แบบ ภ.ง.ด.90",
        "PIT91": "แบบ ภ.ง.ด.91",
        "PIT93": "แบบ ภ.ง.ด.93",
        "PIT94": "แบบ ภ.ง.ด.94",
        "PIT95": "แบบ ภ.ง.ด.95",
    }
    for token, title in known.items():
        if token.lower() in path.stem.lower():
            return f"{title} ปีภาษี {tax_year} ({path.name})"
    if path.suffix.lower() == ".txt":
        return f"ประมวลรัษฎากร หมวดภาษีเงินได้ ({path.name})"
    return f"เอกสารประกอบภาษี ปีภาษี {tax_year} ({path.name})"


def extract_text_pages(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_pages(path)
    text = clean_text(path.read_text(encoding="utf-8-sig"))
    return ([{"page_number": 1, "text": text}], [] if len(text) >= 50 else [1])


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    files = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
    )
    if not files:
        raise RuntimeError(f"No PDF/TXT files found in {source_dir}")

    prepared: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    for path in files:
        tax_year = infer_tax_year(path, args.default_tax_year)
        title = title_for(path, tax_year)
        pages, possible_scans = extract_text_pages(path)
        chunks = create_chunks(
            pages,
            title=title,
            tax_year=tax_year,
            source_url="",
            document_type=infer_document_type(path),
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        if not chunks:
            print(f"SKIP (no text): {path.name}")
            continue
        print(
            f"Prepared {path.name}: pages={len(pages)}, chunks={len(chunks)}, "
            f"tax_year={tax_year}, low_text_pages={len(possible_scans)}"
        )
        prepared.append(
            {
                "path": path,
                "title": title,
                "tax_year": tax_year,
                "document_type": infer_document_type(path),
                "chunks": chunks,
            }
        )
        all_chunks.extend(chunks)

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model once: {args.model}")
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        [chunk["embedded_content"] for chunk in all_chunks],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dimension = int(vectors.shape[1])
    if args.upload_supabase and dimension != 1024:
        raise RuntimeError(f"Supabase schema expects 1024 dimensions, got {dimension}")
    for chunk, vector in zip(all_chunks, vectors):
        chunk["embedding_model"] = args.model
        chunk["embedding_dimension"] = dimension
        chunk["embedding"] = vector.tolist()

    upload_queue: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []
    for item in prepared:
        path = item["path"]
        metadata = {
            "title": item["title"],
            "source_url": "",
            "tax_year": item["tax_year"],
            "document_type": item["document_type"],
            "source_file": path.name,
            "source_hash": file_sha256(path),
            "embedding_model": args.model,
            "embedding_dimension": dimension,
        }
        output_path = output_dir / f"{path.stem}_embeddings.jsonl"
        write_jsonl(output_path, item["chunks"], metadata, overwrite=args.overwrite)
        print(f"Saved {output_path.name}: {len(item['chunks'])} chunks")
        upload_queue.append((metadata, item["chunks"], path.name))

    # Save every local artifact before starting network writes. If Supabase is
    # unavailable, a retry tool can upload the complete JSONL set without losing work.
    if args.upload_supabase:
        for metadata, chunks, source_name in upload_queue:
            upload_to_supabase(metadata, chunks)
            print(f"Uploaded {source_name}")

    print(
        f"Completed: documents={len(prepared)}, chunks={len(all_chunks)}, "
        f"dimension={dimension}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
