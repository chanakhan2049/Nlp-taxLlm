"""Run the embedding pipeline for every PDF/TXT file in a folder."""

import re
import sys
from pathlib import Path

from create_embeddings import (
    CorruptDocumentError,
    add_embeddings,
    connect_supabase,
    create_chunks,
    load_embedding_model,
    search_similar_chunks,
    upload_document,
)


# Run `python batch_ingest.py` for ./data, or pass another folder as an argument.
DATA_FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
CURRENT_TAX_YEAR = 2568


def get_document_year(file_path):
    """Infer the document's publication year from a DDMMYY filename value."""
    dates = re.findall(r"(?<!\d)(\d{6})(?!\d)", file_path.stem)
    return 2500 + int(dates[-1][-2:]) if dates else CURRENT_TAX_YEAR


def get_document_type(file_path):
    name = file_path.stem.lower()
    if file_path.suffix.lower() == ".txt":
        return "revenue_code"
    if name.startswith("ins"):
        return "tax_form_instruction"
    if "pit" in name:
        return "tax_form"
    return "tax_form_attachment"


def get_title(file_path, document_year):
    """Create a readable title from common Revenue Department filenames."""
    lower_name = file_path.stem.lower()
    if lower_name.startswith("ly03") or "loryor03" in lower_name:
        return f"แบบ ล.ย.03 (เอกสารเผยแพร่ปี {document_year}; {file_path.name})"
    if lower_name.startswith("ly04-1"):
        return f"เอกสารแนบแบบ ล.ย.04 (เอกสารเผยแพร่ปี {document_year}; {file_path.name})"
    if lower_name.startswith("ly04"):
        return f"แบบ ล.ย.04 (เอกสารเผยแพร่ปี {document_year}; {file_path.name})"

    names = {
        "ins90": "วิธีกรอกแบบ ภ.ง.ด.90",
        "ins91": "วิธีกรอกแบบ ภ.ง.ด.91",
        "ins94": "วิธีกรอกแบบ ภ.ง.ด.94",
        "pit90": "แบบ ภ.ง.ด.90",
        "pit91": "แบบ ภ.ง.ด.91",
        "pit93": "แบบ ภ.ง.ด.93",
        "pit94": "แบบ ภ.ง.ด.94",
        "pit95": "แบบ ภ.ง.ด.95",
    }
    for keyword, title in names.items():
        if keyword in lower_name:
            return f"{title} (เอกสารเผยแพร่ปี {document_year}; {file_path.name})"
    if file_path.suffix.lower() == ".txt":
        return f"ประมวลรัษฎากร หมวดภาษีเงินได้ ({file_path.name})"
    return f"เอกสารประกอบภาษี (เผยแพร่ปี {document_year}; {file_path.name})"


def main():
    files = sorted(
        file_path
        for file_path in DATA_FOLDER.iterdir()
        if file_path.suffix.lower() in {".pdf", ".txt"}
    )

    model = load_embedding_model()
    supabase = connect_supabase()

    for file_path in files:
        document_year = get_document_year(file_path)
        tax_year = CURRENT_TAX_YEAR
        document_type = get_document_type(file_path)
        title = get_title(file_path, document_year)

        try:
            chunks = create_chunks(
                file_path,
                title,
                tax_year,
                document_type,
                document_year=document_year,
            )
        except CorruptDocumentError as error:
            print(f"Skipped {file_path.name}: {error}")
            continue
        chunks = add_embeddings(chunks, model)
        upload_document(
            supabase,
            file_path,
            title,
            tax_year,
            document_type,
            chunks,
            document_year=document_year,
        )
        print(f"Uploaded {file_path.name}: {len(chunks)} chunks")

    # Retrieval test: cosine similarity, top-k = 5.
    question = "ผู้มีเงินได้เฉพาะเงินเดือนต้องยื่นแบบอะไร"
    results = search_similar_chunks(
        supabase,
        model,
        question,
        tax_year=CURRENT_TAX_YEAR,
        k=5,
    )

    print("\nQuestion:", question)
    for result in results:
        print(
            f"{result['similarity']:.3f} | "
            f"{result['title']} | page {result['page_start']}"
        )


if __name__ == "__main__":
    main()
