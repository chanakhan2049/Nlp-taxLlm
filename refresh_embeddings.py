"""Refresh live Supabase embeddings when original source files are unavailable."""

from create_embeddings import (
    add_embeddings,
    connect_supabase,
    has_corrupt_pdf_text,
    load_embedding_model,
)


PAGE_SIZE = 500


def fetch_chunks(client):
    rows = []
    start = 0
    while True:
        page = (
            client.table("document_chunks")
            .select(
                "id,content,page_start,metadata,"
                "documents(title,document_year,tax_year)"
            )
            .order("id")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
            .data
        )
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def embedding_input(row):
    document = row["documents"]
    return (
        f"เอกสาร: {document['title']}\n"
        f"ปีที่เผยแพร่เอกสาร: {document['document_year']}\n"
        f"ใช้สำหรับปีภาษี: {document['tax_year']}\n"
        f"หน้า: {row['page_start']}\n{row['content']}"
    )


def refresh(client, model):
    rows = fetch_chunks(client)
    valid_rows = []
    corrupt_rows = []
    for row in rows:
        (corrupt_rows if has_corrupt_pdf_text(row["content"]) else valid_rows).append(row)

    vectors = model.encode(
        [embedding_input(row) for row in valid_rows],
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for row, vector in zip(valid_rows, vectors):
        document = row["documents"]
        metadata = {
            **(row.get("metadata") or {}),
            "title": document["title"],
            "document_year": document["document_year"],
            "applicable_tax_year": document["tax_year"],
            "text_quality": "ok",
        }
        (
            client.table("document_chunks")
            .update({"embedding": vector.tolist(), "metadata": metadata})
            .eq("id", row["id"])
            .execute()
        )

    for row in corrupt_rows:
        metadata = {**(row.get("metadata") or {}), "text_quality": "corrupt"}
        (
            client.table("document_chunks")
            .update({"metadata": metadata})
            .eq("id", row["id"])
            .execute()
        )

    return len(valid_rows), len(corrupt_rows)


def main():
    valid, corrupt = refresh(connect_supabase(), load_embedding_model())
    print(f"Refreshed {valid} embeddings; flagged {corrupt} corrupt chunks.")


if __name__ == "__main__":
    main()
