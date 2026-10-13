"""Answer a question: retrieve chunks from Supabase, then generate with Ollama.

Usage:
    python ask.py "ผู้มีเงินได้เฉพาะเงินเดือนต้องยื่นแบบอะไร"
    python ask.py                      # interactive mode

Environment (.env or shell):
    OLLAMA_MODEL    model name, default qwen2.5:7b  (run `ollama pull <name>` first)
    OLLAMA_NUM_CTX  context window, default 8192
    OLLAMA_HOST     default http://localhost:11434
"""

import os
import sys

import ollama
from dotenv import load_dotenv

from create_embeddings import (
    connect_supabase,
    load_embedding_model,
    search_similar_chunks,
)

load_dotenv()

DEFAULT_TAX_YEAR = 2568
TOP_K = 5

NOT_FOUND = (
    "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารที่มีอยู่ "
    "กรุณาตรวจสอบกับกรมสรรพากรหรือผู้เชี่ยวชาญด้านภาษีโดยตรง"
)

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับภาษีเงินได้บุคคลธรรมดาของประเทศไทย

กติกา:
1. ตอบจาก "เอกสารอ้างอิง" ที่ให้มาเท่านั้น ห้ามใช้ความรู้นอกเอกสารหรือเดาเพิ่ม
2. อ้างอิงแหล่งที่มาท้ายประโยคที่ใช้ข้อมูลนั้น ในรูปแบบ [1], [2] ตามหมายเลขเอกสาร
3. ถ้าเอกสารอ้างอิงไม่มีข้อมูลเพียงพอ ให้ตอบตรง ๆ ว่าไม่พบข้อมูลในเอกสาร อย่าพยายามตอบ
4. ตัวเลข อัตราภาษี และวันที่ ให้คัดลอกตามเอกสารอย่างระมัดระวัง
5. ตอบเป็นภาษาไทย กระชับ ชัดเจน"""


def format_context(chunks):
    """Number each chunk so the model can cite it as [n]."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        pages = (
            f"หน้า {chunk['page_start']}"
            if chunk["page_start"] == chunk["page_end"]
            else f"หน้า {chunk['page_start']}-{chunk['page_end']}"
        )
        blocks.append(f"[{number}] {chunk['title']} ({pages})\n{chunk['content']}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question, chunks):
    user_prompt = (
        f"เอกสารอ้างอิง:\n\n{format_context(chunks)}\n\n"
        f"คำถาม: {question}\n\n"
        "คำตอบ (พร้อมอ้างอิง [n]):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(question, chunks, stream=True):
    """Send the question and retrieved chunks to Ollama and return the answer."""
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    # Ollama's default context window is small and silently truncates the prompt.
    # Thai text is token-heavy, so 5 chunks of 1,800 characters need a larger window.
    options = {
        "temperature": 0.1,
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    }
    client = ollama.Client()
    messages = build_messages(question, chunks)

    if not stream:
        response = client.chat(model=model_name, messages=messages, options=options)
        return response["message"]["content"]

    parts = []
    for part in client.chat(
        model=model_name, messages=messages, options=options, stream=True
    ):
        text = part["message"]["content"]
        print(text, end="", flush=True)
        parts.append(text)
    print()
    return "".join(parts)


def print_sources(chunks):
    print("\nแหล่งอ้างอิง:")
    for number, chunk in enumerate(chunks, start=1):
        print(
            f"  [{number}] {chunk['title']} | หน้า {chunk['page_start']} "
            f"| similarity {chunk['similarity']:.3f}"
        )


def ask(question, supabase, embedding_model, tax_year=DEFAULT_TAX_YEAR, k=TOP_K):
    chunks = search_similar_chunks(
        supabase, embedding_model, question, tax_year=tax_year, k=k
    )

    # Nothing passed the similarity threshold: skip the LLM instead of letting it guess.
    if not chunks:
        print(NOT_FOUND)
        return {"answer": NOT_FOUND, "sources": []}

    try:
        answer = generate_answer(question, chunks)
    except ConnectionError:
        sys.exit("Cannot reach Ollama. Is it running? Try `ollama serve`.")
    except ollama.ResponseError as error:
        sys.exit(
            f"Ollama error: {error.error}\n"
            f"If the model is missing, run `ollama pull "
            f"{os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')}`."
        )

    print_sources(chunks)
    return {"answer": answer, "sources": chunks}


def main():
    # Keeps Thai output readable in Windows PowerShell.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Load once: the embedding model and Supabase connection are slow to set up.
    embedding_model = load_embedding_model()
    supabase = connect_supabase()

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), supabase, embedding_model)
        return

    print("พิมพ์คำถาม (เว้นว่างหรือ Ctrl+C เพื่อออก)")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(question, supabase, embedding_model)


if __name__ == "__main__":
    main()
