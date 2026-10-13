"""Run a repeatable retrieval and optional answer evaluation set."""

import argparse
import json
import sys
from pathlib import Path

from ask import DEFAULT_TAX_YEAR, NOT_FOUND, generate_answer
from create_embeddings import (
    connect_supabase,
    load_embedding_model,
    search_similar_chunks,
)


DEFAULT_CASES = Path(__file__).with_name("evaluation_questions.json")


def load_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation file must contain a non-empty JSON list.")
    return cases


def source_text(chunks):
    return "\n".join(
        f"{chunk.get('title', '')}\n{chunk.get('content', '')}" for chunk in chunks
    ).lower()


def retrieval_passes(case, chunks):
    if case.get("expect_no_answer"):
        return not chunks
    haystack = source_text(chunks)
    terms = case.get("expected_source_terms", [])
    return bool(chunks) and any(term.lower() in haystack for term in terms)


def main():
    # Keep Thai case names and generated answers readable in Windows shells.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--tax-year", type=int, default=DEFAULT_TAX_YEAR)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    model = load_embedding_model()
    supabase = connect_supabase()
    passed = 0

    for case in cases:
        chunks = search_similar_chunks(
            supabase,
            model,
            case["question"],
            tax_year=args.tax_year,
            k=args.top_k,
        )
        retrieval_ok = retrieval_passes(case, chunks)
        answer = None
        citation_ok = None
        if args.with_llm:
            answer = generate_answer(case["question"], chunks, stream=False) if chunks else NOT_FOUND
            citation_ok = case.get("expect_no_answer", False) or "[" in answer

        passed += retrieval_ok and (citation_ok is not False)
        status = "PASS" if retrieval_ok and citation_ok is not False else "FAIL"
        print(f"{status} {case['id']}: {case['question']}")
        if not retrieval_ok:
            for chunk in chunks[:3]:
                print(
                    f"  {chunk.get('similarity', 0):.3f} | "
                    f"{chunk.get('title', '')} | {chunk.get('content', '')[:120]}"
                )
        if answer is not None:
            print(f"  {answer}")

    print(f"\nPassed {passed}/{len(cases)} cases")
    raise SystemExit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
