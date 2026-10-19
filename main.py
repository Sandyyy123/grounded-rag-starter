"""Run the whole pipeline on the synthetic help-centre corpus.

    python main.py            # clean -> chunk -> index -> answer -> evaluate, writes results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rag_starter.answer import answer
from rag_starter.clean import Doc, clean_corpus
from rag_starter.evaluate import evaluate
from rag_starter.index import BM25Index, chunk_doc

ROOT = Path(__file__).parent


def build(corpus_path: Path = ROOT / "data" / "corpus.json"):
    raw = [Doc(**d) for d in json.loads(corpus_path.read_text(encoding="utf-8"))]
    docs, report = clean_corpus(raw)
    chunks = [c for d in docs for c in chunk_doc(d)]
    return BM25Index(chunks), report, len(chunks)


def main(argv: list[str]) -> int:
    index, report, n_chunks = build()
    golden = [json.loads(line) for line in (ROOT / "data" / "golden.jsonl").read_text().splitlines() if line.strip()]
    metrics = evaluate(index, golden, k=3)
    question = " ".join(argv) if argv else "How long do data export download links stay valid?"
    demo = answer(index, question)
    out = {"cleaning": report.as_dict(), "chunks": n_chunks,
           "demo": {"question": question, "answer": demo.text, "abstained": demo.abstained},
           "metrics": metrics}
    (ROOT / "results.json").write_text(json.dumps(out, indent=2))
    summary = {k: v for k, v in metrics.items() if k != "rows"}
    print(json.dumps({"cleaning": report.as_dict(), "chunks": n_chunks, "metrics": summary}, indent=2))
    print(f"\nQ: {question}\nA: {demo.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
