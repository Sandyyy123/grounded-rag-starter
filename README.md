> **Proprietary - All Rights Reserved.** Licensed to Sandeep Grover. No reuse, copying, modification, or distribution without prior written permission. See LICENSE.

# grounded-rag-starter

A small, runnable sample of the parts of a RAG product that decide whether users can trust it:
**data cleaning before indexing, retrieval, answers that cite their source, abstention when the
documents do not contain the answer, and a golden-set evaluation that is rerun on every change.**

All documents in `data/` are invented (a fictional "Acme Workspace" help centre). No client data.

```mermaid
flowchart LR
  A[Raw documents] --> B[Clean: normalise, strip boilerplate,<br/>drop short, exact + near duplicates]
  B --> C[Sentence-aware chunks<br/>with stable ids]
  C --> D[BM25 index]
  Q[Question] --> D
  D --> E{Evidence strong enough?<br/>score + IDF-weighted coverage}
  E -- no --> F[Abstain]
  E -- yes --> G[Cited answer<br/>optional LLM, rejected if it cites<br/>anything not retrieved]
  H[Golden set] --> I[hit_rate@k, MRR,<br/>answer-doc accuracy, abstention]
  D --> I
```

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # only pytest; the pipeline is standard-library Python 3.10+
python main.py                       # writes results.json
python main.py "Which plans include SAML single sign-on?"
pytest -q                            # 14 tests
```

Output of `python main.py` on the included data (Python 3.11):

| Step | Result |
|---|---|
| Cleaning | 9 documents in, 6 kept: 1 too short, 1 exact duplicate, 1 near duplicate (Jaccard 0.917), 16 boilerplate lines removed |
| Golden set | 13 questions: 10 answerable, 3 deliberately unanswerable |
| Retrieval | hit_rate@3 1.0, MRR 1.0 |
| Answers | cited document correct for 10/10; abstained on 3/3 unanswerable, 0 false abstentions |

These numbers come from a 13-question synthetic set. The abstention thresholds (`min_score`,
`min_coverage`) were chosen on this same small set, so they are a starting point that must be
recalibrated on a held-out golden set built from the real documents.

## Layout

```
rag_starter/clean.py     normalisation, boilerplate removal, short/duplicate/near-duplicate filtering, report
rag_starter/index.py     sentence-aware chunking, light plural folding, BM25
rag_starter/answer.py    cited extractive answer, abstention, guard for an optional LLM generator
rag_starter/evaluate.py  golden-set metrics
main.py                  end-to-end run
tests/                   pytest suite
```

## What a production build changes

BM25 is used so the sample runs offline and deterministically. The `search()` interface is where a
vector store or hybrid retriever (BM25 + embeddings with rank fusion, optional reranker) plugs in; the
`generator` hook is where a hosted or local LLM writes the final answer. The cleaning report, citation
check, abstention rule and golden-set gate stay the same.
