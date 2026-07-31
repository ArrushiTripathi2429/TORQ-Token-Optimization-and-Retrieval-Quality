# TORQ — Context-Budgeting RAG Optimizer

A LangGraph-based Retrieval-Augmented Generation system that treats LLM context as a **financial budget** — retrieving, scoring, and compressing information so that only the most relevant content fits within a fixed token allowance, without sacrificing answer quality.

---

## Problem Statement

Most RAG pipelines retrieve a fixed number of chunks (top-k) and stuff them into the context window regardless of size, relevance density, or cost. This leads to three failure modes:

1. **Wasted tokens** — irrelevant or redundant chunks eat into the context window, increasing cost and latency without improving answer quality.
2. **Truncation** — when retrieved content exceeds the context limit, naive systems either fail or blindly truncate, potentially cutting the most relevant information.
3. **No cost accountability** — most RAG systems don't track *how much* context is actually necessary versus how much is being spent, making it impossible to optimize.

TORQ addresses this by treating the context window as a limited budget that must be actively managed: retrieve broadly, score for relevance, compress intelligently when over budget, and evaluate whether the resulting answer quality holds up — retrying with an adjusted strategy if it doesn't.

**Dataset:** 200 Indian government policy documents (rural development and Uttar Pradesh state policies), used to simulate a realistic, high-volume, long-document retrieval setting where budget pressure is real.

---

## Approach

TORQ is built as a 7-node LangGraph workflow with both fixed and conditional routing:

```
Vector Retrieval → Budget Check ──┬─→ Intelligent Compression ─┐
                                   │                             │
                                   └─────────────────────────────┴─→ Context Assembly
                                                                       │
                                                                       ▼
                                                                Main Reasoning
                                                                       │
                                                                       ▼
                                                              Quality Evaluation ──┐
                                                                       │           │
                                                                       ▼           │ (retry loop:
                                                              Metrics Reporting    │  re-compress /
                                                                                   │  re-retrieve)
                                                                       └───────────┘
```

### Node-by-node breakdown

| Node | Responsibility |
|---|---|
| **Vector Retrieval** | Embeds the query, runs FAISS similarity search over 200 policy docs, returns top-k candidate chunks. Coarse relevance only — fast, cheap, imprecise. |
| **Budget Check** | Counts tokens across retrieved chunks and compares against the configured token budget. Routes to Compression if over budget, or straight to Context Assembly if under. |
| **Intelligent Compression** | Fine-grained relevance scoring (cross-encoder reranker) ranks chunks by actual relevance to the query. Chunks are greedily selected until the budget is filled; if still over budget at the chunk level, a fast/cheap LLM summarizes the selected content further. |
| **Context Assembly** | Assembles the final context string from selected/compressed chunks, in a consistent format for the reasoning model. |
| **Main Reasoning** | The primary (larger, more expensive) LLM generates an answer using the assembled, budget-compliant context. |
| **Quality Evaluation** | Scores the answer for relevance/correctness/completeness. If quality is below threshold, routes back into the loop (adjust budget, re-compress, or re-retrieve) rather than accepting a weak answer. |
| **Metrics Reporting** | Compiles cost, latency, compression ratio, and quality metrics for the run, enabling before/after comparison against a non-budgeted baseline. |

### Why this design

- **Two-stage relevance filtering**: coarse (embedding similarity) at retrieval, fine (cross-encoder + LLM) at compression. This keeps the expensive judgment calls limited to a smaller candidate set instead of scoring everything against the full document store.
- **Retry loop, not hard failure**: if compression degrades answer quality too much, the system adjusts rather than returning a bad answer silently.
- **Metrics-first**: every run is measured against a baseline (uncompressed, unbudgeted) so the value of budgeting is quantified, not assumed.

---

## Architecture Details

### State

A single `RAGState` object flows through all nodes, carrying the query, retrieved documents, token budget/usage, compressed and final context, answer, quality score, retry count, and accumulated metrics. Each node reads what it needs and returns only the fields it updates.

### Conditional edges (where the intelligence lives)

- **Post budget-check**: `under budget → Context Assembly` / `over budget → Compression`
- **Post quality-eval**: `quality acceptable → Metrics Reporting` / `quality poor → back into compression/retrieval with adjusted parameters`

### Models used

- **Embeddings**: multilingual sentence embedding model for FAISS retrieval
- **Reranker**: cross-encoder for fine-grained chunk relevance scoring
- **Compression model**: fast, cheap LLM for summarizing selected chunks when still over budget
- **Reasoning model**: larger LLM for final answer generation

---

## Results & Metrics

> Populate this section once `results/` is generated from actual runs.

| Metric | Baseline (no budgeting) | TORQ (budgeted) | Delta |
|---|---|---|---|
| Avg. tokens per query | — | — | — |
| Avg. cost per query | — | — | — |
| Avg. latency per query | — | — | — |
| Answer quality score | — | — | — |
| Compression ratio | — | — | — |

Detailed breakdowns available in:
- `results/cost_analysis.csv`
- `results/latency_comparison.json`
- `results/quality_metrics.json`
- `results/sample_outputs/`

---

## How to Run

### 1. Setup

```bash
git clone <repo-url>
cd context-budgeting-rag
pip install -r requirements.txt
```

### 2. Prepare data

Place government policy PDFs in `data/policy_pdfs/`, then build the vector index:

```bash
python src/pdf_parser.py --input data/policy_pdfs/ --output data/parsed/
python src/vector_store.py --build --input data/parsed/
```

### 3. Run the workflow

```bash
python src/langgraph_workflow.py --query "your query here" --budget 2000
```

Or explore interactively:

```bash
jupyter notebook notebooks/demo.ipynb
```

### 4. Generate metrics

```bash
python src/metrics_tracker.py --compare-baseline
```

Outputs cost, latency, and quality comparisons to `results/`.

---

## Project Structure

```
context-budgeting-rag/
├── README.md
├── data/
│   ├── policy_pdfs/        # 200 government documents
│   ├── metadata.json       # doc titles, dates, sources
│   └── sample_queries.txt
├── src/
│   ├── pdf_parser.py       # extract text from PDFs
│   ├── vector_store.py     # FAISS setup
│   ├── rag_nodes.py        # all 7 nodes
│   ├── langgraph_workflow.py  # graph + routing
│   └── metrics_tracker.py  # cost, latency, quality
├── notebooks/
│   └── demo.ipynb          # live walkthrough
└── results/
    ├── cost_analysis.csv
    ├── latency_comparison.json
    ├── quality_metrics.json
    └── sample_outputs/
```

---

## Future Work

- Adaptive budget sizing based on query complexity rather than a fixed constant
- Multi-hop retrieval for queries requiring synthesis across multiple documents
- Caching compressed contexts for repeated/similar queries to reduce redundant compression cost
