"""
src/graph/nodes/compression_node.py

Node 3 of the TORQ graph: Intelligent Compression.

Only runs when budget_check_node found retrieved_chunks over budget.
Two-stage process:

1. Rerank: use the cross-encoder (reranker.py) to score every
   retrieved chunk's TRUE relevance to the query — more accurate than
   the coarse FAISS distance from retrieval_node.

2. Greedy select: walk the ranked list from most to least relevant,
   adding chunks to the selection until adding the next one would
   exceed token_budget. This is the actual "budgeting" decision —
   most relevant content survives, the rest is dropped.

If even the single most relevant chunks together still can't all fit
(rare, but possible with a very tight budget), a further LLM-based
summarization step could compress the selected chunks' text further.
That's marked as a TODO below — v1 ships with greedy selection only,
which is already enough to demonstrate the budgeting concept and
produces a defensible baseline before adding LLM summarization complexity.
"""

from src.graph.state import RAGState
from src.models.reranker import rerank_chunks
from src.utils.token_counter import count_tokens, count_tokens_per_chunk


def compression_node(state: RAGState) -> dict:
    """
    Reranks retrieved_chunks by true relevance, then greedily selects
    chunks until the token budget is filled. Returns the selection as
    compressed_chunks.
    """
    query = state["query"]
    chunks = state["retrieved_chunks"]
    budget = state["token_budget"]

    ranked_chunks = rerank_chunks(query, chunks)

    selected_chunks = []
    running_total = 0

    for chunk in ranked_chunks:
        chunk_tokens = count_tokens(chunk["text"])
        if running_total + chunk_tokens > budget:
            continue  # skip this one, but keep checking — a smaller
                      # lower-ranked chunk further down might still fit
        selected_chunks.append(chunk)
        running_total += chunk_tokens

    print(f"Compression: {len(chunks)} retrieved -> {len(selected_chunks)} selected "
          f"({running_total}/{budget} tokens used)")

    # TODO (v2): if selected_chunks is empty or running_total is far
    # under budget because even the top chunk alone exceeds it, add an
    # LLM summarization fallback here using compressor_llm.py to shrink
    # the top 1-2 chunks' text instead of dropping them entirely.

    return {
        "compressed_chunks": selected_chunks,
        "compression_applied": True,
    }