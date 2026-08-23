"""
src/graph/state.py

Defines RAGState — the shared state object that flows through every
node in the TORQ LangGraph workflow. Every node reads from this and
returns partial updates that get merged back in.

This must be defined BEFORE any nodes are written, since every node
function's signature depends on this schema.
"""

import operator
from typing import TypedDict, List, Optional, Annotated


class RAGState(TypedDict):
    # --- Input ---
    query: str

    # --- Retrieval node output ---
    retrieved_chunks: List[dict]        # raw chunks from FAISS: [{chunk_id, source_doc, text, ...}, ...]

    # --- Budget check node output ---
    token_budget: int                   # fixed ceiling, set at graph invocation time (e.g. 2000)
    current_token_count: int            # total tokens across retrieved_chunks

    # --- Compression node output ---
    compressed_chunks: List[dict]       # subset of retrieved_chunks selected by reranker, within budget
    compression_applied: bool           # whether compression node actually ran this turn

    # --- Context assembly node output ---
    final_context: str                  # the assembled text actually sent to the reasoning LLM

    # --- Reasoning node output ---
    answer: str

    # --- Quality eval node output ---
    quality_score: float                # 0.0-1.0, how good the answer was judged to be
    quality_passed: bool                # whether quality_score cleared the threshold

    # --- Retry loop control ---
    retry_count: int                    # incremented each time quality_eval routes back to compression
    max_retries: int                    # ceiling to prevent infinite loops, checked in edges.py

    # --- Metrics (accumulates across retries — needs a reducer) ---
    # operator.add on a list means each node's return value gets APPENDED
    # to this list instead of overwriting it. Critical for the retry loop:
    # without this, only the last retry's metrics would survive.
    metrics_log: Annotated[List[dict], operator.add]


def get_initial_state(query: str, token_budget: int = 2000, max_retries: int = 2) -> RAGState:
    """
    Helper to construct a clean starting state for a new query.
    Use this in main.py rather than hand-building the dict every time,
    so every run starts with consistent defaults.
    """
    return RAGState(
        query=query,
        retrieved_chunks=[],
        token_budget=token_budget,
        current_token_count=0,
        compressed_chunks=[],
        compression_applied=False,
        final_context="",
        answer="",
        quality_score=0.0,
        quality_passed=False,
        retry_count=0,
        max_retries=max_retries,
        metrics_log=[],
    )