"""
src/graph/nodes/retrieval_node.py

Node 1 of the TORQ graph: Vector Retrieval.

Takes the query from state, embeds it using the SAME embedding model
used in embed_and_index.py (must match, otherwise vectors live in
different spaces and similarity scores are meaningless), searches the
FAISS index for the top-k nearest chunks, and returns them into state.

This is coarse relevance only — cosine/L2 distance on embeddings.
Fine-grained relevance scoring happens later in the compression node.
"""

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.graph.state import RAGState

# Must match the model used in embed_and_index.py exactly.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TOP_K = 10

# Loaded once at module import time, not per-query — loading the model
# and index on every single query would be extremely slow.
_embedding_model = None
_faiss_index = None
_chunk_metadata = None


def _load_resources(index_dir: str = "data/index"):
    """
    Lazy-loads the embedding model, FAISS index, and chunk metadata once.
    Subsequent calls reuse the already-loaded objects.
    """
    global _embedding_model, _faiss_index, _chunk_metadata

    if _embedding_model is None:
        print("Loading embedding model for retrieval...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if _faiss_index is None:
        print("Loading FAISS index...")
        _faiss_index = faiss.read_index(f"{index_dir}/faiss_index.bin")

    if _chunk_metadata is None:
        print("Loading chunk metadata...")
        with open(f"{index_dir}/chunk_metadata.pkl", "rb") as f:
            _chunk_metadata = pickle.load(f)


def retrieval_node(state: RAGState, index_dir: str = "data/index", top_k: int = TOP_K) -> dict:
    """
    Embeds the query, searches FAISS for the top_k nearest chunks,
    and returns them with their metadata (source doc, text, etc.)
    plus the L2 distance score, so downstream nodes / metrics can
    see how strong the initial retrieval match was.
    """
    _load_resources(index_dir)

    query = state["query"]
    query_embedding = _embedding_model.encode([query], convert_to_numpy=True).astype("float32")

    # FAISS returns distances and positions (indices), not chunk content directly.
    distances, positions = _faiss_index.search(query_embedding, top_k)

    retrieved_chunks = []
    for dist, pos in zip(distances[0], positions[0]):
        if pos == -1:
            continue  # FAISS pads with -1 if fewer than top_k results exist
        chunk = _chunk_metadata[pos].copy()
        chunk["retrieval_distance"] = float(dist)  # lower = more similar (L2 distance)
        retrieved_chunks.append(chunk)

    print(f"Retrieved {len(retrieved_chunks)} chunks for query: '{query[:60]}...'")

    return {"retrieved_chunks": retrieved_chunks}