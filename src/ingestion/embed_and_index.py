"""
src/ingestion/embed_and_index.py

Stage 0, Step 3 of TORQ pipeline: chunks -> embeddings -> FAISS index.

Takes the chunks.json produced by chunker.py, embeds each chunk's text
using a sentence embedding model, and builds a FAISS index for fast
similarity search at query time.

Also saves a separate metadata store (chunk_id -> full chunk info),
since FAISS itself only stores vectors + an integer position — it
does NOT store the original text or metadata. We need a side lookup
table to go from "FAISS returned position 42" back to "here's the
actual chunk text and which document it came from."

Pipeline position:
    loader.py -> chunker.py -> embed_and_index.py   <-- YOU ARE HERE

This is the final step of the OFFLINE setup. After this runs once,
the FAISS index + metadata store are what the runtime graph's
Retrieval node queries against — chunking and embedding do NOT
happen again per-query.

Run standalone:
    python -m src.ingestion.embed_and_index --input data/chunks/chunks.json --output data/index/
"""

import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Move to src/config.py once that file exists.
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Multilingual because your corpus/queries may mix Hindi and English (Hinglish).


def load_chunks(chunks_path: str) -> list:
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks: list, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 32) -> np.ndarray:
    """
    Embeds every chunk's text. Returns a numpy array of shape
    (num_chunks, embedding_dim).

    Batching matters here — embedding 2000+ chunks one at a time is slow;
    the model processes them much faster in batches.
    """
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.astype("float32")  # FAISS requires float32


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a flat L2 index — exact nearest-neighbor search, no approximation.
    Fine for ~a few thousand chunks (200 docs worth). If the corpus grows
    much larger (50k+ chunks), switch to an approximate index like
    IndexIVFFlat for speed — not needed at this scale.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Built FAISS index: {index.ntotal} vectors, dimension {dimension}")
    return index


def save_index_and_metadata(index: faiss.Index, chunks: list, output_dir: str):
    """
    Saves two things side by side:
    1. faiss_index.bin — the actual vector index
    2. chunk_metadata.pkl — position -> chunk info lookup

    FAISS returns results as integer positions (e.g. "position 42 is
    closest"). chunk_metadata.pkl lets us map that back to the real
    chunk_id, source_doc, and text. Position i in this list corresponds
    exactly to row i in the FAISS index (same order they were added in).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_path / "faiss_index.bin"))

    with open(output_path / "chunk_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index to: {output_path / 'faiss_index.bin'}")
    print(f"Saved chunk metadata to: {output_path / 'chunk_metadata.pkl'}")


def build_index_from_chunks(chunks_path: str, output_dir: str, model_name: str = DEFAULT_MODEL_NAME):
    """Full pipeline: load chunks -> embed -> build FAISS -> save."""
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}\n")

    embeddings = embed_chunks(chunks, model_name)
    index = build_faiss_index(embeddings)
    save_index_and_metadata(index, chunks, output_dir)

    print("\nOffline setup complete. The FAISS index is ready for querying.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and build FAISS index")
    parser.add_argument("--input", type=str, required=True, help="Path to chunks.json (from chunker.py)")
    parser.add_argument("--output", type=str, required=True, help="Folder to save FAISS index + metadata")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    build_index_from_chunks(args.input, args.output, args.model)