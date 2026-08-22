"""
src/ingestion/chunker.py

Stage 0, Step 2 of TORQ pipeline: parsed text -> chunks.

Splits each document's text into overlapping chunks using a recursive,
paragraph/sentence-aware splitter (not fixed character cutting), so
chunks don't break mid-clause in policy documents.

Pipeline position:
    loader.py -> chunker.py -> embed_and_index.py

Run standalone:
    python -m src.ingestion.chunker --input data/parsed/ --output data/chunks/
"""

import os
import json
import argparse
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Defaults — move to src/config.py once that file exists, so budget
# experiments and chunk-size experiments live in one place.
DEFAULT_CHUNK_SIZE = 500       # tokens (approx, since splitter counts chars by default)
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def get_splitter(chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """
    Returns a configured splitter. Tries paragraph breaks first, then
    sentence breaks, then word breaks — keeps policy clauses intact
    as much as possible instead of cutting at arbitrary character counts.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,  # character-based; swap for a token counter if you want exact token control
    )


def chunk_document(text: str, source_filename: str, splitter=None) -> list:
    """
    Splits a single document's text into chunks, attaching metadata
    to each chunk so we can trace it back to its source document later
    (needed for citations and for debugging the compression node).
    """
    if splitter is None:
        splitter = get_splitter()

    raw_chunks = splitter.split_text(text)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": f"{Path(source_filename).stem}_chunk{i}",
            "source_doc": source_filename,
            "chunk_index": i,
            "text": chunk_text,
            "char_count": len(chunk_text),
        })

    return chunks


def chunk_all_documents(input_dir: str, output_dir: str,
                         chunk_size: int = DEFAULT_CHUNK_SIZE,
                         chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """
    Processes every .txt file in input_dir (output of loader.py),
    chunks each one, and saves all chunks to a single chunks.json
    in output_dir — this is what embed_and_index.py will consume next.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splitter = get_splitter(chunk_size, chunk_overlap)

    txt_files = list(input_path.glob("*.txt"))
    print(f"Found {len(txt_files)} parsed text files in {input_dir}\n")

    all_chunks = []

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            print(f"  Skipping {txt_file.name} — empty text (check loader.py output for this file)")
            continue

        doc_chunks = chunk_document(text, txt_file.name, splitter)
        print(f"  {txt_file.name} -> {len(doc_chunks)} chunks")
        all_chunks.extend(doc_chunks)

    output_file = output_path / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_chunks)} total chunks saved to: {output_file}")
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk parsed policy documents")
    parser.add_argument("--input", type=str, required=True, help="Folder with parsed .txt files (e.g. data/parsed/)")
    parser.add_argument("--output", type=str, required=True, help="Folder to save chunks.json (e.g. data/chunks/)")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    args = parser.parse_args()

    chunk_all_documents(args.input, args.output, args.chunk_size, args.chunk_overlap)