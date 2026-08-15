"""
loader.py — TORQ ingestion module

Reads all PDFs from data/raw/, extracts raw text page-by-page,
and saves cleaned text files into data/processed/.

Usage:
    python loader.py
"""

import os
from pathlib import Path
from pypdf import PdfReader


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a single PDF, page by page.
    Returns one combined string with page markers,
    so we know which page each chunk of text came from later.
    """
    reader = PdfReader(str(pdf_path))
    full_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""  # fallback to empty string if extraction fails
        page_text = page_text.strip()

        if page_text:  # skip totally blank pages
            full_text.append(f"\n\n--- PAGE {page_num} ---\n\n{page_text}")

    return "".join(full_text)


def clean_text(text: str) -> str:
    """
    Basic cleanup — collapses excessive blank lines and whitespace.
    Extend this later if you notice garbage characters, headers/footers repeating, etc.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # remove empty lines
    return "\n".join(lines)


def process_all_pdfs():
    """
    Loops through every PDF in data/raw/, extracts + cleans text,
    and writes one .txt file per PDF into data/processed/.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}. Add some PDFs there first.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting extraction...\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(raw_text)

            output_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
            output_path.write_text(cleaned, encoding="utf-8")

            word_count = len(cleaned.split())
            print(f"  -> Saved: {output_path} ({word_count} words)\n")

        except Exception as e:
            print(f"  !! Failed to process {pdf_path.name}: {e}\n")

    print("Done. All processed files are in data/processed/.")


if __name__ == "__main__":
    process_all_pdfs()