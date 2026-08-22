"""
src/ingestion/loader.py

Stage 0, Step 1 of TORQ pipeline: PDF -> raw text extraction.

Handles two cases:
1. Text-based PDFs (normal, fast path) -> PyMuPDF extracts text directly
2. Scanned/image-based PDFs (like PMGSY docs) -> OCR fallback via pytesseract

This is the first stage of the ingestion pipeline:
    loader.py -> chunker.py -> embed_and_index.py

Run standalone:
    python -m src.ingestion.loader --input data/raw/ --output data/parsed/
"""

import os
import json
import argparse
from pathlib import Path

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract


def extract_text_direct(pdf_path: str) -> str:
    """Try direct text extraction first (fast, works for non-scanned PDFs)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_text_ocr(pdf_path: str, dpi: int = 300) -> str:
    """
    Fallback for scanned/image-based PDFs.
    Converts each page to an image, then runs OCR on it.
    Slower — only use when direct extraction returns near-empty text.
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    text = ""
    for i, img in enumerate(images):
        page_text = pytesseract.image_to_string(img, lang="eng")
        text += f"\n--- Page {i + 1} ---\n{page_text}"
    return text.strip()


def is_likely_scanned(extracted_text: str, min_chars_per_page: int = 50) -> bool:
    """
    Heuristic: if direct extraction returns very little text,
    the PDF is probably scanned/image-based and needs OCR.
    """
    return len(extracted_text.strip()) < min_chars_per_page


def load_pdf(pdf_path: str) -> dict:
    """
    Load and extract text from a single PDF, auto-detecting whether it needs OCR.
    Returns a dict with the extracted text and metadata about the method used.
    """
    filename = os.path.basename(pdf_path)
    print(f"Loading: {filename}")

    text = extract_text_direct(pdf_path)
    method = "direct"

    if is_likely_scanned(text):
        print(f"  -> Looks scanned, falling back to OCR (slower)...")
        text = extract_text_ocr(pdf_path)
        method = "ocr"

    return {
        "filename": filename,
        "text": text,
        "char_count": len(text),
        "extraction_method": method,
    }


def load_all_pdfs(input_dir: str, output_dir: str) -> list:
    """
    Process every PDF in input_dir, save extracted text + metadata to output_dir.
    Also builds a single parse_metadata.json summarizing the batch.

    Returns the list of metadata dicts (also used by chunker.py if chaining
    directly in a pipeline script instead of via saved files).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {input_dir}\n")

    all_metadata = []

    for pdf_file in pdf_files:
        try:
            result = load_pdf(str(pdf_file))

            # Save extracted text as a .txt file, same name as PDF
            out_txt_path = output_path / f"{pdf_file.stem}.txt"
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            all_metadata.append({
                "source_pdf": result["filename"],
                "parsed_txt": out_txt_path.name,
                "char_count": result["char_count"],
                "extraction_method": result["extraction_method"],
            })

        except Exception as e:
            print(f"  ERROR loading {pdf_file.name}: {e}")
            all_metadata.append({
                "source_pdf": pdf_file.name,
                "error": str(e),
            })

    # Save batch metadata — useful later for debugging which docs used OCR
    # (OCR'd docs tend to have noisier text, worth tracking for quality analysis)
    metadata_path = output_path / "parse_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Parsed text saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    return all_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and extract text from policy PDFs")
    parser.add_argument("--input", type=str, required=True, help="Folder containing raw PDFs (e.g. data/raw/)")
    parser.add_argument("--output", type=str, required=True, help="Folder to save extracted text (e.g. data/parsed/)")
    args = parser.parse_args()

    load_all_pdfs(args.input, args.output)