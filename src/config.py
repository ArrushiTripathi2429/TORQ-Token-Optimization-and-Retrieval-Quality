"""
src/config.py

Centralizes settings that were previously hardcoded across individual
files (marked with "move to config.py once that exists" comments).

Having these in one place matters specifically for TORQ's budget
experiments — testing multiple token_budget values (the "results/
cost_analysis.csv" comparison) means changing ONE line here instead
of hunting through 7 node files.

Import from here rather than redefining constants locally:
    from src.config import TOKEN_BUDGET, EMBEDDING_MODEL_NAME
"""

import os

# --- Paths ---
DATA_DIR = "data"
RAW_PDF_DIR = f"{DATA_DIR}/raw"
PARSED_TEXT_DIR = f"{DATA_DIR}/parsed"
CHUNKS_PATH = f"{DATA_DIR}/chunks/chunks.json"
INDEX_DIR = f"{DATA_DIR}/index"
RESULTS_DIR = "results"

# --- Chunking (used by chunker.py) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embedding model (used by embed_and_index.py AND retrieval_node.py —
# these two MUST match, otherwise query vectors and stored chunk vectors
# live in different embedding spaces and similarity search breaks silently) ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- Retrieval (used by retrieval_node.py) ---
TOP_K_RETRIEVAL = 10

# --- Reranker (used by reranker.py) ---
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- LLMs (used by reasoning_llm.py and quality_eval_node.py) ---
REASONING_MODEL_NAME = "llama-3.3-70b-versatile"   # larger model, main answer generation
JUDGE_MODEL_NAME = "llama-3.1-8b-instant"          # smaller/cheaper model, quality scoring only
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Budgeting (used by budget_check_node.py, compression_node.py, main.py) ---
# This is the primary knob for your cost/quality experiments — run
# main.py with different values here (or via --budget CLI arg, which
# overrides this default) to populate results/cost_analysis.csv.
DEFAULT_TOKEN_BUDGET = 2000

# --- Quality / retry loop (used by edges.py, quality_eval_node.py) ---
QUALITY_THRESHOLD = 0.7
DEFAULT_MAX_RETRIES = 2

# --- Graph execution safety (used by main.py) ---
RECURSION_LIMIT = 25