# ANLP Assignment 2 — RAG Pipeline for Pittsburgh & CMU QA

## Overview
End-to-end Retrieval Augmented Generation (RAG) system for answering questions about Pittsburgh and Carnegie Mellon University. The pipeline combines hybrid retrieval (dense + sparse) with cross-encoder reranking and generative QA.

## Architecture
1. **Data Collection** — Scraped pages from CMU, Pittsburgh gov, Wikipedia, food/event sites using `requests` + `BeautifulSoup`
2. **Document Parsing & Cleaning** — HTML → clean text with aggressive artifact removal
3. **Chunking** — Sentence-aware chunking (200 words, 40-word overlap)
4. **Dense Retrieval** — `sentence-transformers/all-mpnet-base-v2` embeddings indexed with FAISS (`IndexFlatIP`, cosine similarity)
5. **Sparse Retrieval** — BM25 via `rank_bm25.BM25Okapi` with custom tokenization & stop-word removal
6. **Hybrid Fusion** — Reciprocal Rank Fusion (RRF, k=60) merges dense + sparse results
7. **Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top-100 candidates → top-10
8. **Generation** — `google/flan-t5-large` generates answers from top-10 reranked contexts
9. **Post-processing** — Answer cleaning, extractive fallback, answer-sentence grounding

## Models Used
| Component | Model | Source |
|-----------|-------|--------|
| Embedder | `all-mpnet-base-v2` | HuggingFace sentence-transformers |
| Reranker | `ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder |
| Generator | `flan-t5-large` (780M params) | HuggingFace google |

All models are open-weight, available on HuggingFace, released before January 2025, and ≤32B parameters.

## Setup & Installation

```bash
# Create environment (Python 3.10+)
pip install -r requirements.txt
```

### Requirements
- `beautifulsoup4` — HTML parsing
- `requests` — Web scraping
- `transformers` — Generator model (flan-t5-large)
- `sentence-transformers` — Embedding & reranking
- `faiss-cpu` — Dense vector index
- `rank-bm25` — Sparse retrieval (BM25Okapi)
- `torch` — PyTorch backend (CUDA recommended)
- `numpy` — Numerical operations

## Running the Pipeline

### On the leaderboard queries
```bash
python run_full_pipeline.py
# Uses leaderboard_queries.json by default
# Outputs: submission_Pidgeot.json + system_outputs/leaderboard_output.json
```

### On a custom/test query file
```bash
python run_full_pipeline.py path/to/test_queries.json output.json
# Outputs: output.json + system_outputs/system_output_1.json
```

### Rebuilding the index from scratch
Delete cached files first:
```bash
rm data/chunks_v2.jsonl
rm -rf data/faiss_v2/
python run_full_pipeline.py
```
This will re-parse all documents in `assignment_2_baseline_data/baseline_data/` and `data/docs/`, re-chunk, re-embed, and rebuild the FAISS index.

## Project Structure
```
├── .gitignore                      # git ignore rules
├── run_full_pipeline.py            # Main pipeline (preprocessing, retrieval, generation)
├── requirements.txt                # Python dependencies
├── README.md                       # Short project README (optional)
├── PROJECT_README.md               # This detailed readme
├── report.pdf                      # Compiled project report (PDF)
├── submission_Pidgeot.json         # Latest leaderboard submission
├── test_set_day1.json              # Test queries (day 1)
├── test_set_day1.txt               # Raw text version of test set
├── system_outputs/                 # Saved system outputs for Canvas submission
└── data/                           # data/, index/, and cached artifacts (chunks, faiss)
	├── chunks_v2.jsonl
	└── faiss_v2/index.faiss

```

## Data Sources
- **Wikipedia** — CMU, Pittsburgh history, landmarks, notable people
- **www.cmu.edu** — University info, academics, admissions, traditions
- **visitpittsburgh.com** — Tourism, events, attractions
- **pittsburghpa.gov** — City government, services
- **pittsburghsymphony.org** — PSO info
- **pghcitypaper.com** — Events, culture
- **pghtacofest.com** — Pittsburgh Taco Festival
- **ppgpaintsarena.com** — Arena events
- **carnegiemnh.org** — Carnegie Museum of Natural History
- Curated additional facts documents


- Python: the codebase is developed and tested with Python 3.10+. Use a virtual environment.
- Models: pre-trained weights are downloaded from HuggingFace at runtime; expect significant disk and memory usage for `flan-t5-large` (use GPU if available).
- Data & artifacts: the repository includes cleaned text and a cached FAISS index under `data/` where available. Large raw crawl data and model weights are not checked into the repository — they are downloaded or rebuilt by the pipeline as described below.

## Quick Smoke Test (fast, no model downloads)

If you want to quickly verify preprocessing and chunking without downloading large models or running generation, run the following Python one-liner from the repository root. It will parse the baseline HTML and scraped text, build chunks, and save the `data/chunks_v2.jsonl` file:

```bash
python - <<'PY'
from run_full_pipeline import load_all_documents, build_chunks, save_chunks
docs = load_all_documents()
chunks = build_chunks(docs)
save_chunks(chunks)
print(f"Built {len(chunks)} chunks")
PY
```

This avoids loading FAISS/model weights and is a quick functional check of the preprocessing steps.

## Rebuilding models/indexes

- To rebuild the FAISS index from the newly created `data/chunks_v2.jsonl`:

```bash
rm -rf data/faiss_v2/
python -c "from run_full_pipeline import build_faiss_index, load_chunks; chunks=load_chunks(); build_faiss_index(chunks)"
```

Note: the above command will download embedding models if not already present and may take substantial time depending on environment.

## Submission Checklist (GitHub repo)

- `run_full_pipeline.py` — main pipeline (preprocessing, retrieval, generation)
- `requirements.txt` — exact Python deps
- `PROJECT_README.md` — this file with run instructions
- `report_acl.tex` and `refs.bib` — writeup and references
- `leaderboard_queries.json` and example query files
- `data/chunks_v2.jsonl` and `data/faiss_v2/index.faiss` (or instructions to rebuild them)
- `system_outputs/` — example outputs submitted for Canvas
- small sample data for quick smoke tests (if possible)
