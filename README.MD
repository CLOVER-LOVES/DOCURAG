# DocuChat - RAG Q&A

Production-leaning document Q&A with **hybrid retrieval** (dense + BM25), grounded citations, evaluation utilities, tests, and Docker deployment.

## Features

- PDF ingestion with page-level metadata
- Recursive chunking with configurable size/overlap
- Dense retrieval via ChromaDB
- Optional BM25 keyword retrieval via `EnsembleRetriever`
- Grounded OpenAI chat answers with source attribution
- Per-query latency and source metrics
- Offline evaluation scaffold and RAG metrics
- Pytest test suite
- Dockerfile and GitHub Actions CI

## Setup

```bash
cd DOCUCHAT
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
python -m pytest -q
```

## Docker

```bash
docker build -t docuchat .
docker run -p 8501:8501 --env-file .env docuchat
```

## Evaluation

Process documents first, then run:

```bash
python scripts/run_eval.py --dataset eval/your_dataset.json
```

Dataset item format:

```json
[
  {
    "question": "What is the invoice number?",
    "relevant_sources": ["invoice.pdf"],
    "expected_keywords": ["A123"]
  }
]
```

## Architecture

```text
app.py
├── docurag/ingestion.py      # PDF loading + chunking
├── docurag/retrieval.py      # Dense / hybrid retrieval
├── docurag/qa.py             # Prompt + QA chain
├── docurag/evaluation.py     # RAG metrics
├── docurag/config.py         # Shared config
├── tests/
├── scripts/
└── Dockerfile
```

## Notes

- Scanned/image-only PDFs need OCR before ingestion.
- OpenAI credits are required for embeddings and chat.
- Use **Clear Database** before reprocessing the same corpus to avoid duplicate chunks.
- `.env` is ignored by git; rotate any key that was ever committed or shared.
```