# RAG Chunking Challenge

Smart document chunking strategies for Retrieval-Augmented Generation systems.

## Overview

This project implements and evaluates multiple document chunking strategies to improve retrieval quality in RAG systems, addressing issues like:
- Lost context from mid-sentence splits
- Broken cross-references
- Poor semantic boundaries
- Missing document structure metadata

## Project Structure
```
rag-chunking-challenge/
├── src/                    # Core application code
│   ├── chunkers/          # Chunking strategies
│   ├── context/           # Context preservation methods
│   ├── services/          # Supporting services
│   └── evaluation/        # Evaluation logic
├── evaluation/            # Test queries and results
├── data/                  # Sample documents
├── notebooks/             # Analysis notebooks
└── tests/                 # Unit tests
```

## Setup

1. Install dependencies:
```bash
poetry install
```

2. Run the FastAPI server:
```bash
poetry run uvicorn src.main:app --reload
```

3. Run evaluation:
```bash
poetry run python evaluation/run_evaluation.py --strategy semantic
```

## Chunking Strategies

### 1. Naive (Baseline)
Fixed-size chunks with no overlap or context preservation.

### 2. Semantic
Uses embeddings to detect topic boundaries and create semantically coherent chunks.

### 3. Structural
Respects document structure (headings, sections) when chunking.

## Evaluation

See `evaluation/` directory for test queries and results.

## License

MIT
