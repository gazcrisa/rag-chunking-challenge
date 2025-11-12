# Smart Chunking for RAG: Take-Home Challenge

## Overview

This project implements and evaluates multiple document chunking strategies for RAG.
The goal is to preserve semantic and contextual coherence in each chunk, improving retrieval accuracy and reducing information loss compared to naive fixed-size splitting.

The dataset used for this challenge is Tesla’s 2024 10-K filing, which is long, hierarchical, and context-dense.

### How to Run

#### Prerequisites

- Python **3.11+**
- **Poetry** (for dependency management)
- **Docker** and **Docker Compose** (to run Postgres + API containers)
- OpenAI API key (required for embedding and retrieval)

#### Setup

```bash
poetry install
poetry shell
```

#### Environment Configuration

Before running the scripts, you’ll need to provide your OpenAI API key.

1. Copy the example environment file:
   ```bash
   cp .env.example .env

2. Open `.env` and add your OpenAI API key

#### Start the Database
Using Docker, you can spin up the Postgres database (with pgvector extension) using:

```
docker-compose up -d
```

This will start a Postgres instance configured with the pgvector extension (for vector storage and similarity search), along with a containerized FastAPI service for querying embeddings

#### Step 1 — Seed the Database

Run the seeding script with different chunking strategies to generate and embed chunks of the Tesla 10-K document.

#### 1. Baseline: fixed-size chunking
- `STRATEGY=FIXED poetry run python -m src.services.seed_embeddings`

#### 2. Sentence-based chunking (with overlap)
- `STRATEGY=SENTENCE poetry run python -m src.services.seed_embeddings`

#### 3. Semantic chunking (embedding-aware segmentation)
- `STRATEGY=SEMANTIC poetry run python -m src.services.seed_embeddings`

Each command will:

- Read `data/processed/tesla_10k_2024.txt`
- Chunk the document using the specified strategy
- Embed each chunk using OpenAI's `text-embedding-3-small`
- Insert results into the `document_chunks` table (with pgvector column)

#### Step 2 — Run Evaluations
Once the database is seeded, you can run the retrieval evaluation using the same strategy variable:

```
# Evaluate each chunking method independently
STRATEGY=FIXED poetry run python -m src.evaluation.run_baseline
STRATEGY=SENTENCE poetry run python -m src.evaluation.run_baseline
STRATEGY=SEMANTIC poetry run python -m src.evaluation.run_baseline
```

Each run will:

- Query `document_chunks` filtered by the selected strategy
- Embed evaluation queries from `src/queries.json`
- Compute cosine similarity between query and chunk embeddings
- Save results to `evaluation/results/*.json`


#### Query the API via the browser or CURL

If running via Docker, the API will be available automatically (containerized alongside the Postgres service).
By default, it listens on http://localhost:8000

Once running, you can query the vector database in your browser or with curl, for example:

```
http://localhost:8000/search?q=What are the main components of Tesla’s electric powertrain?

```

This endpoint performs semantic retrieval against the embedded chunks (using cosine similarity), returning the most relevant passages from the document.

You can optionally specify a chunking strategy via the strategy query parameter:

```
http://localhost:8000/search?q=What are the main components of Tesla’s electric powertrain?&strategy=SEMANTIC

```

### Implemented Chunking Strategies

#### 1. Fixed Chunker (Baseline)

File: `src/chunkers/fixed.py`

Splits text into equal-sized word windows (e.g., 500 words).
No awareness of sentence boundaries or meaning — serves as a control for performance and retrieval comparison.

#### 2. Sentence-Based Chunker (Sliding Context Window)

File: `src/chunkers/sentence.py`

Splits text by sentence boundaries using NLTK’s `sent_tokenize`, grouping sentences into chunks up to a target *word count* (e.g., ~500 words per chunk).  
To preserve continuity and avoid context loss between boundaries, it uses a **sliding window** — overlapping the last few sentences between consecutive chunks.

**Example**

```
Chunk 1: Sentences 1–10
Chunk 2: Sentences 9–18  ← overlaps 9–10 for context continuity
```

#### 3. Semantic Chunker (Embedding-Aware)

File: `src/chunkers/semantic.py`

The Semantic Chunker groups adjacent sentences into coherent segments based on **semantic similarity** rather than just size.  
It uses the `all-MiniLM-L6-v2` model from `sentence-transformers` to generate sentence embeddings and measures relatedness via cosine similarity.

A new chunk boundary is created when either:
- The cosine similarity between consecutive sentences falls below a threshold (default: **0.75**), or  
- Adding the next sentence would exceed the target word budget (e.g., ~500 words)

**Example**

```
if cosine_similarity(prev, current) < threshold:
    new_chunk()
```

Each chunker includes strategies to retain document context:

| Strategy           | Context Method                                   | Example                                         |
| ------------------ | ------------------------------------------------ | ----------------------------------------------- |
| **Fixed**          | None                                             | Pure baseline                                   |
| **Sentence-based** | Sliding overlap (previous 1–2 sentences)         | “...As discussed above” references remain valid |
| **Semantic**       | Metadata injection (`section`, `similarity_gap`) | Allows traceability and section-level retrieval |

### Evaluation Methodology

Goal: demonstrate improvement in retrieval quality from fixed → structured → semantic chunking.

**Dataset**

- `data/processed/tesla_10k_2024.txt`: a long, hierarchical SEC filing (150+ pages) chosen for its complex structure and mix of numerical and narrative text.

**Queries**

- Stored in `src/queries.json`.
- Each query corresponds to a factual question (e.g., “How does Tesla describe supply chain risks?”).


#### Evaluation Process

- Chunk the same document using all three strategies.
- Embed chunks using a standard embedding model
- Embed each query and compute cosine similarity with all chunks.

Results are saved under `evaluation/results/`


### Actual Evaluation

The top retrieved chunks were analyzed for factual and contextual relevance.

| Strategy     | # of Chunks | Avg Words/Chunk | Notes                                        |
| ------------- | ------------ | ---------------- | -------------------------------------------- |
| **Fixed**     | 293          | 499              | Arbitrary segmentation, weak coherence       |
| **Sentence**  | 403          | 433              | Preserves sentence structure and syntax      |
| **Semantic**  | 3,454        | 42               | Fine-grained segmentation; higher precision in meaning alignment |

<sub>All embeddings were computed using OpenAI’s `text-embedding-3-small` model for consistent retrieval performance.</sub>

---

#### How *Avg Words/Chunk* Was Calculated

Average word count per chunk was derived directly from the Postgres database using the following SQL query:

```sql
SELECT
    strategy,
    COUNT(*) AS total_chunks,
    ROUND(AVG(array_length(regexp_split_to_array(chunk_text, '\s+'), 1))) AS avg_words_per_chunk
FROM document_chunks
WHERE chunk_text IS NOT NULL AND length(trim(chunk_text)) > 0
GROUP BY strategy
ORDER BY strategy;
```

#### Observations & Adjustments

During evaluation, the semantic strategy initially produced 3,454 chunks averaging only ~42 words each, which was unexpectedly small compared to the other strategies.


I think this reveals an overly strict similarity threshold (`0.75`), which is causing frequent breaks between sentences that were still semantically related.  
In other words, the chunker treated minor topic shifts as hard boundaries instead of grouping coherent paragraphs.

To correct this, the threshold was relaxed to **0.55** in hopes to yield chunk sizes closer to human-meaningful sections (200–500 words) while still detecting true semantic shifts.



### Discussion & Tradeoffs

- Fixed chunking is simple but unsuitable for complex text because it fragments context.

- Semantic chunking most closely approximates human segmentation and is ideal for RAG, but computationally heavier.



### Future Improvements

- **Tune semantic thresholds adaptively:**  
  Instead of using a static similarity threshold (e.g., 0.55), dynamically adjust it based on the document’s internal variance or embedding density.  
  This prevents over-fragmentation in narrative text and under-segmentation in technical or tabular sections.

- **Add hierarchical (heading-aware) chunking:**  
  Use HTML or markdown structure (e.g., `<h1>`, `<h2>`) to anchor chunks at logical section boundaries, preserving context like “Risk Factors” or “Financial Overview.”

- **Experiment with transformer token counts:**  
  Replace approximate word counts with tokenizer-based budgets (`tiktoken` or model-native tokenizers) for more consistent chunk sizing across languages and models.

- **Combine semantic and sliding window methods:**  
  Introduce overlap between semantically-defined chunks to retain cross-boundary continuity — useful for RAG systems that rely on long-context reasoning.

- **Integrate automatic evaluation metrics:**  
  Implement `Precision@k` and `Recall@k` calculations directly in the evaluation script with manual or annotated ground truth for more reproducible scoring.

- **Add automated unit tests and benchmarking:**  
  Include tests to validate chunk size consistency, context overlap integrity, and similarity-based boundary placement across strategies and datasets.

