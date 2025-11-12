from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from src.database import db_session
from src.models import ChunkStrategy
from src.services.retrieval import RetrievalService

QUERIES_PATH = Path("src/queries.json")
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------
# Evaluation Logic
# --------------------------------------------------------------------
def run_evaluation(strategy: ChunkStrategy) -> dict:
    """Run retrieval evaluation for the given chunking strategy."""
    logger.info(f"Running retrieval evaluation (strategy={strategy.value})...")

    with db_session() as session:
        service = RetrievalService(session)

        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            queries = json.load(f)["queries"]

        results = {}
        for q in queries:
            qid = q["id"]
            question = q["question"]

            logger.info(f" → Query {qid}: {question}")
            response = service.search(question, top_k=3, strategy=strategy)
            results[qid] = {
                "question": question,
                "results": response,
            }

        out_path = RESULTS_DIR / f"{strategy.value.lower()}_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation complete! Results saved to {out_path}")

    return results


# --------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Read strategy from environment variable, defaulting to FIXED
    strategy_env = os.getenv("STRATEGY", "FIXED").upper()
    chosen_strategy = ChunkStrategy.__members__.get(strategy_env, ChunkStrategy.FIXED)

    run_evaluation(chosen_strategy)
