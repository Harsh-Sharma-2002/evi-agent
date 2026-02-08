# src/scoring/helpers.py

import math
import time
from typing import Dict, List, Any


# ------------------------------
# Hyperparameters
# ------------------------------
RECENCY_HALFLIFE_YEARS = 8
MIN_DOCS_FOR_CONFIDENCE = 2
MAX_SCORE = 1.0


def recency_score(year: int | None, current_year: int | None = None) -> float:
    if year is None:
        return 0.5

    if current_year is None:
        current_year = time.gmtime().tm_year

    age = max(0, current_year - year)
    return math.exp(-age / RECENCY_HALFLIFE_YEARS)


def similarity_score(similarity: float) -> float:
    return max(0.0, min(1.0, similarity))


def doc_diversity_bonus(num_docs: int) -> float:
    return 1.0 - math.exp(-num_docs / 3)


def compute_retrieval_score(anchor_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    PURE function.

    Takes anchor_chunks and returns:
    {
        retrieval_score: float,
        num_docs: int,
        doc_scores: Dict[str, float],
        confident: bool
    }
    """
    if not anchor_chunks:
        return {
            "retrieval_score": 0.0,
            "num_docs": 0,
            "doc_scores": {},
            "confident": False,
        }

    doc_scores_accumulator: Dict[str, List[float]] = {}

    for chunk in anchor_chunks:
        meta = chunk.get("metadata", {})
        pmid = meta.get("pmid")
        if pmid is None:
            continue

        sim = similarity_score(chunk.get("similarity", 0.0))
        year = meta.get("year")

        chunk_score = (
            0.7 * sim +
            0.3 * recency_score(year)
        )

        doc_scores_accumulator.setdefault(pmid, []).append(chunk_score)

    if not doc_scores_accumulator:
        return {
            "retrieval_score": 0.0,
            "num_docs": 0,
            "doc_scores": {},
            "confident": False,
        }

    doc_scores = {
        pmid: max(scores)
        for pmid, scores in doc_scores_accumulator.items()
    }

    num_docs = len(doc_scores)
    avg_doc_score = sum(doc_scores.values()) / num_docs
    diversity = doc_diversity_bonus(num_docs)

    retrieval_score = min(
        MAX_SCORE,
        avg_doc_score * diversity
    )

    confident = num_docs >= MIN_DOCS_FOR_CONFIDENCE

    return {
        "retrieval_score": retrieval_score,
        "num_docs": num_docs,
        "doc_scores": doc_scores,
        "confident": confident,
    }





