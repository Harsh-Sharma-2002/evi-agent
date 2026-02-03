from typing import Dict, List, Any
from collections import defaultdict
import math
import time

from agent.state import AgentState




RECENCY_HALFLIFE_YEARS = 8
MIN_DOCS_FOR_CONFIDENCE = 2
MAX_SCORE = 1.0




def recency_score(year: int | None, current_year: int | None = None) -> float:
    """
    Exponential decay based on publication year.
    """
    if year is None:
        return 0.5

    if current_year is None:
        current_year = time.gmtime().tm_year

    age = max(0, current_year - year)
    return math.exp(-age / RECENCY_HALFLIFE_YEARS)


def similarity_score(similarity: float) -> float:
    """
    Clamp similarity to [0, 1].
    """
    return max(0.0, min(1.0, similarity))


def doc_diversity_bonus(num_docs: int) -> float:
    """
    Saturating reward for multiple independent documents.
    NOTE: This rewards independence, not agreement.
    """
    return 1.0 - math.exp(-num_docs / 3)




def score_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Consumes anchor_chunks from state
    - Computes document-level and retrieval scores
    - Updates state in-place
    """

    anchor_chunks = state["anchor_chunks"]


    if not anchor_chunks:
        state["retrieval_score"] = 0.0
        state["num_docs"] = 0
        state["doc_scores"] = {}
        state["confident"] = False
        state["prev_retrieval_scores"].append(0.0)
        return state


    doc_scores_accumulator: Dict[str, List[float]] = defaultdict(list)

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

        doc_scores_accumulator[pmid].append(chunk_score)

    # Collapse chunks → per-document score
    doc_scores: Dict[str, float] = {
        pmid: max(scores)
        for pmid, scores in doc_scores_accumulator.items()
    }

    num_docs = len(doc_scores)

    if num_docs == 0:
        state["retrieval_score"] = 0.0
        state["num_docs"] = 0
        state["doc_scores"] = {}
        state["confident"] = False
        state["prev_retrieval_scores"].append(0.0)
        return state

  

    avg_doc_score = sum(doc_scores.values()) / num_docs
    diversity = doc_diversity_bonus(num_docs)

    retrieval_score = min(
        MAX_SCORE,
        avg_doc_score * diversity
    )

    confident = num_docs >= MIN_DOCS_FOR_CONFIDENCE

    # Update state

    state["retrieval_score"] = retrieval_score
    state["num_docs"] = num_docs
    state["doc_scores"] = doc_scores
    state["confident"] = confident

    # Track score evolution for stagnation detection
    state["prev_retrieval_scores"].append(retrieval_score)

    return state


# We clear the agent state for next query in the run_agent() function in main.py or like run.py
