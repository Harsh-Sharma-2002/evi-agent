from typing import Dict, List, Any
from collections import defaultdict
import math
import time

from agent.state import AgentState



# Scoring hyperparameters
RECENCY_HALFLIFE_YEARS = 8          # publication age decay
MIN_DOCS_FOR_CONFIDENCE = 2         # hard confidence gate
MAX_SCORE = 1.0                    # upper bound for retrieval score


# Helper scoring functions
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
    """
    return 1.0 - math.exp(-num_docs / 3)




