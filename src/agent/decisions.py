from  .state import AgentState
from typing import List


STOP_SCORE_THRESHOLD = 0.6 # Min confidence to stop
MIN_CACHE_SCORE = 0.6 # Min quality to reuse later ie enter in cache  

MAX_ITERATIONS = 4 # prevent excess looping
MAX_API_CALLS = 5 # API call control else it would keep calling and exceed rate limit

STAGNATION_TOLERATION = 0.02 # If after calls no better result then stop and say I dont know man


def is_stagnating(scores: List[float],tol: float = STAGNATION_TOLERATION) -> bool:
    """
    - Detect whether retrieval score is no longer improving.
    - Used to stop retrieval when additional fetches
      are unlikely to yield better evidence.
    """
    if len(scores) < 2:
        return False
    
    return abs(scores[-1] - scores[-2]) < tol




def should_cache(state: AgentState) -> bool:
    """
    - Decide whether the final result should be stored
    in the query cache.
    - Only confident, high-quality, non-forced stops
    are admitted.
    """
    return ( state["decision"] == "STOP" and state["stop_reason"] == "score_threshold_met" and state["api_calls"] > 0 and state["retrieval_score"] >= MIN_CACHE_SCORE and state["num_docs"] >= 1 )