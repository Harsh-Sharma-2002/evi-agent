from typing import TypedDict, List, Dict, Any, Optional, Literal
import numpy as np


class AgentState(TypedDict):
    """
    Global mutable state passed between LangGraph nodes.

    This state is:
    - Per-query (cleared every run)
    - The single source of truth for the agent
    - Mutated in-place by graph nodes

    Cross-query memory lives ONLY in the VectorCache.
    """

    query: str
    query_embedding: np.ndarray

    # Control / Loop bookkeeping
    iteration: int                    # number of retrieval loops completed
    api_calls: int                    # number of external API calls made
    decision: Optional[Literal["STOP", "FETCH_MORE"]]
    stop_reason: Optional[str]        # e.g. cache_hit, score_threshold_met, stagnation

    # Track retrieval score evolution for stagnation detection
    prev_retrieval_scores: List[float]


    # Tier 3: Raw retrieval results (PubMed)
    documents: List[Dict[str, Any]]   # raw fetched documents (abstracts, metadata)
    doc_chunks_map: Dict[str, List[Dict[str, Any]]]   # pmid -> ordered list of chunks (for context expansion)

    # Tier 2: Chunk-level evidence
    # Selected high similarity chunks used for scoring and context window expansion
    anchor_chunks: List[Dict[str, Any]]

 
    # Scoring outputs (derived from anchor_chunks)
    retrieval_score: float            # overall quality * diversity score
    num_docs: int                     # number of distinct documents represented
    doc_scores: Dict[str, float]      # "pmid" : "best chunk score"
    confident: bool                   # hard confidence gate (min docs)


    # Context & answer (used only after STOP)
    expanded_context: Optional[Dict[str, List[Dict[str, Any]]]]
    final_answer: Optional[str]

    # Tier 1: Query cache interaction
    cache_hit: bool                   # whether query cache was used
    cache_payload: Optional[Dict[str, Any]]  # reusable payload (answer, evidence, etc.)
    evidence_exhausted: bool

    # Diagnostics (read-only, per run)
    query_cache_size: int
    chunk_store_size: int
    num_anchor_chunks: int
    
    # Logging variables
    query_cache_hits: int        # Tier-1 reuse events
    chunk_store_hits: int        # Tier-2 reuse events
    api_calls_saved: int      
