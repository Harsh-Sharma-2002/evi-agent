from typing import TypedDict, List, Dict, Any, Optional, Literal
import numpy as np


class AgentState(TypedDict):
    """
    Global mutable state passed between LangGraph nodes.
    This is the ONLY place where agent state is defined.
    """
    # Query & embeddings
    query: str
    query_embedding: np.ndarray

    # Control / loop bookkeeping
    iteration: int
    api_calls: int
    decision: Optional[Literal["STOP", "FETCH_MORE"]]

    # Retrieval results
    documents: List[Dict[str, Any]]          # raw PubMed docs
    doc_chunks_map: Dict[str, List[Dict[str, Any]]]  # pmid -> chunks

    # Chunk-level retrieval
    anchor_chunks: List[Dict[str, Any]]


    # Scoring outputs
    retrieval_score: float
    num_docs: int
    doc_scores: Dict[str, float]
    confident: bool

    # Context & synthesis
    expanded_context: Optional[Dict[str, List[Dict[str, Any]]]]
    final_answer: Optional[str]

    # Cache-related
    cache_hit: bool
    cache_payload: Optional[Dict[str, Any]]
