from typing import Dict, List, Any
from collections import defaultdict
import math
import time

RECENCY_HALFLIFE_YEARS = 8
MIN_DOCS_FOR_CONFIDENCE = 2
MAX_SCORE = 1.0

def recency_score(year: int | None, current_year: int | None) -> float:
    """
    Exponential decay based on publication year.
    """
    if year is None:
        return 0.5
    
    if current_year is None:
        current_year = time.gmtime().tm_year

    age = max(0,current_year - year)

    return math.exp(-age/RECENCY_HALFLIFE_YEARS)

def similarity_score(similarity: float) -> float:
    """
    To make sure the the range of score is in 0 to 1
    """

def doc_diversity_bonus(num_docs: int) -> float: #“Number of independently sourced documents that contain at least one high-similarity chunk.” but they can contradich other so fix that logic
    """
    Saturating reward for muktiple indpendent documents.
    """

    return 1 - math.exp(-num_docs/3) # Random half life choice since if in 3 then must be fact and no point exploding score using linear hence saturating

def score_chunks(anchor_chunks: List[Dict[str,Any]]) -> Dict[str,Any]: # return datatype is for fast protytping need to define full Typed dict for this.
    """
    Aggregate chunk-level signals into document-level scores.

    Returns:
        {
          "doc_scores": {pmid: float},
          "num_docs": int
        }
    """
    doc_score = defaultdict(list)

    for chunk in anchor_chunks:
        meta = chunk["metadata"]
        pmid = meta.get("pmid")
        if pmid is None:
            continue
        sim = similarity_score(chunk.get("similarity",0.0))

        year  =  meta.get(("year"))

        score = (0.7 * sim) + (0.3 * recency_score(year))
        doc_score[pmid].append(score)

        final_doc_scores = {
            pmid : max(scores)for pmid,scores in doc_score.items()
        }

        return {
            "doc_scores": final_doc_scores,
            "num_docs": len(final_doc_scores)
        }
    

def score_retieval(anchor_chunks: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Compute overall retrieval score used by the agent to decide STOP / FETCH_MORE.

    Returns:
        {
        "retrieval_score": float,
        "num_docs": int,
        "doc_scores": {pmid: float},
        "confident": bool
        }
     """
    
    if not anchor_chunks:
        return {
            "retrieval_score": 0.0,
            "num_docs": 0,
            "doc_scores": {},
            "confident": False,
        }
    
    doc_result = score_chunks(anchor_chunks)
    doc_scores = doc_result["doc_scores"]
    num_docs = doc_result["nums_docs"]

    if num_docs == 0:
        return {
            "retrieval_score": 0.0,
            "num_docs": 0,
            "doc_scores": {},
            "confident": False,
        }
    
    avg_doc_score = sum(doc_scores.values())/ num_docs
    diversity = doc_diversity_bonus(num_docs)

    retrieval_score = min(MAX_SCORE,avg_doc_score*diversity)

    confidence = num_docs >= MIN_DOCS_FOR_CONFIDENCE

    return {
        "retrieval_score": retrieval_score,
        "num_docs": num_docs,
        "doc_scores": doc_scores,
        "confident": confident,
    }