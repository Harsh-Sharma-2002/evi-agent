from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    """
    Lazy-load the embedding model exactly once.
    """
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        return _embed_model
    
def embed(text:str) -> List[float]:
    """
    Embed text into semantic vector.

    Contract:
    - returns List[float]
    - same embedding space for query +chunks
    - normalised for cosine similarity  
    """
    if not text:
        dim = _get_embed_model().get_sentence_embedding_dimension()
        return [0.0] * dim
    
    model = _get_embed_model()
    vec = model.encode(
        text,
        normalize_embeddings=True
    )

    return vec.tolist()