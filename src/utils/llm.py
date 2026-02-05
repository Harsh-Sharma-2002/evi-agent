from typing import List
import os
from dotenv import load_dotenv
import requests
import numpy as np
from ..agent.state import AgentState

load_dotenv()

# Embedding model (LOCAL)


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


def embed(text: str) -> List[float]:
    """
    Embed text into a semantic vector.

    Contract:
    - returns list[float]
    - same embedding space for queries + chunks
    - normalized for cosine similarity
    """
    if not text:
        dim = _get_embed_model().get_sentence_embedding_dimension()
        return [0.0] * dim

    model = _get_embed_model()
    vec = model.encode(
        text,
        normalize_embeddings=True,
    )
    return vec.tolist()


# LLM (HF Inference API – REMOTE)

_HF_LLM_MODEL = "google/flan-t5-base"
_HF_API_URL = f"https://api-inference.huggingface.co/models/{_HF_LLM_MODEL}"
_HF_TOKEN = os.getenv("HF_TOKEN")


def call_llm(state: AgentState, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Call Hugging Face Inference API.

    This function:
    - consumes AgentState (for future extensions / logging)
    - sends a fully-built prompt
    - returns raw model output text

    Prompt construction MUST happen elsewhere.
    """

    if not _HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN not set. Add it to your environment or .env file."
        )

    headers = {
        "Authorization": f"Bearer {_HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
        },
    }

    response = requests.post(
        _HF_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    # HF API response formats vary
    if isinstance(data, list) and data:
        text = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        text = data.get("generated_text", "")
    else:
        text = ""

    return text.strip()