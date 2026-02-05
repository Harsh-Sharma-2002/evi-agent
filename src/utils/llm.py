from typing import List
import os
from dotenv import load_dotenv
import numpy as np

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from agent.state import AgentState

load_dotenv()

# Embedding model (LOCAL)


_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model


def embed(text: str) -> List[float]:
    if not text:
        dim = _get_embed_model().get_sentence_embedding_dimension()
        return [0.0] * dim

    model = _get_embed_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


# =================================================
# LLM (HF Inference Router – CORRECT WAY)
# =================================================

HF_TOKEN = os.environ.get("_HF_TOKEN") 
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY / HF_TOKEN not found in environment")

# This client automatically routes to the correct provider
_llm_client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN,
)


def call_llm(
    state: AgentState,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Call Mistral-7B-Instruct via Hugging Face Inference Router
    using CHAT COMPLETION (required for this model).
    """

    response = _llm_client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )

    # HF returns structured chat output
    return response.choices[0].message.content.strip()

