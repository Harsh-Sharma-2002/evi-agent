from typing import List, Optional
import gc

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from agent.state import AgentState


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
# Local LLM (FLAN-T5, NO pipeline)
# =================================================

_LLM_MODEL_NAME = "google/flan-t5-base"
_llm_model = None
_llm_tokenizer = None


def _get_llm():
    """
    Lazy-load FLAN-T5 model and tokenizer.
    """
    global _llm_model, _llm_tokenizer
    if _llm_model is None or _llm_tokenizer is None:
        _llm_tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_NAME)
        _llm_model = AutoModelForSeq2SeqLM.from_pretrained(_LLM_MODEL_NAME)
        _llm_model.eval()
    return _llm_model, _llm_tokenizer


def unload_llm():
    """
    Explicitly unload the local LLM to free RAM.
    """
    global _llm_model, _llm_tokenizer
    _llm_model = None
    _llm_tokenizer = None
    gc.collect()


def call_llm(state: AgentState, prompt: str, max_new_tokens: int = 256, node_name: Optional[str] = None,) -> str:
    """
    Local FLAN-T5 inference (CPU, deterministic).
    """
    model, tokenizer = _get_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return text.strip()
