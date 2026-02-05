import requests
import numpy as np
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

from agent.state import AgentState
from retrieval.cache import VectorCache


# -------------------------------------------------
# Internal helper: fetch PubMed docs (abstract + conclusion)
# -------------------------------------------------

def _extract_year(article: ET.Element) -> Optional[int]:
    year_text = article.findtext(".//PubDate/Year")
    if year_text and year_text.isdigit():
        return int(year_text)
    return None


def _extract_abstract_and_conclusion(article: ET.Element) -> tuple[Optional[str], Optional[str]]:
    """
    PubMed XML is messy. AbstractText nodes may:
    - be multiple segments
    - include Label attributes (e.g., CONCLUSIONS)
    - use mixed case

    Strategy:
    - Abstract: concatenate ALL AbstractText segments (best-effort)
    - Conclusion: prefer a labeled segment containing "conclu" (case-insensitive),
      else None.
    """
    nodes = article.findall(".//Abstract/AbstractText")
    if not nodes:
        return None, None

    # Full abstract = concat all segments
    parts: List[str] = []
    labeled_conclusion: Optional[str] = None

    for n in nodes:
        txt = (n.text or "").strip()
        if txt:
            parts.append(txt)

        label = n.attrib.get("Label") or n.attrib.get("NlmCategory") or ""
        label_lc = label.lower()
        if ("conclu" in label_lc) and txt:
            # e.g. "CONCLUSION", "CONCLUSIONS"
            labeled_conclusion = txt

    abstract = " ".join(parts).strip() if parts else None
    conclusion = labeled_conclusion.strip() if labeled_conclusion else None
    return abstract, conclusion


def _fetch_pubmed_docs(
    query: str,
    retmax: int = 5,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Minimal PubMed search + fetch.

    Returns list of:
    {
      "pmid": str,
      "year": int | None,
      "abstract": str,
      "conclusion": str | None
    }
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "retstart": offset,
    }

    search_resp = requests.get(search_url, params=search_params, timeout=10)
    search_resp.raise_for_status()
    pmids = search_resp.json()["esearchresult"]["idlist"]

    if not pmids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=10)
    fetch_resp.raise_for_status()

    root = ET.fromstring(fetch_resp.text)

    docs: List[Dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        if not pmid:
            continue

        year = _extract_year(article)
        abstract, conclusion = _extract_abstract_and_conclusion(article)

        if not abstract:
            continue

        docs.append(
            {
                "pmid": pmid,
                "year": year,
                "abstract": abstract,
                "conclusion": conclusion,
            }
        )

    return docs


# -------------------------------------------------
# Tier 3: PubMed fetch node
# -------------------------------------------------

def pubmed_fetch_node(
    state: AgentState,
    cache: VectorCache,
    embed,  # ✅ injected to ensure SAME embedding policy as query embedding
    retmax: int = 5,
) -> AgentState:
    """
    LangGraph node (Tier 3).

    Fetches new PubMed docs, makes up to TWO chunks per paper:
      - chunk_index 0: abstract
      - chunk_index 1: conclusion (if present)

    Updates:
      - state["documents"]
      - state["doc_chunks_map"]  (for context expansion)
      - cache chunk store        (for Tier 2 reuse)
      - state["api_calls"]

    Does NOT:
      - score
      - decide stop/fetch
    """

    offset = state["iteration"] * retmax
    docs = _fetch_pubmed_docs(state["query"], retmax=retmax, offset=offset)

    # Count API call even if empty result (still cost)
    state["api_calls"] += 1

    if not docs:
        return state

    chunk_texts: List[str] = []
    chunk_embeddings: List[np.ndarray] = []
    chunk_metadatas: List[Dict[str, Any]] = []

    for doc in docs:
        pmid = doc["pmid"]
        year = doc["year"]

        # Ensure doc entry exists for context expansion
        state["doc_chunks_map"].setdefault(pmid, [])

        # -------------------------
        # Chunk 0: abstract
        # -------------------------
        abs_text = doc["abstract"]
        abs_emb = np.asarray(embed(abs_text))

        chunk_texts.append(abs_text)
        chunk_embeddings.append(abs_emb)
        chunk_metadatas.append(
            {
                "pmid": pmid,
                "year": year,
                "chunk_index": 0,
                "section": "abstract",
            }
        )

        state["doc_chunks_map"][pmid].append(
            {
                "text": abs_text,
                "chunk_index": 0,
                "metadata": {  # ✅ include chunk_index inside metadata (MANDATORY)
                    "pmid": pmid,
                    "year": year,
                    "chunk_index": 0,
                    "section": "abstract",
                },
            }
        )

        # -------------------------
        # Chunk 1: conclusion (optional)
        # -------------------------
        concl = doc.get("conclusion")
        if concl:
            concl_text = concl
            concl_emb = np.asarray(embed(concl_text))

            chunk_texts.append(concl_text)
            chunk_embeddings.append(concl_emb)
            chunk_metadatas.append(
                {
                    "pmid": pmid,
                    "year": year,
                    "chunk_index": 1,
                    "section": "conclusion",
                }
            )

            state["doc_chunks_map"][pmid].append(
                {
                    "text": concl_text,
                    "chunk_index": 1,
                    "metadata": {  # ✅ include chunk_index inside metadata (MANDATORY)
                        "pmid": pmid,
                        "year": year,
                        "chunk_index": 1,
                        "section": "conclusion",
                    },
                }
            )

    # Add to Tier 2 chunk store (cross-query memory)
    cache.add_chunks(
        chunks=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas,
    )

    # Keep raw docs (optional debugging / provenance)
    state["documents"].extend(docs)

    return state
