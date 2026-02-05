import requests
import numpy as np
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

from agent.state import AgentState
from retrieval.cache import VectorCache
from utils.llm import embed   #  embed imported, NOT passed


# -------------------------------------------------
# Internal helpers
# -------------------------------------------------

def _extract_year(article: ET.Element) -> Optional[int]:
    year_text = article.findtext(".//PubDate/Year")
    if year_text and year_text.isdigit():
        return int(year_text)
    return None


def _extract_abstract_and_conclusion(
    article: ET.Element,
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract abstract and conclusion from PubMed XML.
    Conclusion is detected via label containing 'conclu' (case-insensitive).
    """
    nodes = article.findall(".//Abstract/AbstractText")
    if not nodes:
        return None, None

    parts: List[str] = []
    conclusion: Optional[str] = None

    for n in nodes:
        txt = (n.text or "").strip()
        if txt:
            parts.append(txt)

        label = (n.attrib.get("Label") or n.attrib.get("NlmCategory") or "").lower()
        if "conclu" in label and txt:
            conclusion = txt

    abstract = " ".join(parts).strip() if parts else None
    return abstract, conclusion


def _fetch_pubmed_docs(
    query: str,
    retmax: int,
    offset: int,
) -> List[Dict[str, Any]]:
    """
    Fetch PubMed docs (abstract + optional conclusion).
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "retstart": offset,
    }

    pmids = requests.get(search_url, params=search_params, timeout=10) \
                    .json()["esearchresult"]["idlist"]

    if not pmids:
        return []

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }

    root = ET.fromstring(
        requests.get(fetch_url, params=fetch_params, timeout=10).text
    )

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
    retmax: int = 5,
) -> AgentState:
    """
    Tier 3 ingestion node.

    - Fetches PubMed docs
    - Creates up to two chunks per paper:
        * chunk_index 0: abstract
        * chunk_index 1: conclusion (if present)
    - Embeds chunks using utils.llm.embed
    - Updates chunk store and AgentState
    """

    offset = state["iteration"] * retmax
    docs = _fetch_pubmed_docs(state["query"], retmax, offset)

    # API call counts even if empty
    state["api_calls"] += 1

    if not docs:
        return state

    chunk_texts: List[str] = []
    chunk_embeddings: List[np.ndarray] = []
    chunk_metadatas: List[Dict[str, Any]] = []

    for doc in docs:
        pmid = doc["pmid"]
        year = doc["year"]

        state["doc_chunks_map"].setdefault(pmid, [])

        # ---- abstract chunk ----
        abs_text = doc["abstract"]
        abs_emb = np.asarray(embed(abs_text))

        chunk_texts.append(abs_text)
        chunk_embeddings.append(abs_emb)
        chunk_metadatas.append({
            "pmid": pmid,
            "year": year,
            "chunk_index": 0,
            "section": "abstract",
        })

        state["doc_chunks_map"][pmid].append({
            "text": abs_text,
            "chunk_index": 0,
            "metadata": {
                "pmid": pmid,
                "year": year,
                "chunk_index": 0,
                "section": "abstract",
            },
        })

        # ---- conclusion chunk (optional) ----
        concl = doc.get("conclusion")
        if concl:
            concl_emb = np.asarray(embed(concl))

            chunk_texts.append(concl)
            chunk_embeddings.append(concl_emb)
            chunk_metadatas.append({
                "pmid": pmid,
                "year": year,
                "chunk_index": 1,
                "section": "conclusion",
            })

            state["doc_chunks_map"][pmid].append({
                "text": concl,
                "chunk_index": 1,
                "metadata": {
                    "pmid": pmid,
                    "year": year,
                    "chunk_index": 1,
                    "section": "conclusion",
                },
            })

    cache.add_chunks(chunk_texts, chunk_embeddings, chunk_metadatas)
    state["documents"].extend(docs)

    return state
