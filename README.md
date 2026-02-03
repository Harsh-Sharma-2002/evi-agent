### Caching Logic


cache.py — Hybrid Vector Cache Design & Rationale
Purpose of this file

cache.py implements the memory subsystem for an agentic retrieval system that operates over scientific literature (e.g., PubMed abstracts).

Its job is not to reason, score, or decide.

Its job is to store and retrieve semantic memory safely and efficiently, so that the agent can:

avoid redundant API calls

reuse high-quality past results

retrieve precise context when needed

scale without memory pollution

This file is intentionally policy-free.
All decisions about what to cache or when to reuse are made by the agent logic, not here.

High-level design

This cache implements a hybrid memory model with two distinct layers:

1. Query Cache (coarse-grained, decision memory)

One vector per query

Represents a solved intent, not raw documents

Used to answer:

“Have I already answered something like this?”

This enables:

early exit

API cost reduction

deterministic stopping behavior

2. Chunk Store (fine-grained, context memory)

Many vectors per document

Each vector represents a large semantic chunk of an abstract

Used to answer:

“Which exact parts of the documents are relevant?”

This enables:

higher semantic precision

better context selection

chunk-level diversity across documents

These two layers solve different problems and must not be merged.

Why a hybrid approach is required

Using only query-level caching causes semantic dilution
Using only chunk-level vectors causes loss of control and no stopping signal

The hybrid design separates concerns cleanly:

Query cache controls agent behavior
Chunk store controls answer quality

This mirrors how real retrieval systems work in production.

Responsibilities of VectorCache
What this file DOES

Stores embeddings in ChromaDB

Enforces:

cosine similarity threshold

TTL (time-to-live)

LRU eviction (query cache only)

Performs bounded, monotonic search

Provides:

query reuse lookup

chunk-level semantic retrieval with diversity control

What this file DOES NOT do

Decide whether a query should be cached

Score evidence quality

Track API usage

Handle conversation memory

Expand context windows

Control agent loops or stopping criteria

All of that belongs elsewhere in the agent.

Query Cache: detailed behavior
What is stored

Each query cache entry stores:

{
  "embedding": <query_embedding>,
  "metadata": {
    "query": "<original text>",
    "created_at": <timestamp>,
    "payload": {
      "docs": [...],
      "chunks": [...],
      "retrieval_score": <float>,
      "answer": "<optional>",
      "citations": [...]
    }
  }
}


The payload is the reusable unit, not the query string.

How lookup works

When a new query arrives:

Embed the query

Search for the top N most similar cached queries (default N = 3)

For each candidate (in descending similarity order):

If similarity < threshold → stop (monotonic guarantee)

If expired → delete and continue

If valid → return payload immediately

If none valid → cache miss

This bounded fallback prevents:

false misses due to expiration

scanning the entire cache

reuse of weakly related queries

TTL and LRU

TTL ensures scientific freshness

LRU ensures bounded memory usage

LRU eviction applies only to query cache entries, because:

query cache is decision memory

chunk store is reference memory

Chunk Store: detailed behavior
What is stored

Each chunk entry represents a large semantic unit, not a sentence:

{
  "embedding": <chunk_embedding>,
  "document": "<chunk_text>",
  "metadata": {
    "pmid": "<document_id>",
    "chunk_index": <int>,
    "section": "<optional>"
  }
}


Chunk size is intentionally large (≈200–350 tokens) to avoid:

excessive homogeneity

multiple tiny chunks from the same document

artificial evidence inflation

How chunk retrieval works

Query Chroma for top-k chunks

Iterate in similarity order

Early-exit once similarity drops below threshold

Enforce per-document chunk limits

Optionally require a minimum number of chunks

Return only anchor chunks

These anchor chunks represent where the evidence lives.

They are not yet expanded for context.

Why context window expansion is NOT here

Context window expansion is a presentation concern, not a retrieval concern.

Retrieval answers:

“Which chunks are relevant?”

Context expansion answers:

“How much surrounding text should I show?”

Therefore:

expansion happens after scoring

expansion happens after agent decides to stop

expansion lives in a separate context_builder.py

Keeping this separation:

preserves clean similarity math

avoids polluting scoring signals

keeps cache semantics simple

Similarity thresholds

Similarity thresholds are configurable

Chunk retrieval can optionally use a looser threshold

Query cache reuse should be strict

This flexibility allows:

conservative reuse

exploratory retrieval

safe agent control

Important architectural rules

Query cache is intent-level memory

Chunk store is content-level memory

Cache does not decide admission

Agent decides what is worth remembering

Expired entries are deleted eagerly

Monotonic similarity enables early exit

Diversity constraints prevent evidence collapse

Violating these rules leads to brittle, unscalable systems.

How this file fits into the full agent

Execution order (simplified):

User Query
  ↓
Embed query
  ↓
Query Cache lookup (this file)
  ↓ hit
Return cached payload
  ↓ miss
Retrieve documents
  ↓
Chunk documents
  ↓
Add chunks to chunk store (this file)
  ↓
Search chunks (this file)
  ↓
Score evidence
  ↓
Agent decides stop / fetch more
  ↓ stop
Context window expansion (elsewhere)
  ↓
Final synthesis
  ↓
Agent may add query to cache (this file)

Why this design scales

Memory remains bounded

Similarity stays meaningful

Cache reuse is safe

Agent behavior is deterministic

Chunk retrieval stays precise

Future extensions (persistent DB, full text, PDFs) are possible without rewriting this file

Summary (one paragraph)

cache.py implements a hybrid vector memory that cleanly separates decision reuse (query cache) from semantic precision (chunk store). It enforces TTL, LRU, similarity thresholds, and diversity constraints, while deliberately avoiding policy decisions. This design allows an agent to scale, stop safely, avoid redundant retrieval, and retrieve high-quality context without semantic dilution.

Chunk Structure

    {
        "text": "...",
        "similarity": 0.60,
        "metadata": {
            "pmid": "12345",
            "year": 2021
        }
    },