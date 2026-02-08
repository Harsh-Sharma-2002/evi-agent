# Agentic Evidence Retrieval System (Scientific Literature)

## Overview

This project implements a **minimal but explicit agentic system** for evidence-grounded question answering over scientific literature (e.g., PubMed abstracts).

Unlike prompt-based RAG pipelines, this system is designed as a **stateful agent** with:

- Explicit control flow  
- Hybrid memory (short-term, long-term, semantic)  
- Tool use (external retrieval)  
- Policy-driven stopping and reuse  
- Clear separation between **policy** and **mechanism**

The goal of this project is **not answer quality or UI polish**, but to explore **agent architecture, control, memory, and robustness**.

This repository is intentionally verbose and inspectable, prioritizing **clarity over abstraction**.

---

## What Makes This Agentic (Not Just RAG)

This system crosses from a pipeline into an **agent** because it:

- Decides **whether to act or stop**
- Iteratively gathers evidence
- Tracks internal state across steps
- Uses memory to avoid redundant work
- Can terminate with *“insufficient evidence”*
- Separates **decision logic** from **execution mechanisms**

In short:

> The system reasons about *what to do next*, not just *how to answer*.

---

## High-Level Agent Loop

User Query
↓
Initialize Agent State
↓
Tier 1: Query Cache Lookup
↓
[Cache Hit] → Prompt + Answer
↓
[Cache Miss]
↓
Tier 2: Chunk Store Search
↓
Score Evidence
↓
Decision Policy
├─ STOP → Context Expansion → Final Answer → Cache Admission
└─ FETCH_MORE → Tier 3: External Retrieval (PubMed) → Loop


The loop is **explicit, bounded, and policy-driven**.

---

## Architectural Principles

### 1. Separation of Policy and Mechanism

- **Policies** decide *what should happen*
- **Mechanisms** implement *how it happens*

Examples:
- *Policy*: When to stop retrieving evidence  
- *Mechanism*: Vector similarity search in ChromaDB  

This separation allows policies to evolve (heuristics → learned policies) without rewriting infrastructure.

---

### 2. State Is the Single Source of Truth

All agent behavior is driven by a mutable `AgentState`, which tracks:

- Retrieval progress
- Evidence quality
- Memory interactions
- Cost and iteration limits
- Termination reasons

There is no hidden control logic.

---

### 3. Memory Has Multiple Time Horizons

The agent uses **three distinct memory types**, each with a clear role.

---

## Memory Design

### Tier 1: Query Cache (Decision Memory)

**Purpose:**  
Reuse previously solved intents safely and deterministically.

**Characteristics:**
- One vector per query
- Stores *payloads*, not raw text
- Enforced via:
  - Cosine similarity threshold
  - TTL (freshness)
  - LRU eviction (bounded memory)

**Key property:**  
The cache is **policy-free** — it does not decide what is worth storing.

The agent controls cache admission explicitly.

---

### Tier 2: Chunk Store (Semantic Memory)

**Purpose:**  
Provide high-precision semantic retrieval at the chunk level.

**Characteristics:**
- Many vectors per document
- Large semantic chunks (≈200–350 tokens)
- Enforces:
  - Similarity threshold
  - Monotonic early exit
  - Per-document diversity constraints

This avoids semantic dilution and evidence collapse.

---

### Conversational Memory (Summary-Only)

**Purpose:**  
Preserve high-level conversational context without polluting retrieval.

**Characteristics:**
- Stores a single rolling summary
- Updated periodically via LLM
- Injected optionally into prompts
- Never stored in `AgentState`

---

## Retrieval Tiers

### Tier 1: Query Cache Lookup

- Checks if a similar query has already been solved
- If valid:
  - Skips retrieval
  - Uses cached payload as reference context
  - Still runs the LLM to answer independently

Cache hits **do not force termination**.

---

### Tier 2: Chunk Store Search

- Searches previously stored document chunks
- Returns *anchor chunks* only
- Anchor chunks represent where evidence lives
- Context expansion happens later

This tier enables reuse without full re-retrieval.

---

### Tier 3: External Retrieval (PubMed)

- Fetches scientific abstracts via PubMed E-Utilities
- Safe by construction:
  - Never crashes
  - Handles network / parse failures
- Each document is chunked and embedded
- New chunks are added to the chunk store

This tier is **cost-aware and bounded**.

---

## Evidence Scoring

Evidence is scored using a combination of:

- Semantic similarity
- Publication recency
- Document-level aggregation
- Diversity bonus across independent sources

The result is a **single retrieval score** used by the agent’s policy.

A hard confidence gate ensures:
- Minimum number of distinct documents
- No overconfidence from single-source evidence

---

## Decision Policy (Core Agent Logic)

The agent decides between `STOP` and `FETCH_MORE` based on:

- Maximum iterations
- API call budget
- Evidence exhaustion
- Stagnation detection
- Retrieval score threshold + confidence

These rules are **explicit, inspectable, and deterministic**.

This system intentionally uses **heuristic policies** rather than learned ones, to make control transparent.

---

## Context Expansion

Once the agent decides to stop:

- Anchor chunks are expanded with neighboring chunks
- Expansion is deterministic and bounded
- Happens **after** scoring and stopping

This preserves clean similarity math and avoids context pollution.

---

## Prompt Construction

The final prompt:

- Enforces strict grounding rules
- Injects:
  - Expanded evidence
  - Optional conversational memory
  - Cached answer (reference-only, never copied)
- Explicitly disallows hallucination

The LLM is treated as a **synthesis mechanism**, not a decision-maker.

---

## Observability & Logging

Every agent run is logged with:

- Stop reason
- Retrieval score
- Evidence count
- Cache reuse metrics
- API usage

Logs are **read-only diagnostics** and never influence behavior.

---

## Why the Cache Is Policy-Free

The cache:
- Stores vectors
- Retrieves by similarity
- Enforces TTL and LRU

It does **not** decide:
- Whether something is correct
- Whether it should be cached
- Whether it should be reused

All such decisions belong to the agent.

This mirrors real production systems and enables future research flexibility.

---

## Limitations (Intentional)

This project is intentionally minimal:

- No symbolic planning
- No learned policies (e.g., RL)
- No UI
- Uses a small local LLM (FLAN-T5) for determinism

The focus is **agent control and architecture**, not raw performance.

---

## Future Directions

Possible extensions include:

- Learned stopping policies
- Planner / executor separation
- Multi-agent collaboration
- Persistent vector stores
- Full-text and PDF ingestion
- Benchmark-driven evaluation

None of these require redesigning the cache or retrieval mechanisms.

---

## Summary

This repository implements a **transparent, inspectable agentic retrieval system** with:

- Explicit control flow
- Hybrid memory
- Policy-driven behavior
- Safe tool use
- Deterministic stopping

It is designed as a **learning and research scaffold**, not a product.

The core idea is simple:

> The agent decides *what to do*; the system provides *how to do it*.
