# Clinical Dosing & Guideline Assistant

> **An enterprise-grade, two-stage Retrieval-Augmented Generation (RAG) pipeline purpose-built for clinical pharmacology — where a wrong answer is not a bug, it is a patient safety risk.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![LLM: Groq / LLaMA-3](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA--3.3--70b-orange)](https://console.groq.com)
[![Embeddings: HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace%20(Free)-green)](https://huggingface.co)
[![Reranker: CrossEncoder](https://img.shields.io/badge/Reranker-BAAI%2Fbge--reranker--base-purple)](https://huggingface.co/BAAI/bge-reranker-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [High-Level Design (HLD)](#3-high-level-design-hld)
4. [Low-Level Design (LLD)](#4-low-level-design-lld)
5. [End-to-End Data Flow](#5-end-to-end-data-flow)
6. [Chunking Strategies (Ingestion Layer)](#6-chunking-strategies-ingestion-layer)
7. [Retrieval Architecture (Two-Stage Pipeline)](#7-retrieval-architecture-two-stage-pipeline)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Project Structure](#9-project-structure)
10. [Tech Stack](#10-tech-stack)
11. [Setup & Installation](#11-setup--installation)
12. [Usage](#12-usage)
13. [API Keys & Cost](#13-api-keys--cost)
14. [Confidence Scoring & Safety Logic](#14-confidence-scoring--safety-logic)

---

## 1. Problem Statement

Standard vector search (dense retrieval) is **inadequate for clinical AI**. Here is why:

| Failure Mode | Example | Impact |
|---|---|---|
| **Numeric blindness** | Query `eGFR 45` matches generic kidney docs, not specific threshold rules | Wrong dosing tier returned |
| **Hallucination** | LLM fabricates a dose when context is sparse | Potential patient harm |
| **Chunk fragmentation** | "Max dose 500mg" separated from "only if eGFR > 30" by the chunker | Orphaned dosing rule |
| **No uncertainty signal** | AI gives confident answer even with irrelevant context | Doctor trusts a wrong answer |

This system is engineered to solve all four failure modes simultaneously.

---

## 2. Solution Overview

We built a **Two-Stage Retrieval Pipeline** that combines the conceptual understanding of neural AI with the exact-match precision of classical keyword search, fused by a mathematical ranking algorithm, and filtered by a Cross-Encoder neural reranker before generating the final cited answer.

**Key Engineering Decisions:**
- ✅ **Hierarchical chunking** keeps dosing rules permanently attached to their qualifying conditions
- ✅ **Hybrid retrieval (Dense + BM25 + RRF)** handles both concepts and exact clinical numbers
- ✅ **Cross-Encoder reranking** provides a deep, accurate second-pass filter before the LLM sees any context
- ✅ **Confidence scoring** explicitly signals when the system is uncertain, preventing the AI from hallucinating
- ✅ **100% Free stack** — Local HuggingFace embeddings, open-source reranker, Groq free tier for LLM

---

## 3. High-Level Design (HLD)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLINICAL RAG SYSTEM                                   │
│                                                                               │
│   ┌──────────────┐     ┌─────────────────────────────────────────────────┐  │
│   │   DATA PLANE │     │                  QUERY PLANE                    │  │
│   │              │     │                                                  │  │
│   │  PDF / Text  │     │  User Query (Doctor / Clinician)                 │  │
│   │      │       │     │        │                                         │  │
│   │      ▼       │     │        ▼                                         │  │
│   │  Hierarchical│     │  ┌─────────────────────────────────────────┐    │  │
│   │   Chunker    │     │  │         HYBRID RETRIEVER                │    │  │
│   │      │       │     │  │                                         │    │  │
│   │      ▼       │     │  │  ┌──────────────┐  ┌────────────────┐  │    │  │
│   │  HuggingFace │     │  │  │ DenseRetriever│  │ BM25Retriever  │  │    │  │
│   │  Embeddings  │     │  │  │ (Semantic AI) │  │ (Keyword Exact)│  │    │  │
│   │      │       │     │  │  └──────┬───────┘  └───────┬────────┘  │    │  │
│   │      ▼       │     │  │         │                   │           │    │  │
│   │  FAISS Index │     │  │         └─────────┬─────────┘           │    │  │
│   │  (Vector DB) │     │  │                   ▼                     │    │  │
│   └──────────────┘     │  │     Reciprocal Rank Fusion (RRF)        │    │  │
│                         │  │          (Top 20 candidates)            │    │  │
│                         │  └─────────────────┬───────────────────────┘   │  │
│                         │                    │                             │  │
│                         │                    ▼                             │  │
│                         │  ┌──────────────────────────────────────────┐   │  │
│                         │  │         CROSS-ENCODER RERANKER           │   │  │
│                         │  │    (BAAI/bge-reranker-base, local)       │   │  │
│                         │  │         Returns Top-K with scores        │   │  │
│                         │  └─────────────────┬────────────────────────┘   │  │
│                         │                    │                             │  │
│                         │                    ▼                             │  │
│                         │  ┌──────────────────────────────────────────┐   │  │
│                         │  │            GROQ LLM (LLaMA-3.3-70b)     │   │  │
│                         │  │  System Prompt: "Answer ONLY from       │   │  │
│                         │  │   context. Cite every dosing rule."     │   │  │
│                         │  └─────────────────┬────────────────────────┘   │  │
│                         │                    │                             │  │
│                         │                    ▼                             │  │
│                         │     Cited Answer + Confidence + Sources          │  │
│                         └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Low-Level Design (LLD)

### 4.1 Ingestion Module (`ingestion/`)

```
ingestion/
├── loader.py          → PyPDFLoader wrapper: PDF → List[Section]
└── chunker.py         → Four strategy implementations

Section schema:
{
  "source":  "Drug_Monograph.pdf",
  "section": "Renal Dosing Adjustments",
  "page":    12,
  "text":    "Raw extracted text..."
}

Chunk schema:
{
  "chunk_id":  "h_c001_s001",
  "text":      "Max dose 500mg for eGFR 30-45",
  "strategy":  "hierarchical",
  "metadata":  { "source", "section", "page", "parent_id" }
}
```

### 4.2 Retrieval Module (`retrieval/`)

```
retrieval/
├── config.py      → Settings dataclass, loaded from .env
├── dense.py       → DenseRetriever: HuggingFace Embeddings + FAISS
├── bm25.py        → BM25Retriever: rank_bm25 (BM25Okapi)
├── hybrid.py      → HybridRetriever: RRF fusion of Dense + BM25
└── reranker.py    → BaseReranker → CrossEncoderReranker / CohereReranker
                     get_reranker() factory reads RERANKER env var

ScoredChunk schema (output of all retrievers):
{
  "chunk_id":     "h_c001_s001",
  "text":         "...",
  "score":        0.032,          ← RRF score (pre-rerank)
  "rerank_score": 0.9986,         ← CrossEncoder score (post-rerank)
  "metadata":     { ... }
}
```

### 4.3 Application Module (`app/`)

```
app/
├── rag_chain.py    → ClinicalRAGChain orchestrator
├── streamlit_app.py → Web UI
└── cli.py          → Terminal interface

ClinicalRAGChain.answer() return schema:
{
  "answer":     "The maximum dose is 500mg... [Source: X, Section: Y]",
  "sources":    [ { "text", "document", "section", "rerank_score" } ],
  "confidence": "high" | "medium" | "low"
}
```

---

## 5. End-to-End Data Flow

### Ingestion Flow (One-time, at startup)

```
PDF File
   │
   ▼
loader.py (PyPDFLoader)
   │  → Extracts text page-by-page
   │  → Returns List[Section dicts]
   ▼
chunker.py (hierarchical_chunker)
   │  → Parent chunk = full section
   │  → Child chunks = individual sentences
   │  → Each child stores parent_id for traceability
   ▼
DenseRetriever.index()
   │  → HuggingFaceEmbeddings("all-MiniLM-L6-v2")
   │  → Converts each chunk text → 384-dim float32 vector
   │  → Stores in FAISS IndexFlatL2 (in-memory, L2 distance)
   ▼
BM25Retriever.index()
   │  → Tokenizes each chunk text (lowercase, whitespace split)
   │  → Builds BM25Okapi inverted index in memory
   ▼
Knowledge Base Ready ✅
```

### Query Flow (Per user request)

```
User Query: "max metformin dose stage 3 CKD"
   │
   ├──────────────────────────────────────────────────────────────┐
   │                                                              │
   ▼                                                             ▼
DenseRetriever.search(query, k=20)                BM25Retriever.search(query, k=20)
  │  → Embed query → 384-dim vector                │  → Tokenize query
  │  → FAISS L2 nearest-neighbor search             │  → BM25Okapi.get_scores()
  │  → Returns 20 semantically similar chunks       │  → Returns 20 exact-match chunks
  │                                                 │
  └──────────────────┬──────────────────────────────┘
                     │
                     ▼
           HybridRetriever (RRF Fusion)
             │  For each chunk from both lists:
             │  rrf_score += 1 / (60 + rank)
             │  → Chunks appearing high in BOTH lists
             │    get double-boosted scores
             │  → Returns unified Top-20 ranked list
             ▼
         CrossEncoderReranker.rerank(query, top_20, top_n=5)
             │  → For each of 20 candidates:
             │    Input: [query, chunk_text] as a pair
             │    Output: relevance_score (0.0 → 1.0)
             │  → Deep cross-attention (reads both together)
             │  → Returns Top-5 sorted by rerank_score desc
             ▼
         ClinicalRAGChain.answer()
             │  → Formats Top-5 as numbered context blocks
             │  → Builds prompt with system instructions
             │  → Calls ChatGroq (LLaMA-3.3-70b-versatile)
             │  → Computes confidence:
             │      HIGH   → top rerank_score ≥ 0.8
             │      MEDIUM → top rerank_score 0.6–0.8
             │      LOW    → top rerank_score < 0.6 OR
             │               LLM replied "INSUFFICIENT INFORMATION"
             ▼
         Final Response
         {
           answer:     "Max 500mg twice daily [Source: ...]",
           sources:    [...],
           confidence: "high"
         }
```

---

## 6. Chunking Strategies (Ingestion Layer)

We implemented and benchmarked four chunking strategies. The chosen production strategy is **Hierarchical**.

| Strategy | Method | Clinical Suitability | Risk |
|---|---|---|---|
| **Fixed-Size** | Naive split every N characters | ❌ Baseline only | Splits mid-rule, separates dose from condition |
| **Recursive** | LangChain `RecursiveCharacterTextSplitter` | ⚠️ Moderate | Respects paragraphs but still misses semantic boundaries |
| **Semantic** | Sentence groups by embedding similarity | ✅ Good | Computationally expensive; may over-merge unrelated rules |
| **Hierarchical** | Parent = Section, Child = Sentence with `parent_id` | ✅ **Production** | Guarantees dosing rule stays attached to its qualifying condition |

### Why Hierarchical is Critical

Consider this clinical rule:
```
Section: Metformin / Renal Adjustments
  → "The maximum dose is 500mg twice daily."    ← Child chunk A
  → "Only if eGFR is between 30 and 45."       ← Child chunk B (parent_id → Section)
```

Fixed-size chunking might put A in chunk 4 and B in chunk 5, separated by other text. A retrieval system returning only chunk A would generate a dangerously incomplete answer. Hierarchical chunking ties both to the same parent so they are always retrieved together.

---

## 7. Retrieval Architecture (Two-Stage Pipeline)

### Stage 1: Hybrid Retrieval

#### Dense Retrieval (The "Brain")
- **Model:** `all-MiniLM-L6-v2` (384-dim, runs locally, 100% free)
- **Vector DB:** FAISS `IndexFlatL2` (exact L2 nearest neighbor, in-memory)
- **Strength:** Understands semantic meaning. "Hypertension" matches "high blood pressure"
- **Weakness:** Poor at exact numbers and clinical codes (eGFR values, drug codes)

#### BM25 Retrieval (The "Librarian")
- **Algorithm:** BM25Okapi (term-frequency weighted keyword ranking)
- **Library:** `rank_bm25`
- **Strength:** Exact character-level matching for numbers, drug names, lab values
- **Weakness:** Zero semantic understanding. "BP" won't match "blood pressure"

#### Reciprocal Rank Fusion (RRF)
Both retrievers return Top-20 candidates. Their scores are **incompatible** (cosine similarity vs. BM25 score). RRF solves this by ignoring raw scores entirely and only using ranks:

```
rrf_score(doc) = Σ  1 / (k + rank_i)
                  i
where k = 60 (smoothing constant)
```

A document ranked #1 by Dense and #2 by BM25 gets a much higher RRF score than one ranked #1 by only one engine. This mathematically forces documents that satisfy **both** semantic and exact-match criteria to the top.

### Stage 2: Cross-Encoder Reranking

The Top-20 RRF results are passed to a **Cross-Encoder** model (`BAAI/bge-reranker-base`).

| | Bi-Encoder (Stage 1 Dense) | Cross-Encoder (Stage 2 Reranker) |
|---|---|---|
| **How it works** | Query and document embedded *separately*, compared by cosine distance | Query and document concatenated, processed *together* by BERT-style model |
| **Speed** | Fast (milliseconds) | Slow (seconds per pair) |
| **Accuracy** | Moderate (no cross-attention) | High (full cross-attention between query and doc) |
| **Use case** | Recall: Find the best 20 from thousands | Precision: Find the best 5 from 20 |

The Cross-Encoder outputs a `rerank_score` (0.0–1.0) per chunk. This score directly feeds the **confidence badge** in the UI.

---

## 8. Evaluation Framework

Located in `eval/`, the evaluation suite provides mathematical proof of pipeline quality.

### Labeled Dataset (`eval/labeled_queries.json`)
- **20 hand-crafted clinical Q&A pairs**
- Covers: `dose_with_condition` (8), `contraindication` (4), `max_dose` (4), `drug_interaction` (2), `monitoring` (2)
- Uses realistic clinical parameters: eGFR 15/30/45/60 thresholds, Child-Pugh A/B/C hepatic staging, pediatric weight bands

### RAGAS Metrics (`eval/benchmark.py`)

| Metric | What it measures | Clinical Interpretation |
|---|---|---|
| **Context Precision** | Are the retrieved chunks relevant? | Did we retrieve the right dosing rule? |
| **Context Recall** | Did we retrieve all necessary chunks? | Did we miss a critical qualifying condition? |
| **Faithfulness** | Does the answer stay within context? | Is the LLM hallucinating? |
| **Answer Relevancy** | Does the answer address the question? | Is the response clinically actionable? |

Run the benchmark with:
```bash
python -m eval.benchmark --strategy all
```

---

## 9. Project Structure

```
Clinical_Dosing_and_Guideline_Assistant/
├── .env                          ← Your API keys (never commit this)
│
└── clinical_rag/
    ├── .env.example              ← Template for required env vars
    ├── requirements.txt
    │
    ├── ingestion/
    │   ├── loader.py             ← PDF → Section dicts (PyPDFLoader)
    │   └── chunker.py            ← Four chunking strategies
    │
    ├── retrieval/
    │   ├── config.py             ← Settings dataclass loaded from .env
    │   ├── dense.py              ← DenseRetriever (HuggingFace + FAISS)
    │   ├── bm25.py               ← BM25Retriever (rank_bm25)
    │   ├── hybrid.py             ← HybridRetriever (RRF fusion)
    │   └── reranker.py           ← CrossEncoderReranker / CohereReranker
    │
    ├── app/
    │   ├── rag_chain.py          ← ClinicalRAGChain orchestrator
    │   ├── streamlit_app.py      ← Web UI
    │   └── cli.py                ← Terminal interface
    │
    └── eval/
        ├── labeled_queries.json  ← 20 clinical Q&A pairs
        ├── benchmark.py          ← RAGAS evaluation harness
        └── results/
            └── {strategy}_scores.json
```

---

## 10. Tech Stack

| Component | Technology | Cost |
|---|---|---|
| **PDF Parsing** | `pypdf` + `langchain-community` | Free |
| **Text Splitting** | `langchain-text-splitters` | Free |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | **Free, local** |
| **Vector Database** | `FAISS` (IndexFlatL2) | **Free, in-memory** |
| **Keyword Search** | `rank_bm25` (BM25Okapi) | **Free** |
| **Reranker** | `BAAI/bge-reranker-base` (CrossEncoder) | **Free, local** |
| **LLM** | Groq API · `llama-3.3-70b-versatile` | **Free tier** |
| **Evaluation** | `RAGAS` framework | Free |
| **UI** | `Streamlit` | Free |
| **Orchestration** | `LangChain` | Free |

**Total cost to run this system: $0.00**

---

## 11. Setup & Installation

### Prerequisites
- Python 3.10+
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/swagatobauri/Clinical_Dosing_and_Guideline_Assistant.git
cd Clinical_Dosing_and_Guideline_Assistant
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
cd clinical_rag
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Your `.env` file should look like this:
```env
# LLM Provider — Free tier at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# Architecture Settings (defaults are already optimal)
RERANKER=crossencoder
LLM_MODEL=llama-3.3-70b-versatile
```

> **Note:** `OPENAI_API_KEY` is NOT required. All embeddings and reranking run locally and for free.

---

## 12. Usage

### Web UI (Streamlit)
```bash
cd clinical_rag
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

The app ships with a built-in demo knowledge base. You can query it immediately or upload your own PDF clinical guideline.

### Terminal CLI
```bash
cd clinical_rag
python -m app.cli "max metformin dose stage 3 CKD"
```

**Sample output:**
```
============================================================
ANSWER:
============================================================
The maximum metformin dose for a patient with eGFR 30-45 mL/min/1.73m2
(stage 3b CKD) is 500mg twice daily; do not exceed 1000mg/day.
[Source: Clinical_Guidelines_2026.pdf, Section: General Dosing]

============================================================
CONFIDENCE: HIGH
============================================================
SOURCES:
[1] Clinical_Guidelines_2026.pdf | General Dosing | Score: 0.9986
    Snippet: The maximum metformin dose for a patient with eGFR 30-45...
```

### Benchmarking Chunking Strategies
```bash
cd clinical_rag
python -m eval.benchmark --strategy all
```

---

## 13. API Keys & Cost

| Key | Required | Where to get | Cost |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) | **Free tier, generous limits** |
| `OPENAI_API_KEY` | ❌ No | — | Not needed |
| `COHERE_API_KEY` | ❌ No | — | Only if switching `RERANKER=cohere` |

---

## 14. Confidence Scoring & Safety Logic

The confidence badge in the UI is computed in `app/rag_chain.py` and is the most critical safety feature of this system.

```python
top_score = top_chunks[0].get("rerank_score", 0.0)

if top_score < 0.6 or "INSUFFICIENT INFORMATION" in answer:
    confidence = "low"     # ← AI doesn't know. DO NOT trust this answer.
elif top_score < 0.8:
    confidence = "medium"  # ← Partial match. Cross-reference before acting.
else:
    confidence = "high"    # ← Strong match found in indexed guidelines.
```

When the system returns `LOW`, the LLM will respond with:
> *"INSUFFICIENT INFORMATION — consult the drug monograph directly."*

This is **not a failure**. This is the system working correctly. A clinical AI that says *"I don't know"* is infinitely safer than one that confidently fabricates a dose.

---

## License

MIT License © 2026 Swagato Bauri

---

> *"For research and educational use only. This system is not a substitute for clinical judgment. Always verify dosing decisions with a licensed pharmacist or physician and the current drug monograph."*
