# 🏛️ DeepClause — Contract Intelligence Platform

> AI-powered RAG system for enterprise legal & sales operations. Ask questions about your contracts in plain English and get precise answers with source citations.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-green)
![Qdrant](https://img.shields.io/badge/Qdrant-1.10-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-Answer_Relevancy_0.79-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 The Problem

Every enterprise drowns in contracts. thousands of PDFs scattered across drives, email attachments.

| Without DeepClause | With DeepClause |
|---|---|
| 3h/week per sales rep hunting clauses | < 5 min per query |
| Renewals missed → lost revenue | 90-day expiry alerts via metadata |
| Legal bottleneck on every clause question | Self-service for 80% of queries |
| New AE needs months to learn contracts | Answers on day one |

---

## 🏗️ Architecture

```
User Query
    ↓
[1] Query Rewriting         — Qwen3-32B transforms vague questions into legal search terms
    ↓
[2] Metadata Filtering      — LLM extracts party names, dates, contract types as Qdrant filters
    ↓
[3] Hybrid Search            — Dense (bge-large-en-v1.5) + Sparse (BM25) with RRF fusion
    ↓
[4] Reranking               — Cohere rerank-english-v3.0 scores query+clause jointly
    ↓
[5] Generation              — llama-3.3-70b-versatile answers from retrieved context only
    ↓
Structured Answer + Citations (file, page, clause)
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **Embeddings** | `BAAI/bge-large-en-v1.5` (1024 dims) |
| **Sparse Search** | `Qdrant/bm25` via FastEmbed |
| **Vector Store** | Qdrant (local Docker) |
| **Reranker** | Cohere `rerank-english-v3.0` |
| **LLM** | Groq — `llama-3.3-70b-versatile` |
| **Query Rewriter** | Groq — `Qwen3-32B` |
| **Metadata Filters** | LangChain Query Constructor |
| **API** | FastAPI + SSE streaming |
| **UI** | Gradio |
| **Evaluation** | RAGAS on CUAD benchmark |
| **Containerization** | Docker + Docker Compose |

---

## 📁 Project Structure

```
DeepClause/
├── api/
│   └── routers/
│       └── query.py            # /query and /query/stream endpoints
├── generation/
│   ├── llm_client.py           # Groq LLM wrapper + result formatter
│   ├── question_answering.py   # System + user prompt templates
│   └── guardrails.py           # Hallucination + PII detection
├── retrieval/
│   ├── hybridSearch.py         # Dense + sparse + RRF fusion
│   ├── reranker.py             # Cohere / local cross-encoder
│   ├── query_rewriter.py       # LLM query rewriting
│   └── filters.py              # LLM-powered metadata filter extraction
├── ingestion/
│   └── ingestionPipeline.py    # Load → chunk → embed → store
├── evals/
│   ├── build_golden_dataset.py # CUAD Q&A pair sampling
│   └── ragas.py                # RAGAS evaluation runner
├── pipeline.py                 # RAG pipeline orchestrator
├── fastApiMain.py              # FastAPI app factory
├── main.py                     # Local dev entry point (Gradio + CLI)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/deepclause.git
cd deepclause
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your keys
```

```bash
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxx
COHERE_API_KEY=xxxxxxxxxxxx       # optional, falls back to local reranker
API_KEY=your_secret_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Start Qdrant

```bash
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 4. Run ingestion pipeline

```bash
python ingestionPipeline.py --data_dir ./data/full_contract_pdf
```

### 5. Run locally (Gradio UI)

```bash
python main.py
# Open http://localhost:7860
```

### 6. Run API

```bash
uvicorn fastApiMain:app --reload --port 8000
# Open http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

```bash
# Start everything (Qdrant + FastAPI)
docker compose up -d

# Check health
curl http://localhost:8000/health

# Query the API
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: your_secret_key_here" \
  -H "Content-Type: application/json" \
  -d '{"query": "When does the Zogenix distributor agreement expire?"}'
```

---

## 🔍 Example Query

**Input:**
```
"In the Zogenix distributor agreement, who are the parties and what is the expiration date?"
```

**Output:**
```json
{
  "answer": "The agreement is between Zogenix, Inc. and Nippon Shinyaku Company, Ltd.
             It expires on September 1, 2045 per Section 8 of the agreement.",
  "sources": [
    {
      "filename": "ZogenixInc_20190509_10-Q_EX-10.2_Distributor Agreement.pdf",
      "contract_type": "DISTRIBUTORSHIP AGREEMENT",
      "party_1": "ZOGENIX, INC.",
      "party_2": "Nippon Shinyaku Company, Ltd.",
      "agreement_date": "2045-09-01",
      "effective_date": "2045-09-01",
      "expiration_date": "2045-09-01",
      "pages": [1, 2],
      "governing_law": "New York",
      "notice_period_to_terminate": "2045-09-01"
    }
  ],
}
```

---

## 📊 Evaluation (RAGAS on CUAD)

Evaluated on 30 Q&A pairs sampled.

| Metric | Score |
|---|---|
| **Answer Relevancy** | 0.79 |
| **Dataset** | CUAD v1 (510 contracts, 13,000+ labels) |
| **Sample size** | 50 Q&A pairs |

---

## 📦 Dataset

This project uses the [CUAD (Contract Understanding Atticus Dataset)](https://www.atticusprojectai.org/cuad):
- 510 commercial contracts
- 13,000+ expert-annotated labels
- 41 clause categories
- Licensed under CC BY 4.0

---

## 🗺️ Roadmap

- [ ] Conversation memory (multi-turn Q&A)
- [ ] Automatic renewal alerts
- [ ] JWT + RBAC authentication
- [ ] Frontend dashboard (Next.js)
- [ ] Faithfulness and Context Recall metrics

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ using Python, Langchain, FastAPI, Qdrant, and Groq</p>
