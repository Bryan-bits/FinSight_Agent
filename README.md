# FinSight Agent

**A multi-step financial question-answering agent over SEC filings, designed around a domain-specific evaluation framework.**

Most LLM-based financial QA demos achieve surface-level fluency but fail systematically on multi-document reasoning, numerical precision, and domain terminology. FinSight Agent is built end-to-end — from document ingestion to agent orchestration — with **evaluation as a first-class design constraint, not an afterthought**.

---

## TL;DR

- **Task**: Answer real-world financial questions (e.g. *"How did 3M's operating margin change YoY in FY2022, and what drove the change?"*) over SEC 10-K / 10-Q filings.
- **Approach**: Multi-step agent combining domain-aware retrieval, query decomposition, tool-augmented calculation, and a custom Transformer encoder for earnings-call sentiment scoring.
- **Differentiator**: A structured evaluation framework that diagnoses failure modes (retrieval gaps, numerical errors, GAAP/Non-GAAP confusion, hallucinations) rather than reporting a single accuracy number.
- **Benchmark**: [FinanceBench](https://github.com/patronus-ai/financebench) (Patronus AI) — 150 expert-annotated QA pairs over real 10-K filings.

---

## Motivation

Production LLM systems in finance fail in specific, recurring ways:

1. **Multi-hop retrieval failures** — the system fetches one year's filing when the question requires year-over-year comparison.
2. **Numerical precision failures** — confidently produces a wrong number off-by-a-decimal.
3. **Domain terminology failures** — substitutes GAAP for Non-GAAP, organic for total growth, net for gross.
4. **Hallucinated reasoning** — produces a plausible-sounding answer with no grounding in the source document.

The original FinanceBench paper reported that GPT-4-Turbo with a vector store incorrectly answered or refused **81% of questions**. This project treats that gap as the central engineering problem: not "build another RAG demo," but "build a system whose failure modes you can characterize, measure, and reduce."

---

## System Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           User Question                   │
                    │  e.g. "Compare 3M's operating margin     │
                    │        in FY2022 vs FY2021"              │
                    └────────────────┬─────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────┐
              │       Query Decomposer (LLM)             │
              │  Plans atomic sub-queries + tool calls   │
              └────────────────┬─────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────────┐
        ▼                      ▼                          ▼
┌───────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   Retrieval   │    │   Calculation    │    │  Sentiment Module  │
│   Sub-Agent   │    │   Sub-Agent      │    │  (Custom Encoder)  │
│               │    │                  │    │                    │
│ Domain-aware  │    │ Python tool for  │    │ Scratch Transformer│
│ chunking +    │    │ arithmetic on    │    │ trained on Financial│
│ hybrid search │    │ retrieved values │    │ PhraseBank         │
└───────┬───────┘    └────────┬─────────┘    └─────────┬──────────┘
        │                     │                        │
        └─────────────────────┼────────────────────────┘
                              ▼
              ┌──────────────────────────────────────┐
              │        Synthesizer (LLM)             │
              │  Composes final grounded answer      │
              └────────────────┬─────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────────┐
              │      Evaluation Harness              │
              │  • Step-level accuracy               │
              │  • Tool-call correctness             │
              │  • Hallucination detection           │
              │  • Domain-specific metrics           │
              └──────────────────────────────────────┘
```

---

## Roadmap & Status

The project is structured in five phases, each producing concrete deliverables before the next phase starts.

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 0** | Dataset exploration, naive baseline, failure taxonomy v1 | 🚧 In progress |
| **Sentiment Module** | Custom Transformer encoder (built in parallel with Phase 0) | 🚧 In progress |
| **Phase 1** | Production-grade financial retrieval pipeline | 📋 Planned |
| **Phase 2** | Multi-step agent orchestration with tool use | 📋 Planned |
| **Phase 3** | Evaluation framework & failure-mode deep dive | 📋 Planned |
| **Phase 4** | Observability, cost/latency analysis, demo & writeup | 📋 Planned |

A three-day LLM ecosystem warmup (API patterns, minimal RAG, agent loop with tool use) is **complete** and feeds into the components below. The scratch Transformer encoder for the sentiment module is being implemented in parallel with Phase 0.

---

## Components

### 1. Financial Document Retrieval Pipeline *(Planned — Phase 1)*

- SEC EDGAR API ingestion of 10-K / 10-Q filings
- Domain-aware chunking: separate strategies for financial statements (table-aware), MD&A, and Risk Factors sections
- Retrieval ablation study across: embedding model (general vs financial), chunking strategy, retrieval method (dense / BM25 / hybrid + RRF), reranker (none / cross-encoder)
- Evaluation: Recall@k, MRR, nDCG, and a domain-specific *answer-bearing chunk recall* metric

### 2. Multi-Step Agent Orchestration *(Planned — Phase 2)*

- Framework: LangGraph for explicit state-graph control
- Sub-agents: Query Decomposer, Retrieval Sub-Agent, Calculation Sub-Agent, Sentiment Module, Synthesizer
- Trajectory logging for downstream evaluation

### 3. Custom Transformer Sentiment Module *(In progress)*

A from-scratch PyTorch Transformer encoder, to be used as a sentiment-scoring component for earnings-call passages within the agent system.

- **Built from scratch (no `nn.TransformerEncoder`)**: scaled dot-product attention, multi-head attention with concat + projection, sinusoidal positional encoding, encoder block with residual stream
- **Training data**: [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) — ~4,200 sentences from financial news, expert-annotated for sentiment
- **Design rationale**: Implementing the encoder from scratch makes architectural decisions (residual stream as persistent information highway, multi-head projection mechanics, hyperparameter taxonomy) explicit rather than opaque library calls — and enables direct attention-pattern instrumentation when diagnosing the agent's reasoning failures
- **Integration target**: Exposed via a `SentimentScorer.predict(text)` wrapper for invocation by the agent's sub-agents in Phase 2

Lightweight baselines (XGBoost + TF-IDF, fine-tuned FinBERT) will be included as reference points; the scratch model is the integrated component.

### 4. Evaluation Framework *(Planned — Phase 3, project's core deliverable)*

The component that distinguishes this project from typical RAG demos.

- **Failure taxonomy** — structured categorization across retrieval, calculation, reasoning, domain-terminology, and refusal failures, derived from manual annotation
- **Multi-level metrics** — step-level accuracy, tool-call correctness, end-to-end answer accuracy, hallucination detection (NLI-based + SelfCheckGPT-style)
- **Domain-specific metrics** — designed to catch failures that generic metrics miss: GAAP/Non-GAAP confusion, unit mismatches ($M vs $B), period misattribution
- **Failure-mode case studies** — annotated trajectories showing where and why the agent failed, with attribution to specific pipeline stages

### 5. Observability & Production Polish *(Planned — Phase 4)*

- Trace visualization (LangSmith or Braintrust integration)
- Cost / latency / quality tradeoff analysis across model tiers
- Lightweight demo UI (Streamlit or Gradio)

---

## Datasets

| Dataset | Role | Status |
|---------|------|--------|
| [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) (150 QA pairs over 10-K filings) | Primary benchmark | 🚧 Phase 0 manual annotation in progress |
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (~4,200 sentences) | Training data for sentiment module | 🚧 In progress |
| SEC EDGAR filings (10-K / 10-Q, ~10 companies) | Retrieval corpus | 📋 Phase 1 |

---

## Tech Stack

- **Modeling**: PyTorch (custom Transformer, no `nn.TransformerEncoder`); HuggingFace Transformers (FinBERT baseline)
- **Agent orchestration**: LangGraph
- **Retrieval**: SEC EDGAR API; embedding models (OpenAI, BGE); BM25 (rank_bm25); reranking (Cohere or cross-encoder)
- **LLM APIs**: OpenAI, Anthropic
- **Evaluation**: Custom harness; optional integration with LangSmith / Braintrust for trace observability
- **Baselines**: scikit-learn, XGBoost

---

## Repository Structure

```
finsight-agent/
├── warmup/                          # LLM ecosystem quickstart (complete)
│   ├── day1_api_prompting.py
│   ├── day2_rag_pipeline.py
│   └── day3_agents_tools.py
│
├── src/
│   ├── sentiment/                   # Scratch Transformer (in progress)
│   │   ├── attention.py
│   │   ├── encoder.py
│   │   ├── positional.py
│   │   └── model.py
│   ├── retrieval/                   # Planned — Phase 1
│   ├── agent/                       # Planned — Phase 2
│   └── evaluation/                  # Planned — Phase 3
│
├── notebooks/
│   ├── 00_dataset_exploration.ipynb # Phase 0
│   ├── 01_naive_baseline.ipynb      # Phase 0
│   ├── 02_train_sentiment.ipynb     # Sentiment module training (in progress)
│   ├── 03_retrieval_ablation.ipynb  # Planned — Phase 1
│   ├── 04_agent_trajectories.ipynb  # Planned — Phase 2
│   └── 05_failure_analysis.ipynb    # Planned — Phase 3
│
├── docs/
│   ├── failure_taxonomy.md          # Living document, updated each phase
│   ├── retrieval_design.md          # Planned
│   └── agent_design.md              # Planned
│
├── eval/                            # Evaluation harness — Planned, Phase 3
│   ├── datasets/
│   ├── metrics/
│   └── reports/
│
├── results/                         # Metrics, plots, attention heatmaps
├── requirements.txt
└── README.md
```

---

## Evaluation Methodology

The evaluation strategy is the project's intellectual core. The principle: **a single accuracy number on a benchmark tells you nothing useful for production**.

The framework is structured around three questions:

1. **Where does the system fail?**
   Failure modes are categorized into a taxonomy derived from manual annotation, not invented top-down. Each category is associated with example trajectories and an estimated incidence rate.

2. **Why does it fail there?**
   Failures are attributed to specific pipeline stages (retrieval vs reasoning vs calculation vs formatting), enabling targeted intervention rather than broad model swaps.

3. **What changes when we intervene?**
   Each pipeline modification is evaluated against the full taxonomy, not just on aggregate accuracy. A change that improves aggregate accuracy while introducing new failure modes is flagged, not celebrated.

Detailed methodology and findings will live in `docs/failure_taxonomy.md` and `notebooks/05_failure_analysis.ipynb` as Phase 3 completes.

---

## Setup

```bash
git clone https://github.com/<your-username>/finsight-agent.git
cd finsight-agent
pip install -r requirements.txt
```

Environment variables (place in `.env`):
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

---

## Author

**Bryan (Bolai) Yin**
M.S. in Computer Science (AI focus), Northeastern University — Khoury College of Computer Sciences
Background: 15 years in HR / recruitment including J.P. Morgan, with applied understanding of financial-services workflows and document types relevant to this project.
