# Transformer from Scratch

Building a Transformer encoder from scratch in PyTorch, then applying it to financial sentiment analysis — comparing a hand-built model against traditional ML (XGBoost) and fine-tuned pretrained models (FinBERT).

## Motivation

Most ML practitioners use Transformers through high-level APIs without understanding the internal mechanics. This project implements every component from the ground up — scaled dot-product attention, multi-head attention, positional encoding, encoder blocks — with full dimension annotations at each layer. The goal is to demonstrate deep architectural understanding, not just library proficiency.

## Project Structure

```
├── warmup/                     # Week 0: LLM ecosystem quickstart
│   ├── day1_api_prompting.py   # API calls + prompt engineering patterns
│   ├── day2_rag_pipeline.py    # Minimal RAG: PDF → chunk → embed → retrieve → generate
│   └── day3_agents_tools.py    # Agent loop with tool use
│
├── src/                        # Core Transformer implementation
│   ├── attention.py            # ScaledDotProductAttention, MultiHeadAttention
│   ├── encoder.py              # TransformerEncoderBlock
│   ├── positional.py           # PositionalEncoding (sinusoidal)
│   ├── model.py                # TransformerClassifier (full model)
│   └── utils.py                # Tokenization, data loading, metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_scratch_transformer.ipynb
│   ├── 03_train_xgboost_baseline.ipynb
│   ├── 04_finetune_finbert.ipynb
│   └── 05_attention_visualization.ipynb
│
├── results/                    # Metrics, plots, attention heatmaps
├── requirements.txt
└── README.md
```

## Architecture

```
Input token IDs
    │
    ▼
TokenEmbedding ──► (batch, seq_len, d_model)
    │
    ▼
PositionalEncoding ──► (batch, seq_len, d_model)
    │
    ▼
TransformerEncoderBlock × N
    ├── MultiHeadAttention + LayerNorm + Residual
    └── FeedForward + LayerNorm + Residual
    │
    ▼
Mean Pooling ──► (batch, d_model)
    │
    ▼
ClassificationHead ──► (batch, 3)   # positive / negative / neutral
```

## Experiment: Three-Way Model Comparison

| Model | Description |
|-------|-------------|
| **Scratch Transformer** | Hand-built encoder, trained from random initialization |
| **XGBoost + TF-IDF** | Traditional ML baseline |
| **FinBERT (fine-tuned)** | Pretrained on financial corpus, fine-tuned on task data |

**Dataset:** [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) — ~4,200 sentences from financial news, annotated for sentiment by finance professionals (75% annotator agreement split).

**Metrics:** Accuracy, Macro F1, per-class Precision/Recall, Confusion Matrix

## Key Takeaways

*(To be updated after experiments)*

- How much does domain-specific pretraining (FinBERT) improve over training from scratch?
- Where does the scratch Transformer fail compared to FinBERT — and what does attention visualization reveal?
- Can traditional ML (XGBoost) still compete on small-data NLP tasks?

## Tech Stack

- PyTorch (model implementation, no `nn.TransformerEncoder`)
- HuggingFace Transformers & Datasets (FinBERT fine-tuning, data loading)
- scikit-learn, XGBoost (baseline)
- matplotlib, seaborn (visualization)

## Setup

```bash
git clone https://github.com/<your-username>/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
```

## Author

**Bryan (Bolai) Yin**
M.S. Machine Learning, Northeastern University (Khoury College of Computer Sciences)
