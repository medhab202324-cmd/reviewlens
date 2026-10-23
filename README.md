# 🔍 ReviewLens: GenAI-Powered Product Review Intelligence

**[▶ Live demo](https://reviewlens-medha.streamlit.app)** ·
[Sentiment model](https://huggingface.co/Medha-56/reviewlens-sentiment-distilbert) ·
[Aspect model](https://huggingface.co/Medha-56/reviewlens-aspects-distilbert) ·
[App data](https://huggingface.co/datasets/Medha-56/reviewlens-data)

Brands receive thousands of reviews and can't read them all. ReviewLens processes
**701,528 Amazon beauty reviews** end to end and answers three questions for a brand manager:

1. **What are customers saying?** Fine-tuned DistilBERT models tag every review's sentiment and the aspects it discusses.
2. **Is something going wrong?** Statistical tests flag products whose complaints rise significantly, after correcting for multiple testing.
3. **Why?** A RAG assistant answers plain-English questions using only real reviews, with citations.

## Architecture

```
Amazon Reviews 2023 (701K reviews, 112K products)
        |
        v
[1] PySpark ingestion + profiling --> [2] Cleaning, quality checks, EDA (634,229 reviews)
        |
        +--> [3] Baseline: TF-IDF + Logistic Regression (time-based split)
        +--> [4] LLM aspect labeling (Gemini) + second-LLM agreement check
        +--> [5] Fine-tuned DistilBERT: sentiment + 7-aspect multi-label (Colab GPU)
        |            |
        |            v  labels 123K reviews of the top 500 products
        +--> [6] Trend alerts: two-proportion z-tests + Benjamini-Hochberg FDR
        +--> [7] RAG: MiniLM embeddings + FAISS + LangChain + Gemini, with evaluation
        v
[8] Streamlit dashboard (Streamlit Community Cloud; models + data on Hugging Face Hub)
```

## Results

**Sentiment** (trained on reviews up to 2020, tested on 20,000 reviews from 2022-2023)

| Model | Macro-F1 | Accuracy | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| Always predict "positive" | 0.264 | n/a | n/a | n/a | n/a |
| TF-IDF + Logistic Regression | 0.713 | 0.839 | 0.823 | 0.395 | 0.922 |
| **DistilBERT (fine-tuned)** | **0.748** | **0.865** | **0.855** | **0.449** | **0.941** |

**Aspect detection** (DistilBERT distilled from 1,900 LLM-labeled reviews; LLM label agreement Cohen's κ = 0.85)
Test micro-F1 **0.821**, macro-F1 **0.757**

| Aspect | DistilBERT F1 | Keyword rules F1 |
|---|---|---|
| Quality / effectiveness | 0.89 | n/a |
| Price / value | 0.92 | 0.83 |
| Scent | 0.92 | 0.86 |
| Packaging | 0.66 | 0.52 |
| Shipping / delivery | 0.69 | 0.40 |
| Ease of use | 0.74 | n/a |
| Skin / hair reaction | 0.48 | 0.55 |

**Statistical alerting:** 997 product-quarter tests → 138 "significant" at p<0.05 →
**16 actionable alerts** after FDR correction and a ≥10-point rule. Category-wide, the odds of a review
being negative rise ×1.064 per year.

**RAG evaluation**

| Metric | Result |
|---|---|
| Precision@5, generic questions | 0.83 (random: 0.06) |
| Precision@5, product questions | 0.77 (random: 0.36) |
| Hit@5 (generic / product) | 1.00 / 1.00 |
| Faithfulness (claims supported by sources, LLM judge) | 1.0 |
| Valid citations | 86.7% |
| Unanswerable questions correctly refused | 3/3 |

## Key engineering decisions

- **Label leakage found and removed:** 57,580 reviews had auto-generated titles like "Five Stars" that revealed the rating.
- **Time-based splits** (train on the past, test on the future) to mirror production and avoid leakage.
- **Macro-F1** as the main metric, because 70% of reviews are positive.
- **LLM-to-small-model distillation:** an LLM labels a small sample once; DistilBERT then labels all reviews for free.
- **Label quality measured** with a second, stronger LLM as an independent annotator (Cohen's κ).
- **Multiple-testing correction** (Benjamini-Hochberg) plus a practical-significance threshold, so alerts are both real and large.
- **RAG evaluated, not just demoed:** retrieval precision vs a random baseline, faithfulness, citation validity and abstention.

## Tech stack

PySpark · pandas · NumPy · SciPy · statsmodels · scikit-learn · PyTorch · Hugging Face Transformers ·
DistilBERT · sentence-transformers · FAISS · LangChain · Gemini API · Plotly · Streamlit ·
Hugging Face Hub · Streamlit Community Cloud · Google Colab (T4 GPU) · Git

## Project structure

```
reviewlens/
|-- notebooks/
|   |-- 01_ingestion_profiling.ipynb          # full pipeline, phases 1-8
|   |-- 02_colab_finetune_distilbert.ipynb    # DistilBERT fine-tuning on Colab GPU
|-- app/                                      # Streamlit app (Streamlit Community Cloud)
|-- reports/                                  # metrics (JSON/CSV) and interactive charts (HTML)
|-- requirements.txt
|-- README.md
```

## Run it locally

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (Python 3.11, Java 17 for PySpark)
pip install -r requirements.txt
# add GEMINI_API_KEY and HF_TOKEN to a .env file
# run the notebook top to bottom (DistilBERT fine-tuning runs on Google Colab)
streamlit run app/app.py
```

## Data

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab, UC San Diego. Category: All_Beauty.

---
Built by **Medha Bhandari** · [GitHub](https://github.com/medhab202324-cmd)
