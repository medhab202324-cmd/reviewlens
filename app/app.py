"""ReviewLens - GenAI-powered product review intelligence (Streamlit app)."""
import os, re, json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------- config
APP_DIR = Path(__file__).resolve().parent
LOCAL_MODELS = APP_DIR.parent / "models"
HF_USER = "Medha-56"
DATA_REPO = f"{HF_USER}/reviewlens-data"

@st.cache_resource(show_spinner="Downloading app data (first start only)...")
def get_data_dir():
    """Laptop: use app/data. Online: download the data once from Hugging Face."""
    local = APP_DIR / "data"
    if (local / "reviews.parquet").exists():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=DATA_REPO, repo_type="dataset"))

DATA_DIR = get_data_dir()

def model_source(local_name, hub_name):
    """Use the local model folder if it exists (laptop), otherwise Hugging Face Hub (online)."""
    p = LOCAL_MODELS / local_name
    return str(p) if (p / "config.json").exists() else f"{HF_USER}/{hub_name}"

SENT_SRC = model_source("sentiment_distilbert", "reviewlens-sentiment-distilbert")
ASP_SRC = model_source("aspect_distilbert", "reviewlens-aspects-distilbert")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    from dotenv import load_dotenv
    load_dotenv(APP_DIR.parent / ".env")
except Exception:
    pass
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:                      # online: read the key from Streamlit's secret settings
    try:
        GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
MAX_QUESTIONS = 15                      # per visitor session, protects the free Gemini quota

ASPECTS = ["quality_effectiveness", "price_value", "scent", "packaging",
           "shipping_delivery", "ease_of_use", "skin_hair_reaction"]
ASPECT_LABELS = {"quality_effectiveness": "Quality / effectiveness", "price_value": "Price / value",
                 "scent": "Scent", "packaging": "Packaging", "shipping_delivery": "Shipping / delivery",
                 "ease_of_use": "Ease of use", "skin_hair_reaction": "Skin / hair reaction"}
LABELS = ["negative", "neutral", "positive"]
COLORS = {"negative": "#E4572E", "neutral": "#F2C14E", "positive": "#2E86AB"}

# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading review data...")
def load_data():
    reviews = pd.read_parquet(DATA_DIR / "reviews.parquet")
    quarterly = pd.read_parquet(DATA_DIR / "product_quarterly.parquet")
    quarterly["alert"] = quarterly["alert"].fillna(False).astype(bool)
    alerts = pd.read_csv(DATA_DIR / "alerts.csv")
    results = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in DATA_DIR.glob("phase*.json")}
    products = (reviews.groupby("parent_asin")
                .agg(product_title=("product_title", "first"), store=("store", "first"),
                     n_reviews=("review_id", "size"),
                     pct_negative=("sentiment", lambda s: (s == "negative").mean() * 100),
                     avg_rating=("rating", "mean"))
                .sort_values("n_reviews", ascending=False).reset_index())
    return reviews, quarterly, alerts, results, products

@st.cache_resource(show_spinner="Loading the semantic search index...")
def load_search():
    import faiss
    from sentence_transformers import SentenceTransformer
    emb = np.load(DATA_DIR / "embeddings_fp16.npy").astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return emb, index, SentenceTransformer(EMB_MODEL, device="cpu")

@st.cache_resource(show_spinner="Loading the DistilBERT models...")
def load_classifiers():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok_s = AutoTokenizer.from_pretrained(SENT_SRC)
    m_s = AutoModelForSequenceClassification.from_pretrained(SENT_SRC).eval()
    tok_a = AutoTokenizer.from_pretrained(ASP_SRC)
    m_a = AutoModelForSequenceClassification.from_pretrained(ASP_SRC).eval()
    thr_path = DATA_DIR / "aspect_thresholds.json"
    thresholds = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.exists() else {}
    return torch, tok_s, m_s, tok_a, m_a, thresholds

@st.cache_resource
def gemini_client():
    from google import genai
    return genai.Client(api_key=GEMINI_KEY)

reviews, quarterly, alerts, results, products = load_data()
R3 = results.get("phase3_baseline", {})
R4 = results.get("phase4_aspects", {})
R5 = results.get("phase5_results", {})
R6 = results.get("phase6_trends", {})
R7 = results.get("phase7_rag_eval", {})

# ---------------------------------------------------------------- models
def analyze_text(text):
    torch, tok_s, m_s, tok_a, m_a, thr = load_classifiers()
    with torch.no_grad():
        enc = tok_s(text, truncation=True, max_length=160, return_tensors="pt")
        sp = torch.softmax(m_s(**enc).logits, dim=-1)[0].numpy()
        enc = tok_a(text, truncation=True, max_length=160, return_tensors="pt")
        ap = torch.sigmoid(m_a(**enc).logits)[0].numpy()
    sent = pd.DataFrame({"sentiment": LABELS, "probability": sp})
    asp = pd.DataFrame({"aspect": [ASPECT_LABELS[a] for a in ASPECTS], "probability": ap,
                        "threshold": [thr.get(a, 0.5) for a in ASPECTS]})
    asp["detected"] = asp["probability"] >= asp["threshold"]
    return sent, asp

# ---------------------------------------------------------------- RAG
COMPLAINT_WORDS = re.compile(r"\b(unhappy|complain\w*|problems?|issues?|negative|bad|worst|"
                             r"disappoint\w*|hate|wrong|fail\w*|broke\w*|returns?)\b", re.I)

def retrieve(question, k=8, product=None, sentiment=None):
    emb, index, embedder = load_search()
    qv = embedder.encode([question], normalize_embeddings=True).astype("float32")
    if product is None and sentiment is None:
        scores, ids = index.search(qv, k)
        ids, scores = ids[0], scores[0]
    else:
        mask = np.ones(len(reviews), dtype=bool)
        if product:
            mask &= (reviews["parent_asin"] == product).values
        if sentiment:
            mask &= (reviews["sentiment"] == sentiment).values
        allowed = np.where(mask)[0]
        if len(allowed) == 0:
            return reviews.iloc[[]].assign(score=[])
        sims = emb[allowed] @ qv[0]
        top = np.argsort(-sims)[:k]
        ids, scores = allowed[top], sims[top]
    return reviews.iloc[ids].assign(score=np.round(scores, 3)).reset_index(drop=True)

def product_stats(asin):
    p = reviews[reviews["parent_asin"] == asin]
    neg = p[p["sentiment"] == "negative"]
    lines = [f"Product: {p['product_title'].iloc[0]}", f"Total reviews: {len(p):,}",
             f"Negative reviews: {len(neg) / len(p) * 100:.1f}%"]
    if len(neg):
        top = (neg[ASPECTS].mean() * 100).sort_values(ascending=False).head(3)
        lines.append("Most-mentioned aspects in negative reviews: " + ", ".join(
            f"{ASPECT_LABELS[a]} ({v:.0f}% of negative reviews)" for a, v in top.items()))
    pa = alerts[alerts["parent_asin"] == asin]
    if len(pa):
        r = pa.sort_values("quarter").iloc[-1]
        lines.append(f"Latest statistical alert: {r['quarter']}, negative share rose from "
                     f"{r['rate_before'] * 100:.0f}% to {r['rate_now'] * 100:.0f}% "
                     f"(main driver: {ASPECT_LABELS.get(r['driver_aspect'], r['driver_aspect'])})")
    return "PRODUCT STATISTICS:\n" + "\n".join(lines)

def format_docs(docs):
    return "\n\n".join(
        f"[{i + 1}] (rating {int(r.rating)}/5, {r.review_date}) {' '.join(str(r.full_text).split()[:120])}"
        for i, r in enumerate(docs.itertuples()))

RAG_SYSTEM = (
    "You are ReviewLens, an assistant that helps brand managers understand customer reviews.\n"
    "Answer ONLY using the numbered customer reviews provided and the product statistics, if given.\n"
    "Rules:\n"
    "- Cite the reviews supporting each point with their numbers in square brackets, e.g. [2] or [1][4].\n"
    "- Never invent facts, numbers or reviews, and never use outside knowledge.\n"
    "- If the reviews do not contain enough information to answer, reply exactly: "
    "\"Not enough evidence in the reviews.\"\n"
    "- Format: one-sentence summary, then 2-4 bullet points, then a line starting with "
    "'Suggested action:' giving one practical recommendation for the brand."
)
MODEL_PREFS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-flash-latest"]

def call_gemini(prompt_value):
    from google.genai import types
    msgs = prompt_value.to_messages()
    errors = []
    for model in MODEL_PREFS:
        for think in ["minimal", "low", None]:
            try:
                cfg = dict(system_instruction=msgs[0].content, temperature=0.2, max_output_tokens=1500)
                if think:
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_level=think)
                r = gemini_client().models.generate_content(
                    model=model, contents=msgs[1].content, config=types.GenerateContentConfig(**cfg))
                if r.text:
                    return {"answer": r.text, "model": model}
                errors.append(f"{model}: empty response")
                break
            except Exception as e:
                if "thinking" in str(e).lower():
                    continue
                errors.append(f"{model}: {str(e)[:120]}")
                break
    raise RuntimeError(" | ".join(errors))

@st.cache_resource
def build_chain():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("human", "{stats}\n\nCUSTOMER REVIEWS:\n{context}\n\nQUESTION: {question}")])

    def smart_retrieve(x):
        sentiment = "negative" if COMPLAINT_WORDS.search(x["question"]) else None
        docs = retrieve(x["question"], k=8, product=x["product"], sentiment=sentiment)
        if len(docs) == 0 and sentiment:
            docs, sentiment = retrieve(x["question"], k=8, product=x["product"]), None
        return {"docs": docs, "filter": sentiment}

    return (RunnablePassthrough.assign(retrieved=RunnableLambda(smart_retrieve))
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(x["retrieved"]["docs"]),
                stats=lambda x: product_stats(x["product"]) if x["product"] else "")
            | RunnablePassthrough.assign(result=prompt | RunnableLambda(call_gemini)))

# ---------------------------------------------------------------- cached charts
@st.cache_data
def overall_trend():
    d = reviews[reviews["quarter"] >= "2015Q1"]
    t = (d.groupby("quarter").agg(pct_negative=("sentiment", lambda s: (s == "negative").mean() * 100),
                                  n=("review_id", "size")).reset_index())
    t = t.merge(alerts.groupby("quarter").size().rename("alerts"),
                left_on="quarter", right_index=True, how="left").fillna({"alerts": 0})
    return t

@st.cache_data
def aspect_trend():
    d = reviews[reviews["review_date"].str[:4] >= "2015"].copy()
    d["year"] = d["review_date"].str[:4].astype(int)
    neg = (d["sentiment"] == "negative").astype(int)
    rows = []
    for a in ASPECTS:
        s = (neg * d[a]).groupby(d["year"]).mean() * 100
        rows += [{"year": y, "aspect": ASPECT_LABELS[a], "pct": v} for y, v in s.items()]
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- header
st.title("🔍 ReviewLens")
st.caption("GenAI-powered product review intelligence: 123K Amazon beauty reviews from the top 500 products. "
           "Built with PySpark, fine-tuned DistilBERT, statistical alerting and RAG (FAISS + LangChain + Gemini).")

tab_over, tab_prod, tab_analyze, tab_ask, tab_perf = st.tabs(
    ["📊 Overview", "🔎 Product explorer", "🧪 Analyze a review", "💬 Ask ReviewLens", "📈 Model performance"])

# ---------------------------------------------------------------- tab 1: overview
with tab_over:
    c = st.columns(5)
    c[0].metric("Reviews analysed", f"{len(reviews):,}")
    c[1].metric("Products", f"{reviews['parent_asin'].nunique():,}")
    c[2].metric("Complaint alerts", f"{len(alerts)}",
                help="Product-quarters where negative reviews rose significantly (z-test, FDR-corrected, +10pp or more)")
    if R5 and R3:
        f1 = R5["sentiment"]["test_macro_f1"]
        c[3].metric("Sentiment macro-F1", f"{f1:.3f}", delta=f"{f1 - R3['test_macro_f1']:+.3f} vs baseline")
    if R7.get("faithfulness_mean") is not None:
        c[4].metric("RAG faithfulness", f"{R7['faithfulness_mean']:.2f}")

    t = overall_trend()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=t["quarter"], y=t["alerts"], name="alerts fired", marker_color="#F2C14E",
                opacity=0.6, secondary_y=True)
    fig.add_scatter(x=t["quarter"], y=t["pct_negative"], name="% negative reviews",
                    mode="lines+markers", line_color="#E4572E", secondary_y=False)
    fig.update_layout(title="% negative reviews per quarter, with complaint alerts fired",
                      template="plotly_white", height=400, legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="% negative", secondary_y=False)
    fig.update_yaxes(title_text="alerts", secondary_y=True)
    st.plotly_chart(fig)

    fig = px.line(aspect_trend(), x="year", y="pct", color="aspect", markers=True,
                  labels={"pct": "% of all reviews"},
                  title="What customers complain about: % of reviews that are negative AND mention each aspect")
    fig.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig)

    st.subheader("🚨 Complaint alerts")
    st.caption("Each row is a product whose share of negative reviews jumped significantly versus the previous "
               "two quarters. 'Driver' is the aspect whose complaints grew the most.")
    if len(alerts):
        show = alerts.assign(product=alerts["product_title"].astype(str).str[:60],
                             before=(alerts["rate_before"] * 100).round(1),
                             now=(alerts["rate_now"] * 100).round(1),
                             increase_pp=alerts["uplift_pp"].round(1),
                             q_value=alerts["q_value"].round(4),
                             driver=alerts["driver_aspect"].map(ASPECT_LABELS))
        st.dataframe(show[["product", "quarter", "n", "before", "now", "increase_pp", "q_value", "driver"]]
                     .rename(columns={"n": "reviews", "before": "% neg before", "now": "% neg now"}),
                     hide_index=True)

# ---------------------------------------------------------------- tab 2: product explorer
with tab_prod:
    options = products["parent_asin"].tolist()
    names = dict(zip(products["parent_asin"],
                     products["product_title"].astype(str).str[:80] + "  ·  " +
                     products["n_reviews"].astype(str) + " reviews"))
    default = alerts.iloc[0]["parent_asin"] if len(alerts) else options[0]
    asin = st.selectbox("Choose a product", options, index=options.index(default),
                        format_func=lambda a: names[a])
    prow = products[products["parent_asin"] == asin].iloc[0]
    p = reviews[reviews["parent_asin"] == asin]
    c = st.columns(4)
    c[0].metric("Reviews", f"{prow['n_reviews']:,}")
    c[1].metric("Average rating", f"{prow['avg_rating']:.2f} ★")
    c[2].metric("% negative", f"{prow['pct_negative']:.1f}%")
    c[3].metric("Alerts", int((alerts["parent_asin"] == asin).sum()))

    pq = quarterly[(quarterly["parent_asin"] == asin) & (quarterly["n"] >= 5)].copy()
    pq["pct_negative"] = pq["neg_rate"] * 100
    fig = px.line(pq, x="quarter", y="pct_negative", markers=True,
                  title="% negative reviews per quarter (quarters with 5+ reviews)")
    hit = pq[pq["alert"]]
    if len(hit):
        fig.add_scatter(x=hit["quarter"], y=hit["pct_negative"], mode="markers", name="ALERT",
                        marker=dict(color="#E4572E", size=14, symbol="diamond"))
    fig.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig)

    prof = []
    for s in ["negative", "positive"]:
        sub = p[p["sentiment"] == s]
        if len(sub):
            for a in ASPECTS:
                prof.append({"aspect": ASPECT_LABELS[a], "sentiment": s, "pct": sub[a].mean() * 100})
    if prof:
        fig = px.bar(pd.DataFrame(prof), x="aspect", y="pct", color="sentiment", barmode="group",
                     color_discrete_map=COLORS, text_auto=".0f",
                     title="What negative vs positive reviews talk about (% mentioning each aspect)")
        fig.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig)

    st.subheader("Most recent negative reviews")
    for r in p[p["sentiment"] == "negative"].sort_values("review_date", ascending=False).head(5).itertuples():
        with st.container(border=True):
            st.markdown(f"**{'★' * int(r.rating)}{'☆' * (5 - int(r.rating))}** · {r.review_date}")
            st.write(str(r.full_text)[:600])

# ---------------------------------------------------------------- tab 3: analyze a review
with tab_analyze:
    st.write("Paste any product review. The two fine-tuned DistilBERT models predict its "
             "**sentiment** and which **aspects** it talks about.")
    text = st.text_area("Review text", height=130, value=(
        "The scent is lovely, but the pump broke after a week and half the bottle leaked in the box. "
        "Honestly not worth the price."))
    if st.button("Analyze", type="primary") and text.strip():
        sent, asp = analyze_text(text.strip())
        top = sent.sort_values("probability", ascending=False).iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted sentiment", top["sentiment"].capitalize(),
                      f"{top['probability'] * 100:.0f}% confidence")
            fig = px.bar(sent, x="probability", y="sentiment", orientation="h", color="sentiment",
                         color_discrete_map=COLORS, text_auto=".2f", range_x=[0, 1])
            fig.update_layout(template="plotly_white", height=250, showlegend=False)
            st.plotly_chart(fig)
        with col2:
            found = asp[asp["detected"]]["aspect"].tolist()
            st.metric("Aspects detected", len(found))
            st.write(", ".join(f"`{a}`" for a in found) if found else "No specific aspect detected.")
            fig = px.bar(asp, x="probability", y="aspect", orientation="h", text_auto=".2f", range_x=[0, 1],
                         color="detected", color_discrete_map={True: "#2E86AB", False: "#C9D1D9"})
            fig.update_layout(template="plotly_white", height=300, showlegend=False)
            st.plotly_chart(fig)

# ---------------------------------------------------------------- tab 4: ask
with tab_ask:
    st.write("Ask a question in plain English. ReviewLens finds the most relevant reviews with semantic "
             "search (FAISS) and an LLM writes an answer **using only those reviews**, citing them like [2].")
    if not GEMINI_KEY:
        st.warning("GEMINI_API_KEY is not set, so the assistant is unavailable.")
    else:
        q_options = ["ALL"] + options
        q_asin = st.selectbox("Product (optional)", q_options,
                              index=q_options.index(default) if default in q_options else 0,
                              format_func=lambda a: "All 500 products" if a == "ALL" else names[a],
                              key="ask_product")
        st.caption("Try: *Why are customers unhappy with this product?* · *Is it worth the money?* · "
                   "*What do people say about the smell?*")
        with st.form("ask_form"):
            question = st.text_input("Your question", value="Why are customers unhappy with this product?")
            submitted = st.form_submit_button("Ask", type="primary")
        st.session_state.setdefault("n_asked", 0)
        if submitted and question.strip():
            if st.session_state["n_asked"] >= MAX_QUESTIONS:
                st.error(f"Demo limit reached ({MAX_QUESTIONS} questions per session).")
            else:
                st.session_state["n_asked"] += 1
                with st.spinner("Retrieving reviews and writing the answer..."):
                    try:
                        out = build_chain().invoke({"question": question.strip(),
                                                    "product": None if q_asin == "ALL" else q_asin})
                        st.session_state["last_answer"] = out
                    except Exception as e:
                        st.session_state.pop("last_answer", None)
                        st.error("The LLM is unavailable right now (free-tier limit). Please try again later.")
                        st.caption(str(e)[:300])
        out = st.session_state.get("last_answer")
        if out:
            st.markdown(out["result"]["answer"])
            docs, filt = out["retrieved"]["docs"], out["retrieved"]["filter"]
            st.caption(f"Answered by {out['result']['model']} from {len(docs)} retrieved reviews"
                       + (" (complaint question: searched negative reviews only)" if filt else ""))
            with st.expander("Show the source reviews"):
                st.dataframe(docs.assign(ref=[f"[{i + 1}]" for i in range(len(docs))],
                                         review=docs["full_text"].astype(str).str[:300])
                             [["ref", "rating", "review_date", "sentiment", "score", "review"]],
                             hide_index=True)

# ---------------------------------------------------------------- tab 5: performance
with tab_perf:
    st.subheader("Sentiment: classical baseline vs fine-tuned DistilBERT")
    st.caption("Time-based split: trained on reviews up to 2020, tested on 20,000 reviews from 2022-2023.")
    if R3 and R5:
        perf = pd.DataFrame([
            {"model": "TF-IDF + Logistic Regression", "macro-F1": R3["test_macro_f1"],
             "accuracy": R3["test_accuracy"], **{f"F1 {l}": R3["test_f1_per_class"][l] for l in LABELS}},
            {"model": "DistilBERT (fine-tuned)", "macro-F1": R5["sentiment"]["test_macro_f1"],
             "accuracy": R5["sentiment"]["test_accuracy"],
             **{f"F1 {l}": R5["sentiment"]["f1_per_class"][l] for l in LABELS}}])
        st.dataframe(perf.round(3), hide_index=True)

    st.subheader("Aspect detection (distilled from LLM labels)")
    if R5:
        a = R5["aspects"]
        st.caption(f"DistilBERT trained on {a['n_train']:,} LLM-labeled reviews · test micro-F1 "
                   f"{a['test_micro_f1']:.3f}, macro-F1 {a['test_macro_f1']:.3f}"
                   + (f" · LLM label agreement (Cohen's kappa) {R4['mean_kappa']:.2f}" if R4.get("mean_kappa") else ""))
        st.dataframe(pd.DataFrame([{"aspect": ASPECT_LABELS[k], "DistilBERT F1": v,
                                    "keyword rules F1": a["keyword_baseline_f1"].get(k)}
                                   for k, v in a["f1_per_aspect"].items()]).round(3), hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistical alerting")
        if R6:
            st.write(f"- Product-quarter tests run: **{R6['n_tests']:,}**\n"
                     f"- 'Significant' at p<0.05 without correction: **{R6['significant_before_correction']}**\n"
                     f"- Alerts after Benjamini-Hochberg FDR + ≥10pp rule: **{R6['alerts_after_fdr_and_10pp']}**\n"
                     f"- Category trend: odds of a negative review **×{R6['category_trend_odds_ratio_per_year']:.3f} per year**")
    with col2:
        st.subheader("RAG evaluation")
        if R7:
            for r in R7.get("retrieval", []):
                st.write(f"- {r['type'].capitalize()} questions: precision@5 **{r['precision_at_5']:.2f}** "
                         f"(random {r['random_baseline']:.2f}), hit@5 {r['hit_at_5']:.2f}")
            st.write(f"- Faithfulness: **{R7.get('faithfulness_mean')}** · valid citations: "
                     f"{R7.get('citations_valid_pct')}% · abstention test: {R7.get('abstention_pass')}")

st.divider()
st.caption("Data: Amazon Reviews 2023 (McAuley Lab, UCSD), All_Beauty category. Built by Medha Bhandari.")
