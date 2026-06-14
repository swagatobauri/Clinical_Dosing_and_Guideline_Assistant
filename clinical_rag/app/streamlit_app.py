# pyrefly: ignore [missing-import]
import streamlit as st
import os, json, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.rag_chain import ClinicalRAGChain
from retrieval.config import settings

st.set_page_config(page_title="Clinical Dosing Assistant")
st.title("Clinical Dosing Assistant")
st.subheader("Answers cited from drug monographs and clinical guidelines")

st.sidebar.header("Settings")
strategy = st.sidebar.selectbox("Strategy", ["hierarchical","semantic","recursive","fixed"], index=0)
reranker = st.sidebar.selectbox("Reranker", ["crossencoder","cohere"], index=0)
top_k = st.sidebar.slider("Top-k", 3, 10, value=5)
show_scores = st.sidebar.checkbox("Show RAGAS scores", value=True)

# Dynamic Settings Override
os.environ["RERANKER"] = reranker
settings.top_k = top_k
settings.reranker = reranker

SAMPLE_CLINICAL_TEXT = """
Metformin / Renal Dosing Adjustments. The maximum metformin dose for a patient with eGFR 30-45 mL/min/1.73 m2 (stage 3b CKD) is 500mg twice daily; do not exceed 1000mg/day.
Lisinopril is an ACE inhibitor used commonly for hypertension and heart failure.
"""

@st.cache_resource(show_spinner=False)
def get_chain(strat, rerank_name):
    # Initializes the chain completely dynamically
    chain = ClinicalRAGChain(strategy=strat)
    # Pre-load demo data so the app works instantly out-of-the-box
    chain.index([{"source": "Default_Demo_Data.txt", "section": "General Dosing", "page": 1, "text": SAMPLE_CLINICAL_TEXT}])
    return chain

chain = get_chain(strategy, reranker)

# 1. DYNAMIC DATA INGESTION
st.markdown("### 1. Upload Clinical Knowledge")
st.info("💡 The database is pre-loaded with a small demo sample for the Metformin query. You can upload a real clinical PDF below to expand the AI's knowledge!")
import tempfile
from ingestion.loader import load_pdf

uploaded_file = st.file_uploader("Upload a Clinical Guideline (PDF)", type="pdf")
if uploaded_file is not None:
    if st.button("Process & Index PDF"):
        with st.spinner("Reading PDF and extracting sections..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            sections = load_pdf(tmp_path)
            
        with st.spinner(f"Chunking and Indexing {len(sections)} sections..."):
            chain.index(sections)
            os.remove(tmp_path)
            
        st.success(f"Successfully indexed '{uploaded_file.name}'!")

st.divider()

# 2. DYNAMIC QUESTION ANSWERING
st.markdown("### 2. Ask the Assistant")
query = st.text_input("Ask a clinical question (e.g. max metformin dose in stage 3 CKD)")
if st.button("Submit Query") and query:
    with st.spinner("Searching indexed monographs..."):
        res = chain.answer(query)
        st.success(res["answer"])
        
        c = res["confidence"]
        color = "green" if c == "high" else "orange" if c == "medium" else "red"
        st.markdown(f"**Confidence:** :{color}[{c.upper()}]")
        
        with st.expander("Sources"):
            for s in res["sources"]:
                st.markdown(f"**{s['document']} - {s['section']}**")
                score_val = min(max(float(s['rerank_score']), 0.0), 1.0)
                st.progress(score_val, text=f"Score: {s['rerank_score']:.2f}")
                st.caption(s["text"])

if show_scores:
    score_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "results", f"{strategy}_scores.json")
    if os.path.exists(score_path):
        with open(score_path) as f:
            st.dataframe([json.load(f)])
    else:
        st.info(f"No RAGAS evaluation scores available for {strategy} strategy yet.")

st.caption("For research use only. Not a substitute for clinical judgment.")
