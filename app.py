"""
app.py — Streamlit UI for YourWords.

Run locally:   streamlit run app.py
Deploy free on Streamlit Community Cloud, then set LLM_API_KEY (and optionally
LLM_BASE_URL / LLM_MODEL) in the app's Secrets.
"""
from __future__ import annotations

import difflib
import html
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from pipeline import WritingAssistancePipeline
from retriever import GrammarRuleRetriever

load_dotenv()

HERE = Path(__file__).parent
INDEX_PATH = HERE / "grammar_rules.index"
META_PATH = HERE / "grammar_rules_meta.json"

st.set_page_config(page_title="YourWords — AI Grammar Assistant", page_icon="✍️")


@st.cache_resource(show_spinner="Loading grammar rules...")
def load_pipeline() -> WritingAssistancePipeline:
    retriever = None
    if INDEX_PATH.exists() and META_PATH.exists():
        retriever = GrammarRuleRetriever(index_path=INDEX_PATH, meta_path=META_PATH)
    return WritingAssistancePipeline(retriever=retriever)


def render_diff(original: str, corrected: str) -> str:
    """Word-level before/after diff as highlighted HTML."""
    before, after = original.split(), corrected.split()
    matcher = difflib.SequenceMatcher(a=before, b=after)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(" ".join(before[i1:i2])))
            continue
        if i2 > i1:  # removed / replaced words
            parts.append(
                '<span style="background:#fadbd8;color:#922;'
                'text-decoration:line-through;padding:0 2px;">'
                f'{html.escape(" ".join(before[i1:i2]))}</span>'
            )
        if j2 > j1:  # added / replacement words
            parts.append(
                '<span style="background:#d5f5e3;color:#1e7e34;padding:0 2px;">'
                f'{html.escape(" ".join(after[j1:j2]))}</span>'
            )
    return " ".join(p for p in parts if p)


st.title("✍️ YourWords")
st.caption("A retrieval-augmented grammar assistant — fixes spelling and grammar while keeping your voice.")

text = st.text_area("Your text", height=180, placeholder="Paste a paragraph to check...")

if st.button("Check writing", type="primary"):
    if not text.strip():
        st.warning("Enter some text first.")
        st.stop()

    try:
        pipeline = load_pipeline()
        with st.spinner("Checking..."):
            result = pipeline.process_text(text)
    except Exception as exc:
        st.exception(exc)
        #st.error(f"Something went wrong: {exc}")
        st.stop()

    st.subheader("Corrected text")
    st.write(result["final_text"])

    if result["final_text"].strip() != text.strip():
        st.subheader("What changed")
        st.markdown(render_diff(text, result["final_text"]), unsafe_allow_html=True)
    else:
        st.info("No changes needed — looks clean.")

    with st.expander("Spelling issues"):
        st.write(result["spelling_issues"])
    with st.expander("Grammar suggestions"):
        st.write(result["grammar_suggestions"])
    if result["retrieved_rules"]:
        with st.expander("Rules the grammar check retrieved (RAG)"):
            for rule in result["retrieved_rules"]:
                st.write(f"- {rule}")
