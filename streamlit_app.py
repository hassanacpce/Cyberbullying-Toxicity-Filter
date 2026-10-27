"""Streamlit UI for the Cyberbullying & Toxicity Filter.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud -> main file path = streamlit_app.py
Optional LLM:  add ANTHROPIC_API_KEY to .streamlit/secrets.toml (local) or the
               app's Secrets box (cloud). Without it the app runs fully offline.
"""

import html
import os

import pandas as pd
import streamlit as st

# Put the API key from Streamlit secrets into the environment BEFORE importing the filter
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:  # no secrets file -> run offline
    pass

import toxicity_filter as tf  # noqa: E402

st.set_page_config(page_title="Cyberbullying & Toxicity Filter", page_icon="🛡️", layout="wide")

COLORS = {"safe": "#1a7f37", "mildly rude": "#b8860b", "toxic": "#cf222e"}


@st.cache_resource(show_spinner="Loading model...")
def get_bundle():
    try:
        return tf.load_bundle()
    except Exception:  # e.g. scikit-learn version mismatch on the server -> retrain
        return tf.train(pd.read_csv(tf.SAMPLE_DATA))


def highlight_html(pieces):
    out = []
    for text, hit in pieces:
        t = html.escape(text)
        out.append(
            f'<mark style="background:#ffd33d80;border-radius:4px;padding:0 2px">{t}</mark>' if hit else t
        )
    return "".join(out)


bundle = get_bundle()
llm_available = bool(os.getenv("ANTHROPIC_API_KEY"))

st.title("🛡️ Cyberbullying & Toxicity Filter")
st.caption("Classifies comments as safe, mildly rude or toxic, with a severity score, highlighted "
           "words, an explanation and a suggested moderation action.")

with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("LLM second opinion when unsure", value=llm_available, disabled=not llm_available)
    st.markdown("**Pipeline:** normalise → tokenize → stop-words → stem → TF-IDF (word + char) "
                "→ Logistic Regression → rules → severity → explanation")

tab1, tab2 = st.tabs(["Single comment", "Batch (CSV)"])

with tab1:
    examples = {
        "Pick an example...": "",
        "Friendly": "Great job on the presentation, really enjoyed it!",
        "Mild insult": "You are so stupid, stop talking.",
        "Obfuscated": "u r an id10t and a l0ser",
        "Threat": "I know where you live, you will pay for this.",
        "Negated (safe)": "You are not stupid, keep going!",
    }
    choice = st.selectbox("Try an example", list(examples))
    text = st.text_area("Comment", value=examples[choice], height=120, placeholder="Type or paste a comment...")

    if st.button("Analyze", type="primary", disabled=not text.strip()):
        with st.spinner("Analyzing..."):
            r = tf.analyze(text, bundle, use_llm=use_llm)

        color = COLORS[r["category"]]
        st.markdown(f"### Result: <span style='color:{color}'>{r['category'].upper()}</span>",
                    unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Severity (0-100)", r["severity_score"])
        c2.metric("Toxicity probability", f"{r['toxicity_probability']:.0%}")
        c3.metric("Model confidence", f"{r['confidence']:.0%}")
        c4.metric("Decided by", "LLM" if r["source"] == "llm" else "ML + rules")
        st.progress(min(int(r["severity_score"]), 100) / 100)

        st.markdown("**Comment with flagged words**")
        st.markdown(f"<div style='padding:12px;border:1px solid #8884;border-radius:8px;font-size:1.05rem'>"
                    f"{highlight_html(r['highlighted'])}</div>", unsafe_allow_html=True)
        st.markdown(f"**Why:** {r['explanation']}")
        st.markdown(f"**Suggested action:** {r['action']}")
        with st.expander("Class probabilities"):
            st.bar_chart(pd.Series(r["probabilities"]))

with tab2:
    st.write("Upload a CSV with a **text** column (or choose the column below). "
             "The LLM is not used for batches, to keep it fast and free.")
    f = st.file_uploader("CSV file", type="csv")
    if f:
        df = pd.read_csv(f)
        col = st.selectbox("Text column", df.columns, index=list(df.columns).index("text") if "text" in df else 0)
        limit = st.slider("Max rows", 10, 2000, min(500, len(df)))
        if st.button("Analyze file"):
            rows = []
            bar = st.progress(0.0)
            sub = df.head(limit)
            for i, t in enumerate(sub[col].astype(str)):
                r = tf.analyze(t, bundle, use_llm=False)
                rows.append({"text": t, "category": r["category"], "severity": r["severity_score"],
                             "keywords": ", ".join(r["keywords"]), "action": r["action"]})
                bar.progress((i + 1) / len(sub))
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.bar_chart(out["category"].value_counts())
            st.download_button("Download results", out.to_csv(index=False).encode(), "results.csv", "text/csv")
