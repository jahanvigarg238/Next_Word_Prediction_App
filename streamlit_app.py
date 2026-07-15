"""
Next-word prediction demo — self-contained Streamlit app.

No separate backend: models are loaded directly into the Streamlit
process. This trades away the "swap frontends without retraining"
flexibility of the FastAPI version for a single deployable file, which
is what you want for Streamlit Community Cloud (push repo -> live app,
no server to manage).

Run locally:  streamlit run streamlit_app.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_pipeline import load_tokenizer  # noqa: E402
from generate import generate_text, predict_topk  # noqa: E402
from inference import SavedModelPredictor  # noqa: E402

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_LABELS = {
    "bilstm": "BiLSTM",
    "bigru_attention": "BiGRU + Attention (from scratch)",
    "mini_transformer": "Mini Transformer (from scratch)",
}

st.set_page_config(page_title="Next-Word Prediction | Architecture Comparison", page_icon="📝", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit reruns the whole script on every interaction,
# so anything expensive (loading models) must be cached.
#
# Note: models are loaded from artifacts/{name}_savedmodel/, a frozen
# TensorFlow SavedModel export -- NOT the .keras files. This is
# deliberate: SavedModel loading doesn't require reconstructing the
# custom AdditiveAttention / TransformerBlock Python classes, which is
# exactly the step that breaks when TensorFlow/Keras versions differ
# between the machine that trained the model and the machine running
# this app. See README "Why SavedModel" for the full explanation.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    meta_path = os.path.join(ARTIFACTS_DIR, "meta.json")
    if not os.path.exists(meta_path):
        return None, None, {}, {}

    with open(meta_path) as f:
        meta = json.load(f)
    tokenizer = load_tokenizer(os.path.join(ARTIFACTS_DIR, "tokenizer.pickle"))

    models, comparison = {}, {}
    for name in MODEL_LABELS:
        path = os.path.join(ARTIFACTS_DIR, f"{name}_savedmodel")
        if os.path.isdir(path):
            models[name] = SavedModelPredictor(path)

    cmp_path = os.path.join(ARTIFACTS_DIR, "comparison.json")
    if os.path.exists(cmp_path):
        with open(cmp_path) as f:
            comparison = {row["model"]: row for row in json.load(f)}

    return tokenizer, meta, models, comparison


@st.cache_data(show_spinner=False)
def get_embedding_projection(model_name: str, top_n: int = 300):
    """PCA-project the saved embedding weight matrix down to 2D, for the
    most frequent `top_n` vocabulary words, so we can visualize whether
    the model learned any semantic/syntactic clustering. Loaded from a
    plain .npy file saved during training -- independent of the model
    format, so this always works regardless of the inference backend.
    """
    emb_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_embedding.npy")
    weights = np.load(emb_path)
    n = min(top_n, weights.shape[0] - 1)
    coords = PCA(n_components=2).fit_transform(weights[1 : n + 1])
    return coords  # index i corresponds to tokenizer word_index value i+1


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
tokenizer, meta, models, comparison = load_artifacts()

st.title("📝 Next-Word Prediction: A From-Scratch Architecture Comparison")
st.caption(
    "BiLSTM vs. BiGRU with hand-built additive attention vs. a from-scratch mini-Transformer — "
    "trained on Shakespeare's *Hamlet* with a leak-free train/val/test split, compared on "
    "perplexity and top-k accuracy rather than accuracy alone."
)

if not models:
    st.warning(
        "No trained models found in `artifacts/`. Run `python src/train.py` first "
        "(see README) — training needs a GPU/Colab for reasonable speed, so this app "
        "expects pre-trained `.keras` files checked into `artifacts/` before deployment."
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    available = list(models.keys())
    model_choice = st.selectbox("Active model", available, format_func=lambda m: MODEL_LABELS[m])
    st.divider()
    if comparison:
        st.subheader("Held-out test performance")
        df_cmp = pd.DataFrame(comparison.values())
        df_cmp["model"] = df_cmp["model"].map(MODEL_LABELS)
        st.dataframe(
            df_cmp.set_index("model")[["perplexity", "top1_acc", "top5_acc"]], use_container_width=True
        )
    st.caption(
        "Not a pretrained LLM — a controlled comparison of three architectures "
        "trained from scratch under identical conditions."
    )

tab_predict, tab_generate, tab_compare, tab_insights = st.tabs(
    ["🔮 Predict", "✍️ Generate", "⚖️ Compare models", "🔬 Model insights"]
)

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    col_input, col_result = st.columns([1, 1.3])
    with col_input:
        text = st.text_input("Enter a sequence of words", "To be or not to", key="predict_text")
        k = st.slider("Show top-k predictions", 1, 10, 5)
        predict_clicked = st.button("Predict next word", type="primary")

    with col_result:
        if predict_clicked and text.strip():
            preds = predict_topk(models[model_choice], tokenizer, text, meta["max_sequence_len"], k=k)
            df = pd.DataFrame(preds, columns=["word", "probability"]).sort_values("probability")
            fig = px.bar(
                df,
                x="probability",
                y="word",
                orientation="h",
                color="probability",
                color_continuous_scale="Blues",
                title=f"Top-{k} predictions — {MODEL_LABELS[model_choice]}",
            )
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            st.metric(
                "Most likely next word", df.iloc[-1]["word"], f"{df.iloc[-1]['probability']:.1%} confidence"
            )

# ---------------------------------------------------------------------------
# Tab 2: Generate
# ---------------------------------------------------------------------------
with tab_generate:
    seed = st.text_input("Seed text", "To be or not to be", key="gen_seed")
    c1, c2, c3 = st.columns(3)
    n_words = c1.slider("Words to generate", 1, 50, 15)
    temperature = c2.slider(
        "Temperature", 0.1, 2.0, 0.8, help="Lower = safer/more repetitive, higher = more random"
    )
    top_k_sample = c3.slider("Top-k sampling", 0, 50, 10, help="0 disables filtering")

    if st.button("Generate text", type="primary"):
        with st.spinner("Generating..."):
            generated = generate_text(
                models[model_choice],
                tokenizer,
                seed,
                meta["max_sequence_len"],
                n_words=n_words,
                temperature=temperature,
                top_k=top_k_sample,
            )
        st.markdown(f"> {generated}")

# ---------------------------------------------------------------------------
# Tab 3: Compare models side by side
# ---------------------------------------------------------------------------
with tab_compare:
    text_cmp = st.text_input("Enter a sequence of words", "To be or not to", key="cmp_text")
    if st.button("Compare all models", type="primary") and text_cmp.strip():
        cols = st.columns(len(models))
        for col, name in zip(cols, models):
            with col:
                st.subheader(MODEL_LABELS[name])
                preds = predict_topk(models[name], tokenizer, text_cmp, meta["max_sequence_len"], k=5)
                df = pd.DataFrame(preds, columns=["word", "probability"]).sort_values("probability")
                fig = px.bar(df, x="probability", y="word", orientation="h", height=280)
                fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Model insights — embedding space + metrics
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Learned embedding space (PCA projection)")
    st.caption(
        "Each point is a word's learned embedding vector, projected from its full "
        "dimensionality down to 2D. Words the model uses similarly tend to cluster."
    )

    coords = get_embedding_projection(model_choice)
    index_to_word = {v: w for w, v in tokenizer.word_index.items()}
    words = [index_to_word.get(i + 1, "") for i in range(len(coords))]

    fig = go.Figure(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=words,
            textposition="top center",
            marker=dict(size=6, color=coords[:, 0], colorscale="Viridis"),
            textfont=dict(size=9),
        )
    )
    fig.update_layout(height=600, title=f"Top-300 vocabulary — {MODEL_LABELS[model_choice]}")
    st.plotly_chart(fig, use_container_width=True)

    if comparison:
        st.subheader("Full comparison table")
        st.dataframe(pd.DataFrame(comparison.values()), use_container_width=True)

st.divider()
st.caption(
    "Source & methodology: see README.md — leak-free split, from-scratch attention/"
    "Transformer implementation, perplexity + top-k evaluation on a held-out test set."
)
