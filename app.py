"""Production Streamlit interface for the AI versus human text detector."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Verity | AI Text Detector",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).with_name("ai_detector_pipeline.pkl")


def inject_styles() -> None:
    """Add a theme-aware visual system without changing Streamlit's controls."""
    st.markdown(
        """
        <style>
        :root {
            --verity-ink: #e9eef5;
            --verity-muted: #94a3b8;
            --verity-panel: rgba(15, 23, 42, 0.72);
            --verity-panel-light: rgba(248, 250, 252, 0.88);
            --verity-border: rgba(148, 163, 184, 0.22);
            --verity-accent: #f43f5e;
            --verity-teal: #14b8a6;
        }
        .stApp { background: radial-gradient(circle at 10% 0%, rgba(244,63,94,.12), transparent 32rem), var(--background-color); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { border-right: 1px solid var(--verity-border); }
        [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem; }
        .verity-brand { display:flex; align-items:center; gap:.75rem; margin-bottom:2.2rem; }
        .verity-logo { display:grid; place-items:center; width:2.4rem; height:2.4rem; border-radius:12px; color:#fff; background:linear-gradient(135deg,#f43f5e,#fb7185); box-shadow:0 8px 24px rgba(244,63,94,.32); font-size:1.35rem; }
        .verity-brand strong { font-size:1.15rem; letter-spacing:.02em; }
        .verity-brand small, .eyebrow { color:var(--verity-muted); text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; }
        .hero { padding:1.5rem 0 1rem; }
        .eyebrow { color:#fb7185; font-weight:700; margin-bottom:.8rem; }
        .hero h1 { font-size:clamp(2.1rem, 4vw, 4rem); line-height:1.02; margin:0; letter-spacing:-.04em; }
        .hero p { color:var(--verity-muted); max-width:42rem; font-size:1.05rem; margin-top:1rem; }
        .panel { background:var(--verity-panel); border:1px solid var(--verity-border); border-radius:18px; padding:1.35rem; box-shadow:0 16px 50px rgba(2,6,23,.12); }
        @media (prefers-color-scheme: light) { .panel { background:var(--verity-panel-light); } }
        .panel-title { font-size:.78rem; font-weight:700; color:var(--verity-muted); letter-spacing:.12em; text-transform:uppercase; margin-bottom:.8rem; }
        .status-pill { display:inline-flex; align-items:center; gap:.45rem; border-radius:999px; padding:.35rem .7rem; font-size:.75rem; font-weight:700; background:rgba(20,184,166,.12); color:#2dd4bf; }
        .status-dot { width:.42rem; height:.42rem; border-radius:50%; background:currentColor; box-shadow:0 0 12px currentColor; }
        .result-card { border-radius:18px; padding:1.5rem; border:1px solid; margin:1rem 0; transition:transform .2s ease, box-shadow .2s ease; }
        .result-card:hover { transform:translateY(-2px); }
        .result-card.ai { background:linear-gradient(135deg,rgba(244,63,94,.15),rgba(244,63,94,.03)); border-color:rgba(244,63,94,.55); box-shadow:0 0 30px rgba(244,63,94,.14); }
        .result-card.human { background:linear-gradient(135deg,rgba(20,184,166,.15),rgba(20,184,166,.03)); border-color:rgba(20,184,166,.55); box-shadow:0 0 30px rgba(20,184,166,.14); }
        .result-label { color:var(--verity-muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; }
        .result-value { font-size:1.8rem; font-weight:800; margin-top:.45rem; }
        .ai .result-value { color:#fb7185; } .human .result-value { color:#2dd4bf; }
        div[data-testid="stMetric"] { background:var(--verity-panel); border:1px solid var(--verity-border); border-radius:14px; padding:1rem; transition:border-color .2s ease, transform .2s ease; }
        div[data-testid="stMetric"]:hover { border-color:rgba(244,63,94,.5); transform:translateY(-2px); }
        div[data-testid="stMetricLabel"] { color:var(--verity-muted); }
        .helper { color:var(--verity-muted); font-size:.82rem; margin-top:.5rem; }
        .stButton > button { border-radius:10px; transition:all .2s ease; }
        .stButton > button:hover { transform:translateY(-1px); border-color:#fb7185; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def extract_features(text: str) -> dict[str, object]:
    """Build the raw text and linguistic features expected by the pipeline."""
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    return {
        "text": text,
        "word_count": word_count,
        "lexical_diversity": len({word.lower() for word in words}) / word_count if word_count else 0.0,
        "avg_word_len": float(np.mean([len(word) for word in words])) if words else 0.0,
        "punct_count": sum(character in '.,!?;:"()' for character in text),
    }


@st.cache_resource
def load_model():
    """Load the fitted pipeline once and retain it across Streamlit reruns."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    try:
        from train_model import LinguisticFeatureExtractor

        sys.modules["__main__"].LinguisticFeatureExtractor = LinguisticFeatureExtractor
    except ImportError:
        pass
    return joblib.load(MODEL_PATH)


def is_ai_prediction(prediction: object) -> bool:
    """Interpret the dataset's numeric or textual positive class labels."""
    normalized = str(prediction).strip().lower()
    return normalized in {"1", "1.0", "ai", "generated", "true", "yes"}


def prediction_confidence(model, input_data: pd.DataFrame, prediction: object) -> float | None:
    """Return the probability for the predicted class when supported."""
    probabilities = get_probability_distribution(model, input_data)
    if probabilities is None:
        return None
    return probabilities["ai"] if is_ai_prediction(prediction) else probabilities["human"]


def get_probability_distribution(model, input_data: pd.DataFrame) -> dict[str, float] | None:
    """Normalize model probabilities into the two product-facing classes."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = np.asarray(model.predict_proba(input_data)[0], dtype=float)
        classes = getattr(model, "classes_", None)
        if classes is None or len(classes) != len(probabilities):
            return None
        ai_probability = sum(
            probability for probability, model_class in zip(probabilities, classes)
            if is_ai_prediction(model_class)
        )
        human_probability = max(0.0, 1.0 - ai_probability)
        return {"ai": float(ai_probability), "human": float(human_probability)}
    except Exception:
        return None


def render_probability_chart(probabilities: dict[str, float]) -> None:
    """Render a compact, theme-friendly donut chart."""
    figure = go.Figure(
        go.Pie(
            labels=["AI generated", "Human written"],
            values=[probabilities["ai"], probabilities["human"]],
            hole=0.72,
            marker={"colors": ["#f43f5e", "#14b8a6"], "line": {"color": "rgba(255,255,255,.12)", "width": 2}},
            textinfo="none",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    figure.update_layout(
        height=280,
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        showlegend=True,
        legend={"orientation": "h", "y": -0.04, "x": 0.5, "xanchor": "center"},
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[{"text": f"{max(probabilities.values()):.0%}<br><span style='font-size:12px'>confidence</span>", "showarrow": False}],
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="verity-brand"><div class="verity-logo">◈</div><div><strong>VERITY</strong><br><small>Text intelligence</small></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="status-pill"><span class="status-dot"></span>Model online</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("#### Model specs")
        st.caption("Accuracy  ·  98.17%")
        st.caption("Training corpus  ·  500k+ texts")
        st.caption("Pipeline  ·  Linguistic + n-gram")
        st.divider()
        st.markdown("#### Try a sample")
        samples = {
            "Professional note": "The quarterly report outlines measurable improvements in customer retention and operational efficiency.",
            "Creative passage": "Moonlight spilled across the quiet station, turning every empty window into a small silver story.",
        }
        for label, sample in samples.items():
            if st.button(label, use_container_width=True):
                st.session_state.input_text = sample
        st.caption("Paste any passage in the workspace to inspect its linguistic signals.")


def render_metrics(features: dict[str, object]) -> None:
    columns = st.columns(4)
    metrics = [
        ("Total word count", f"{features['word_count']:,}"),
        ("Lexical diversity", f"{float(features['lexical_diversity']):.1%}"),
        ("Average word length", f"{float(features['avg_word_len']):.1f}"),
        ("Punctuation count", f"{features['punct_count']:,}"),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            st.metric(label, value)


def main() -> None:
    inject_styles()
    render_sidebar()
    st.markdown('<div class="hero"><div class="eyebrow">Signal intelligence / 01</div><h1>Know what you\'re reading.</h1><p>Analyze the linguistic fingerprint of any passage with a fast, transparent AI detection model.</p></div>', unsafe_allow_html=True)

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Model artifact not found. Run `train_model.py` first to create `ai_detector_pipeline.pkl`.")
        st.stop()
    except Exception as error:
        st.error(f"Model loading failed: {error}")
        st.stop()

    st.markdown('<div class="panel"><div class="panel-title">Text workspace</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        "Text to analyze",
        key="input_text",
        height=220,
        label_visibility="collapsed",
        placeholder="Paste an article, email, or passage here...",
    )
    action_column, hint_column = st.columns([1, 2])
    with action_column:
        analyze = st.button("Analyze passage  →", type="primary", use_container_width=True)
    with hint_column:
        st.markdown('<div class="helper">Your text is analyzed locally by the loaded pipeline.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not input_text.strip():
            st.warning("Add some text before starting the analysis.")
            return
        features = extract_features(input_text)
        input_data = pd.DataFrame([features])
        try:
            prediction = model.predict(input_data)[0]
            probabilities = get_probability_distribution(model, input_data)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

        st.divider()
        st.markdown('<div class="eyebrow">Analysis complete / 02</div>', unsafe_allow_html=True)
        result_is_ai = is_ai_prediction(prediction)
        result_class = "ai" if result_is_ai else "human"
        result_title = "Likely AI generated" if result_is_ai else "Likely human written"
        result_icon = "🤖" if result_is_ai else "👤"
        confidence = prediction_confidence(model, input_data, prediction)
        st.markdown(f'<div class="result-card {result_class}"><div class="result-label">Classification</div><div class="result-value">{result_icon} {result_title}</div><div class="helper">Model confidence: {confidence:.1%}</div></div>', unsafe_allow_html=True) if confidence is not None else st.markdown(f'<div class="result-card {result_class}"><div class="result-label">Classification</div><div class="result-value">{result_icon} {result_title}</div></div>', unsafe_allow_html=True)

        left_column, right_column = st.columns([1, 1.15])
        with left_column:
            st.markdown('<div class="panel"><div class="panel-title">Probability distribution</div>', unsafe_allow_html=True)
            if probabilities is not None:
                render_probability_chart(probabilities)
            else:
                st.info("This model does not expose probability scores.")
            st.markdown("</div>", unsafe_allow_html=True)
        with right_column:
            st.markdown('<div class="panel"><div class="panel-title">Text feature analytics</div>', unsafe_allow_html=True)
            render_metrics(features)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
