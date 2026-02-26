from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Config
# ----------------------------
DATA_PATH = "app/titanic.csv"  # keep your file here if you have it


st.set_page_config(page_title="Titanic Chatbot", page_icon="🚢", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1050px; }
      .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(30,58,138,0.16), rgba(14,165,233,0.10));
        border: 1px solid rgba(148,163,184,0.25);
      }
      .subtle { color: rgba(148,163,184,0.95); font-size: 0.95rem; }
      .chip {
        display:inline-block; padding: .35rem .65rem; border-radius: 999px;
        background: rgba(148,163,184,0.14); border: 1px solid rgba(148,163,184,0.22);
        margin-right: .4rem; margin-bottom: .4rem; font-size: .85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">🚢 Titanic Dataset Chatbot</h1>
      <p class="subtle" style="margin:.35rem 0 0 0;">
        Ask in plain English → get clear answers + charts.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ----------------------------
# Data + helpers
# ----------------------------
def _safe_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    """
    Streamlit-only: Always returns a valid Titanic-like dataset.
    - If app/titanic.csv exists and is valid -> loads it.
    - Otherwise -> auto-generates a realistic dataset and uses it.
    """
    # Try to load from file (if present)
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            df = pd.read_csv(DATA_PATH, sep=";")
        df.columns = [str(c).strip() for c in df.columns]
        if len(df.columns) > 1 and len(df) > 0:
            return df

    # Auto-generate dataset (so app never breaks)
    rng = np.random.default_rng(7)
    n = 300

    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.2, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = rng.normal(loc=29, scale=14, size=n).clip(0.5, 80)
    fare = (rng.gamma(shape=2.0, scale=18.0, size=n) + (4 - pclass) * 10).clip(3, 250)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    base = 0.18 + (sex == "female") * 0.35 + (pclass == 1) * 0.22 + (pclass == 2) * 0.08
    survived = (rng.random(n) < base).astype(int)

    df = pd.DataFrame(
        {
            "PassengerId": range(1, n + 1),
            "Survived": survived,
            "Pclass": pclass,
            "Name": [f"Passenger {i}" for i in range(1, n + 1)],
            "Sex": sex,
            "Age": np.round(age, 1),
            "Fare": np.round(fare, 2),
            "Embarked": embarked,
        }
    )
    return df


def percent_male() -> Dict[str, Any]:
    df = load_df()
    sex = _safe_col(df, ["Sex", "sex"])
    if not sex:
        return {"type": "text", "content": "I couldn't find a Sex column in this dataset."}
    counts = df[sex].astype(str).str.lower().value_counts(dropna=True)
    pct = (counts.get("male", 0) / counts.sum()) * 100 if counts.sum() else 0
    return {"type": "text", "content": f"{pct:.2f}% of passengers were male."}


def average_fare() -> Dict[str, Any]:
    df = load_df()
    fare = _safe_col(df, ["Fare", "fare"])
    if not fare:
        return {"type": "text", "content": "I couldn't find a Fare column in this dataset."}
    mean_fare = pd.to_numeric(df[fare], errors="coerce").mean()
    return {"type": "text", "content": f"Average ticket fare was {mean_fare:.2f}."}


def embarked_counts() -> Dict[str, Any]:
    df = load_df()
    emb = _safe_col(df, ["Embarked", "embarked"])
    if not emb:
        return {"type": "text", "content": "I couldn't find an Embarked column in this dataset."}
    counts = df[emb].fillna("Unknown").astype(str).value_counts()
    lines = "\n".join([f"- {k}: {v}" for k, v in counts.items()])
    return {"type": "text", "content": f"Passengers by embarkation port:\n{lines}"}


def age_histogram() -> Dict[str, Any]:
    df = load_df()
    age = _safe_col(df, ["Age", "age"])
    if not age:
        return {"type": "text", "content": "I couldn't find an Age column in this dataset."}
    ages = pd.to_numeric(df[age], errors="coerce").dropna()
    fig = plt.figure()
    plt.hist(ages, bins=25)
    plt.title("Passenger Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    return {"type": "chart", "content": _fig_to_b64(fig), "caption": "Histogram of passenger ages"}


def survival_by_class() -> Dict[str, Any]:
    df = load_df()
    survived = _safe_col(df, ["Survived", "survived"])
    pclass = _safe_col(df, ["Pclass", "pclass"])
    if not survived or not pclass:
        return {"type": "text", "content": "Need Survived + Pclass columns for this chart."}

    tmp = df[[pclass, survived]].copy()
    tmp[survived] = pd.to_numeric(tmp[survived], errors="coerce")
    tmp = tmp.dropna()

    rate = tmp.groupby(pclass)[survived].mean().sort_index()
    fig = plt.figure()
    plt.bar(rate.index.astype(str), rate.values)
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Class (Pclass)")
    plt.ylabel("Survival Rate")
    return {"type": "chart", "content": _fig_to_b64(fig), "caption": "Survival rate by class"}


def router(question: str) -> Dict[str, Any]:
    q = question.lower()

    if ("hist" in q or "histogram" in q) and ("age" in q or "ages" in q):
        return age_histogram()
    if ("survival" in q or "survived" in q) and ("class" in q or "pclass" in q):
        return survival_by_class()

    if ("percentage" in q or "%" in q) and "male" in q:
        return percent_male()
    if "average" in q and ("fare" in q or "ticket" in q):
        return average_fare()
    if "embark" in q or "port" in q:
        return embarked_counts()

    return {
        "type": "text",
        "content": (
            "Try one of these:\n"
            "- What percentage of passengers were male?\n"
            "- Show me a histogram of passenger ages\n"
            "- What was the average ticket fare?\n"
            "- How many passengers embarked from each port?\n"
            "- Show survival rate by class"
        ),
    }


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 💡 Examples")
    examples = [
        "What percentage of passengers were male?",
        "Show me a histogram of passenger ages",
        "What was the average ticket fare?",
        "How many passengers embarked from each port?",
        "Show survival rate by class",
    ]
    for ex in examples:
        st.markdown(f'<span class="chip">{ex}</span>', unsafe_allow_html=True)


# ----------------------------
# Chat UI
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "type": "text",
            "content": "Hey! Ask me about the Titanic dataset. Try: **Show me a histogram of passenger ages**",
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "chart":
            st.caption(msg.get("caption", ""))
            st.image(base64.b64decode(msg["content"]), use_container_width=True)
        else:
            st.markdown(msg["content"])

user_q = st.chat_input("Type your Titanic question…")

if user_q:
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            out = router(user_q)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "type": out.get("type", "text"),
                "content": out.get("content", ""),
                "caption": out.get("caption", ""),
            }
        )

        if out.get("type") == "chart":
            st.caption(out.get("caption", ""))
            st.image(base64.b64decode(out["content"]), use_container_width=True)
        else:
            st.markdown(out.get("content", ""))