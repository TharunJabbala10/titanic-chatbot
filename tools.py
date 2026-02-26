from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = "app/titanic.csv"


def _safe_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    """
    Always returns a valid Titanic-like dataset.
    - If app/titanic.csv exists and is valid -> loads it.
    - Otherwise -> auto-generates a realistic dataset (300 rows) and saves it.
    """
    import numpy as np

    # Use existing CSV if present & non-empty
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
        # Try comma CSV, then semicolon CSV
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            df = pd.read_csv(DATA_PATH, sep=";")

        df.columns = [str(c).strip() for c in df.columns]
        if len(df.columns) > 1 and len(df) > 0:
            return df

    # Otherwise: generate demo Titanic-like data
    rng = np.random.default_rng(7)
    n = 300

    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.2, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = rng.normal(loc=29, scale=14, size=n).clip(0.5, 80)
    fare = (rng.gamma(shape=2.0, scale=18.0, size=n) + (4 - pclass) * 10).clip(3, 250)
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    # Survival probability (roughly higher for females + higher class)
    base = 0.18 + (sex == "female") * 0.35 + (pclass == 1) * 0.22 + (pclass == 2) * 0.08
    survived = (rng.random(n) < base).astype(int)

    df = pd.DataFrame(
        {
            "PassengerId": range(1, n + 1),
            "Survived": survived,
            "Pclass": pclass,
            "Name": [f"Passenger {i}" for i in range(1, n + 1)],
            "Sex": sex,
            "Age": age.round(1),
            "Fare": fare.round(2),
            "Embarked": embarked,
        }
    )

    # Save it so next run is stable
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    return df


# ---------- Text responses ----------

def dataset_columns() -> Dict[str, Any]:
    df = load_df()
    cols = ", ".join(df.columns.tolist())
    return {"type": "text", "content": f"Dataset columns:\n{cols}"}


def percent_male() -> Dict[str, Any]:
    df = load_df()
    sex = _safe_col(df, ["Sex", "sex"])
    if not sex:
        return {"type": "text", "content": f"'Sex' column not found. Columns: {df.columns.tolist()}"}

    counts = df[sex].astype(str).str.lower().value_counts(dropna=True)
    if counts.sum() == 0 or "male" not in counts:
        return {"type": "text", "content": "Couldn't compute male percentage (no 'male' values found)."}

    pct = (counts["male"] / counts.sum()) * 100
    return {"type": "text", "content": f"{pct:.2f}% of passengers were male."}


def average_fare() -> Dict[str, Any]:
    df = load_df()
    fare = _safe_col(df, ["Fare", "fare"])
    if not fare:
        return {"type": "text", "content": f"'Fare' column not found. Columns: {df.columns.tolist()}"}

    mean_fare = pd.to_numeric(df[fare], errors="coerce").mean()
    return {"type": "text", "content": f"Average ticket fare was {mean_fare:.2f}."}


def embarked_counts() -> Dict[str, Any]:
    df = load_df()
    emb = _safe_col(df, ["Embarked", "embarked"])
    if not emb:
        return {"type": "text", "content": f"'Embarked' column not found. Columns: {df.columns.tolist()}"}

    counts = df[emb].fillna("Unknown").astype(str).value_counts()
    lines = "\n".join([f"- {k}: {v}" for k, v in counts.items()])
    return {"type": "text", "content": f"Passengers by embarkation port:\n{lines}"}


# ---------- Charts ----------

def age_histogram() -> Dict[str, Any]:
    df = load_df()
    age = _safe_col(df, ["Age", "age"])
    if not age:
        return {"type": "text", "content": f"'Age' column not found. Columns: {df.columns.tolist()}"}

    ages = pd.to_numeric(df[age], errors="coerce").dropna()
    if ages.empty:
        return {"type": "text", "content": "No valid age values found to plot."}

    fig = plt.figure()
    plt.hist(ages, bins=25)
    plt.title("Passenger Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")

    return {"type": "chart", "content": _fig_to_base64(fig), "caption": "Histogram of passenger ages"}


def survival_by_class() -> Dict[str, Any]:
    df = load_df()
    survived = _safe_col(df, ["Survived", "survived"])
    pclass = _safe_col(df, ["Pclass", "pclass"])
    if not survived or not pclass:
        return {"type": "text", "content": f"Need 'Survived' + 'Pclass'. Columns: {df.columns.tolist()}"}

    tmp = df[[pclass, survived]].copy()
    tmp[survived] = pd.to_numeric(tmp[survived], errors="coerce")
    tmp = tmp.dropna()

    rate = tmp.groupby(pclass)[survived].mean().sort_index()

    fig = plt.figure()
    plt.bar(rate.index.astype(str), rate.values)
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Class (Pclass)")
    plt.ylabel("Survival Rate")

    return {"type": "chart", "content": _fig_to_base64(fig), "caption": "Survival rate by class"}