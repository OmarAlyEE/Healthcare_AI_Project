import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    EXPLAINED_CLAIMS_CSV,
    FEATURE_COLUMNS,
    ISOLATION_FOREST_PATH,
    N_CLUSTERS,
    SCALER_PATH,
    SCORED_CLAIMS_CSV,
)


from plot_utils import (
    enrich_time_columns,
    has_time_columns,
    plot_anomaly_spikes,
    plot_cluster_summary,
    plot_clusters,
    plot_provider_monthly_trend,
    run_kmeans,
)

st.set_page_config(page_title="Healthcare Payment Integrity AI", layout="wide")


@st.cache_resource
def load_model():
    if not ISOLATION_FOREST_PATH.exists() or not SCALER_PATH.exists():
        return None, None
    return joblib.load(ISOLATION_FOREST_PATH), joblib.load(SCALER_PATH)


@st.cache_data
def load_default_data():
    path = EXPLAINED_CLAIMS_CSV if EXPLAINED_CLAIMS_CSV.exists() else SCORED_CLAIMS_CSV
    if not path.exists():
        return None, path.name
    df = pd.read_csv(path)
    return prepare_dataframe(df, already_scored=True), path.name


def explain_row(row: pd.Series) -> str:
    reasons = []

    if row.get("claim_vs_provider_avg", 0) > 2:
        reasons.append(
            f"Cost Anomaly: Claim is {row.get('claim_vs_provider_avg', 0):.1f}x higher than provider average"
        )
    if row.get("high_cost_flag", 0) == 1:
        reasons.append("Cost Anomaly: Flagged as a statistically high-cost claim")
    if row.get("claim_difference_from_provider_avg", 0) > 10000:
        reasons.append(
            f"Cost Anomaly: Claim is ${row.get('claim_difference_from_provider_avg', 0):,.2f} above provider average"
        )
    if row.get("provider_claims_per_month", 0) > 50:
        reasons.append(
            f"Frequency Anomaly: Provider has an unusually high claim frequency "
            f"({row.get('provider_claims_per_month', 0):.0f} per month)"
        )
    if row.get("high_frequency_flag", 0) == 1:
        reasons.append("Frequency Anomaly: Provider flagged for top 5% frequency behavior")
    if row.get("provider_claim_count", 0) > 100 and row.get("provider_avg_claim", 0) > 5000:
        reasons.append("Provider Anomaly: High volume and high average cost provider")
    if row.get("risk_score", 0) >= 80:
        reasons.append(f"Model Anomaly: High Isolation Forest risk score ({row.get('risk_score', 0):.1f}/100)")
    elif row.get("anomaly_flag", 0) == 1:
        reasons.append("Model Anomaly: Claim flagged as outlier by Isolation Forest")

    return "; ".join(reasons) if reasons else "No strong anomaly indicators detected"


def apply_risk_columns(df: pd.DataFrame, raw_scores: np.ndarray, preds: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["raw_risk_score"] = raw_scores
    span = raw_scores.max() - raw_scores.min() + 1e-8
    df["risk_score"] = np.clip(100 * (1 - (raw_scores - raw_scores.min()) / span), 0, 100)
    df["risk_percentile"] = df["risk_score"].rank(pct=True)
    df["risk_category"] = "low risk"
    df.loc[df["risk_percentile"] >= 0.95, "risk_category"] = "medium risk"
    df.loc[df["risk_percentile"] >= 0.99, "risk_category"] = "high risk"
    df["anomaly_flag"] = (preds == -1).astype(int)
    df["explanation"] = df.apply(explain_row, axis=1)
    return df


def score_dataframe(df: pd.DataFrame, model, scaler) -> tuple[pd.DataFrame | None, list[str]]:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        return None, missing

    features = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = scaler.transform(features)
    preds = model.predict(scaled)
    raw_scores = model.decision_function(scaled)
    return apply_risk_columns(df, raw_scores, preds), []


def prepare_dataframe(df: pd.DataFrame, already_scored: bool = False) -> pd.DataFrame:
    df = df.copy()
    if "anomaly_flag" in df.columns:
        df["anomaly_flag"] = pd.to_numeric(df["anomaly_flag"], errors="coerce").fillna(0).astype(int)
    if "risk_category" not in df.columns and "risk_score" in df.columns:
        df["risk_percentile"] = df["risk_score"].rank(pct=True)
        df["risk_category"] = "low risk"
        df.loc[df["risk_percentile"] >= 0.95, "risk_category"] = "medium risk"
        df.loc[df["risk_percentile"] >= 0.99, "risk_category"] = "high risk"
    if "explanation" not in df.columns:
        df["explanation"] = df.apply(explain_row, axis=1)
    return df


def dataset_stats(df: pd.DataFrame) -> dict:
    return {
        "global_avg": float(df["claim_amount"].mean()),
        "claim_amount_mean": float(df["claim_amount"].mean()),
        "provider_claims_per_month_mean": float(df["provider_claims_per_month"].mean()),
        "provider_claim_count_median": float(df["provider_claim_count"].median()),
        "provider_avg_claim_median": float(df["provider_avg_claim"].median()),
        "patient_claim_count_median": float(df["patient_claim_count"].median()),
        "raw_risk_min": float(df["raw_risk_score"].min()),
        "raw_risk_max": float(df["raw_risk_score"].max()),
        "frequency_threshold": float(df["provider_claims_per_month"].quantile(0.95)),
    }


def normalize_risk_score(raw_score: float, stats: dict) -> float:
    span = stats["raw_risk_max"] - stats["raw_risk_min"] + 1e-8
    return float(np.clip(100 * (1 - (raw_score - stats["raw_risk_min"]) / span), 0, 100))


def build_feature_vector(
    claim_amount: float,
    age: int,
    claim_duration_days: int,
    provider_claims_per_month: float,
    stats: dict,
    provider_claim_count: float | None = None,
    provider_avg_claim: float | None = None,
    patient_claim_count: float | None = None,
) -> pd.DataFrame:
    global_avg = stats["global_avg"]
    provider_avg = provider_avg_claim if provider_avg_claim is not None else stats["provider_avg_claim_median"]
    provider_count = provider_claim_count if provider_claim_count is not None else stats["provider_claim_count_median"]
    patient_count = patient_claim_count if patient_claim_count is not None else stats["patient_claim_count_median"]

    claim_vs_provider_avg = claim_amount / (provider_avg + 1)
    claim_vs_global_avg = claim_amount / (global_avg + 1)
    claim_difference = claim_amount - provider_avg

    row = {
        "claim_duration_days": claim_duration_days,
        "age": age,
        "claim_amount": claim_amount,
        "claim_log_amount": np.log1p(claim_amount),
        "annual_claim_amount": claim_amount,
        "annual_claim_amount_log": np.log1p(max(claim_amount, 0)),
        "annual_claim_frequency_proxy": 1,
        "provider_claims_30d": provider_claims_per_month,
        "provider_claims_90d": provider_claims_per_month * 3,
        "provider_claim_count": provider_count,
        "provider_claim_count_log": np.log1p(provider_count),
        "provider_avg_claim": provider_avg,
        "provider_total_claim": provider_avg * provider_count,
        "claim_vs_provider_avg": claim_vs_provider_avg,
        "claim_vs_global_avg": claim_vs_global_avg,
        "claim_difference_from_provider_avg": claim_difference,
        "high_cost_flag": int(claim_vs_provider_avg > 3),
        "provider_claims_per_month": provider_claims_per_month,
        "provider_claims_per_month_log": np.log1p(provider_claims_per_month),
        "high_frequency_flag": int(provider_claims_per_month > stats["frequency_threshold"]),
        "patient_claim_count": patient_count,
        "patient_claim_count_log": np.log1p(patient_count),
        "patient_avg_claim": global_avg,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def score_features(features: pd.DataFrame, model, scaler, stats: dict) -> tuple[float, float, int]:
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    raw_score = float(model.decision_function(scaled)[0])
    risk_score = normalize_risk_score(raw_score, stats)
    anomaly_flag = int(pred == -1)
    return risk_score, raw_score, anomaly_flag


def explain_inputs(
    claim_amount: float,
    provider_claims_per_month: float,
    features: pd.Series,
    risk_score: float,
    anomaly_flag: int,
    stats: dict,
) -> list[str]:
    reasons = []

    if features["claim_vs_provider_avg"] > 2:
        reasons.append(
            f"Cost Anomaly: Claim is {features['claim_vs_provider_avg']:.1f}x higher than provider average"
        )
    if features["high_cost_flag"] == 1:
        reasons.append("Cost Anomaly: Flagged as a statistically high-cost claim")
    if features["claim_difference_from_provider_avg"] > 10000:
        reasons.append(
            f"Cost Anomaly: Claim is ${features['claim_difference_from_provider_avg']:,.2f} above provider average"
        )
    if provider_claims_per_month > 50:
        reasons.append(
            f"Frequency Anomaly: Unusually high provider frequency ({provider_claims_per_month:.0f}/month)"
        )
    if features["high_frequency_flag"] == 1:
        reasons.append("Frequency Anomaly: Provider flagged for top 5% frequency behavior")
    if claim_amount > stats["claim_amount_mean"] * 3:
        reasons.append("Cost Anomaly: Claim amount is more than 3x the dataset average")
    if provider_claims_per_month > stats["provider_claims_per_month_mean"] * 2:
        reasons.append("Frequency Anomaly: Submission frequency is more than 2x the dataset average")
    if risk_score >= 80:
        reasons.append(f"Model Anomaly: High Isolation Forest risk score ({risk_score:.1f}/100)")
    elif anomaly_flag == 1:
        reasons.append("Model Anomaly: Claim flagged as outlier by Isolation Forest")

    return reasons or ["No strong anomaly indicators detected"]


def risk_label(risk_score: float) -> str:
    if risk_score >= 80:
        return "HIGH RISK"
    if risk_score >= 50:
        return "MEDIUM RISK"
    return "LOW RISK"


def render_visual_dashboard(df: pd.DataFrame) -> None:
    plot_df = df.sample(n=min(10_000, len(df)), random_state=42)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(df["claim_amount"], bins=50, edgecolor="white")
    ax1.set_title("Claim Amount Distribution")
    ax1.set_xlabel("Claim Amount")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)
    plt.close()

    if "Provider" in df.columns:
        st.subheader("Top Risky Providers")
        provider_risk = (
            df.groupby("Provider")
            .agg(claim_count=("anomaly_flag", "size"), anomaly_rate=("anomaly_flag", "mean"))
            .query("claim_count >= 5")
            .sort_values("anomaly_rate", ascending=False)
            .head(10)
        )

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        provider_risk["anomaly_rate"].plot(kind="bar", ax=ax2)
        ax2.set_title("Top 10 Providers by Anomaly Rate")
        ax2.set_xlabel("Provider")
        ax2.set_ylabel("Anomaly Rate")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)
        plt.close()

    st.subheader("Claim Amount vs Risk Score")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    colors = np.where(plot_df["anomaly_flag"] == 1, "red", "blue")
    ax3.scatter(plot_df["claim_amount"], plot_df["risk_score"], c=colors, alpha=0.5, s=10)
    ax3.set_xlabel("Claim Amount")
    ax3.set_ylabel("Risk Score")
    ax3.set_title("Normal (blue) vs Anomaly (red)")
    st.pyplot(fig3)
    plt.close()


def render_clustering_and_timeseries(df: pd.DataFrame) -> None:
    st.subheader("KMeans Clustering")
    st.caption(f"KMeans with k={N_CLUSTERS} on engineered features (PCA visualization)")

    try:
        clustered_df, _, _ = run_kmeans(df, n_clusters=N_CLUSTERS)
    except ValueError as exc:
        st.error(str(exc))
        return

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_clusters(clustered_df, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_cluster_summary(clustered_df, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    cluster_profile = (
        clustered_df.groupby("cluster")[FEATURE_COLUMNS[:6]]
        .mean()
        .round(2)
    )
    st.write("Cluster profiles (mean of key features):")
    st.dataframe(cluster_profile)

    time_df = enrich_time_columns(clustered_df)
    if not has_time_columns(time_df):
        st.info("Time-series charts need ClaimStartDt or ClaimID (merged from clean_claims.csv).")
        return

    st.subheader("Provider Monthly Trend")
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_provider_monthly_trend(time_df, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Anomaly Spikes Over Time")
    fig, ax = plt.subplots(figsize=(9, 4))
    plot_anomaly_spikes(time_df, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


def init_session_state(default_df: pd.DataFrame | None, default_name: str) -> None:
    if "active_df" not in st.session_state:
        st.session_state.active_df = default_df
        st.session_state.data_source = default_name
        st.session_state.upload_key = None


model, scaler = load_model()
default_df, default_name = load_default_data()

st.title("Healthcare Payment Integrity AI Dashboard")

if default_df is None and model is None:
    st.error(
        "No data or model found. Run the pipeline first:\n\n"
        "`python src/data_cleaning.py` → `python src/feature_engineering.py` → "
        "`python src/train_model.py`"
    )
    st.stop()

init_session_state(default_df, default_name)

# --- Sidebar: upload drives the active dataset for all pages ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")

if uploaded_file is not None:
    upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.upload_key != upload_key:
        if model is None or scaler is None:
            st.sidebar.error("Model not found. Run `python src/train_model.py` first.")
        else:
            raw_upload = pd.read_csv(uploaded_file)
            scored_df, missing = score_dataframe(raw_upload, model, scaler)
            if missing:
                st.sidebar.error(
                    "Upload must include engineered features. "
                    "Run feature_engineering.py first, or upload scored_claims.csv."
                )
                st.sidebar.caption(f"Missing {len(missing)} columns.")
            else:
                st.session_state.active_df = scored_df
                st.session_state.data_source = uploaded_file.name
                st.session_state.upload_key = upload_key
                st.sidebar.success(f"Loaded & scored: {uploaded_file.name}")

if default_df is not None and st.sidebar.button("Reset to default dataset"):
    st.session_state.active_df = default_df
    st.session_state.data_source = default_name
    st.session_state.upload_key = None
    st.rerun()

df = st.session_state.active_df
if df is None:
    st.warning("Upload a CSV with engineered features, or run the pipeline to create the default dataset.")
    st.stop()

if model is None or scaler is None:
    st.error("Model artifacts not found. Run `python src/train_model.py` to train and save the model.")
    st.stop()

df = prepare_dataframe(df)
stats = dataset_stats(df)

source_label = "uploaded" if st.session_state.upload_key else "default"
st.sidebar.info(f"Active dataset ({source_label}): **{st.session_state.data_source}**\n\n{len(df):,} rows")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Upload",
        "Model Insights",
        "Interactive Analysis",
        "Visual Dashboard",
        "Clustering & Trends",
        "Demo Scenarios",
    ],
)

if menu == "Dataset Upload":
    st.header("Dataset Upload")
    st.caption(f"Active dataset: `{st.session_state.data_source}` ({len(df):,} rows)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", f"{len(df):,}")
    col2.metric("Anomalies", f"{(df['anomaly_flag'] == 1).sum():,}")
    col3.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")

    st.subheader("Preview")
    preview_cols = [c for c in ["Provider", "claim_amount", "risk_score", "anomaly_flag", "risk_category", "explanation"] if c in df.columns]
    st.dataframe(df[preview_cols].head(25))

    st.subheader("Quick Charts (from active dataset)")
    render_visual_dashboard(df)

elif menu == "Model Insights":
    st.header("Model Summary")
    st.caption(f"Based on: `{st.session_state.data_source}`")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Claims", f"{len(df):,}")
    col2.metric("Anomalies", f"{(df['anomaly_flag'] == 1).sum():,}")
    col3.metric("Normal Claims", f"{(df['anomaly_flag'] == 0).sum():,}")

    st.subheader("Risk Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    df["risk_category"].value_counts().plot(kind="bar", ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"])
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Count")
    ax.set_title("Claims by Risk Category")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close()

    st.subheader("Sample High-Risk Explanations")
    high_risk = df[df["anomaly_flag"] == 1].sort_values("risk_score", ascending=False).head(5)
    if high_risk.empty:
        st.write("No anomalies detected in the active dataset.")
    else:
        for _, row in high_risk.iterrows():
            provider = row.get("Provider", "N/A")
            st.write(f"**Provider {provider}** — score {row['risk_score']:.1f}: {row['explanation']}")

elif menu == "Interactive Analysis":
    st.header("Analyze a Single Claim")
    st.caption(f"Thresholds computed from: `{st.session_state.data_source}`")

    col1, col2 = st.columns(2)
    with col1:
        claim_amount = st.number_input("Claim Amount ($)", min_value=0, max_value=500_000, value=500)
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=65)
        claim_duration_days = st.number_input("Claim Duration (days)", min_value=0, max_value=365, value=1)
    with col2:
        provider_claims_per_month = st.number_input("Provider Claims per Month", min_value=0, max_value=500, value=10)
        provider_avg_claim = st.number_input(
            "Provider Average Claim ($)",
            min_value=0,
            max_value=500_000,
            value=int(stats["provider_avg_claim_median"]),
        )
        provider_claim_count = st.number_input(
            "Provider Total Claim Count",
            min_value=0,
            max_value=100_000,
            value=int(stats["provider_claim_count_median"]),
        )

    if st.button("Analyze Claim", type="primary"):
        features = build_feature_vector(
            claim_amount=claim_amount,
            age=age,
            claim_duration_days=claim_duration_days,
            provider_claims_per_month=provider_claims_per_month,
            stats=stats,
            provider_claim_count=provider_claim_count,
            provider_avg_claim=provider_avg_claim,
        )
        risk_score, _, anomaly_flag = score_features(features, model, scaler, stats)
        reasons = explain_inputs(
            claim_amount,
            provider_claims_per_month,
            features.iloc[0],
            risk_score,
            anomaly_flag,
            stats,
        )

        st.subheader("Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{risk_score:.1f}/100")
        m2.metric("Status", risk_label(risk_score))
        m3.metric("Model Flag", "ANOMALY" if anomaly_flag else "NORMAL")

        st.subheader("Explanation")
        for reason in reasons:
            st.write("-", reason)

elif menu == "Visual Dashboard":
    st.header("Visual Dashboard")
    st.caption(f"Charts computed from: `{st.session_state.data_source}`")
    render_visual_dashboard(df)

elif menu == "Clustering & Trends":
    st.header("Clustering & Time Series")
    st.caption(f"Analysis based on: `{st.session_state.data_source}`")
    render_clustering_and_timeseries(df)

elif menu == "Demo Scenarios":
    st.header("Prebuilt Scenarios")
    st.caption(f"Compared against: `{st.session_state.data_source}`")

    scenarios = {
        "Normal Case": {"claim_amount": 120, "age": 70, "claim_duration_days": 1, "provider_claims_per_month": 2},
        "Suspicious Case": {"claim_amount": 8000, "age": 55, "claim_duration_days": 3, "provider_claims_per_month": 20},
        "Extreme Anomaly": {"claim_amount": 50000, "age": 45, "claim_duration_days": 10, "provider_claims_per_month": 80},
    }

    for name, values in scenarios.items():
        st.subheader(name)
        features = build_feature_vector(stats=stats, **values)
        risk_score, _, anomaly_flag = score_features(features, model, scaler, stats)
        reasons = explain_inputs(
            values["claim_amount"],
            values["provider_claims_per_month"],
            features.iloc[0],
            risk_score,
            anomaly_flag,
            stats,
        )
        st.write("Risk Score:", round(risk_score, 2))
        st.write("Status:", risk_label(risk_score))
        st.write("Prediction:", "ANOMALY" if anomaly_flag else "NORMAL")
        for reason in reasons:
            st.write("-", reason)
