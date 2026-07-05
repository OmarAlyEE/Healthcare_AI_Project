import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CLEAN_CLAIMS_CSV, FEATURE_COLUMNS, KMEANS_SAMPLE_SIZE, N_CLUSTERS


def enrich_time_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "claim_period" in df.columns:
        return df

    if "ClaimStartDt" in df.columns:
        df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
    elif "ClaimID" in df.columns and CLEAN_CLAIMS_CSV.exists():
        clean = pd.read_csv(CLEAN_CLAIMS_CSV, usecols=["ClaimID", "ClaimStartDt"])
        clean["ClaimStartDt"] = pd.to_datetime(clean["ClaimStartDt"], errors="coerce")
        df = df.merge(clean, on="ClaimID", how="left")

    if "ClaimStartDt" in df.columns:
        df["claim_year"] = df["ClaimStartDt"].dt.year
        df["claim_month"] = df["ClaimStartDt"].dt.month

    if "claim_year" in df.columns and "claim_month" in df.columns:
        df["claim_period"] = pd.to_datetime(
            df["claim_year"].astype("Int64").astype(str)
            + "-"
            + df["claim_month"].astype("Int64").astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )

    return df


def has_time_columns(df: pd.DataFrame) -> bool:
    return "claim_period" in df.columns and df["claim_period"].notna().any()


def run_kmeans(
    df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
    sample_size: int = KMEANS_SAMPLE_SIZE,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Fit KMeans on engineered features and assign cluster labels."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for clustering: {missing}")

    work = df.copy()
    features = work[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    if len(work) > sample_size:
        rng = np.random.default_rng(random_state)
        fit_idx = rng.choice(len(work), size=sample_size, replace=False)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(scaled[fit_idx])
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(scaled)

    work["cluster"] = kmeans.predict(scaled)
    return work, kmeans, scaler


def plot_clusters(df: pd.DataFrame, ax=None, sample_size: int = KMEANS_SAMPLE_SIZE):

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if "cluster" not in df.columns:
        df, _, _ = run_kmeans(df)

    features = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    plot_df = df.copy()
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)

    idx = plot_df.index
    X = features.loc[idx]
    scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(scaled)

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=plot_df["cluster"],
        cmap="tab10",
        alpha=0.6,
        s=12,
    )
    ax.set_title("KMeans Clusters (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    return ax


def plot_provider_monthly_trend(df: pd.DataFrame, ax=None, top_n: int = 5):

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df = enrich_time_columns(df)
    if not has_time_columns(df):
        ax.text(0.5, 0.5, "No date columns available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    time_df = df.dropna(subset=["claim_period", "Provider"])
    top_providers = time_df["Provider"].value_counts().head(top_n).index
    monthly = (
        time_df[time_df["Provider"].isin(top_providers)]
        .groupby(["claim_period", "Provider"])
        .size()
        .reset_index(name="claim_count")
    )

    for provider in top_providers:
        subset = monthly[monthly["Provider"] == provider]
        ax.plot(subset["claim_period"], subset["claim_count"], marker="o", label=str(provider)[:12])

    ax.set_title(f"Provider Monthly Claim Trend (Top {top_n})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Claim Count")
    ax.legend(loc="upper left", fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    return ax


def plot_anomaly_spikes(df: pd.DataFrame, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df = enrich_time_columns(df)
    if not has_time_columns(df):
        ax.text(0.5, 0.5, "No date columns available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    if "anomaly_flag" not in df.columns:
        ax.text(0.5, 0.5, "No anomaly_flag column", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    time_df = df.dropna(subset=["claim_period"]).copy()
    time_df["anomaly_flag"] = pd.to_numeric(time_df["anomaly_flag"], errors="coerce").fillna(0).astype(int)

    monthly = (
        time_df.groupby("claim_period")
        .agg(total_claims=("anomaly_flag", "size"), anomalies=("anomaly_flag", "sum"))
        .reset_index()
    )
    monthly["anomaly_rate"] = monthly["anomalies"] / monthly["total_claims"]

    ax.bar(monthly["claim_period"], monthly["anomalies"], alpha=0.7, label="Anomaly count")
    ax.plot(
        monthly["claim_period"],
        monthly["anomaly_rate"] * monthly["anomalies"].max(),
        color="red",
        marker="o",
        linewidth=2,
        label="Spike intensity (rate scaled)",
    )
    ax.set_title("Anomaly Spikes Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Anomaly Count")
    ax.legend()
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    return ax


def plot_cluster_summary(df: pd.DataFrame, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if "cluster" not in df.columns:
        df, _, _ = run_kmeans(df)

    counts = df["cluster"].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, ax=ax, palette="tab10", legend=False)
    ax.set_title("Claims per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    return ax
