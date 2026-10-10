"""
Audio Telemetry Intelligence Dashboard (Streamlit)
=====================================================
Interactive dashboard for exploring telemetry insights,
model predictions, and data quality metrics.

Maps to JD: "Improve data accessibility and reliability through
automation and internal data tooling."
"""

import os
import json
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Config & Data Loading
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Audio Telemetry Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS = os.path.join(BASE_DIR, "data", "models")

# Small bundled dataset (built by scripts/build_sample_data.py). Used when the
# full pipeline hasn't been run, e.g. on a fresh clone or a hosted deployment.
SAMPLE_PROCESSED = os.path.join(BASE_DIR, "sample_data", "processed")
SAMPLE_MODELS = os.path.join(BASE_DIR, "sample_data", "models")


def _data_dirs() -> tuple[str, str, bool]:
    """Return (processed_dir, models_dir, is_sample)."""
    if os.path.exists(os.path.join(PROCESSED, "sessions.parquet")):
        return PROCESSED, MODELS, False
    return SAMPLE_PROCESSED, SAMPLE_MODELS, True


@st.cache_data
def load_data():
    processed, models, is_sample = _data_dirs()
    data = {"is_sample": is_sample}
    for name in ["sessions", "hourly_agg", "device_profile", "env_analysis"]:
        path = os.path.join(processed, f"{name}.parquet")
        if os.path.exists(path):
            data[name] = pd.read_parquet(path)

    # Load model results
    results_path = os.path.join(models, "model_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data["model_results"] = json.load(f)

    # Load quality report
    quality_path = os.path.join(processed, "quality_report.json")
    if os.path.exists(quality_path):
        with open(quality_path) as f:
            data["quality_report"] = json.load(f)

    return data


# ---------------------------------------------------------------------------
# Dashboard Pages
# ---------------------------------------------------------------------------

def overview_page(data: dict):
    """Main overview with KPIs and trends."""
    st.title("🎧 Audio Telemetry Intelligence")
    st.markdown("*Device experience optimization through telemetry analytics*")

    sessions = data.get("sessions", pd.DataFrame())
    profiles = data.get("device_profile", pd.DataFrame())
    quality = data.get("quality_report", {})

    if sessions.empty:
        st.warning("No data loaded. Run the pipeline first.")
        return

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Sessions", f"{len(sessions):,}")
    col2.metric("Unique Devices", f"{sessions['device_id'].nunique():,}")
    col3.metric("Avg Satisfaction", f"{sessions['satisfaction_proxy'].mean():.1f}")
    col4.metric("Avg Quality Index", f"{sessions['audio_quality_index'].mean():.1f}")
    col5.metric("Data Quality", f"{quality.get('quality_score', 'N/A')}%")

    st.divider()

    # Charts row 1
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sessions by Device Type")
        device_counts = sessions["device_type"].value_counts()
        fig = px.bar(
            x=device_counts.index, y=device_counts.values,
            labels={"x": "Device", "y": "Sessions"},
            color=device_counts.values,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Satisfaction by Environment")
        env_sat = sessions.groupby("environment_type")["satisfaction_proxy"].mean().sort_values()
        fig = px.bar(
            x=env_sat.values, y=env_sat.index, orientation="h",
            labels={"x": "Avg Satisfaction", "y": "Environment"},
            color=env_sat.values,
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    # Charts row 2
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Hourly Usage Pattern")
        hourly = sessions.groupby("hour_of_day").agg(
            sessions=("session_id", "count"),
            avg_satisfaction=("satisfaction_proxy", "mean"),
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=hourly["hour_of_day"], y=hourly["sessions"],
                   name="Sessions", marker_color="#636EFA"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=hourly["hour_of_day"], y=hourly["avg_satisfaction"],
                       name="Satisfaction", line=dict(color="#EF553B", width=2)),
            secondary_y=True,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    with c4:
        st.subheader("ANC Level vs Ambient Noise")
        sample = sessions[sessions["anc_level"] > 0].sample(
            n=min(5000, len(sessions[sessions["anc_level"] > 0])),
            random_state=42,
        )
        fig = px.scatter(
            sample, x="ambient_noise_db", y="anc_level",
            color="environment_type", opacity=0.5,
            labels={"ambient_noise_db": "Ambient Noise (dB)",
                    "anc_level": "ANC Level"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")


def model_insights_page(data: dict):
    """ML model results and insights."""
    st.title("🤖 Model Insights")

    results = data.get("model_results", {})
    if not results:
        st.warning("No model results found. Run the ML pipeline first.")
        return

    # ANC Predictor
    st.subheader("Adaptive ANC Predictor")
    anc = results.get("anc_predictor", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", anc.get("r2_score", "N/A"))
    c2.metric("MAE", anc.get("mae", "N/A"))
    c3.metric("RMSE", anc.get("rmse", "N/A"))
    c4.metric("CV R² (mean)", anc.get("cv_r2_mean", "N/A"))

    # Feature importance
    importance = results.get("anc_feature_importance", {})
    if importance:
        st.markdown("**Top Predictive Features**")
        imp_df = pd.DataFrame(
            list(importance.items()), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # User Segmentation
    st.subheader("User Behavior Segments")
    clusters = results.get("cluster_profiles", {})
    seg_metrics = results.get("user_segmentation", {})
    if clusters:
        st.metric("Silhouette Score", seg_metrics.get("silhouette_score", "N/A"))
        cluster_df = pd.DataFrame(clusters).T.reset_index()
        cluster_df.columns = ["Segment", "Size", "% of Users",
                               "Sessions/Day", "Avg Duration (min)",
                               "Satisfaction", "Preferred ANC"]
        st.dataframe(cluster_df, width="stretch")

        fig = px.scatter(
            cluster_df, x="Sessions/Day", y="Avg Duration (min)",
            size="Size", color="Segment", hover_data=["Satisfaction"],
            size_max=60,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # Anomaly Detection
    st.subheader("Audio Quality Anomaly Detection")
    anomaly = results.get("anomaly_detection", {})
    comparison = results.get("anomaly_comparison", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies Detected", f"{anomaly.get('anomalies_detected', 0):,}")
    c2.metric("Anomaly Rate", f"{anomaly.get('anomaly_rate', 0)*100:.1f}%")
    c3.metric("Total Sessions", f"{anomaly.get('total_sessions', 0):,}")

    if comparison:
        comp_df = pd.DataFrame(comparison).T
        st.dataframe(comp_df, width="stretch")

    st.divider()

    # Duration Forecaster
    st.subheader("Session Duration Forecaster")
    dur = results.get("duration_forecaster", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", dur.get("r2_score", "N/A"))
    c2.metric("MAE (seconds)", dur.get("mae_seconds", "N/A"))
    c3.metric("Median Error (sec)", dur.get("median_abs_error_seconds", "N/A"))


def data_quality_page(data: dict):
    """Data quality monitoring dashboard."""
    st.title("📊 Data Quality Report")

    quality = data.get("quality_report", {})
    sessions = data.get("sessions", pd.DataFrame())

    if not quality:
        st.warning("No quality report found.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Records", f"{quality.get('total_raw', 0):,}")
    c2.metric("Clean Records", f"{quality.get('clean_count', 0):,}")
    c3.metric("Quality Score", f"{quality.get('quality_score', 0)}%")
    c4.metric("Duplicates", f"{quality.get('duplicates', 0):,}")

    st.divider()

    # Null analysis
    st.subheader("Null Value Analysis")
    nulls = quality.get("nulls", {})
    if nulls:
        null_df = pd.DataFrame(
            list(nulls.items()), columns=["Column", "Null Count"]
        )
        fig = px.bar(null_df, x="Column", y="Null Count",
                     color="Null Count", color_continuous_scale="Reds")
        st.plotly_chart(fig, width="stretch")

    # Distribution checks
    if not sessions.empty:
        st.subheader("Feature Distributions")
        col = st.selectbox(
            "Select feature",
            ["duration_seconds", "anc_level", "ambient_noise_db",
             "volume_level", "latency_ms", "satisfaction_proxy",
             "audio_quality_index"],
        )
        fig = px.histogram(
            sessions.sample(n=min(50000, len(sessions)), random_state=42),
            x=col, nbins=80, marginal="box",
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")


def exploration_page(data: dict):
    """Interactive data exploration."""
    st.title("🔍 Data Explorer")

    sessions = data.get("sessions", pd.DataFrame())
    if sessions.empty:
        st.warning("No session data loaded.")
        return

    st.sidebar.subheader("Filters")
    device_types = st.sidebar.multiselect(
        "Device Type", sessions["device_type"].unique(),
        default=sessions["device_type"].unique()[:3],
    )
    env_types = st.sidebar.multiselect(
        "Environment", sessions["environment_type"].unique(),
        default=sessions["environment_type"].unique(),
    )
    audio_modes = st.sidebar.multiselect(
        "Audio Mode", sessions["audio_mode"].unique(),
        default=sessions["audio_mode"].unique(),
    )

    filtered = sessions[
        (sessions["device_type"].isin(device_types)) &
        (sessions["environment_type"].isin(env_types)) &
        (sessions["audio_mode"].isin(audio_modes))
    ]

    st.metric("Filtered Sessions", f"{len(filtered):,}")

    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X Axis", ["ambient_noise_db", "anc_level",
                                          "volume_level", "duration_seconds"])
    with c2:
        y_axis = st.selectbox("Y Axis", ["satisfaction_proxy",
                                          "audio_quality_index", "latency_ms",
                                          "battery_drain_rate_pct_hr"])

    color_by = st.selectbox("Color By", ["device_type", "environment_type",
                                          "audio_mode", "engagement_tier",
                                          "noise_category"])

    sample = filtered.sample(n=min(8000, len(filtered)), random_state=42)
    fig = px.scatter(
        sample, x=x_axis, y=y_axis, color=color_by, opacity=0.5,
        hover_data=["device_type", "environment_type", "audio_mode"],
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    # Summary table
    st.subheader("Summary Statistics")
    summary = filtered.select_dtypes("number").describe().round(2)
    st.dataframe(summary, width="stretch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_data()

    if data.get("is_sample"):
        st.info(
            "**Demo dataset** — you're viewing a small bundled sample "
            "(synthetic data, ~40K sessions). Run `python run_pipeline.py` "
            "to analyze the full 500K-session dataset locally."
        )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Overview", "🤖 Model Insights",
         "📋 Data Quality", "🔍 Explorer"],
    )

    if page == "📊 Overview":
        overview_page(data)
    elif page == "🤖 Model Insights":
        model_insights_page(data)
    elif page == "📋 Data Quality":
        data_quality_page(data)
    elif page == "🔍 Explorer":
        exploration_page(data)

    st.sidebar.divider()
    st.sidebar.caption(
        "Audio Telemetry Intelligence Platform v1.0\n\n"
        "Built with PySpark, scikit-learn, Streamlit"
    )


if __name__ == "__main__":
    main()
