"""
Predictive & Analytical Models for Audio Telemetry
=====================================================
1. Adaptive ANC Predictor — predicts optimal noise cancellation level
   based on environment, device state, and user behavior patterns.
2. User Behavior Segmentation — clusters users by audio usage patterns.
3. Audio Quality Anomaly Detection — identifies sessions with degraded
   audio experience for proactive intervention.
4. Session Duration Forecasting — predicts session length for
   battery/resource optimization.

Tech: scikit-learn, XGBoost, scipy, statsmodels
"""

import os
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.config import load_config, resolve_path

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    name: str
    metrics: dict
    model: object
    artifacts: dict


class ANCPredictor:
    """
    Predicts optimal ANC attenuation level based on environment context.
    Maps to JD: "Explore opportunities for predictive or adaptive system
    behaviors that improve device audio experience."
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_cols = []

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Engineer features for ANC prediction."""
        # Filter to ANC-capable devices
        anc_df = df[df["anc_level"] > 0].copy()

        cat_cols = ["audio_mode", "environment_type", "device_type",
                    "noise_category", "engagement_tier", "time_bucket"]

        for col in cat_cols:
            le = LabelEncoder()
            anc_df[f"{col}_enc"] = le.fit_transform(anc_df[col].astype(str))
            self.label_encoders[col] = le

        self.feature_cols = [
            "ambient_noise_db", "volume_level", "hour_of_day",
            "day_of_week", "is_weekend", "duration_seconds",
            "battery_drain_rate_pct_hr", "latency_ms",
            "anc_demand_score",
        ] + [f"{c}_enc" for c in cat_cols]

        anc_df["is_weekend"] = anc_df["is_weekend"].astype(int)

        X = anc_df[self.feature_cols].values
        y = anc_df["anc_level"].values

        return X, y

    def train(self, df: pd.DataFrame) -> ModelResult:
        """Train XGBoost-style gradient boosting for ANC prediction."""
        print("\n[MODEL] Training Adaptive ANC Predictor...")

        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=self.random_state,
        )
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring="r2"
        )

        # Feature importance
        importance = dict(zip(
            self.feature_cols,
            self.model.feature_importances_.tolist()
        ))
        top_features = dict(sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )[:10])

        metrics = {
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "r2_score": round(r2_score(y_test, y_pred), 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        print(f"  R² = {metrics['r2_score']}, MAE = {metrics['mae']}, "
              f"CV R² = {metrics['cv_r2_mean']} ± {metrics['cv_r2_std']}")

        return ModelResult(
            name="adaptive_anc_predictor",
            metrics=metrics,
            model=self.model,
            artifacts={"feature_importance": top_features,
                       "scaler": self.scaler},
        )


def _name_segments(stats: pd.DataFrame) -> dict:
    """
    Derive segment names from cluster characteristics (K-Means label order is
    arbitrary, so names must come from the profiles, not the cluster index).

    `stats` is indexed by cluster id with columns `preferred_anc_level` and
    `total_sessions` (per-cluster means) and `size` (devices in the cluster).
    """
    names = {}
    remaining = list(stats.index)

    # A cluster whose ANC usage clearly dominates the rest is the ANC-heavy one.
    if len(stats) > 1:
        anc = stats["preferred_anc_level"]
        top = anc.idxmax()
        if anc[top] >= max(0.15, 2 * anc.drop(top).max()):
            names[top] = "ANC-Heavy Listeners"
            remaining.remove(top)

    # Label the rest by session volume relative to the population average, so
    # a cluster is only "high"/"low" when it is meaningfully different.
    overall = np.average(stats["total_sessions"], weights=stats["size"])
    for c in remaining:
        ratio = stats.loc[c, "total_sessions"] / max(overall, 1e-9)
        names[c] = ("High-Activity Users" if ratio >= 1.25
                    else "Light Users" if ratio <= 0.75
                    else "Standard Users")

    # Guarantee unique names (several "Standard" clusters when k is large).
    seen = {}
    for c in sorted(names):
        base = names[c]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[c] = f"{base} {seen[base]}"
    return names


class UserBehaviorSegmentation:
    """
    Segments users into behavioral archetypes via K-Means clustering
    on device-level usage profiles.
    Maps to JD: "Analyze large-scale audio session and device telemetry
    data to identify patterns in user behavior."
    """

    def __init__(self, n_clusters: int = 6):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self, device_profile: pd.DataFrame) -> np.ndarray:
        """Build clustering features from device profiles."""
        feature_cols = [
            "total_sessions", "avg_session_duration", "preferred_anc_level",
            "preferred_volume", "avg_satisfaction", "avg_quality_index",
            "spatial_audio_sessions", "avg_battery_drain",
            "sessions_per_day",
        ]

        X = device_profile[feature_cols].fillna(0).values
        return X, feature_cols

    def train(self, device_profile: pd.DataFrame) -> ModelResult:
        """Train K-Means clustering."""
        print("\n[MODEL] Training User Behavior Segmentation...")

        X, feature_cols = self.prepare_features(device_profile)
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal k using silhouette analysis
        silhouette_scores = {}
        for k in range(3, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X)))
            silhouette_scores[k] = round(sil, 4)

        best_k = max(silhouette_scores, key=silhouette_scores.get)
        print(f"  Silhouette scores: {silhouette_scores}")
        print(f"  Best k = {best_k}")

        # Train final model with selected k
        self.model = KMeans(
            n_clusters=min(self.n_clusters, best_k),
            random_state=42, n_init=10,
        )
        labels = self.model.fit_predict(X_scaled)

        # Cluster profiles
        device_profile = device_profile.copy()
        device_profile["cluster"] = labels
        cluster_stats = device_profile.groupby("cluster")[
            ["preferred_anc_level", "total_sessions"]
        ].mean()
        cluster_stats["size"] = device_profile.groupby("cluster").size()
        segment_names = _name_segments(cluster_stats)
        cluster_profiles = {}
        for c in range(self.model.n_clusters):
            mask = device_profile["cluster"] == c
            cluster_profiles[segment_names.get(c, f"Segment_{c}")] = {
                "size": int(mask.sum()),
                "pct": round(mask.mean() * 100, 1),
                "avg_sessions_per_day": round(
                    device_profile.loc[mask, "sessions_per_day"].mean(), 2
                ),
                "avg_session_duration_min": round(
                    device_profile.loc[mask, "avg_session_duration"].mean() / 60, 1
                ),
                "avg_satisfaction": round(
                    device_profile.loc[mask, "avg_satisfaction"].mean(), 1
                ),
                "preferred_anc": round(
                    device_profile.loc[mask, "preferred_anc_level"].mean(), 3
                ),
            }

        metrics = {
            "n_clusters": self.model.n_clusters,
            "silhouette_score": silhouette_scores.get(self.model.n_clusters, 0),
            "inertia": round(self.model.inertia_, 2),
            "silhouette_all": silhouette_scores,
        }

        print(f"  Clusters: {self.model.n_clusters}, "
              f"Silhouette: {metrics['silhouette_score']}")
        for name, profile in cluster_profiles.items():
            print(f"    {name}: {profile['size']:,} devices ({profile['pct']}%)")

        return ModelResult(
            name="user_behavior_segmentation",
            metrics=metrics,
            model=self.model,
            artifacts={"cluster_profiles": cluster_profiles,
                       "scaler": self.scaler,
                       "labels": labels.tolist()},
        )


class AudioQualityAnomalyDetector:
    """
    Detects anomalous audio sessions indicating degraded experience.
    Maps to JD: "Support experimentation and data-driven decision making
    across audio and device interaction scenarios."
    """

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()

    def train(self, sessions_df: pd.DataFrame) -> ModelResult:
        """Train Isolation Forest for anomaly detection."""
        print("\n[MODEL] Training Audio Quality Anomaly Detector...")

        feature_cols = [
            "duration_seconds", "anc_level", "ambient_noise_db",
            "volume_level", "audio_glitches", "latency_ms",
            "battery_drain_rate_pct_hr", "audio_quality_index",
            "satisfaction_proxy",
        ]

        X = sessions_df[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=150,
            max_samples="auto",
            random_state=42,
        )
        predictions = self.model.fit_predict(X_scaled)

        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)

        # Analyze anomalies
        sessions_with_labels = sessions_df.copy()
        sessions_with_labels["is_anomaly"] = predictions == -1

        anomaly_stats = {
            "by_device_type": sessions_with_labels.groupby("device_type")["is_anomaly"]
            .mean().round(4).to_dict(),
            "by_environment": sessions_with_labels.groupby("environment_type")["is_anomaly"]
            .mean().round(4).to_dict(),
            "by_audio_mode": sessions_with_labels.groupby("audio_mode")["is_anomaly"]
            .mean().round(4).to_dict(),
        }

        # Anomalous vs normal comparison
        anomalous = sessions_with_labels[sessions_with_labels["is_anomaly"]]
        normal = sessions_with_labels[~sessions_with_labels["is_anomaly"]]

        comparison = {}
        for col in ["latency_ms", "audio_glitches", "satisfaction_proxy",
                     "audio_quality_index"]:
            comparison[col] = {
                "anomalous_mean": round(anomalous[col].mean(), 2),
                "normal_mean": round(normal[col].mean(), 2),
                "ratio": round(anomalous[col].mean() / max(normal[col].mean(), 0.01), 2),
            }

        metrics = {
            "total_sessions": len(predictions),
            "anomalies_detected": int(n_anomalies),
            "anomaly_rate": round(anomaly_rate, 4),
            "contamination_param": self.contamination,
        }

        print(f"  Anomalies: {n_anomalies:,} ({anomaly_rate*100:.1f}%)")
        print(f"  Avg latency (anomalous vs normal): "
              f"{comparison['latency_ms']['anomalous_mean']}ms vs "
              f"{comparison['latency_ms']['normal_mean']}ms")

        return ModelResult(
            name="audio_quality_anomaly_detector",
            metrics=metrics,
            model=self.model,
            artifacts={
                "anomaly_stats": anomaly_stats,
                "comparison": comparison,
                "scaler": self.scaler,
                "predictions": predictions.tolist(),
            },
        )


class SessionDurationForecaster:
    """
    Predicts session duration for battery/resource pre-allocation.
    Maps to JD: "Explore opportunities for predictive or adaptive
    system behaviors."
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def train(self, sessions_df: pd.DataFrame) -> ModelResult:
        """Train gradient boosting for duration prediction."""
        print("\n[MODEL] Training Session Duration Forecaster...")

        df = sessions_df.copy()
        cat_cols = ["audio_mode", "environment_type", "device_type", "time_bucket"]

        for col in cat_cols:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        feature_cols = [
            "ambient_noise_db", "anc_level", "volume_level",
            "hour_of_day", "day_of_week", "is_weekend",
            "anc_demand_score",
        ] + [f"{c}_enc" for c in cat_cols]

        df["is_weekend"] = df["is_weekend"].astype(int)

        X = df[feature_cols].values
        y = np.log1p(df["duration_seconds"].values)  # log-transform target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        self.model.fit(X_train_s, y_train)
        y_pred = self.model.predict(X_test_s)

        # Convert back from log space
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)

        metrics = {
            "mae_seconds": round(mean_absolute_error(y_test_orig, y_pred_orig), 1),
            "rmse_seconds": round(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)), 1),
            "r2_score": round(r2_score(y_test, y_pred), 4),
            "median_abs_error_seconds": round(
                np.median(np.abs(y_test_orig - y_pred_orig)), 1
            ),
        }

        print(f"  R² = {metrics['r2_score']}, MAE = {metrics['mae_seconds']}s")

        return ModelResult(
            name="session_duration_forecaster",
            metrics=metrics,
            model=self.model,
            artifacts={"scaler": self.scaler},
        )


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------

def _effect_label(value: float, cutoffs: tuple) -> str:
    """Cohen-style label for variance-explained effect sizes (η², ε²)."""
    small, medium, large = cutoffs
    if value < small:
        return "negligible"
    if value < medium:
        return "small"
    if value < large:
        return "medium"
    return "large"


def run_statistical_analysis(sessions_df: pd.DataFrame) -> dict:
    """Behavioral pattern analysis using statistical methods."""
    print("\n[ANALYSIS] Running statistical analysis...")

    results = {}

    # 1. ANC effectiveness by environment (ANOVA)
    anc_groups = []
    env_types = sessions_df[sessions_df["anc_level"] > 0]["environment_type"].unique()
    for env in env_types:
        group = sessions_df[
            (sessions_df["environment_type"] == env) &
            (sessions_df["anc_level"] > 0)
        ]["satisfaction_proxy"].values
        if len(group) > 30:
            anc_groups.append(group)

    if len(anc_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*anc_groups)
        # Eta-squared: share of total variance explained by environment.
        all_vals = np.concatenate(anc_groups)
        grand_mean = all_vals.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in anc_groups)
        ss_total = ((all_vals - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total
        results["anc_environment_anova"] = {
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
            "eta_squared": round(float(eta_sq), 4),
            "effect_size": _effect_label(eta_sq, (0.01, 0.06, 0.14)),
            "n": int(len(all_vals)),
            "interpretation": (
                "Significant difference in satisfaction across environments"
                if p_val < 0.05 else "No significant difference"
            ),
        }
        print(f"  ANOVA (ANC × environment): F={f_stat:.2f}, p={p_val:.6f}, "
              f"η²={eta_sq:.4f} ({results['anc_environment_anova']['effect_size']})")

    # 2. Correlation matrix for key metrics
    corr_cols = [
        "duration_seconds", "anc_level", "ambient_noise_db",
        "volume_level", "audio_quality_index", "satisfaction_proxy",
        "latency_ms", "battery_drain_rate_pct_hr",
    ]
    corr_matrix = sessions_df[corr_cols].corr().round(4)
    results["correlation_matrix"] = corr_matrix.to_dict()

    # 3. Time-of-day usage patterns
    hourly = sessions_df.groupby("hour_of_day").agg(
        session_count=("session_id", "count"),
        avg_duration=("duration_seconds", "mean"),
        avg_satisfaction=("satisfaction_proxy", "mean"),
    ).round(2)
    results["hourly_patterns"] = hourly.to_dict()

    # 4. Device type comparison (Kruskal-Wallis)
    device_groups = []
    device_names = []
    for dt in sessions_df["device_type"].unique():
        group = sessions_df[sessions_df["device_type"] == dt]["satisfaction_proxy"].values
        if len(group) > 50:
            device_groups.append(group)
            device_names.append(dt)

    if len(device_groups) >= 2:
        h_stat, p_val = stats.kruskal(*device_groups)
        # Epsilon-squared (rank-based effect size): (H - k + 1) / (n - k).
        n_total = sum(len(g) for g in device_groups)
        k = len(device_groups)
        eps_sq = max(0.0, (h_stat - k + 1) / (n_total - k))
        results["device_satisfaction_kruskal"] = {
            "h_statistic": round(float(h_stat), 4),
            "p_value": round(float(p_val), 6),
            "devices_compared": device_names,
            "significant": bool(p_val < 0.05),
            "epsilon_squared": round(float(eps_sq), 4),
            "effect_size": _effect_label(eps_sq, (0.01, 0.06, 0.14)),
            "n": int(n_total),
        }
        print(f"  Kruskal-Wallis (device satisfaction): H={h_stat:.2f}, "
              f"p={p_val:.6f}, ε²={eps_sq:.4f} "
              f"({results['device_satisfaction_kruskal']['effect_size']})")

    return results


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Serialize numpy scalars/arrays natively (np.bool_ -> true, not "True")."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_all_models(data_dir: str, output_dir: str, model_cfg: dict | None = None):
    """Execute all models and save results."""
    model_cfg = model_cfg or load_config()["models"]
    print("=" * 60)
    print("Audio Telemetry ML Pipeline")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    sessions_full = pd.read_parquet(os.path.join(data_dir, "sessions.parquet"))
    device_profile = pd.read_parquet(os.path.join(data_dir, "device_profile.parquet"))

    # Sample for training efficiency (full dataset used in production)
    sessions = sessions_full.sample(
        n=min(model_cfg["sample_size"], len(sessions_full)), random_state=42
    )
    print(f"Loaded {len(sessions_full):,} sessions (sampled {len(sessions):,}), "
          f"{len(device_profile):,} device profiles")

    # Run models
    anc = ANCPredictor(**model_cfg["anc_predictor"])
    anc_result = anc.train(sessions)

    seg = UserBehaviorSegmentation(
        n_clusters=model_cfg["user_segmentation"]["max_clusters"]
    )
    seg_result = seg.train(device_profile)

    anomaly = AudioQualityAnomalyDetector(
        contamination=model_cfg["anomaly_detection"]["contamination"]
    )
    anomaly_result = anomaly.train(sessions)

    duration = SessionDurationForecaster()
    duration_result = duration.train(sessions)

    # Statistical analysis
    stat_results = run_statistical_analysis(sessions)

    # Save all results
    all_results = {
        "anc_predictor": anc_result.metrics,
        "user_segmentation": seg_result.metrics,
        "anomaly_detection": anomaly_result.metrics,
        "duration_forecaster": duration_result.metrics,
        "statistical_analysis": {
            k: v for k, v in stat_results.items()
            if k != "correlation_matrix"
        },
        "cluster_profiles": seg_result.artifacts.get("cluster_profiles", {}),
        "anomaly_comparison": anomaly_result.artifacts.get("comparison", {}),
        "anc_feature_importance": anc_result.artifacts.get("feature_importance", {}),
    }

    with open(os.path.join(output_dir, "model_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=_json_default)

    # Save models
    joblib.dump(anc_result.model, os.path.join(output_dir, "anc_predictor.joblib"))
    joblib.dump(seg_result.model, os.path.join(output_dir, "user_segmentation.joblib"))
    joblib.dump(anomaly_result.model, os.path.join(output_dir, "anomaly_detector.joblib"))
    joblib.dump(duration_result.model, os.path.join(output_dir, "duration_forecaster.joblib"))

    print("\n" + "=" * 60)
    print("All models trained and saved!")
    print("=" * 60)

    return all_results


def main():
    cfg = load_config()
    run_all_models(
        resolve_path(cfg["pipeline"]["processed_data_path"]),
        resolve_path(cfg["pipeline"]["model_output_path"]),
        cfg["models"],
    )


if __name__ == "__main__":
    main()
