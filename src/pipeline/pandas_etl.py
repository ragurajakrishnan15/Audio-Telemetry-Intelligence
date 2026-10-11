"""
Audio Telemetry ETL Pipeline (Pandas fallback)
=================================================
Mirrors the PySpark pipeline using Pandas for environments
where PySpark is unavailable. The PySpark version (spark_etl.py)
is the production-grade implementation.

Transforms raw telemetry data into analytical datasets:
  1. Data quality validation & cleansing
  2. Feature engineering for ML consumption
  3. Aggregated analytical views
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path


class AudioTelemetryPipelinePandas:
    """Pandas-based ETL pipeline (dev/demo fallback)."""

    def __init__(self, raw_path: str, output_path: str,
                 iqr_multiplier: float = 3.0):
        self.raw_path = raw_path
        self.output_path = output_path
        self.iqr_multiplier = iqr_multiplier
        self.quality_report = {}

    def extract(self) -> dict[str, pd.DataFrame]:
        print("\n[EXTRACT] Loading raw datasets...")
        dfs = {}
        for name in ["devices", "sessions", "system_adjustments"]:
            path = os.path.join(self.raw_path, f"{name}.parquet")
            df = pd.read_parquet(path)
            dfs[name] = df
            print(f"  {name}: {len(df):,} rows × {len(df.columns)} cols")
        return dfs

    def validate_quality(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        print("\n[QUALITY] Running data quality checks...")
        sessions = dfs["sessions"]
        total = len(sessions)

        critical_cols = ["session_id", "device_id", "session_start",
                         "duration_seconds", "audio_mode"]
        null_counts = {col: int(sessions[col].isna().sum()) for col in critical_cols}
        print(f"  Null checks: {null_counts}")

        dup_count = int(sessions.duplicated(subset=["session_id"]).sum())
        print(f"  Duplicate sessions: {dup_count}")

        range_violations = int(((sessions["duration_seconds"] < 0) |
                                (sessions["duration_seconds"] > 86400) |
                                (sessions["volume_level"] < 0) |
                                (sessions["volume_level"] > 100) |
                                (sessions["anc_level"] < 0) |
                                (sessions["anc_level"] > 1.0) |
                                (sessions["ambient_noise_db"] < 0) |
                                (sessions["ambient_noise_db"] > 140)).sum())
        print(f"  Range violations: {range_violations}")

        q1 = sessions["duration_seconds"].quantile(0.25)
        q3 = sessions["duration_seconds"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        sessions_clean = sessions[
            (sessions["duration_seconds"] >= lower) &
            (sessions["duration_seconds"] <= upper)
        ].drop_duplicates(subset=["session_id"])

        removed = total - len(sessions_clean)
        print(f"  Records removed: {removed} ({removed/total*100:.2f}%)")

        self.quality_report = {
            "total_raw": total,
            "nulls": null_counts,
            "duplicates": dup_count,
            "range_violations": range_violations,
            "outliers_removed": int(removed),
            "clean_count": len(sessions_clean),
            "quality_score": round((1 - removed / total) * 100, 2),
        }

        dfs["sessions"] = sessions_clean
        return dfs

    def engineer_features(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        print("\n[TRANSFORM] Engineering features...")
        s = dfs["sessions"].copy()

        s["session_ts"] = pd.to_datetime(s["session_start"])
        s["hour_of_day"] = s["session_ts"].dt.hour
        s["day_of_week"] = s["session_ts"].dt.dayofweek
        s["month"] = s["session_ts"].dt.month
        s["is_weekend"] = s["day_of_week"].isin([5, 6])

        s["time_bucket"] = pd.cut(
            s["hour_of_day"],
            bins=[-1, 5, 11, 17, 22, 24],
            labels=["night", "morning", "afternoon", "evening", "night_late"],
        ).astype(str).replace("night_late", "night")

        s["audio_quality_index"] = (
            (100 - s["audio_glitches"] * 15) *
            (1 - np.minimum(s["latency_ms"] / 100, 1.0)) *
            s["voice_isolation_score"].fillna(85).clip(0, 100) / 100
        ).round(2)

        s["anc_effectiveness"] = "low"
        s.loc[s["anc_level"] == 0, "anc_effectiveness"] = "none"
        s.loc[
            (s["anc_level"] > 0) & (s["ambient_noise_db"] > 60) &
            (s["audio_glitches"] == 0), "anc_effectiveness"
        ] = "high"
        s.loc[
            (s["anc_level"] > 0) & (s["ambient_noise_db"] > 40) &
            (s["anc_effectiveness"] != "high"), "anc_effectiveness"
        ] = "medium"

        s["engagement_tier"] = pd.cut(
            s["duration_seconds"],
            bins=[-1, 120, 900, 3600, float("inf")],
            labels=["brief", "casual", "engaged", "power_user"],
        ).astype(str)

        s["noise_category"] = pd.cut(
            s["ambient_noise_db"],
            bins=[-float("inf"), 35, 50, 65, 80, float("inf")],
            labels=["silent", "quiet", "moderate", "loud", "very_loud"],
            right=False,  # [lo, hi) — matches Spark's `< threshold` chain
        ).astype(str)

        mode_mult = s["audio_mode"].isin(
            ["music_streaming", "podcast", "media_playback"]
        ).astype(float) * 0.3 + 1.0
        env_mult = s["environment_type"].isin(
            ["commute_transit", "airplane", "gym"]
        ).astype(float) * 0.5 + 1.0
        s["anc_demand_score"] = (
            s["ambient_noise_db"] / 100 * mode_mult * env_mult
        ).round(3)

        dfs["sessions"] = s
        print("  Added 10 engineered features")
        return dfs

    def build_aggregations(self, dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        print("\n[TRANSFORM] Building aggregated views...")
        s = dfs["sessions"]

        # Hourly rollup
        # Group on a temporary Series so no `date` column leaks into sessions.
        hourly = s.groupby(
            [s["session_ts"].dt.date.rename("date"), "hour_of_day", "device_type"]
        ).agg(
            session_count=("session_id", "count"),
            avg_duration_sec=("duration_seconds", "mean"),
            avg_anc_level=("anc_level", "mean"),
            avg_ambient_noise_db=("ambient_noise_db", "mean"),
            avg_volume=("volume_level", "mean"),
            avg_quality_index=("audio_quality_index", "mean"),
            avg_satisfaction=("satisfaction_proxy", "mean"),
            total_glitches=("audio_glitches", "sum"),
            avg_latency_ms=("latency_ms", "mean"),
            avg_battery_drain=("battery_drain_rate_pct_hr", "mean"),
            unique_devices=("device_id", "nunique"),
        ).reset_index()
        dfs["hourly_agg"] = hourly
        print(f"  hourly_agg: {len(hourly):,} rows")

        # Device profile
        profile = s.groupby(["device_id", "device_type", "region"]).agg(
            total_sessions=("session_id", "count"),
            total_listening_sec=("duration_seconds", "sum"),
            avg_session_duration=("duration_seconds", "mean"),
            preferred_anc_level=("anc_level", "mean"),
            preferred_volume=("volume_level", "mean"),
            avg_satisfaction=("satisfaction_proxy", "mean"),
            avg_quality_index=("audio_quality_index", "mean"),
            primary_audio_mode=("audio_mode", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
            spatial_audio_sessions=("spatial_audio_enabled", "sum"),
            conversation_awareness_events=("conversation_awareness_triggered", "sum"),
            avg_battery_drain=("battery_drain_rate_pct_hr", "mean"),
            total_glitches=("audio_glitches", "sum"),
            first_session=("session_ts", "min"),
            last_session=("session_ts", "max"),
        ).reset_index()
        # Calendar-day differences (same as Spark's datediff), not elapsed 24h blocks.
        first_day = profile["first_session"].dt.normalize()
        profile["days_active"] = (
            profile["last_session"].dt.normalize() - first_day
        ).dt.days
        # Sessions per day over the device's observed window (first session ->
        # end of dataset). Dividing by days_active instead makes single-session
        # devices look like heavy users (1 session / max(0,1) days).
        observed_days = (s["session_ts"].max().normalize() - first_day).dt.days + 1
        profile["sessions_per_day"] = (
            profile["total_sessions"] / observed_days.clip(lower=1)
        ).round(3)
        dfs["device_profile"] = profile
        print(f"  device_profile: {len(profile):,} rows")

        # Environment analysis
        env = s.groupby(["environment_type", "noise_category", "device_type"]).agg(
            session_count=("session_id", "count"),
            avg_anc_level=("anc_level", "mean"),
            avg_voice_isolation=("voice_isolation_score", "mean"),
            avg_quality_index=("audio_quality_index", "mean"),
            avg_satisfaction=("satisfaction_proxy", "mean"),
            avg_duration=("duration_seconds", "mean"),
            total_glitches=("audio_glitches", "sum"),
            avg_anc_demand=("anc_demand_score", "mean"),
        ).reset_index()
        dfs["env_analysis"] = env
        print(f"  env_analysis: {len(env):,} rows")

        return dfs

    def load(self, dfs: dict[str, pd.DataFrame]):
        print("\n[LOAD] Writing processed datasets...")
        os.makedirs(self.output_path, exist_ok=True)

        for name in ["sessions", "hourly_agg", "device_profile", "env_analysis"]:
            if name in dfs:
                path = os.path.join(self.output_path, f"{name}.parquet")
                # Microsecond timestamps: Spark 3.5 cannot read pandas 3's
                # default nanosecond parquet timestamps.
                dfs[name].to_parquet(
                    path, index=False, coerce_timestamps="us",
                    allow_truncated_timestamps=True,
                )
                print(f"  {name} → {path}")

        report_path = os.path.join(self.output_path, "quality_report.json")
        with open(report_path, "w") as f:
            json.dump(self.quality_report, f, indent=2)
        print(f"  quality_report → {report_path}")

    def run(self):
        print("=" * 60)
        print("Audio Telemetry ETL Pipeline (Pandas)")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)

        dfs = self.extract()
        dfs = self.validate_quality(dfs)
        dfs = self.engineer_features(dfs)
        dfs = self.build_aggregations(dfs)
        self.load(dfs)

        print(f"\nQuality score: {self.quality_report.get('quality_score', 'N/A')}%")
        print("Pipeline completed!")


def main():
    cfg = load_config()["pipeline"]
    pipeline = AudioTelemetryPipelinePandas(
        resolve_path(cfg["raw_data_path"]),
        resolve_path(cfg["processed_data_path"]),
        iqr_multiplier=cfg["quality_checks"]["iqr_multiplier"],
    )
    pipeline.run()


if __name__ == "__main__":
    main()
