"""
Audio Telemetry ETL Pipeline (PySpark)
========================================
Transforms raw telemetry data into analytical datasets:
  1. Data quality validation & cleansing
  2. Feature engineering for ML consumption
  3. Aggregated analytical views (hourly, daily, device-level)
  4. Session behavior segmentation features

Tech stack: PySpark, Parquet, SQL-based transformations.
"""

import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from src.config import load_config, resolve_path


class AudioTelemetryPipeline:
    """End-to-end ETL pipeline for audio telemetry data."""

    def __init__(self, raw_path: str, output_path: str,
                 iqr_multiplier: float = 3.0, spark_cfg: dict | None = None):
        self.raw_path = raw_path
        self.output_path = output_path
        self.iqr_multiplier = iqr_multiplier
        self.spark = self._create_spark_session(spark_cfg or {})
        self.quality_report = {}

    @staticmethod
    def _create_spark_session(cfg: dict) -> SparkSession:
        return (
            SparkSession.builder
            .appName(cfg.get("app_name", "AudioTelemetryPipeline"))
            .master(cfg.get("master", "local[*]"))
            .config("spark.sql.shuffle.partitions",
                    str(cfg.get("shuffle_partitions", 8)))
            .config("spark.driver.memory", cfg.get("driver_memory", "4g"))
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------

    def extract(self) -> dict[str, DataFrame]:
        """Load raw parquet files into Spark DataFrames."""
        print("\n[EXTRACT] Loading raw datasets...")

        dfs = {}
        for name in ["devices", "sessions", "system_adjustments"]:
            path = os.path.join(self.raw_path, f"{name}.parquet")
            df = self.spark.read.parquet(path)
            dfs[name] = df
            print(f"  {name}: {df.count():,} rows × {len(df.columns)} cols")

        return dfs

    # ------------------------------------------------------------------
    # Data Quality Validation
    # ------------------------------------------------------------------

    def validate_quality(self, dfs: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """Run data quality checks and flag/remove bad records."""
        print("\n[QUALITY] Running data quality checks...")

        sessions = dfs["sessions"]
        total = sessions.count()

        # 1. Null checks on critical fields
        critical_cols = ["session_id", "device_id", "session_start",
                         "duration_seconds", "audio_mode"]
        null_counts = {}
        for col in critical_cols:
            nc = sessions.filter(F.col(col).isNull()).count()
            null_counts[col] = nc
        print(f"  Null checks: {null_counts}")

        # 2. Duplicate session IDs
        dup_count = total - sessions.dropDuplicates(["session_id"]).count()
        print(f"  Duplicate sessions: {dup_count}")

        # 3. Range validation
        range_violations = sessions.filter(
            (F.col("duration_seconds") < 0) |
            (F.col("duration_seconds") > 86400) |
            (F.col("volume_level") < 0) |
            (F.col("volume_level") > 100) |
            (F.col("anc_level") < 0) |
            (F.col("anc_level") > 1.0) |
            (F.col("ambient_noise_db") < 0) |
            (F.col("ambient_noise_db") > 140)
        ).count()
        print(f"  Range violations: {range_violations}")

        # 4. Outlier detection using IQR on duration
        # relativeError=0.0 -> exact quantiles, so the fences (and therefore the
        # kept rows) match pandas_etl. Raise it (e.g. 0.01) for very large inputs.
        quantiles = sessions.approxQuantile("duration_seconds", [0.25, 0.75], 0.0)
        iqr = quantiles[1] - quantiles[0]
        lower_bound = quantiles[0] - self.iqr_multiplier * iqr
        upper_bound = quantiles[1] + self.iqr_multiplier * iqr

        sessions_clean = sessions.filter(
            (F.col("duration_seconds") >= lower_bound) &
            (F.col("duration_seconds") <= upper_bound)
        ).dropDuplicates(["session_id"])

        removed = total - sessions_clean.count()
        print(f"  Records removed (outliers + duplicates): {removed} ({removed/total*100:.2f}%)")

        self.quality_report = {
            "total_raw": total,
            "nulls": null_counts,
            "duplicates": dup_count,
            "range_violations": range_violations,
            "outliers_removed": removed,
            "clean_count": sessions_clean.count(),
            "quality_score": round((1 - removed / total) * 100, 2),
        }

        dfs["sessions"] = sessions_clean
        return dfs

    # ------------------------------------------------------------------
    # Transform — Feature Engineering
    # ------------------------------------------------------------------

    def engineer_features(self, dfs: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """Create analytical features from raw telemetry."""
        print("\n[TRANSFORM] Engineering features...")

        sessions = dfs["sessions"]

        # Temporal features
        sessions = (
            sessions
            .withColumn("session_ts", F.to_timestamp("session_start"))
            .withColumn("hour_of_day", F.hour("session_ts"))
            # Spark dayofweek: 1=Sun..7=Sat -> pandas convention 0=Mon..6=Sun
            .withColumn("day_of_week", (F.dayofweek("session_ts") + 5) % 7)
            .withColumn("month", F.month("session_ts"))
            .withColumn("is_weekend", F.when(
                F.dayofweek("session_ts").isin(1, 7), True
            ).otherwise(False))
            .withColumn("time_bucket", F.when(
                F.col("hour_of_day").between(6, 11), "morning"
            ).when(
                F.col("hour_of_day").between(12, 17), "afternoon"
            ).when(
                F.col("hour_of_day").between(18, 22), "evening"
            ).otherwise("night"))
        )

        # Audio experience quality index (composite metric)
        sessions = sessions.withColumn(
            "audio_quality_index",
            F.round(
                (100 - F.col("audio_glitches") * 15) *
                (1 - F.least(F.col("latency_ms") / 100, F.lit(1.0))) *
                F.when(
                    F.col("voice_isolation_score").isNotNull(),
                    F.col("voice_isolation_score") / 100
                ).otherwise(F.lit(0.85)),
                2
            )
        )

        # ANC effectiveness category
        sessions = sessions.withColumn(
            "anc_effectiveness",
            F.when(F.col("anc_level") == 0, "none")
            .when(
                (F.col("anc_level") > 0) & (F.col("ambient_noise_db") > 60) &
                (F.col("audio_glitches") == 0), "high"
            ).when(
                (F.col("anc_level") > 0) & (F.col("ambient_noise_db") > 40), "medium"
            ).otherwise("low")
        )

        # Session engagement tier
        sessions = sessions.withColumn(
            "engagement_tier",
            F.when(F.col("duration_seconds") > 3600, "power_user")
            .when(F.col("duration_seconds") > 900, "engaged")
            .when(F.col("duration_seconds") > 120, "casual")
            .otherwise("brief")
        )

        # Noise environment category
        sessions = sessions.withColumn(
            "noise_category",
            F.when(F.col("ambient_noise_db") < 35, "silent")
            .when(F.col("ambient_noise_db") < 50, "quiet")
            .when(F.col("ambient_noise_db") < 65, "moderate")
            .when(F.col("ambient_noise_db") < 80, "loud")
            .otherwise("very_loud")
        )

        # Feature interaction: ANC demand score
        sessions = sessions.withColumn(
            "anc_demand_score",
            F.round(
                F.col("ambient_noise_db") / 100 *
                F.when(F.col("audio_mode").isin(
                    "music_streaming", "podcast", "media_playback"
                ), 1.3).otherwise(1.0) *
                F.when(F.col("environment_type").isin(
                    "commute_transit", "airplane", "gym"
                ), 1.5).otherwise(1.0),
                3
            )
        )

        dfs["sessions"] = sessions
        print("  Added 10 engineered features")
        return dfs

    # ------------------------------------------------------------------
    # Transform — Aggregations
    # ------------------------------------------------------------------

    def build_aggregations(self, dfs: dict[str, DataFrame]) -> dict[str, DataFrame]:
        """Build analytical aggregate datasets."""
        print("\n[TRANSFORM] Building aggregated views...")

        sessions = dfs["sessions"]

        # 1. Hourly telemetry rollup
        hourly_agg = (
            sessions
            .withColumn("date", F.to_date("session_ts"))
            .groupBy("date", "hour_of_day", "device_type")
            .agg(
                F.count("*").alias("session_count"),
                F.avg("duration_seconds").alias("avg_duration_sec"),
                F.avg("anc_level").alias("avg_anc_level"),
                F.avg("ambient_noise_db").alias("avg_ambient_noise_db"),
                F.avg("volume_level").alias("avg_volume"),
                F.avg("audio_quality_index").alias("avg_quality_index"),
                F.avg("satisfaction_proxy").alias("avg_satisfaction"),
                F.sum("audio_glitches").alias("total_glitches"),
                F.avg("latency_ms").alias("avg_latency_ms"),
                F.avg("battery_drain_rate_pct_hr").alias("avg_battery_drain"),
                F.countDistinct("device_id").alias("unique_devices"),
            )
        )
        dfs["hourly_agg"] = hourly_agg
        print(f"  hourly_agg: {hourly_agg.count():,} rows")

        # 2. Device-level behavioral profile
        dataset_end = sessions.agg(F.max("session_ts")).first()[0]
        device_profile = (
            sessions
            .groupBy("device_id", "device_type", "region")
            .agg(
                F.count("*").alias("total_sessions"),
                F.sum("duration_seconds").alias("total_listening_sec"),
                F.avg("duration_seconds").alias("avg_session_duration"),
                F.avg("anc_level").alias("preferred_anc_level"),
                F.avg("volume_level").alias("preferred_volume"),
                F.avg("satisfaction_proxy").alias("avg_satisfaction"),
                F.avg("audio_quality_index").alias("avg_quality_index"),

                # Feature adoption
                F.sum(F.when(F.col("spatial_audio_enabled"), 1).otherwise(0))
                .alias("spatial_audio_sessions"),
                F.sum(F.when(F.col("conversation_awareness_triggered"), 1).otherwise(0))
                .alias("conversation_awareness_events"),

                F.avg("battery_drain_rate_pct_hr").alias("avg_battery_drain"),
                F.sum("audio_glitches").alias("total_glitches"),
                F.min("session_ts").alias("first_session"),
                F.max("session_ts").alias("last_session"),
            )
            .withColumn(
                "days_active",
                F.datediff("last_session", "first_session")
            )
            # Sessions per day over the observed window (first session -> end of
            # dataset); matches pandas_etl. Dividing by days_active would make
            # single-session devices look like heavy users.
            .withColumn(
                "observed_days",
                F.greatest(F.datediff(F.lit(dataset_end), "first_session") + 1,
                           F.lit(1))
            )
            .withColumn(
                "sessions_per_day",
                F.round(F.col("total_sessions") / F.col("observed_days"), 3)
            )
            .drop("observed_days")
        )
        # Most common audio mode per device (ties -> alphabetically first,
        # matching pandas' Series.mode().iloc[0])
        mode_window = Window.partitionBy("device_id").orderBy(
            F.col("n").desc(), F.col("audio_mode").asc()
        )
        primary_mode = (
            sessions.groupBy("device_id", "audio_mode").agg(F.count("*").alias("n"))
            .withColumn("rn", F.row_number().over(mode_window))
            .filter(F.col("rn") == 1)
            .select("device_id", F.col("audio_mode").alias("primary_audio_mode"))
        )
        device_profile = device_profile.join(primary_mode, "device_id", "left")

        dfs["device_profile"] = device_profile
        print(f"  device_profile: {device_profile.count():,} rows")

        # 3. Environment impact analysis
        env_analysis = (
            sessions
            .groupBy("environment_type", "noise_category", "device_type")
            .agg(
                F.count("*").alias("session_count"),
                F.avg("anc_level").alias("avg_anc_level"),
                F.avg("voice_isolation_score").alias("avg_voice_isolation"),
                F.avg("audio_quality_index").alias("avg_quality_index"),
                F.avg("satisfaction_proxy").alias("avg_satisfaction"),
                F.avg("duration_seconds").alias("avg_duration"),
                F.sum("audio_glitches").alias("total_glitches"),
                F.avg("anc_demand_score").alias("avg_anc_demand"),
            )
        )
        dfs["env_analysis"] = env_analysis
        print(f"  env_analysis: {env_analysis.count():,} rows")

        return dfs

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, dfs: dict[str, DataFrame]):
        """Write processed datasets to parquet."""
        print("\n[LOAD] Writing processed datasets...")

        os.makedirs(self.output_path, exist_ok=True)

        datasets_to_write = [
            "sessions", "hourly_agg", "device_profile", "env_analysis"
        ]

        for name in datasets_to_write:
            if name in dfs:
                path = os.path.join(self.output_path, f"{name}.parquet")
                # Microsecond timestamps keep the files readable by Spark 3.5
                # (pandas 3 would otherwise write nanosecond timestamps).
                dfs[name].toPandas().to_parquet(
                    path, index=False, coerce_timestamps="us",
                    allow_truncated_timestamps=True,
                )
                print(f"  {name} → {path}")

        # Write quality report
        import json
        report_path = os.path.join(self.output_path, "quality_report.json")
        with open(report_path, "w") as f:
            json.dump(self.quality_report, f, indent=2)
        print(f"  quality_report → {report_path}")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full ETL pipeline."""
        print("=" * 60)
        print("Audio Telemetry ETL Pipeline")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)

        dfs = self.extract()
        dfs = self.validate_quality(dfs)
        dfs = self.engineer_features(dfs)
        dfs = self.build_aggregations(dfs)
        self.load(dfs)

        self.spark.stop()

        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Quality score: {self.quality_report.get('quality_score', 'N/A')}%")
        print("=" * 60)


def main():
    cfg = load_config()["pipeline"]
    pipeline = AudioTelemetryPipeline(
        resolve_path(cfg["raw_data_path"]),
        resolve_path(cfg["processed_data_path"]),
        iqr_multiplier=cfg["quality_checks"]["iqr_multiplier"],
        spark_cfg=cfg["spark"],
    )
    pipeline.run()


if __name__ == "__main__":
    main()
