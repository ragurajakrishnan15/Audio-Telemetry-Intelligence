"""
Spark <-> Pandas ETL parity.

Skipped automatically unless PySpark is installed AND a JVM is on PATH. These
tests have not been executed in the original development environment (no JVM),
so a first run on a Spark-capable machine may surface real differences.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java runtime not found; Spark tests need a JVM",
                allow_module_level=True)

from src.pipeline.spark_etl import AudioTelemetryPipeline  # noqa: E402


@pytest.fixture(scope="module")
def spark_run(small_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("spark_out")
    AudioTelemetryPipeline(
        small_run["raw"], str(out),
        spark_cfg={"master": "local[2]", "shuffle_partitions": 2,
                   "driver_memory": "1g"},
    ).run()
    return {
        "sessions": pd.read_parquet(out / "sessions.parquet"),
        "device_profile": pd.read_parquet(out / "device_profile.parquet"),
    }


FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "time_bucket",
            "anc_effectiveness", "engagement_tier", "noise_category"]


def test_row_sets_identical(small_run, spark_run):
    # Spark computes exact quantiles (relativeError=0), so the IQR fences and
    # therefore the retained rows must match pandas exactly.
    assert set(small_run["sessions"]["session_id"]) == set(
        spark_run["sessions"]["session_id"]
    )


def test_categorical_features_match(small_run, spark_run):
    m = small_run["sessions"].merge(
        spark_run["sessions"], on="session_id", suffixes=("_p", "_s")
    )
    for col in FEATURES:
        assert (m[f"{col}_p"].astype(str) == m[f"{col}_s"].astype(str)).all(), col


def test_numeric_features_match(small_run, spark_run):
    m = small_run["sessions"].merge(
        spark_run["sessions"], on="session_id", suffixes=("_p", "_s")
    )
    for col in ["audio_quality_index", "anc_demand_score"]:
        np.testing.assert_allclose(m[f"{col}_p"], m[f"{col}_s"], atol=0.011)


def test_device_profile_parity(small_run, spark_run):
    m = small_run["device_profile"].merge(
        spark_run["device_profile"], on="device_id", suffixes=("_p", "_s")
    )
    assert len(m) > 0.99 * len(small_run["device_profile"])
    assert (m["primary_audio_mode_p"] == m["primary_audio_mode_s"]).mean() > 0.99
    np.testing.assert_allclose(
        m["sessions_per_day_p"], m["sessions_per_day_s"], atol=0.002
    )
