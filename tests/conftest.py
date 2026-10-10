"""Shared fixtures: a small synthetic dataset pushed through the Pandas ETL once."""

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_generation.generate_telemetry import (  # noqa: E402
    generate_devices, generate_sessions, generate_system_adjustments,
)
from src.pipeline.pandas_etl import AudioTelemetryPipelinePandas  # noqa: E402

N_DEVICES = 300
N_SESSIONS = 4000


@pytest.fixture(scope="session")
def small_run(tmp_path_factory):
    """Generate a small dataset, run the Pandas ETL, return paths and frames."""
    base = tmp_path_factory.mktemp("telemetry")
    raw, processed = base / "raw", base / "processed"
    raw.mkdir()

    devices = generate_devices(N_DEVICES)
    sessions = generate_sessions(devices, N_SESSIONS)
    adjustments = generate_system_adjustments(sessions)
    devices.to_parquet(raw / "devices.parquet", index=False)
    sessions.to_parquet(raw / "sessions.parquet", index=False)
    adjustments.to_parquet(raw / "system_adjustments.parquet", index=False)

    pipeline = AudioTelemetryPipelinePandas(str(raw), str(processed))
    pipeline.run()

    return {
        "raw": str(raw),
        "processed": str(processed),
        "pipeline": pipeline,
        "raw_sessions": sessions,
        "sessions": pd.read_parquet(processed / "sessions.parquet"),
        "device_profile": pd.read_parquet(processed / "device_profile.parquet"),
        "hourly_agg": pd.read_parquet(processed / "hourly_agg.parquet"),
        "env_analysis": pd.read_parquet(processed / "env_analysis.parquet"),
    }
