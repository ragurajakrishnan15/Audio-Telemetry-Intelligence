"""Tests for the Pandas ETL: feature logic, quality scoring, aggregations."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.pipeline.pandas_etl import AudioTelemetryPipelinePandas


def _pipe() -> AudioTelemetryPipelinePandas:
    return AudioTelemetryPipelinePandas("unused", "unused")


def _row(**overrides) -> dict:
    base = {
        "session_start": "2024-01-08T12:00:00",  # a Monday
        "duration_seconds": 600,
        "audio_glitches": 0,
        "latency_ms": 10.0,
        "voice_isolation_score": 80.0,
        "anc_level": 0.0,
        "ambient_noise_db": 50.0,
        "audio_mode": "phone_call",
        "environment_type": "quiet_indoor",
    }
    base.update(overrides)
    return base


def _features(rows: list[dict]) -> pd.DataFrame:
    out = _pipe().engineer_features({"sessions": pd.DataFrame(rows)})
    return out["sessions"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestFeatureLogic:
    @pytest.mark.parametrize("hour,expected", [
        (0, "night"), (5, "night"), (6, "morning"), (11, "morning"),
        (12, "afternoon"), (17, "afternoon"), (18, "evening"),
        (22, "evening"), (23, "night"),
    ])
    def test_time_bucket(self, hour, expected):
        df = _features([_row(session_start=f"2024-01-08T{hour:02d}:30:00")])
        assert df.loc[0, "time_bucket"] == expected

    def test_weekday_convention_is_monday_zero(self):
        df = _features([
            _row(session_start="2024-01-06T10:00:00"),  # Saturday
            _row(session_start="2024-01-07T10:00:00"),  # Sunday
            _row(session_start="2024-01-08T10:00:00"),  # Monday
        ])
        assert df["day_of_week"].tolist() == [5, 6, 0]
        assert df["is_weekend"].tolist() == [True, True, False]

    @pytest.mark.parametrize("duration,expected", [
        (120, "brief"), (121, "casual"), (900, "casual"),
        (901, "engaged"), (3600, "engaged"), (3601, "power_user"),
    ])
    def test_engagement_tier_boundaries(self, duration, expected):
        df = _features([_row(duration_seconds=duration)])
        assert df.loc[0, "engagement_tier"] == expected

    @pytest.mark.parametrize("noise,expected", [
        (34.9, "silent"), (35.0, "quiet"), (49.9, "quiet"), (50.0, "moderate"),
        (64.9, "moderate"), (65.0, "loud"), (79.9, "loud"), (80.0, "very_loud"),
    ])
    def test_noise_category_boundaries(self, noise, expected):
        # Lower bound inclusive, matching Spark's `< threshold` chain.
        df = _features([_row(ambient_noise_db=noise)])
        assert df.loc[0, "noise_category"] == expected

    def test_anc_effectiveness(self):
        df = _features([
            _row(anc_level=0.0),
            _row(anc_level=0.5, ambient_noise_db=70, audio_glitches=0),
            _row(anc_level=0.5, ambient_noise_db=70, audio_glitches=2),
            _row(anc_level=0.5, ambient_noise_db=30),
            _row(anc_level=0.5, ambient_noise_db=45, audio_glitches=0),
        ])
        assert df["anc_effectiveness"].tolist() == [
            "none", "high", "medium", "low", "medium"
        ]

    def test_anc_demand_score(self):
        df = _features([
            _row(ambient_noise_db=50, audio_mode="music_streaming",
                 environment_type="gym"),
            _row(ambient_noise_db=50),
        ])
        assert df.loc[0, "anc_demand_score"] == pytest.approx(0.975)
        assert df.loc[1, "anc_demand_score"] == pytest.approx(0.5)

    def test_audio_quality_index(self):
        df = _features([
            _row(audio_glitches=0, latency_ms=0, voice_isolation_score=100),
            _row(audio_glitches=0, latency_ms=50, voice_isolation_score=np.nan),
            _row(audio_glitches=2, latency_ms=500, voice_isolation_score=90),
        ])
        assert df.loc[0, "audio_quality_index"] == pytest.approx(100.0)
        # null voice score defaults to 85 -> 100 * 0.5 * 0.85
        assert df.loc[1, "audio_quality_index"] == pytest.approx(42.5)
        # latency saturates at 100 ms -> index floors at 0
        assert df.loc[2, "audio_quality_index"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

def _quality_frame(n: int = 1000) -> pd.DataFrame:
    return pd.DataFrame({
        "session_id": [f"s{i}" for i in range(n)],
        "device_id": [f"d{i % 50}" for i in range(n)],
        "session_start": "2024-01-01T00:00:00",
        "duration_seconds": (np.arange(n) % 900) + 100,
        "audio_mode": "podcast",
        "volume_level": 50.0,
        "anc_level": 0.0,
        "ambient_noise_db": 40.0,
    })


class TestQualityValidation:
    def test_clean_data_loses_nothing(self):
        pipe = _pipe()
        out = pipe.validate_quality({"sessions": _quality_frame()})
        assert len(out["sessions"]) == 1000
        assert pipe.quality_report["quality_score"] == 100.0
        assert pipe.quality_report["duplicates"] == 0

    def test_outlier_and_duplicate_are_removed_and_scored(self):
        df = _quality_frame()
        extreme = df.iloc[[0]].copy()
        extreme["session_id"] = "extreme"
        extreme["duration_seconds"] = 10_000_000
        dupe = df.iloc[[5]].copy()  # same session_id as an existing row
        df = pd.concat([df, extreme, dupe], ignore_index=True)

        pipe = _pipe()
        out = pipe.validate_quality({"sessions": df})
        clean = out["sessions"]

        assert "extreme" not in set(clean["session_id"])
        assert clean["session_id"].is_unique
        report = pipe.quality_report
        assert report["total_raw"] == 1002
        assert report["duplicates"] == 1
        assert report["clean_count"] == len(clean) == 1000
        assert report["outliers_removed"] == 2
        assert report["quality_score"] == pytest.approx(99.8)

    def test_range_violations_counted(self):
        df = _quality_frame()
        df.loc[0, "volume_level"] = 150
        df.loc[1, "anc_level"] = -0.1
        df.loc[2, "ambient_noise_db"] = 200
        pipe = _pipe()
        pipe.validate_quality({"sessions": df})
        assert pipe.quality_report["range_violations"] == 3

    def test_null_counts_reported(self):
        df = _quality_frame()
        df.loc[:9, "audio_mode"] = None
        pipe = _pipe()
        pipe.validate_quality({"sessions": df})
        assert pipe.quality_report["nulls"]["audio_mode"] == 10

    def test_iqr_multiplier_is_respected(self):
        df = _quality_frame()
        df.loc[0, "duration_seconds"] = 1500  # inside 3.0*IQR fence, outside 0.5*IQR
        loose = AudioTelemetryPipelinePandas("", "", iqr_multiplier=3.0)
        tight = AudioTelemetryPipelinePandas("", "", iqr_multiplier=0.5)
        n_loose = len(loose.validate_quality({"sessions": df.copy()})["sessions"])
        n_tight = len(tight.validate_quality({"sessions": df.copy()})["sessions"])
        assert n_tight < n_loose


# ---------------------------------------------------------------------------
# End-to-end on a small dataset
# ---------------------------------------------------------------------------

class TestPipelineOutputs:
    def test_all_outputs_written(self, small_run):
        for name in ["sessions", "hourly_agg", "device_profile", "env_analysis"]:
            assert os.path.exists(
                os.path.join(small_run["processed"], f"{name}.parquet")
            )
        with open(os.path.join(small_run["processed"], "quality_report.json")) as f:
            report = json.load(f)
        assert report["total_raw"] == len(small_run["raw_sessions"])

    def test_engineered_columns_present(self, small_run):
        expected = {
            "hour_of_day", "day_of_week", "month", "is_weekend", "time_bucket",
            "audio_quality_index", "anc_effectiveness", "engagement_tier",
            "noise_category", "anc_demand_score",
        }
        assert expected.issubset(small_run["sessions"].columns)

    def test_parquet_timestamps_are_spark_readable(self, small_run):
        import pyarrow as pa
        import pyarrow.parquet as pq

        for name in ["sessions", "device_profile"]:
            schema = pq.read_schema(
                os.path.join(small_run["processed"], f"{name}.parquet")
            )
            for field in schema:
                if pa.types.is_timestamp(field.type):
                    assert field.type.unit != "ns", (name, field.name)

    def test_categorical_features_have_no_unknowns(self, small_run):
        s = small_run["sessions"]
        assert set(s["time_bucket"]) <= {"night", "morning", "afternoon", "evening"}
        assert set(s["engagement_tier"]) <= {"brief", "casual", "engaged", "power_user"}
        assert set(s["noise_category"]) <= {
            "silent", "quiet", "moderate", "loud", "very_loud"
        }
        assert set(s["anc_effectiveness"]) <= {"none", "low", "medium", "high"}

    def test_aggregations_conserve_session_counts(self, small_run):
        n = len(small_run["sessions"])
        assert small_run["hourly_agg"]["session_count"].sum() == n
        assert small_run["device_profile"]["total_sessions"].sum() == n
        assert small_run["env_analysis"]["session_count"].sum() == n

    def test_device_profile_keys_unique(self, small_run):
        assert small_run["device_profile"]["device_id"].is_unique

    def test_sessions_per_day_uses_observed_window(self, small_run):
        s, p = small_run["sessions"], small_run["device_profile"]
        end = pd.to_datetime(s["session_start"]).max().normalize()
        observed = (end - p["first_session"].dt.normalize()).dt.days + 1
        expected = (p["total_sessions"] / observed.clip(lower=1)).round(3)
        np.testing.assert_allclose(p["sessions_per_day"], expected, atol=1e-9)
        # Never inflated by single-session devices (the old days_active bug).
        assert (p["sessions_per_day"] <= p["total_sessions"]).all()
        single = p[p["total_sessions"] == 1]
        if len(single):
            assert (single["sessions_per_day"] < 1.0).mean() > 0.9

    def test_primary_audio_mode_is_the_mode(self, small_run):
        s, p = small_run["sessions"], small_run["device_profile"]
        expected = s.groupby("device_id")["audio_mode"].agg(
            lambda x: x.mode().iloc[0]
        )
        got = p.set_index("device_id")["primary_audio_mode"]
        assert got.sort_index().equals(expected.sort_index())

    def test_clean_sessions_are_subset_of_raw(self, small_run):
        assert set(small_run["sessions"]["session_id"]) <= set(
            small_run["raw_sessions"]["session_id"]
        )
