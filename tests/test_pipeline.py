"""
Unit tests for the Audio Telemetry Intelligence pipeline.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation.generate_telemetry import (
    generate_devices, generate_sessions, generate_system_adjustments,
)


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_devices_count(self):
        devices = generate_devices(n_devices=100)
        assert len(devices) == 100

    def test_device_columns(self):
        devices = generate_devices(n_devices=10)
        expected = {"device_id", "device_type", "firmware_version",
                    "os_version", "region", "registration_date",
                    "has_anc", "has_spatial_audio", "has_transparency",
                    "has_conversation_awareness", "has_conversation_boost"}
        assert set(devices.columns) == expected

    def test_device_ids_unique(self):
        devices = generate_devices(n_devices=500)
        assert devices["device_id"].nunique() == 500

    def test_session_generation(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=200)
        assert len(sessions) == 200
        assert sessions["device_id"].isin(devices["device_id"]).all()

    def test_session_duration_bounds(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=500)
        assert sessions["duration_seconds"].min() >= 5
        assert sessions["duration_seconds"].max() <= 28800

    def test_anc_only_on_capable_devices(self):
        devices = generate_devices(n_devices=100)
        sessions = generate_sessions(devices, n_sessions=1000)
        non_anc_devices = devices[~devices["has_anc"]]["device_id"]
        non_anc_sessions = sessions[sessions["device_id"].isin(non_anc_devices)]
        assert (non_anc_sessions["anc_level"] == 0).all()

    def test_satisfaction_proxy_range(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=500)
        assert sessions["satisfaction_proxy"].min() >= 0
        assert sessions["satisfaction_proxy"].max() <= 100

    def test_system_adjustments(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=500)
        adjustments = generate_system_adjustments(sessions, max_adjustments_per_session=3)
        assert len(adjustments) > 0
        assert adjustments["session_id"].isin(sessions["session_id"]).all()

    def test_volume_range(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=500)
        assert sessions["volume_level"].min() >= 5
        assert sessions["volume_level"].max() <= 100


class TestDataQuality:
    """Tests for data quality validation."""

    def test_no_nulls_in_critical_fields(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=200)
        for col in ["session_id", "device_id", "session_start", "duration_seconds"]:
            assert sessions[col].isna().sum() == 0

    def test_no_duplicate_session_ids(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=500)
        assert sessions["session_id"].duplicated().sum() == 0

    def test_valid_audio_modes(self):
        devices = generate_devices(n_devices=50)
        sessions = generate_sessions(devices, n_sessions=500)
        valid_modes = {
            "media_playback", "phone_call", "facetime", "siri_interaction",
            "notification", "spatial_audio_content", "voice_memo",
            "music_streaming", "podcast", "video_conference",
        }
        assert set(sessions["audio_mode"].unique()).issubset(valid_modes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
