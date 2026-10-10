"""Tests for configuration loading and the adjustment time series."""

import copy
import os

import pytest

from src.config import DEFAULTS, ROOT, load_config, resolve_path


class TestConfig:
    def test_missing_file_returns_defaults(self, tmp_path):
        assert load_config(str(tmp_path / "nope.yaml")) == DEFAULTS

    def test_partial_override_keeps_sibling_defaults(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("pipeline:\n  quality_checks:\n    iqr_multiplier: 1.5\n"
                     "models:\n  sample_size: 123\n")
        cfg = load_config(str(p))
        assert cfg["pipeline"]["quality_checks"]["iqr_multiplier"] == 1.5
        assert cfg["models"]["sample_size"] == 123
        # untouched siblings survive the merge
        assert cfg["pipeline"]["spark"] == DEFAULTS["pipeline"]["spark"]
        assert cfg["models"]["anomaly_detection"] == DEFAULTS["models"]["anomaly_detection"]

    def test_loading_does_not_mutate_defaults(self, tmp_path):
        before = copy.deepcopy(DEFAULTS)
        p = tmp_path / "c.yaml"
        p.write_text("data_generation:\n  num_devices: 7\n")
        load_config(str(p))
        assert DEFAULTS == before

    def test_empty_file_is_ok(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert load_config(str(p)) == DEFAULTS

    def test_repo_config_is_valid_and_complete(self):
        cfg = load_config()
        assert cfg["data_generation"]["num_sessions"] > 0
        assert set(DEFAULTS) <= set(cfg)

    def test_resolve_path(self):
        assert resolve_path("data/raw") == os.path.join(ROOT, "data/raw")
        absolute = os.path.abspath(os.sep)
        assert resolve_path(absolute) == absolute


class TestAdjustmentTimeSeries:
    def test_offsets_ordered_and_inside_session(self, small_run):
        import pandas as pd

        adj = pd.read_parquet(
            os.path.join(small_run["raw"], "system_adjustments.parquet")
        )
        sess = small_run["raw_sessions"].set_index("session_id")
        adj["ts"] = pd.to_datetime(adj["timestamp"])
        start = pd.to_datetime(sess.loc[adj["session_id"], "session_start"]).values
        end = pd.to_datetime(sess.loc[adj["session_id"], "session_end"]).values
        assert (adj["ts"].values > start).all()
        assert (adj["ts"].values <= end).all()

        ordered = adj.groupby("session_id")["ts"].apply(
            lambda s: s.is_monotonic_increasing and s.is_unique
        )
        assert ordered.all()
