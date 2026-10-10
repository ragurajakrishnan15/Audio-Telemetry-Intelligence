"""Smoke and behavior tests for the models, statistics and segment naming."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.models.predictive_models import (
    _effect_label, _json_default, _name_segments, run_all_models,
    run_statistical_analysis,
)

MODEL_CFG = {
    "sample_size": 3000,
    "anc_predictor": {"test_size": 0.2, "random_state": 42},
    "user_segmentation": {"max_clusters": 6},
    "anomaly_detection": {"contamination": 0.05},
}


@pytest.fixture(scope="module")
def model_run(small_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    results = run_all_models(small_run["processed"], str(out), MODEL_CFG)
    return {"results": results, "out": str(out)}


class TestSegmentNaming:
    @staticmethod
    def _stats(anc, sessions, size):
        return pd.DataFrame({
            "preferred_anc_level": anc,
            "total_sessions": sessions,
            "size": size,
        })

    def test_dominant_anc_cluster_is_named_anc_heavy(self):
        names = _name_segments(self._stats([0.4, 0.02, 0.03], [10, 10, 10],
                                           [100, 100, 100]))
        assert names[0] == "ANC-Heavy Listeners"

    def test_names_follow_traits_not_index(self):
        a = _name_segments(self._stats([0.05, 0.05, 0.05], [20, 10, 2],
                                       [100, 100, 100]))
        b = _name_segments(self._stats([0.05, 0.05, 0.05], [2, 10, 20],
                                       [100, 100, 100]))
        assert a == {0: "High-Activity Users", 1: "Standard Users", 2: "Light Users"}
        assert b == {0: "Light Users", 1: "Standard Users", 2: "High-Activity Users"}

    def test_small_gaps_do_not_create_high_or_low(self):
        names = _name_segments(self._stats([0.05, 0.05], [10.0, 10.5], [100, 100]))
        assert set(names.values()) <= {"Standard Users", "Standard Users 2"}

    def test_names_are_unique(self):
        names = _name_segments(self._stats([0.0] * 5, [10] * 5, [10] * 5))
        assert len(set(names.values())) == 5

    def test_no_anc_label_when_all_similar(self):
        names = _name_segments(self._stats([0.05, 0.06, 0.05], [10, 10, 10],
                                           [10, 10, 10]))
        assert "ANC-Heavy Listeners" not in names.values()


class TestHelpers:
    def test_json_default_unwraps_numpy(self):
        assert _json_default(np.bool_(True)) is True
        assert _json_default(np.float64(1.5)) == 1.5
        assert _json_default(np.array([1, 2])) == [1, 2]

    @pytest.mark.parametrize("value,label", [
        (0.005, "negligible"), (0.02, "small"), (0.10, "medium"), (0.30, "large"),
    ])
    def test_effect_label(self, value, label):
        assert _effect_label(value, (0.01, 0.06, 0.14)) == label


class TestStatistics:
    def test_effect_sizes_are_valid(self, small_run):
        res = run_statistical_analysis(small_run["sessions"])
        anova = res["anc_environment_anova"]
        assert 0.0 <= anova["eta_squared"] <= 1.0
        assert isinstance(anova["significant"], bool)
        kw = res["device_satisfaction_kruskal"]
        assert 0.0 <= kw["epsilon_squared"] <= 1.0
        assert kw["effect_size"] in {"negligible", "small", "medium", "large"}


class TestModelRun:
    def test_all_result_sections_present(self, model_run):
        r = model_run["results"]
        for key in ["anc_predictor", "user_segmentation", "anomaly_detection",
                    "duration_forecaster", "statistical_analysis",
                    "cluster_profiles", "anc_feature_importance"]:
            assert key in r

    def test_artifacts_written(self, model_run):
        for name in ["model_results.json", "anc_predictor.joblib",
                     "user_segmentation.joblib", "anomaly_detector.joblib",
                     "duration_forecaster.joblib"]:
            assert os.path.exists(os.path.join(model_run["out"], name))

    def test_results_json_has_real_booleans(self, model_run):
        with open(os.path.join(model_run["out"], "model_results.json")) as f:
            saved = json.load(f)
        assert saved["statistical_analysis"]["anc_environment_anova"][
            "significant"] in (True, False)

    def test_anc_predictor_learns_the_noise_relationship(self, model_run):
        m = model_run["results"]["anc_predictor"]
        assert m["r2_score"] > 0.5
        assert 0 <= m["mae"] < 0.2

    def test_anomaly_rate_matches_contamination(self, model_run):
        m = model_run["results"]["anomaly_detection"]
        assert m["anomaly_rate"] == pytest.approx(0.05, abs=0.005)

    def test_segments_partition_all_devices(self, model_run, small_run):
        profiles = model_run["results"]["cluster_profiles"]
        assert sum(p["size"] for p in profiles.values()) == len(
            small_run["device_profile"]
        )
        assert len(profiles) == model_run["results"]["user_segmentation"]["n_clusters"]

    def test_feature_importance_sums_below_one(self, model_run):
        imp = model_run["results"]["anc_feature_importance"]
        assert 0 < sum(imp.values()) <= 1.0 + 1e-9
