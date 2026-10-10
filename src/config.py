"""
Central configuration loader.

Reads configs/pipeline_config.yaml and deep-merges it over DEFAULTS, so the
pipeline still runs if the file is missing or only partially specified.
"""

import copy
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT, "configs", "pipeline_config.yaml")

DEFAULTS = {
    "data_generation": {
        "num_devices": 50_000,
        "num_sessions": 500_000,
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
    },
    "pipeline": {
        "raw_data_path": "data/raw",
        "processed_data_path": "data/processed",
        "model_output_path": "data/models",
        "spark": {
            "app_name": "AudioTelemetryPipeline",
            "master": "local[*]",
            "shuffle_partitions": 8,
            "driver_memory": "4g",
        },
        "quality_checks": {"iqr_multiplier": 3.0},
    },
    "models": {
        "sample_size": 100_000,
        "anc_predictor": {"test_size": 0.2, "random_state": 42},
        "user_segmentation": {"max_clusters": 6},
        "anomaly_detection": {"contamination": 0.05},
    },
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str = CONFIG_PATH) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def resolve_path(relative: str) -> str:
    """Resolve a config path against the project root."""
    return relative if os.path.isabs(relative) else os.path.join(ROOT, relative)
