"""
Build the small bundled demo dataset used by the hosted dashboard.

Runs the same generate -> ETL -> models flow as the full pipeline, but at
~40K sessions, and writes only what the dashboard reads into `sample_data/`:

    sample_data/processed/*.parquet, quality_report.json
    sample_data/models/model_results.json

Deterministic (the generator seeds NumPy/random/Faker at import).

Usage: python scripts/build_sample_data.py [--devices 5000] [--sessions 40000]
"""

import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.data_generation.generate_telemetry import (  # noqa: E402
    generate_devices, generate_sessions, generate_system_adjustments,
)
from src.models.predictive_models import run_all_models  # noqa: E402
from src.pipeline.pandas_etl import AudioTelemetryPipelinePandas  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--devices", type=int, default=5_000)
    parser.add_argument("--sessions", type=int, default=40_000)
    args = parser.parse_args()

    out_root = os.path.join(ROOT, "sample_data")
    processed = os.path.join(out_root, "processed")
    models = os.path.join(out_root, "models")
    shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(models)

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)

        print(f"Generating {args.devices:,} devices / {args.sessions:,} sessions...")
        devices = generate_devices(args.devices)
        sessions = generate_sessions(devices, args.sessions)
        adjustments = generate_system_adjustments(sessions)
        devices.to_parquet(os.path.join(raw, "devices.parquet"), index=False)
        sessions.to_parquet(os.path.join(raw, "sessions.parquet"), index=False)
        adjustments.to_parquet(
            os.path.join(raw, "system_adjustments.parquet"), index=False
        )

        AudioTelemetryPipelinePandas(raw, processed).run()

        model_dir = os.path.join(tmp, "models")
        run_all_models(
            processed, model_dir,
            {
                "sample_size": args.sessions,
                "anc_predictor": {"test_size": 0.2, "random_state": 42},
                "user_segmentation": {"max_clusters": 6},
                "anomaly_detection": {"contamination": 0.05},
            },
        )
        # Dashboard only needs the metrics JSON, not the pickled models.
        shutil.copy(os.path.join(model_dir, "model_results.json"), models)

    total = 0
    print("\nsample_data/")
    for folder, _, files in os.walk(out_root):
        for name in sorted(files):
            size = os.path.getsize(os.path.join(folder, name))
            total += size
            rel = os.path.relpath(os.path.join(folder, name), out_root)
            print(f"  {rel:40s} {size / 1024:8.0f} KB")
    print(f"  {'TOTAL':40s} {total / 1024 / 1024:8.1f} MB")


if __name__ == "__main__":
    main()
