"""
Master Pipeline Runner
========================
Orchestrates the full pipeline: data generation → ETL → ML models.
Usage: python run_pipeline.py [--step generate|etl|models|sql|all]
"""

import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Pipeline output uses non-ASCII symbols (→, ×, —); Windows consoles default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def run_generation():
    print("\n" + "=" * 60)
    print("STEP 1: Generating Synthetic Telemetry Data")
    print("=" * 60)
    from src.data_generation.generate_telemetry import main as gen_main
    gen_main()


def run_etl():
    print("\n" + "=" * 60)
    print("STEP 2: Running ETL Pipeline")
    print("=" * 60)
    try:
        from src.pipeline.spark_etl import main as etl_main
        etl_main()
    except Exception as e:  # ImportError, or Spark/JVM startup failure
        print(f"  Spark ETL unavailable ({type(e).__name__}: {str(e)[:100]}) "
              "— falling back to Pandas ETL")
        from src.pipeline.pandas_etl import main as etl_main
        etl_main()


def run_models():
    print("\n" + "=" * 60)
    print("STEP 3: Training Predictive Models")
    print("=" * 60)
    from src.models.predictive_models import main as model_main
    model_main()


def run_sql():
    """Run the Spark SQL analytics queries (requires PySpark + a JVM)."""
    print("\n" + "=" * 60)
    print("STEP 4: Advanced SQL Analytics (Spark SQL)")
    print("=" * 60)
    from pyspark.sql import SparkSession
    from src.analysis.sql_analytics import run_sql_analytics

    from src.config import load_config, resolve_path

    processed = resolve_path(load_config()["pipeline"]["processed_data_path"])
    spark = (
        SparkSession.builder.appName("SQLAnalytics").master("local[*]")
        .config("spark.sql.session.timeZone", "UTC").getOrCreate()
    )
    try:
        results = run_sql_analytics(
            spark, os.path.join(processed, "sessions.parquet")
        )
    finally:
        spark.stop()

    out = os.path.join(processed, "sql_analytics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Audio Telemetry Pipeline")
    parser.add_argument(
        "--step", choices=["generate", "etl", "models", "sql", "all"],
        default="all", help="Pipeline step to run",
    )
    args = parser.parse_args()

    start = time.time()

    if args.step in ("generate", "all"):
        run_generation()

    if args.step in ("etl", "all"):
        run_etl()

    if args.step in ("models", "all"):
        run_models()

    if args.step == "sql":
        run_sql()
    elif args.step == "all":
        # Needs Spark + JVM; optional in a full run so Pandas-only setups still finish.
        try:
            run_sql()
        except Exception as e:
            print(f"  SQL analytics skipped ({type(e).__name__}: {str(e)[:100]})")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Total pipeline time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
