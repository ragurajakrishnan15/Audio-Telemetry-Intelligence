# Audio Telemetry Intelligence Platform

[![CI](https://github.com/ragurajakrishnan15/Audio-Telemetry-Intelligence/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ragurajakrishnan15/Audio-Telemetry-Intelligence/actions/workflows/ci.yml)

**End-to-end data science pipeline for analyzing large-scale audio device telemetry — from raw session data through ETL, statistical analysis, ML modeling, and interactive dashboards.**

Built as a portfolio project mapping to the requirements of a *Data Scientist, Audio Telemetry Intelligence* role.

> **Read this first:** all data is **synthetic**. Relationships in it (e.g. ANC level vs. ambient noise) were
> written into the generator, so model scores here demonstrate pipeline mechanics, not real-world
> predictive power. See [Limitations](#limitations).

---

## Architecture

```
audio-telemetry-intelligence/
├── configs/
│   └── pipeline_config.yaml         # Paths, sizes, Spark and model settings (read by src/config.py)
├── src/
│   ├── config.py                    # Config loader: YAML deep-merged over defaults
│   ├── data_generation/
│   │   └── generate_telemetry.py    # Synthetic telemetry data generator
│   ├── pipeline/
│   │   ├── spark_etl.py             # PySpark ETL (optional, needs a JVM)
│   │   └── pandas_etl.py            # Pandas ETL (default; used when Spark is unavailable)
│   ├── models/
│   │   └── predictive_models.py     # ML models + statistical analysis
│   ├── analysis/
│   │   └── sql_analytics.py         # Advanced SQL analytics (PySpark SQL)
│   └── dashboard/
│       ├── app.py                   # Streamlit interactive dashboard
│       └── requirements.txt         # Dashboard-only dependencies (used by hosting)
├── scripts/
│   └── build_sample_data.py         # Rebuilds the bundled demo dataset
├── sample_data/                     # Small committed dataset (~40K sessions) for the hosted dashboard
├── tests/
│   ├── conftest.py                  # Small end-to-end fixture shared by the suites
│   ├── test_pipeline.py             # Data generation and data quality
│   ├── test_etl.py                  # Feature logic, quality scoring, aggregations
│   ├── test_models.py               # Models, statistics, segment naming
│   ├── test_config.py               # Config loading, adjustment time series
│   ├── test_dashboard.py            # Dashboard renders from the bundled sample
│   └── test_spark_parity.py         # Spark vs Pandas parity (skipped without a JVM)
├── .github/workflows/ci.yml         # pytest (Java 17 + PySpark; fails on any skip) on push / PR
├── Makefile                         # install | pipeline | test | dashboard | clean
├── run_pipeline.py                  # Master orchestrator
├── requirements.txt                 # Core, pinned to tested versions
├── requirements-spark.txt           # Optional Spark dependency
└── README.md
```

## Data at Scale

| Dataset | Records | Description |
|---|---|---|
| Device Registry | 50,000 | 10 device types with feature capabilities (ANC, spatial audio, transparency, conversation awareness) |
| Audio Sessions | 500,000 | Per-session telemetry: ANC level, spatial audio, voice isolation, Bluetooth quality, battery drain, codec, latency, glitches |
| System Adjustments | 450,327 | Time-series of automatic adjustments within 100,000 sampled ANC sessions (1–8 each, timestamps ordered within session) |

Sizes and the date range are set in `configs/pipeline_config.yaml`.

## Pipeline Stages

### 1. Data Generation
Device capabilities constrain session attributes: AirPods Pro support ANC while iPhones don't, so non-ANC devices always have `anc_level = 0`.

### 2. ETL (Pandas default, PySpark optional)

- **Extract:** load parquet files
- **Validate:** null checks on critical columns, duplicate session IDs, range checks, IQR outlier removal on `duration_seconds` (fences at Q1 − 3·IQR / Q3 + 3·IQR, configurable)
- **Transform:** 10 engineered columns (below)
- **Aggregate:** hourly rollups, per-device profiles, environment analysis
- **Load:** parquet files + `quality_report.json`

**Quality score: 95.5%.** 22,486 of 500,000 sessions (4.5%) fell outside the duration fences. The generator produces no nulls, duplicates or range violations, so those checks report zero. The score therefore reflects the duration outlier rule only.

### 3. Feature Engineering

| Feature | Type | Logic |
|---|---|---|
| `hour_of_day`, `day_of_week`, `month`, `is_weekend` | Temporal | From session timestamp (`day_of_week`: 0 = Monday) |
| `time_bucket` | Categorical | night / morning / afternoon / evening |
| `audio_quality_index` | Composite | f(glitches, latency, voice isolation) |
| `anc_effectiveness` | Categorical | none / low / medium / high from ANC level, noise and glitches |
| `engagement_tier` | Ordinal | brief (≤2 min) / casual / engaged / power_user (>1 h) |
| `noise_category` | Ordinal | silent / quiet / moderate / loud / very_loud |
| `anc_demand_score` | Continuous | noise × audio-mode weight × environment weight |

Device profiles also carry `sessions_per_day`, computed over the device's observed window (first session → end of dataset).

### 4. ML Models

Trained on a 100,000-session sample (`models.sample_size`).

| Model | Algorithm | Task | Result |
|---|---|---|---|
| **ANC Predictor** | Gradient Boosting | Predict ANC level on ANC-capable sessions | R² = 0.886, MAE = 0.068, CV R² = 0.888 ± 0.001 — *see Limitations* |
| **User Segmentation** | K-Means (k chosen by silhouette) | Segment devices by behavior | k = 3, silhouette = 0.171 (weak separation) |
| **Anomaly Detector** | Isolation Forest | Flag degraded sessions | 5.0% flagged (set by `contamination`); anomalous latency 2.9× and glitches 2.7× normal |
| **Duration Forecaster** | Gradient Boosting | Predict session duration | R² = −0.005 — no better than the mean; synthetic duration is independent of the features |

**Segments found**

| Segment | Devices | Preferred ANC | Avg satisfaction |
|---|---|---|---|
| Standard Users | 41.1% | 0.03 | 66.8 |
| High-Activity Users | 29.8% | 0.10 | 67.6 |
| ANC-Heavy Listeners | 29.1% | 0.40 | 71.1 |

Segment names are derived from cluster statistics, not from the arbitrary K-Means label order. The only strong separator is ANC usage.

### 5. Statistical Analysis

| Test | Statistic | p | Effect size |
|---|---|---|---|
| ANOVA — satisfaction across environments (ANC sessions, n = 38,912) | F = 1021.77 | < 0.001 | η² = 0.191 (large) |
| Kruskal-Wallis — satisfaction across device types (n = 100,000) | H = 6523.59 | < 0.001 | ε² = 0.065 (medium) |

With samples this large p-values are always tiny; the effect sizes are the informative part. Both effects were built into the generator.

### 6. Advanced SQL Analytics (PySpark SQL)

Five queries in `sql_analytics.py` using window functions, CTEs and cohort analysis:
environment ANC-demand ranking (`RANK`, running totals), device retention cohorts (`MONTHS_BETWEEN`),
quality degradation (`LAG`, rolling averages), feature adoption funnel, and power-user deciles (`NTILE`).

Run with `python run_pipeline.py --step sql`; results go to `data/processed/sql_analytics.json`.
Cohort retention is `NULL` (not 0%) for windows that extend past the end of the data.

### 7. Interactive Dashboard (Streamlit + Plotly)

Four pages: **Overview** (KPIs, device mix, satisfaction by environment, hourly pattern),
**Model Insights** (metrics, feature importance, segments, anomalies), **Data Quality**
(quality report, null analysis, distributions), **Explorer** (filter by device/environment/mode, custom scatter plots).

### Hosted deployment (Streamlit Community Cloud)

The dashboard falls back to `sample_data/` (a ~40K-session dataset, 4.7 MB) when `data/` doesn't exist,
and shows a "Demo dataset" banner, so it works on a fresh clone or hosted deploy with no pipeline run.

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → repository `ragurajakrishnan15/Audio-Telemetry-Intelligence`, branch `main`,
   main file `src/dashboard/app.py`.
3. **Advanced settings → Python version: 3.11** (the version the pins were tested on).
4. Deploy. Dependencies come from `src/dashboard/requirements.txt` (streamlit, pandas, numpy, pyarrow, plotly only).

Rebuild the sample after changing the generator or ETL: `python scripts/build_sample_data.py`.

## Quick Start

```bash
pip install -r requirements.txt        # Python 3.11

python run_pipeline.py --step all      # generate → ETL → models (→ SQL if Spark is installed)
python run_pipeline.py --step generate # or any single step: etl | models | sql

streamlit run src/dashboard/app.py
pytest tests/ -v                       # 77 tests in a few seconds
```

A full run takes about 2–4 minutes on a laptop (≈2 min with the Pandas ETL, ≈3.5 min with Spark).

**Optional Spark path:** install Java 17 (Temurin works) and make sure `java` is on `PATH`
(or set `JAVA_HOME`), then `pip install -r requirements-spark.txt`. When PySpark starts, it is used
for the ETL and the SQL step runs. If PySpark is missing or fails to start, `run_pipeline.py`
falls back to the Pandas ETL and skips the SQL step. On Windows, Spark logs a harmless
`winutils.exe` warning.

Row order differs between the two ETLs, so the sampled training set (and therefore model metrics)
varies slightly between a Spark run and a Pandas run; segment sizes and statistics are the same.

## Verification status

| Component | Status |
|---|---|
| Data generation, ETL (Pandas **and** Spark), models, statistics | Run end-to-end at full scale; results above are from the Spark run |
| Spark ↔ Pandas parity | Verified on all 477,514 sessions and 49,998 devices: identical row sets, **0** mismatches in categorical features, `primary_audio_mode` and `day_of_week`; numeric differences ≤ 0.01 (rounding mode) |
| `sql_analytics.py` | All 5 queries execute on the full dataset; results spot-checked (cohort sizes, decile counts, NULL handling) |
| Dashboard (all 4 pages) | Rendered headlessly without errors |
| Tests | 82 passing (including 4 Spark parity tests), 0 skipped; CI runs them on Linux with `FAIL_ON_SKIP=1`, so a skipped test fails the build |

Tested with Python 3.11, Java 17 (Temurin) and PySpark 3.5.3 alongside pandas 3.0 / numpy 2.4.

## Requirement Alignment

| Requirement | Where demonstrated |
|---|---|
| Large-scale audio session telemetry | 500K-session synthetic dataset modeled on device audio telemetry |
| Data pipelines (ETL/ELT) | Extract → validate → transform → aggregate → load, in Pandas and PySpark |
| Apache Spark | PySpark ETL and Spark SQL queries (implemented; see verification status) |
| Python | Generation, ETL, ML, dashboard |
| Advanced SQL | Window functions, CTEs, cohort retention, NTILE, LAG, running totals |
| Statistical analysis | ANOVA, Kruskal-Wallis, effect sizes, correlation matrix |
| ML / predictive modeling | Gradient boosting, K-Means with silhouette selection, Isolation Forest |
| Data quality | Null / duplicate / range / IQR-outlier checks with a quality report |
| Internal data tooling | Streamlit dashboard |

## Limitations

- **Synthetic data.** Every relationship was put there by the generator. Results show that the pipeline works, not that these patterns exist in real telemetry.
- **The ANC predictor is largely circular.** The target (`anc_level`) is generated from `ambient_noise_db` by a formula, and 97% of the model's feature importance sits on that one column. The R² of 0.886 measures how well the model recovered the generator's formula.
- **The duration forecaster does not work**, by design of the data: duration is drawn independently of every feature. It is kept as an honest negative result.
- **Anomalies are not all "bad" sessions.** Duration is one of the Isolation Forest inputs, so unusually *long* sessions are flagged; anomalous sessions have slightly *higher* satisfaction (73.0 vs 68.0).
- **Segmentation is weak** (silhouette 0.17); k = 3 was the best of 3–8 but clusters overlap heavily outside ANC usage.
- **Quality score covers one rule.** Only duration outliers are removed; the generator emits clean data otherwise.
- **Spark and Pandas ETL agree on logic and rows**, but rounding differs slightly (Spark rounds half-up, NumPy half-even; ≤ 0.01 on `audio_quality_index`). The Spark ETL uses *exact* quantiles for the IQR fences; at much larger scale you may prefer `approxQuantile` with a small relative error.
- **Pandas-3 / Spark-3.5 interop is held together by two details:** parquet timestamps are written at microsecond precision (Spark 3.5 cannot read pandas 3's nanosecond default), and PySpark emits `Pandas4Warning` deprecation notices on `toPandas()`. Both work today; revisit when upgrading Spark.
- **Retention and funnel numbers are flat/saturated by construction** (sessions are uniformly random over the year; each device has ~10 sessions), so they illustrate the queries, not real behavior.

## Key Design Decisions

1. **Two ETL implementations** — PySpark for scale, Pandas for local runs. Known parity points (day-of-week convention, modal audio mode, noise-category boundaries, range checks, `sessions_per_day`) are aligned deliberately.
2. **Features are computed in the ETL layer**, so every downstream consumer (models, SQL, dashboard) shares one definition.
3. **Parquet throughout** — columnar storage for analytical queries and Spark compatibility.
4. **Config-driven** — paths, sizes, Spark settings, outlier fences and model parameters live in `configs/pipeline_config.yaml`, with in-code defaults.
5. **Capability-constrained generation** — no ANC data for devices without ANC hardware, so data behaves like plausible telemetry.
