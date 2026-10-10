"""
SQL-Based Analytical Queries (PySpark SQL)
=============================================
Demonstrates advanced SQL proficiency against the telemetry dataset.
All queries run via Spark SQL on registered temporary views.

Maps to JD: "Advanced SQL skills for querying and analyzing
large-scale datasets."
"""

from pyspark.sql import SparkSession


QUERIES = {
    "top_environments_by_anc_demand": """
        -- Rank environments by ANC demand with running totals
        SELECT
            environment_type,
            noise_category,
            COUNT(*) AS session_count,
            ROUND(AVG(anc_level), 4) AS avg_anc_level,
            ROUND(AVG(ambient_noise_db), 1) AS avg_noise_db,
            ROUND(AVG(satisfaction_proxy), 2) AS avg_satisfaction,
            ROUND(
                SUM(CASE WHEN anc_level > 0.7 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*), 1
            ) AS high_anc_pct,
            SUM(COUNT(*)) OVER (
                ORDER BY AVG(anc_level) DESC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_sessions
        FROM sessions
        WHERE anc_level > 0
        GROUP BY environment_type, noise_category
        ORDER BY avg_anc_level DESC
    """,

    "device_retention_cohort": """
        -- Monthly cohort retention: what % of devices return each month
        WITH activity AS (
            SELECT DISTINCT
                device_id,
                DATE_TRUNC('month', session_ts) AS active_month
            FROM sessions
        ),
        first_use AS (
            SELECT
                device_id,
                MIN(active_month) AS cohort_month
            FROM activity
            GROUP BY device_id
        ),
        bounds AS (
            SELECT MAX(active_month) AS last_month FROM activity
        )
        -- NULL (not 0%) when the cohort has not yet had N months to return
        SELECT
            f.cohort_month,
            COUNT(DISTINCT f.device_id) AS cohort_size,
            CASE WHEN ADD_MONTHS(f.cohort_month, 1) > b.last_month THEN NULL
            ELSE ROUND(
                COUNT(DISTINCT CASE
                    WHEN MONTHS_BETWEEN(a.active_month, f.cohort_month) = 1
                    THEN a.device_id END) * 100.0
                / COUNT(DISTINCT f.device_id), 1
            ) END AS month_1_retention,
            CASE WHEN ADD_MONTHS(f.cohort_month, 3) > b.last_month THEN NULL
            ELSE ROUND(
                COUNT(DISTINCT CASE
                    WHEN MONTHS_BETWEEN(a.active_month, f.cohort_month) = 3
                    THEN a.device_id END) * 100.0
                / COUNT(DISTINCT f.device_id), 1
            ) END AS month_3_retention,
            CASE WHEN ADD_MONTHS(f.cohort_month, 6) > b.last_month THEN NULL
            ELSE ROUND(
                COUNT(DISTINCT CASE
                    WHEN MONTHS_BETWEEN(a.active_month, f.cohort_month) = 6
                    THEN a.device_id END) * 100.0
                / COUNT(DISTINCT f.device_id), 1
            ) END AS month_6_retention
        FROM first_use f
        CROSS JOIN bounds b
        LEFT JOIN activity a ON f.device_id = a.device_id
        GROUP BY f.cohort_month, b.last_month
        ORDER BY f.cohort_month
    """,

    "audio_quality_degradation_pattern": """
        -- Identify time windows with quality degradation
        WITH hourly_quality AS (
            SELECT
                DATE(session_ts) AS dt,
                HOUR(session_ts) AS hr,
                device_type,
                AVG(audio_quality_index) AS avg_quality,
                AVG(latency_ms) AS avg_latency,
                SUM(audio_glitches) AS total_glitches,
                COUNT(*) AS sessions
            FROM sessions
            GROUP BY DATE(session_ts), HOUR(session_ts), device_type
        ),
        with_lag AS (
            SELECT *,
                LAG(avg_quality, 1) OVER (
                    PARTITION BY device_type ORDER BY dt, hr
                ) AS prev_quality,
                AVG(avg_quality) OVER (
                    PARTITION BY device_type
                    ORDER BY dt, hr
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) AS rolling_avg_quality
            FROM hourly_quality
        )
        SELECT
            dt, hr, device_type,
            ROUND(avg_quality, 2) AS quality,
            ROUND(rolling_avg_quality, 2) AS rolling_quality,
            ROUND(avg_quality - prev_quality, 2) AS quality_delta,
            ROUND(avg_latency, 1) AS latency_ms,
            total_glitches,
            sessions,
            CASE
                WHEN avg_quality < rolling_avg_quality * 0.85 THEN 'DEGRADED'
                WHEN avg_quality < rolling_avg_quality * 0.95 THEN 'WARNING'
                ELSE 'NORMAL'
            END AS status
        FROM with_lag
        WHERE prev_quality IS NOT NULL
        ORDER BY quality_delta ASC
        LIMIT 50
    """,

    "feature_adoption_funnel": """
        -- Feature adoption funnel across device ecosystem
        SELECT
            device_type,
            COUNT(DISTINCT device_id) AS total_devices,
            COUNT(DISTINCT CASE WHEN spatial_audio_enabled THEN device_id END)
                AS spatial_audio_users,
            COUNT(DISTINCT CASE WHEN anc_level > 0 THEN device_id END)
                AS anc_users,
            COUNT(DISTINCT CASE WHEN conversation_awareness_triggered THEN device_id END)
                AS conv_awareness_users,
            COUNT(DISTINCT CASE WHEN transparency_mode_active THEN device_id END)
                AS transparency_users,
            ROUND(
                COUNT(DISTINCT CASE WHEN spatial_audio_enabled THEN device_id END)
                * 100.0 / COUNT(DISTINCT device_id), 1
            ) AS spatial_adoption_pct,
            ROUND(
                COUNT(DISTINCT CASE WHEN anc_level > 0 THEN device_id END)
                * 100.0 / COUNT(DISTINCT device_id), 1
            ) AS anc_adoption_pct,
            ROUND(AVG(satisfaction_proxy), 2) AS avg_satisfaction
        FROM sessions
        GROUP BY device_type
        ORDER BY total_devices DESC
    """,

    "power_user_behavior_profile": """
        -- Deep-dive into power user segment behavior
        WITH user_stats AS (
            SELECT
                device_id,
                device_type,
                region,
                COUNT(*) AS total_sessions,
                SUM(duration_seconds) / 3600.0 AS total_hours,
                AVG(satisfaction_proxy) AS avg_satisfaction,
                AVG(anc_level) AS avg_anc,
                PERCENTILE_APPROX(duration_seconds, 0.5) AS median_duration
            FROM sessions
            GROUP BY device_id, device_type, region
        ),
        ranked AS (
            SELECT *,
                NTILE(10) OVER (ORDER BY total_hours DESC) AS usage_decile
            FROM user_stats
        )
        SELECT
            usage_decile,
            COUNT(*) AS user_count,
            ROUND(AVG(total_sessions), 0) AS avg_sessions,
            ROUND(AVG(total_hours), 1) AS avg_hours,
            ROUND(AVG(median_duration) / 60, 1) AS avg_median_session_min,
            ROUND(AVG(avg_satisfaction), 2) AS avg_satisfaction,
            ROUND(AVG(avg_anc), 3) AS avg_anc_level
        FROM ranked
        GROUP BY usage_decile
        ORDER BY usage_decile
    """,
}


def run_sql_analytics(spark: SparkSession, sessions_path: str) -> dict:
    """Execute all SQL analytics queries and return results."""
    print("\n[SQL ANALYTICS] Running analytical queries...")

    sessions = spark.read.parquet(sessions_path)
    sessions.createOrReplaceTempView("sessions")

    results = {}
    for name, query in QUERIES.items():
        try:
            df = spark.sql(query)
            pdf = df.toPandas()
            results[name] = {
                "rows": len(pdf),
                "columns": list(pdf.columns),
                "data": pdf.head(20).to_dict("records"),
            }
            print(f"  ✓ {name}: {len(pdf)} rows")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  ✗ {name}: {e}")

    return results


if __name__ == "__main__":
    import os
    spark = (
        SparkSession.builder
        .appName("SQLAnalytics")
        .master("local[*]")
        .getOrCreate()
    )
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    results = run_sql_analytics(
        spark, os.path.join(base, "data", "processed", "sessions.parquet")
    )
    spark.stop()
