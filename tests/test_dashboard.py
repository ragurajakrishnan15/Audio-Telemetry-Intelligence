"""
Dashboard smoke tests.

The app is copied into a temp tree that has `sample_data/` but no `data/`,
mimicking a fresh clone / hosted deployment, so the sample-data fallback is
what gets exercised.
"""

import os
import re
import shutil

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAGES = ["📊 Overview", "🤖 Model Insights", "📋 Data Quality", "🔍 Explorer"]


def test_sample_data_is_committed():
    for name in ["sessions", "hourly_agg", "device_profile", "env_analysis"]:
        assert os.path.exists(
            os.path.join(ROOT, "sample_data", "processed", f"{name}.parquet")
        ), f"missing sample_data/processed/{name}.parquet"
    assert os.path.exists(
        os.path.join(ROOT, "sample_data", "processed", "quality_report.json")
    )
    assert os.path.exists(
        os.path.join(ROOT, "sample_data", "models", "model_results.json")
    )


@pytest.fixture(scope="module")
def fresh_clone_app(tmp_path_factory):
    tree = tmp_path_factory.mktemp("clone")
    (tree / "src" / "dashboard").mkdir(parents=True)
    shutil.copy(
        os.path.join(ROOT, "src", "dashboard", "app.py"),
        tree / "src" / "dashboard" / "app.py",
    )
    shutil.copytree(os.path.join(ROOT, "sample_data"), tree / "sample_data")
    assert not (tree / "data").exists()
    return str(tree / "src" / "dashboard" / "app.py")


def test_all_pages_render_from_sample_data(fresh_clone_app):
    at = AppTest.from_file(fresh_clone_app, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]

    # Banner telling visitors this is the demo dataset
    assert any("Demo dataset" in i.value for i in at.info)

    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert not at.warning, (page, [w.value for w in at.warning])


def test_overview_shows_real_numbers(fresh_clone_app):
    at = AppTest.from_file(fresh_clone_app, default_timeout=120).run()
    metrics = {m.label: m.value for m in at.metric}
    assert int(re.sub(r"\D", "", metrics["Total Sessions"])) > 10_000
    assert metrics["Data Quality"].endswith("%")
