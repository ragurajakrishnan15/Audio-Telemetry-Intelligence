"""Static-site sanity checks (no browser needed)."""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HTML = open(os.path.join(ROOT, "site", "index.html"), encoding="utf-8").read()


def test_site_is_self_contained():
    # Nothing is fetched at load time: no external scripts, styles, fonts or images.
    assert not re.search(r"<script[^>]+src=", HTML)
    assert not re.search(r"<link[^>]+rel=[\"']stylesheet", HTML)
    assert not re.search(r"@import|url\(\s*[\"']?https?:", HTML)


def test_internal_anchors_resolve():
    ids = set(re.findall(r'id="([^"]+)"', HTML))
    missing = [h for h in re.findall(r'href="#([^"]+)"', HTML) if h and h not in ids]
    assert not missing, missing


def test_independent_project_disclaimer_present():
    assert "Not affiliated with" in HTML


def test_demo_button_hidden_until_url_is_set():
    assert 'id="demo-link"' in HTML and "hidden" in HTML
    assert re.search(r'var DEMO_URL = "[^"]*";', HTML)


def test_headline_numbers_match_the_repo():
    # Guard against the page drifting from the documented results.
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    for fact in ["95.5%", "22,486", "477,514", "49,998"]:
        assert fact in HTML and fact in readme, fact
