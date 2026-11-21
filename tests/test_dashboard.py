"""Dashboard smoke tests (skipped when Streamlit is unavailable)."""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_dashboard_loads():
    app = AppTest.from_file(APP)
    app.run(timeout=600)
    assert not app.exception


def test_dashboard_has_three_tabs():
    app = AppTest.from_file(APP)
    app.run(timeout=600)
    assert not app.exception
    assert len(app.tabs) == 3
