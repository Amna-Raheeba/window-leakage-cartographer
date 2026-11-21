import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation.json")


def _load():
    with open(RESULTS) as fh:
        return json.load(fh)


def test_stored_results_exist_and_parse():
    assert os.path.isfile(RESULTS)
    payload = _load()
    for key in ("seeds", "configs", "n_runs", "runs", "per_config",
                "pooled", "headline"):
        assert key in payload, key
    assert payload["n_runs"] == 20


def test_stored_headline_keys():
    headline = _load()["headline"]
    for key in (
        "mean_leaked_fraction",
        "detection_precision",
        "detection_recall",
        "baseline_recall",
        "mean_inflation_rmse",
        "mean_inflation_purged",
        "inflation_pct",
        "mean_recovered_share",
        "mean_mass_leaked",
        "mean_optimism_leaked",
        "mean_optimism_clean",
        "mean_rmse_full",
        "mean_rmse_purged",
        "mean_rmse_louo",
        "mean_reference_rmse",
        "inflation_control",
        "leaked_control",
    ):
        assert key in headline, key


def test_stored_headline_values_sane():
    headline = _load()["headline"]
    # The graph recovers the planted overlap exactly...
    assert headline["detection_precision"] >= 0.99
    assert headline["detection_recall"] >= 0.99
    # ...while exact-duplicate counting is blind to partial overlaps.
    assert headline["baseline_recall"] <= 0.01
    # Most test windows leak at overlapping geometries; the
    # non-overlapping control leaks none, by construction.
    assert 0.5 < headline["mean_leaked_fraction"] < 0.9
    assert headline["leaked_control"] == 0.0
    # The leaky score is optimistic, and the replay recovers a large
    # share - but not all - of the MSE optimism.
    assert headline["mean_inflation_rmse"] > 0.0
    assert 0.4 < headline["mean_recovered_share"] < 1.2
    assert headline["mean_optimism_leaked"] > headline["mean_optimism_clean"]
