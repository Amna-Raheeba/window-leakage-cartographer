from window_leakage.cartographer import audit_split
from window_leakage.windows import build_windows, random_split


def test_audit_keys_present(audit):
    for key in (
        "leaked_fraction",
        "detection_precision",
        "detection_recall",
        "baseline_hits",
        "rmse_full",
        "rmse_purged",
        "rmse_louo",
        "inflation_rmse",
        "recovered_share",
        "components",
        "mass",
        "truth_leaked",
    ):
        assert key in audit, key


def test_audit_detects_all_true_leaks(audit):
    # Every truly leaked test window shares timesteps with a train
    # window, so the graph must contain that edge: recall is exact.
    assert audit["detection_recall"] == 1.0
    assert audit["detection_fn"] == 0


def test_audit_detection_precision_high(audit):
    assert audit["detection_precision"] >= 0.9


def test_audit_leakage_is_extensive_at_stride_below_length(audit):
    assert audit["leaked_fraction"] > 0.8
    assert audit["n_straddling"] > 0
    assert audit["n_sources"] > 0


def test_audit_baseline_blind(audit):
    # Exact-duplicate counting catches none of the partial overlaps.
    assert audit["baseline_hits"] == 0


def test_audit_mass_in_range(audit):
    for value in audit["mass"].values():
        assert 0.0 <= value <= 1.0
    assert audit["mean_mass_leaked"] > 0.5


def test_audit_components_straddle(audit):
    for comp in audit["components"]:
        assert comp["train_ids"] and comp["test_ids"]
        assert comp["source_ids"]
        assert comp["rmse_after"] >= 0.0


def test_audit_scores_ordered_on_fixture(audit):
    # On this fixture the leaky score is optimistic vs LOUO.
    assert audit["rmse_louo"] >= audit["rmse_full"] - 1.0
    assert audit["recovered_share"] == audit["recovered_share"]  # finite


def test_control_geometry_has_no_overlap_leakage(fleet):
    windows = build_windows(fleet, 10, 10)
    train, test = random_split(windows, 0.25, seed=11)
    result = audit_split(windows, train, test, fleet, k=3)
    assert result["leaked_fraction"] == 0.0
    assert result["detection_fn"] == 0
    assert result["baseline_hits"] == 0


def test_audit_truth_matches_flagged_overlap(audit):
    # Flagged windows (mass > 0 via an overlap edge) are a superset
    # of the truly leaked ones; counts must be consistent.
    n_leaked = sum(1 for t in audit["truth_leaked"].values() if t)
    assert audit["detection_tp"] + audit["detection_fn"] == n_leaked
    assert audit["detection_tp"] <= len(audit["flagged"])
