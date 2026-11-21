from window_leakage.evaluate import headline_numbers, run_evaluation, run_single_seed


def test_single_seed_keys():
    result = run_single_seed(7)
    assert result["seed"] == 7
    assert result["length"] == 20 and result["stride"] == 5
    assert result["reference_rmse"] > 0.0
    assert result["leaked_fraction"] > 0.9


def test_small_evaluation_structure():
    evaluation = run_evaluation(seeds=(201,), configs=((10, 5), (20, 20)),
                                n_units=8)
    assert evaluation["pooled"]["n_runs"] == 2
    assert set(evaluation["per_config"]) == {"10x5", "20x20"}
    assert evaluation["per_config"]["20x20"]["mean_leaked_fraction"] == 0.0
    assert evaluation["per_config"]["10x5"]["mean_leaked_fraction"] > 0.5


def test_headline_numbers_keys():
    evaluation = run_evaluation(seeds=(201,), configs=((10, 5), (20, 20)),
                                n_units=8)
    headline = headline_numbers(evaluation)
    for key in (
        "mean_leaked_fraction",
        "detection_precision",
        "detection_recall",
        "baseline_recall",
        "mean_inflation_rmse",
        "inflation_pct",
        "mean_recovered_share",
        "mean_rmse_full",
        "mean_rmse_louo",
        "leaked_control",
        "inflation_control",
    ):
        assert key in headline, key
    assert headline["leaked_control"] == 0.0
    assert headline["detection_recall"] == 1.0
