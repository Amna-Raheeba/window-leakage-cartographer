from window_leakage.evaluate import headline_numbers, run_evaluation, run_single_seed
from window_leakage.report import (
    render_evaluation_summary,
    render_single_seed_report,
)


def test_single_seed_report_content():
    text = render_single_seed_report(run_single_seed(7))
    assert "Leaked test windows" in text
    assert "Exact-duplicate baseline" in text
    assert "leave-own-unit-out" in text
    assert "Largest leakage components" in text


def test_evaluation_summary_content():
    evaluation = run_evaluation(seeds=(201,), configs=((10, 5), (20, 20)),
                                n_units=8)
    text = render_evaluation_summary(evaluation, headline_numbers(evaluation))
    assert "# Evaluation summary" in text
    assert "10x5" in text and "20x20" in text
    assert "adjacency premium" in text
