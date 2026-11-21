"""Command-line interface.

Usage:
    python -m window_leakage demo
    python -m window_leakage evaluate --out results_dir
"""

import argparse
import csv
import json
import os
import sys

from .evaluate import headline_numbers, run_evaluation, run_single_seed
from .report import render_evaluation_summary, render_single_seed_report


def _jsonable(obj):
    if isinstance(obj, dict):
        return dict((k, _jsonable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _slim_run(result):
    keys = (
        "seed",
        "length",
        "stride",
        "n_windows",
        "n_train",
        "n_test",
        "n_edges",
        "n_components",
        "n_straddling",
        "leaked_fraction",
        "mean_mass_leaked",
        "detection_precision",
        "detection_recall",
        "baseline_hits",
        "rmse_full",
        "rmse_purged",
        "rmse_louo",
        "reference_rmse",
        "inflation_rmse",
        "inflation_purged",
        "recovered_share",
        "optimism_leaked",
        "optimism_clean",
    )
    return dict((k, result[k]) for k in keys)


def cmd_demo(args):
    result = run_single_seed(7)
    sys.stdout.write(render_single_seed_report(result))
    return 0


def cmd_evaluate(args):
    evaluation = run_evaluation()
    headline = headline_numbers(evaluation)
    out_dir = args.out
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    slim = {
        "seeds": evaluation["seeds"],
        "configs": evaluation["configs"],
        "n_units": evaluation["n_units"],
        "n_runs": evaluation["pooled"]["n_runs"],
        "runs": [_slim_run(r) for r in evaluation["runs"]],
        "per_config": evaluation["per_config"],
        "pooled": evaluation["pooled"],
        "headline": headline,
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w") as fh:
        json.dump(_jsonable(slim), fh, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "evaluation.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "config",
                "length",
                "stride",
                "mean_leaked_fraction",
                "mean_rmse_full",
                "mean_rmse_purged",
                "mean_rmse_louo",
                "mean_inflation_rmse",
                "recovered_share",
            ]
        )
        for key in evaluation["configs"]:
            row = evaluation["per_config"][key]
            writer.writerow(
                [
                    key,
                    row["length"],
                    row["stride"],
                    "%.4f" % row["mean_leaked_fraction"],
                    "%.4f" % row["mean_rmse_full"],
                    "%.4f" % row["mean_rmse_purged"],
                    "%.4f" % row["mean_rmse_louo"],
                    "%.4f" % row["mean_inflation_rmse"],
                    "%.4f" % row["recovered_share"],
                ]
            )
    with open(os.path.join(out_dir, "evaluation_summary.md"), "w") as fh:
        fh.write(render_evaluation_summary(evaluation, headline))
    with open(os.path.join(out_dir, "HEADLINE.md"), "w") as fh:
        fh.write("# Headline results\n\n")
        fh.write(
            "Random 75/25 splits of sliding windows leak %.1f%% of test "
            "windows on average (pooled over %d seeded runs, 5 window "
            "geometries including a non-overlapping control). The "
            "overlap graph detects them at precision %.3f / recall "
            "%.3f; exact-duplicate counting recalls %.3f. The leaky "
            "kNN RUL score is optimistic by %.2f RMSE (leaky %.2f vs "
            "leave-own-unit-out %.2f; purged %.2f, unit-disjoint split "
            "reference %.2f), and per-component removal replay "
            "recovers %.1f%% of the MSE optimism, component by "
            "component - the rest is the same-unit adjacency premium.\n"
            % (
                100.0 * headline["mean_leaked_fraction"],
                evaluation["pooled"]["n_runs"],
                headline["detection_precision"],
                headline["detection_recall"],
                headline["baseline_recall"],
                headline["mean_inflation_rmse"],
                headline["mean_rmse_full"],
                headline["mean_rmse_louo"],
                headline["mean_rmse_purged"],
                headline["mean_reference_rmse"],
                100.0 * headline["mean_recovered_share"],
            )
        )
    sys.stdout.write("Wrote evaluation results to %s\n" % out_dir)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="window_leakage")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
