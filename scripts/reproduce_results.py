"""Reproduce the headline results and guard against drift.

Re-runs the full evaluation on the same seed set that produced the
stored results/evaluation.json and compares the headline numbers.
The pipeline is fully seeded, so reproduction should be near-exact;
exits non-zero if a headline number moved by more than the tolerance,
so a silent change in the generator, graph, or predictor cannot
pass unnoticed.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from window_leakage.evaluate import headline_numbers, run_evaluation

TOLERANCE = 0.02
CHECK_KEYS = (
    "mean_leaked_fraction",
    "detection_precision",
    "detection_recall",
    "baseline_recall",
    "mean_inflation_rmse",
    "mean_recovered_share",
)


def main():
    stored_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "evaluation.json")
    with open(stored_path) as fh:
        stored = json.load(fh)
    stored_headline = stored["headline"]
    fresh = headline_numbers(run_evaluation())
    drifted = []
    for key in CHECK_KEYS:
        old = stored_headline[key]
        new = fresh[key]
        print("%s: stored %.4f, reproduced %.4f" % (key, old, new))
        if abs(old - new) > TOLERANCE:
            drifted.append(key)
    if drifted:
        print("DRIFT DETECTED in: %s" % ", ".join(drifted))
        return 1
    print("Headline reproduced within tolerance %.2f." % TOLERANCE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
