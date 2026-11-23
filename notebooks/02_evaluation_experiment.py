"""Notebook 02 - the headline evaluation experiment.

Runs the full seeded evaluation (24 seeds x 5 window geometries) and
prints the summary: leaked fraction, graph detection vs the
exact-duplicate baseline, score inflation, per-component replay
recovery, and the per-geometry breakdown. Runnable as a plain
script: python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from window_leakage.evaluate import headline_numbers, run_evaluation
from window_leakage.report import render_evaluation_summary


def main():
    evaluation = run_evaluation()
    print(render_evaluation_summary(evaluation, headline_numbers(evaluation)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
