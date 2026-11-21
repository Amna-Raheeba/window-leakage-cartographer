"""Notebook 01 - demo run.

Audits one random split of the demo fleet (seed 7, windows 20 long,
stride 5) and prints the leakage map summary: how many test windows
leaked, how the graph detection compares with the exact-duplicate
baseline, the leaky vs decontaminated score, and the largest leakage
components. Runnable as a plain script: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from window_leakage.evaluate import run_single_seed
from window_leakage.report import render_single_seed_report


def main():
    result = run_single_seed(7)
    print(render_single_seed_report(result))
    n_leaked = sum(1 for t in result["truth_leaked"].values() if t)
    print(
        "Ground truth: %d of %d test windows share raw timesteps with "
        "a train window; the baseline caught %d of them."
        % (n_leaked, result["n_test"], result["baseline_hits"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
