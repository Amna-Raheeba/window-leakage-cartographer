"""Example: audit one random split and print the leakage report."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from window_leakage.evaluate import run_single_seed
from window_leakage.report import render_single_seed_report


def main():
    result = run_single_seed(7)
    print(render_single_seed_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
