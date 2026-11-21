"""Build the seeded demo data stored under data/.

data/cases.json is a compact summary of the demo fleet (seed 7):
one row per unit with its lifetime and window count at the demo
geometry. data/scenarios.json stores the evaluation geometries and
split parameters. Full trajectories are always regenerated
deterministically from the seed, never stored.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from window_leakage.evaluate import CONFIGS, N_UNITS, TEST_FRACTION
from window_leakage.generate import generate_fleet
from window_leakage.windows import build_windows


def main():
    fleet = generate_fleet(7, N_UNITS)
    demo_windows = build_windows(fleet, 20, 5)
    counts = {}
    for w in demo_windows:
        counts[w.unit_id] = counts.get(w.unit_id, 0) + 1
    payload = {
        "seed": 7,
        "n_units": len(fleet),
        "demo_geometry": {"length": 20, "stride": 5},
        "units": [
            {
                "unit_id": t.unit_id,
                "lifetime": t.lifetime,
                "n_windows_demo_geometry": counts.get(t.unit_id, 0),
            }
            for t in fleet
        ],
    }
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "cases.json"), "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    scenarios = {
        "configs": [
            {"length": length, "stride": stride} for length, stride in CONFIGS
        ],
        "test_fraction": TEST_FRACTION,
        "n_units": N_UNITS,
        "predictor": "knn_regressor_k3",
        "label": "remaining_useful_life_at_window_end",
    }
    with open(os.path.join(data_dir, "scenarios.json"), "w") as fh:
        json.dump(scenarios, fh, indent=1, sort_keys=True)
    print("Wrote data/cases.json (%d units) and data/scenarios.json" % len(fleet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
