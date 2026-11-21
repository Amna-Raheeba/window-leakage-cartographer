"""Evaluation: seeded fleets, several window geometries, two splits.

Each run generates one fleet, slices it at one (length, stride)
geometry, and audits a seeded random split with the cartographer.
A unit-disjoint split of the same windows is scored as the honest
reference: its test windows come from units the model never saw.

Headline quantities, all measured against planted ground truth:
- how many test windows a random split leaks;
- whether the overlap graph detects them (precision/recall), and how
  the exact-duplicate baseline compares;
- the score inflation (decontaminated RMSE minus leaky RMSE, same
  test set) and how much of it the per-component removal replay
  recovers;
- how inflation grows as windows lengthen and strides shrink.
"""

from .cartographer import audit_split
from .generate import generate_fleet
from .models import mean, rmse
from .predictor import KNNRegressor
from .windows import build_windows, random_split, unit_disjoint_split

EVAL_SEEDS = (101, 102, 103, 104)
# The last geometry is the control: stride == length, so windows tile
# without overlapping and shared-timestep leakage is zero by
# construction. Any remaining gap to the unit-disjoint reference is
# unit-identity leakage, a different phenomenon the map does not claim.
CONFIGS = ((10, 5), (20, 5), (20, 10), (30, 10), (20, 20))
N_UNITS = 24
TEST_FRACTION = 0.25
K = 3


def _reference_rmse(windows, seed):
    train, test = unit_disjoint_split(windows, TEST_FRACTION, seed=seed + 77)
    model = KNNRegressor(k=K)
    model.fit([w.features for w in train], [w.label for w in train])
    preds = model.predict([w.features for w in test])
    return rmse([p - w.label for p, w in zip(preds, test)])


def run_single_seed(seed=7, length=20, stride=5, n_units=N_UNITS):
    """One fleet, one geometry, one random split - audited end to end."""
    fleet = generate_fleet(seed, n_units)
    windows = build_windows(fleet, length, stride)
    train, test = random_split(windows, TEST_FRACTION, seed=seed * 1000 + 1)
    result = audit_split(windows, train, test, fleet, k=K)
    result["seed"] = seed
    result["length"] = length
    result["stride"] = stride
    result["reference_rmse"] = _reference_rmse(windows, seed)
    return result


def run_evaluation(seeds=EVAL_SEEDS, configs=CONFIGS, n_units=N_UNITS):
    """The full seeded evaluation. Returns per-run and pooled data."""
    runs = []
    for seed in seeds:
        fleet = generate_fleet(seed, n_units)
        for cfg_idx, (length, stride) in enumerate(configs):
            windows = build_windows(fleet, length, stride)
            train, test = random_split(
                windows, TEST_FRACTION, seed=seed * 1000 + cfg_idx + 1
            )
            result = audit_split(windows, train, test, fleet, k=K)
            result["seed"] = seed
            result["length"] = length
            result["stride"] = stride
            result["reference_rmse"] = _reference_rmse(windows, seed)
            runs.append(result)

    per_config = {}
    for length, stride in configs:
        key = "%dx%d" % (length, stride)
        group = [
            r for r in runs if r["length"] == length and r["stride"] == stride
        ]
        gain = sum(r["recovered_se_gain"] for r in group)
        total = sum(r["louo_se_gain"] for r in group)
        per_config[key] = {
            "length": length,
            "stride": stride,
            "n_runs": len(group),
            "mean_leaked_fraction": mean([r["leaked_fraction"] for r in group]),
            "mean_inflation_rmse": mean([r["inflation_rmse"] for r in group]),
            "mean_rmse_full": mean([r["rmse_full"] for r in group]),
            "mean_rmse_purged": mean([r["rmse_purged"] for r in group]),
            "mean_rmse_louo": mean([r["rmse_louo"] for r in group]),
            "recovered_share": (gain / total) if total > 1e-9 else 0.0,
        }

    total_tp = sum(r["detection_tp"] for r in runs)
    total_fp = sum(r["detection_fp"] for r in runs)
    total_fn = sum(r["detection_fn"] for r in runs)
    total_leaked = sum(
        sum(1 for t in r["truth_leaked"].values() if t) for r in runs
    )
    total_baseline_hits = sum(r["baseline_hits"] for r in runs)

    total_gain = sum(r["recovered_se_gain"] for r in runs)
    total_louo_gain = sum(r["louo_se_gain"] for r in runs)
    pooled = {
        "n_runs": len(runs),
        "mean_leaked_fraction": mean([r["leaked_fraction"] for r in runs]),
        "detection_precision": (
            total_tp / float(total_tp + total_fp) if (total_tp + total_fp) else 1.0
        ),
        "detection_recall": (
            total_tp / float(total_tp + total_fn) if (total_tp + total_fn) else 1.0
        ),
        "baseline_recall": (
            total_baseline_hits / float(total_leaked) if total_leaked else 0.0
        ),
        "mean_inflation_rmse": mean([r["inflation_rmse"] for r in runs]),
        "mean_inflation_purged": mean([r["inflation_purged"] for r in runs]),
        "mean_rmse_full": mean([r["rmse_full"] for r in runs]),
        "mean_rmse_purged": mean([r["rmse_purged"] for r in runs]),
        "mean_rmse_louo": mean([r["rmse_louo"] for r in runs]),
        "mean_reference_rmse": mean([r["reference_rmse"] for r in runs]),
        "recovered_share": (
            total_gain / total_louo_gain if total_louo_gain > 1e-9 else 0.0
        ),
        "mean_mass_leaked": mean([r["mean_mass_leaked"] for r in runs]),
        "mean_optimism_leaked": mean([r["optimism_leaked"] for r in runs]),
        "mean_optimism_clean": mean([r["optimism_clean"] for r in runs]),
        "mean_straddling": mean([r["n_straddling"] for r in runs]),
    }
    pooled["inflation_pct"] = (
        pooled["mean_inflation_rmse"] / pooled["mean_rmse_louo"]
        if pooled["mean_rmse_louo"]
        else 0.0
    )
    return {
        "seeds": list(seeds),
        "configs": ["%dx%d" % cfg for cfg in configs],
        "n_units": n_units,
        "runs": runs,
        "per_config": per_config,
        "pooled": pooled,
    }


def headline_numbers(evaluation):
    """Flat headline dict - the numbers the README and guard check."""
    pooled = evaluation["pooled"]
    per_config = evaluation["per_config"]
    fallback = {
        "mean_inflation_rmse": pooled["mean_inflation_rmse"],
        "mean_leaked_fraction": pooled["mean_leaked_fraction"],
    }
    dense = per_config.get("10x5", fallback)
    sparse = per_config.get("30x10", fallback)
    control = per_config.get("20x20", fallback)
    return {
        "mean_leaked_fraction": pooled["mean_leaked_fraction"],
        "detection_precision": pooled["detection_precision"],
        "detection_recall": pooled["detection_recall"],
        "baseline_recall": pooled["baseline_recall"],
        "mean_inflation_rmse": pooled["mean_inflation_rmse"],
        "mean_inflation_purged": pooled["mean_inflation_purged"],
        "inflation_pct": pooled["inflation_pct"],
        "mean_recovered_share": pooled["recovered_share"],
        "mean_mass_leaked": pooled["mean_mass_leaked"],
        "mean_optimism_leaked": pooled["mean_optimism_leaked"],
        "mean_optimism_clean": pooled["mean_optimism_clean"],
        "mean_rmse_full": pooled["mean_rmse_full"],
        "mean_rmse_purged": pooled["mean_rmse_purged"],
        "mean_rmse_louo": pooled["mean_rmse_louo"],
        "mean_reference_rmse": pooled["mean_reference_rmse"],
        "inflation_dense": dense["mean_inflation_rmse"],
        "inflation_sparse": sparse["mean_inflation_rmse"],
        "leaked_dense": dense["mean_leaked_fraction"],
        "leaked_sparse": sparse["mean_leaked_fraction"],
        "inflation_control": control["mean_inflation_rmse"],
        "leaked_control": control["mean_leaked_fraction"],
    }
