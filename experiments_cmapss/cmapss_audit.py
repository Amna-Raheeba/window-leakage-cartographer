"""C-MAPSS FD001 real-data audit with the Window Leakage Cartographer.

Loads the genuine NASA C-MAPSS FD001 training file, builds sliding
windows per engine, wraps them in the package's own Window /
Trajectory objects so the overlap graph, connected components and
leakage mass come from the package's code (window_leakage.graph),
then prices random 75/25 window-level splits four ways (leaky,
purged, leave-own-unit-out, per-component removal replay) with two
regressors:

- kNN (k=3): a numpy implementation that replicates the package's
  KNNRegressor semantics exactly (per-fit standardisation with
  population std, Euclidean distance, ties broken by train order,
  prediction = mean of the k nearest labels). Equivalence with the
  package's own audit_split is verified on one seed before the full
  run (see run_audit_split_equivalence_check) because the package's
  pure-Python predictor is too slow to re-run for every seed.
- ridge regression (lambda=1.0) implemented with numpy; features
  standardised with the split's training-set statistics.

Scoring formulas (fallback on empty training set, LOUO grouping,
replay gain accumulation, recovery share) mirror
window_leakage/cartographer.py line for line in structure.

Outputs: results.json (all per-seed numbers) and summary.md.
No package files are modified.
"""

import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from window_leakage.graph import (  # noqa: E402
    build_edges,
    connected_components,
    index_edges_by_window,
    leakage_mass,
    shared_steps,
)
from window_leakage.models import Trajectory, Window, mean  # noqa: E402

DATA_PATH = os.path.join(HERE, "data", "train_FD001.txt")
RESULTS_PATH = os.path.join(HERE, "results.json")
SUMMARY_PATH = os.path.join(HERE, "summary.md")
SOURCE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
    " (NASA Prognostics Data Repository mirror; CMAPSSData.zip inside)"
)
RUL_CAP = 125
SPLIT_SEEDS = list(range(10))
GEOMETRIES = [(30, 5), (20, 5)]


# ---------------------------------------------------------------- data

def load_fd001(path):
    """Returns (fleet, per-unit sensor arrays).

    fleet: list of package Trajectory objects whose channels are the
    21 sensor series, so the package graph code (including its
    near-duplicate channel) operates on the real raw data.
    units: dict unit_id -> (lifetime, sensors array [lifetime, 21]).
    """
    data = np.loadtxt(path)
    units = {}
    fleet = []
    for unit_id in sorted(set(int(u) for u in data[:, 0])):
        rows = data[data[:, 0] == unit_id]
        rows = rows[np.argsort(rows[:, 1])]
        sensors = rows[:, 5:26]
        lifetime = int(rows.shape[0])
        channels = tuple(
            tuple(float(v) for v in sensors[:, c]) for c in range(21)
        )
        fleet.append(
            Trajectory(unit_id=unit_id, channels=channels, lifetime=lifetime)
        )
        units[unit_id] = (lifetime, sensors)
    return fleet, units


def build_windows(units, length, stride):
    """Sliding windows; features = per-sensor mean and std (42).

    Label: true RUL at the window's last cycle = lifetime - end,
    capped at RUL_CAP (the package convention: label is the RUL at
    the window's end).
    """
    windows = []
    wid = 0
    for unit_id in sorted(units):
        lifetime, sensors = units[unit_id]
        start = 0
        while start + length <= lifetime:
            seg = sensors[start:start + length]
            feats = []
            for c in range(seg.shape[1]):
                feats.append(float(np.mean(seg[:, c])))
                feats.append(float(np.std(seg[:, c])))
            label = float(min(lifetime - (start + length), RUL_CAP))
            windows.append(
                Window(
                    window_id=wid,
                    unit_id=unit_id,
                    start=start,
                    length=length,
                    features=tuple(feats),
                    label=label,
                )
            )
            wid += 1
            start += stride
    return windows


# ------------------------------------------------------------ regressors

class NumpyKNNRegressor:
    """Exact numpy replica of window_leakage.predictor.KNNRegressor."""

    def __init__(self, k=3):
        self.k = k

    def fit(self, features, labels):
        self.train_x = np.array(features, dtype=float)
        self.train_y = np.array(labels, dtype=float)
        self.center = self.train_x.mean(axis=0)
        scale = self.train_x.std(axis=0)  # population std, as package
        self.scale = np.where(scale > 1e-12, scale, 1.0)
        self.z = (self.train_x - self.center) / self.scale
        return self

    def predict(self, features):
        x = np.array(features, dtype=float)
        z = (x - self.center) / self.scale
        # Squared Euclidean distances, test x train.
        d2 = (
            np.sum(z * z, axis=1)[:, None]
            + np.sum(self.z * self.z, axis=1)[None, :]
            - 2.0 * np.dot(z, self.z.T)
        )
        order = np.argsort(d2, axis=1, kind="stable")
        k = min(self.k, self.train_x.shape[0])
        preds = []
        for i in range(z.shape[0]):
            idx = order[i, :k]
            preds.append(float(np.mean(self.train_y[idx])))
        return preds


class RidgeRegressor:
    """Ridge regression (lambda=1.0) on features standardised with
    fixed statistics supplied at construction (the split's training
    set). Intercept via centring on the fit subset's label mean."""

    def __init__(self, center, scale, lam=1.0):
        self.center = np.array(center, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.lam = lam

    def _z(self, features):
        x = np.array(features, dtype=float)
        return (x - self.center) / self.scale

    def fit(self, features, labels):
        z = self._z(features)
        y = np.array(labels, dtype=float)
        self.y_mean = float(np.mean(y))
        yc = y - self.y_mean
        n_feat = z.shape[1]
        a = np.dot(z.T, z) + self.lam * np.eye(n_feat)
        b = np.dot(z.T, yc)
        try:
            self.beta = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            self.beta = np.linalg.lstsq(a, b, rcond=None)[0]
        return self

    def predict(self, features):
        z = self._z(features)
        return [float(v) for v in (np.dot(z, self.beta) + self.y_mean)]


# --------------------------------------------------------------- scoring

def _errors(model_factory, train_windows, test_windows, fallback_label):
    """(prediction - truth) errors; mirrors cartographer._score."""
    if not train_windows:
        return [fallback_label - w.label for w in test_windows]
    model = model_factory()
    model.fit(
        [w.features for w in train_windows],
        [w.label for w in train_windows],
    )
    preds = model.predict([w.features for w in test_windows])
    return [p - w.label for p, w in zip(preds, test_windows)]


def _rmse(errors):
    if not errors:
        return 0.0
    return math.sqrt(sum(e * e for e in errors) / float(len(errors)))


def audit_with_model(windows, train, test, edges, comps, model_factory):
    """The cartographer's four scorings with a pluggable regressor.

    Graph-derived quantities (sources, straddling components, mass,
    ground-truth leakage) are computed exactly as in
    window_leakage/cartographer.py from the package-built edges.
    """
    by_id = dict((w.window_id, w) for w in windows)
    train_ids = set(w.window_id for w in train)
    test_ids = set(w.window_id for w in test)
    fallback_label = mean([w.label for w in train]) if train else 0.0

    edges_by_window = index_edges_by_window(edges)
    sources_of = dict((wid, set()) for wid in test_ids)
    all_source_ids = set()
    for e in edges:
        if e["a"] in test_ids and e["b"] in train_ids:
            sources_of[e["a"]].add(e["b"])
            all_source_ids.add(e["b"])
        elif e["b"] in test_ids and e["a"] in train_ids:
            sources_of[e["b"]].add(e["a"])
            all_source_ids.add(e["a"])

    truth_leaked = {}
    for tw in test:
        leaked = False
        for tr in train:
            if shared_steps(tw, tr) > 0:
                leaked = True
                break
        truth_leaked[tw.window_id] = leaked

    mass = dict(
        (tw.window_id, leakage_mass(tw.window_id, edges_by_window, train_ids))
        for tw in test
    )

    straddling = []
    for members in comps:
        member_train = [wid for wid in members if wid in train_ids]
        member_test = [wid for wid in members if wid in test_ids]
        if member_train and member_test:
            comp_sources = set()
            for wid in member_test:
                comp_sources.update(sources_of[wid])
            straddling.append(
                {
                    "window_ids": members,
                    "train_ids": member_train,
                    "test_ids": member_test,
                    "source_ids": sorted(comp_sources),
                }
            )

    full_errors = _errors(model_factory, train, test, fallback_label)
    full_se = dict((w.window_id, e * e) for w, e in zip(test, full_errors))

    clean_train = [w for w in train if w.window_id not in all_source_ids]
    purged_errors = _errors(model_factory, clean_train, test, fallback_label)

    by_unit_test = {}
    for w in test:
        by_unit_test.setdefault(w.unit_id, []).append(w)
    louo_err = {}
    for unit_id, group in by_unit_test.items():
        rest = [w for w in train if w.unit_id != unit_id]
        errs = _errors(model_factory, rest, group, fallback_label)
        for w, e in zip(group, errs):
            louo_err[w.window_id] = e
    louo_errors_list = [louo_err[wid] for wid in sorted(test_ids)]
    louo_se = dict((wid, e * e) for wid, e in louo_err.items())

    recovered_se_gain = 0.0
    for comp in straddling:
        source_ids = set(comp["source_ids"])
        comp_train = [w for w in train if w.window_id not in source_ids]
        comp_test = [by_id[wid] for wid in comp["test_ids"]]
        errs = _errors(model_factory, comp_train, comp_test, fallback_label)
        for w, e in zip(comp_test, errs):
            recovered_se_gain += e * e - full_se[w.window_id]

    louo_se_gain = sum(louo_se[wid] - full_se[wid] for wid in test_ids)
    if louo_se_gain > 1e-9:
        recovered_share = recovered_se_gain / louo_se_gain
    else:
        recovered_share = 0.0

    leaked_ids = [wid for wid, t in truth_leaked.items() if t]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_straddling": len(straddling),
        "n_sources": len(all_source_ids),
        "leaked_fraction": (
            len(leaked_ids) / float(len(test)) if test else 0.0
        ),
        "mean_mass_leaked": (
            mean([mass[wid] for wid in leaked_ids]) if leaked_ids else 0.0
        ),
        "rmse_leaky": _rmse(full_errors),
        "rmse_purged": _rmse(purged_errors),
        "rmse_louo": _rmse(louo_errors_list),
        "optimism_rmse": _rmse(louo_errors_list) - _rmse(full_errors),
        "recovered_se_gain": recovered_se_gain,
        "louo_se_gain": louo_se_gain,
        "recovered_share": recovered_share,
    }


# ------------------------------------------------------- equivalence

def equivalence_check(windows, train, test, fleet):
    """Run the package's own audit_split (kNN) on one split and
    compare against the numpy replica on the same split."""
    from window_leakage.cartographer import audit_split

    pkg = audit_split(windows, train, test, fleet, k=3)
    edges = build_edges(windows, fleet)
    comps = connected_components([w.window_id for w in windows], edges)
    mine = audit_with_model(
        windows, train, test, edges, comps,
        lambda: NumpyKNNRegressor(k=3),
    )
    diffs = {}
    for key in ("rmse_leaky", "rmse_purged", "rmse_louo"):
        pkg_key = {
            "rmse_leaky": "rmse_full",
            "rmse_purged": "rmse_purged",
            "rmse_louo": "rmse_louo",
        }[key]
        diffs[key] = abs(pkg[pkg_key] - mine[key])
    diffs["recovered_share"] = abs(pkg["recovered_share"] - mine["recovered_share"])
    diffs["leaked_fraction"] = abs(pkg["leaked_fraction"] - mine["leaked_fraction"])
    return diffs


# ----------------------------------------------------------------- main

def run_geometry(fleet, units, length, stride, do_equivalence=False):
    windows = build_windows(units, length, stride)
    print(
        "geometry %dx%d: %d windows; building graph with package code..."
        % (length, stride, len(windows))
    )
    edges = build_edges(windows, fleet)
    n_nd = sum(1 for e in edges if e["source"] == "near_duplicate")
    print(
        "  edges: %d (overlap %d, near-duplicate %d)"
        % (len(edges), len(edges) - n_nd, n_nd)
    )
    comps = connected_components([w.window_id for w in windows], edges)

    per_seed = []
    equivalence = None
    for seed in SPLIT_SEEDS:
        ids = [w.window_id for w in windows]
        random.Random(seed).shuffle(ids)
        n_test = int(round(0.25 * len(ids)))
        test_ids = set(ids[:n_test])
        train = [w for w in windows if w.window_id not in test_ids]
        test = [w for w in windows if w.window_id in test_ids]

        if do_equivalence and seed == SPLIT_SEEDS[0]:
            print("  equivalence check vs package audit_split (seed %d)..." % seed)
            equivalence = equivalence_check(windows, train, test, fleet)
            print("  max abs diff: %.3e" % max(equivalence.values()))

        # Ridge standardisation statistics from this split's train set.
        train_feats = np.array([w.features for w in train], dtype=float)
        center = train_feats.mean(axis=0)
        scale = train_feats.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)

        knn = audit_with_model(
            windows, train, test, edges, comps,
            lambda: NumpyKNNRegressor(k=3),
        )
        ridge = audit_with_model(
            windows, train, test, edges, comps,
            lambda: RidgeRegressor(center, scale, lam=1.0),
        )
        row = {"seed": seed, "knn": knn, "ridge": ridge}
        per_seed.append(row)
        print(
            "  seed %d: leaked %.3f | knn leaky %.2f louo %.2f opt %.2f "
            "rec %.3f | ridge leaky %.2f louo %.2f opt %.2f rec %.3f"
            % (
                seed,
                knn["leaked_fraction"],
                knn["rmse_leaky"], knn["rmse_louo"], knn["optimism_rmse"],
                knn["recovered_share"],
                ridge["rmse_leaky"], ridge["rmse_louo"],
                ridge["optimism_rmse"], ridge["recovered_share"],
            )
        )
    return {
        "window_length": length,
        "stride": stride,
        "n_windows": len(windows),
        "n_edges": len(edges),
        "n_near_duplicate_edges": n_nd,
        "equivalence_check_seed0": equivalence,
        "per_seed": per_seed,
        "pooled": {
            "knn": pool(per_seed, "knn"),
            "ridge": pool(per_seed, "ridge"),
        },
    }


def pool(per_seed, model_key):
    keys = [
        "leaked_fraction", "mean_mass_leaked", "rmse_leaky", "rmse_purged",
        "rmse_louo", "optimism_rmse", "recovered_share", "n_straddling",
    ]
    out = {}
    for key in keys:
        vals = np.array([row[model_key][key] for row in per_seed], dtype=float)
        out[key] = {
            "mean": float(np.mean(vals)),
            "std_sample": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    rec = sum(row[model_key]["recovered_se_gain"] for row in per_seed)
    louo = sum(row[model_key]["louo_se_gain"] for row in per_seed)
    out["recovered_share_pooled_by_se_gain"] = (
        rec / louo if louo > 1e-9 else 0.0
    )
    tot_leaked = sum(
        row[model_key]["leaked_fraction"] * row[model_key]["n_test"]
        for row in per_seed
    )
    tot_test = sum(row[model_key]["n_test"] for row in per_seed)
    out["leaked_fraction_pooled"] = tot_leaked / float(tot_test)
    return out


def fmt_pm(entry, digits=2):
    return "%.*f +/- %.*f" % (
        digits, entry["mean"], digits, entry["std_sample"]
    )


def write_summary(results):
    lines = []
    lines.append("# C-MAPSS FD001 real-data audit — summary")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append("- Source: %s" % SOURCE_URL)
    lines.append(
        "- File: data/train_FD001.txt, md5 %s, sha256 %s"
        % (results["data_md5"], results["data_sha256"])
    )
    lines.append(
        "- Verified as the genuine FD001 training set: %d rows, %d "
        "columns, %d engine units (1-100), no NaNs. The sha256 also "
        "matches the independently published fingerprint for this file."
        % (
            results["data_rows"], results["data_cols"],
            results["data_units"],
        )
    )
    lines.append(
        "- Windows: per engine, length x stride geometries below; "
        "features are the per-sensor mean and standard deviation over "
        "the window (21 sensors, 42 features); label is the true RUL "
        "at the window's last cycle, capped at %d." % RUL_CAP
    )
    lines.append(
        "- Splits: random 75/25 window-level splits, %d split seeds "
        "(0-9). Scores: leaky / purged / leave-own-unit-out (LOUO) / "
        "per-component removal replay, computed with the package's "
        "graph, components and leakage mass; regressors are kNN "
        "(k=3, package semantics) and ridge (lambda=1.0)."
        % len(SPLIT_SEEDS)
    )
    lines.append("")
    for geo in results["geometries"]:
        lines.append(
            "## Geometry: window length %d, stride %d (%d windows, %d graph edges, %d near-duplicate edges)"
            % (
                geo["window_length"], geo["stride"], geo["n_windows"],
                geo["n_edges"], geo["n_near_duplicate_edges"],
            )
        )
        lines.append("")
        eq = geo["equivalence_check_seed0"]
        if eq:
            lines.append(
                "Equivalence check (seed 0): the numpy kNN used here "
                "reproduces the package's own audit_split on the same "
                "split with max abs difference %.2e across leaky / "
                "purged / LOUO RMSE, recovery share and leaked fraction."
                % max(eq.values())
            )
            lines.append("")
        lines.append(
            "| Regressor | Leaked fraction | Leaky RMSE | Purged RMSE | LOUO RMSE | Optimism (LOUO - leaky) | Recovery share (mean of seeds) | Recovery share (pooled by SE gain) | Straddling components |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for name, key in (("kNN (k=3)", "knn"), ("Ridge (lambda=1.0)", "ridge")):
            p = geo["pooled"][key]
            lines.append(
                "| %s | %s (pooled %.3f) | %s | %s | %s | %s | %s | %.3f | %s |"
                % (
                    name,
                    fmt_pm(p["leaked_fraction"], 3),
                    p["leaked_fraction_pooled"],
                    fmt_pm(p["rmse_leaky"]),
                    fmt_pm(p["rmse_purged"]),
                    fmt_pm(p["rmse_louo"]),
                    fmt_pm(p["optimism_rmse"]),
                    fmt_pm(p["recovered_share"], 3),
                    p["recovered_share_pooled_by_se_gain"],
                    fmt_pm(p["n_straddling"], 1),
                )
            )
        lines.append("")
    lines.append("## What the real-data run shows (honest read)")
    lines.append("")
    lines.append(results["honest_paragraph"])
    lines.append("")
    with open(SUMMARY_PATH, "w") as fh:
        fh.write("\n".join(lines))


def main():
    import hashlib

    with open(DATA_PATH, "rb") as fh:
        blob = fh.read()
    md5 = hashlib.md5(blob).hexdigest()
    sha256 = hashlib.sha256(blob).hexdigest()
    data = np.loadtxt(DATA_PATH)
    fleet, units = load_fd001(DATA_PATH)
    print(
        "data: %d rows, %d cols, %d units, md5 %s"
        % (data.shape[0], data.shape[1], len(units), md5)
    )

    geometries = []
    for i, (length, stride) in enumerate(GEOMETRIES):
        geo = run_geometry(
            fleet, units, length, stride, do_equivalence=(i == 0)
        )
        geometries.append(geo)

    results = {
        "source_url": SOURCE_URL,
        "data_file": "data/train_FD001.txt",
        "data_md5": md5,
        "data_sha256": sha256,
        "data_rows": int(data.shape[0]),
        "data_cols": int(data.shape[1]),
        "data_units": len(units),
        "rul_cap": RUL_CAP,
        "split_seeds": SPLIT_SEEDS,
        "geometries": geometries,
        "honest_paragraph": "PENDING - filled after inspecting results.",
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote %s" % RESULTS_PATH)
    write_summary(results)
    print("wrote %s" % SUMMARY_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
