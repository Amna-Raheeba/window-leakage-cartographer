"""The cartographer: map a split's leakage and price it per component.

Given windows already assigned to train/test, the audit:

1. builds the window-overlap graph (graph.py) and finds the
   connected components that straddle the split - the leakage
   components;
2. computes each test window's leakage mass and checks detection
   against ground truth (a test window is truly leaked iff it shares
   at least one raw timestep with a train window - known exactly
   from the window ranges, independently of the graph);
3. runs the exact-duplicate baseline for comparison;
4. prices the leakage with a kNN regressor, four ways on the SAME
   test windows:
   - leaky: the full training set - the score the benchmark reports;
   - purged: every train window that is a direct leakage source
     (joined by a graph edge to any test window) is removed - the
     purged-cross-validation estimate. It overstates the honest
     score a little, because purging also removes legitimate
     cross-unit neighbours and shrinks the training set;
   - leave-own-unit-out (LOUO): each test window is scored by a
     model trained on the other units only - the ground-truth
     honest score for these exact windows, matching what a
     unit-disjoint benchmark claims to measure;
   - per-component removal replay: for each straddling component,
     remove only that component's direct leakage sources and
     re-score that component's test windows.
   The replay gains, summed over components, are the inflation the
   map itself explains; divided by the total LOUO inflation they
   give the recovery share. The remainder is the same-unit adjacency
   premium - windows that share no timesteps still resemble their
   own trajectory - which no overlap audit can remove; only a
   unit-disjoint split prices it. All three quantities are reported,
   not blended.
"""

from .baseline import exact_duplicate_flags
from .graph import (
    build_edges,
    connected_components,
    index_edges_by_window,
    leakage_mass,
    shared_steps,
)
from .models import mean, mse, rmse
from .predictor import KNNRegressor


def _score(train_windows, test_windows, k, fallback_label):
    """Errors (prediction - truth) of kNN on the test windows.

    If the training set is empty (total purge), fall back to
    predicting the full training set's mean label - never a silent
    perfect score.
    """
    if not train_windows:
        return [fallback_label - w.label for w in test_windows]
    model = KNNRegressor(k=k)
    model.fit([w.features for w in train_windows], [w.label for w in train_windows])
    preds = model.predict([w.features for w in test_windows])
    return [p - w.label for p, w in zip(preds, test_windows)]


def _louo_errors(train, test, k, fallback_label):
    """Leave-own-unit-out errors: each test window is scored by a
    model that never saw its unit. One fit per test unit, cached."""
    by_unit_test = {}
    for w in test:
        by_unit_test.setdefault(w.unit_id, []).append(w)
    errors = {}
    for unit_id, group in by_unit_test.items():
        rest = [w for w in train if w.unit_id != unit_id]
        errs = _score(rest, group, k, fallback_label)
        for w, e in zip(group, errs):
            errors[w.window_id] = e
    return errors


def audit_split(windows, train, test, fleet, k=3):
    """Full leakage audit of one split. Returns a result dict."""
    by_id = dict((w.window_id, w) for w in windows)
    train_ids = set(w.window_id for w in train)
    test_ids = set(w.window_id for w in test)
    fallback_label = mean([w.label for w in train]) if train else 0.0

    edges = build_edges(windows, fleet)
    edges_by_window = index_edges_by_window(edges)
    comps = connected_components([w.window_id for w in windows], edges)

    # Direct leakage sources: train windows joined by an edge to a
    # test window, and the reverse map per test window.
    sources_of = dict((wid, set()) for wid in test_ids)
    all_source_ids = set()
    for e in edges:
        if e["a"] in test_ids and e["b"] in train_ids:
            sources_of[e["a"]].add(e["b"])
            all_source_ids.add(e["b"])
        elif e["b"] in test_ids and e["a"] in train_ids:
            sources_of[e["b"]].add(e["a"])
            all_source_ids.add(e["a"])

    # Ground truth, computed straight from the ranges - not from the
    # graph - so detection precision/recall are honest measurements.
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
    flagged = set(wid for wid, m in mass.items() if m > 0.0)

    tp = sum(1 for wid in flagged if truth_leaked[wid])
    fp = sum(1 for wid in flagged if not truth_leaked[wid])
    fn = sum(1 for wid, t in truth_leaked.items() if t and wid not in flagged)
    precision = tp / float(tp + fp) if (tp + fp) else 1.0
    recall = tp / float(tp + fn) if (tp + fn) else 1.0

    baseline_flagged = exact_duplicate_flags(train, test, fleet)
    baseline_hits = sum(1 for wid in baseline_flagged if truth_leaked[wid])

    # Straddling components: at least one train and one test window.
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
                    "unit_ids": sorted(set(by_id[wid].unit_id for wid in members)),
                }
            )

    # Scores on the same test windows: leaky, purged, LOUO.
    full_errors = _score(train, test, k, fallback_label)
    full_se = dict((w.window_id, e * e) for w, e in zip(test, full_errors))
    clean_train = [w for w in train if w.window_id not in all_source_ids]
    purged_errors = _score(clean_train, test, k, fallback_label)
    purged_se = dict((w.window_id, e * e) for w, e in zip(test, purged_errors))
    louo_err = _louo_errors(train, test, k, fallback_label)
    louo_se = dict((wid, e * e) for wid, e in louo_err.items())

    # Per-component removal replay.
    recovered_se_gain = 0.0
    for comp in straddling:
        source_ids = set(comp["source_ids"])
        comp_train = [w for w in train if w.window_id not in source_ids]
        comp_test = [by_id[wid] for wid in comp["test_ids"]]
        errs = _score(comp_train, comp_test, k, fallback_label)
        comp["rmse_before"] = (
            sum(full_se[wid] for wid in comp["test_ids"])
            / float(len(comp["test_ids"]))
        ) ** 0.5
        comp["rmse_after"] = rmse(errs)
        comp["mse_gain"] = mean([e * e for e in errs]) - mean(
            [full_se[wid] for wid in comp["test_ids"]]
        )
        comp["mean_mass"] = mean([mass[wid] for wid in comp["test_ids"]])
        for w, e in zip(comp_test, errs):
            recovered_se_gain += e * e - full_se[w.window_id]

    louo_se_gain = sum(louo_se[wid] - full_se[wid] for wid in test_ids)
    if louo_se_gain > 1e-9:
        recovered_share = recovered_se_gain / louo_se_gain
    else:
        recovered_share = 0.0

    leaked_ids = [wid for wid, t in truth_leaked.items() if t]
    clean_ids = [wid for wid, t in truth_leaked.items() if not t]
    louo_errors_list = [louo_err[wid] for wid in sorted(test_ids)]

    return {
        "n_windows": len(windows),
        "n_train": len(train),
        "n_test": len(test),
        "n_edges": len(edges),
        "n_components": len(comps),
        "n_straddling": len(straddling),
        "n_sources": len(all_source_ids),
        "components": straddling,
        "mass": mass,
        "truth_leaked": truth_leaked,
        "flagged": sorted(flagged),
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_tp": tp,
        "detection_fp": fp,
        "detection_fn": fn,
        "baseline_flagged": sorted(baseline_flagged),
        "baseline_hits": baseline_hits,
        "leaked_fraction": (
            len(leaked_ids) / float(len(test)) if test else 0.0
        ),
        "mean_mass_leaked": (
            mean([mass[wid] for wid in leaked_ids]) if leaked_ids else 0.0
        ),
        "rmse_full": rmse(full_errors),
        "rmse_purged": rmse(purged_errors),
        "rmse_louo": rmse(louo_errors_list),
        "inflation_rmse": rmse(louo_errors_list) - rmse(full_errors),
        "inflation_purged": rmse(purged_errors) - rmse(full_errors),
        "inflation_mse": mse(louo_errors_list) - mse(full_errors),
        "recovered_share": recovered_share,
        "recovered_se_gain": recovered_se_gain,
        "louo_se_gain": louo_se_gain,
        "optimism_leaked": (
            mean([louo_se[wid] - full_se[wid] for wid in leaked_ids])
            if leaked_ids
            else 0.0
        ),
        "optimism_clean": (
            mean([louo_se[wid] - full_se[wid] for wid in clean_ids])
            if clean_ids
            else 0.0
        ),
    }
