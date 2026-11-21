"""Streamlit dashboard for the Window Leakage Cartographer.

Three tabs:
1. Leakage map - one seeded random split audited end to end: how
   many test windows leaked, their leakage mass distribution, and
   the leaky vs purged vs leave-own-unit-out scores.
2. Components - the straddling components of that split, each with
   the score inflation the removal replay attributes to it.
3. Evaluation - the stored multi-seed headline results (from
   results/evaluation.json): detection vs the exact-duplicate
   baseline, optimism, replay recovery, per-geometry breakdown.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from window_leakage.evaluate import CONFIGS, run_single_seed

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results", "evaluation.json")


@st.cache_data(show_spinner=False)
def load_audit(seed, length, stride):
    return run_single_seed(seed, length, stride)


@st.cache_data(show_spinner=False)
def load_evaluation():
    if not os.path.isfile(RESULTS):
        return None
    with open(RESULTS) as fh:
        return json.load(fh)


def _geometry_picker(key_prefix):
    labels = ["%d x %d" % cfg for cfg in CONFIGS]
    choice = st.selectbox("Window geometry (length x stride)", labels,
                          index=1, key=key_prefix)
    return CONFIGS[labels.index(choice)]


def tab_map():
    st.subheader("Leakage map of one random split")
    length, stride = _geometry_picker("map_geometry")
    seed = st.selectbox("Fleet seed", [7, 21, 42], index=0, key="map_seed")
    result = load_audit(int(seed), int(length), int(stride))
    st.write(
        "A random 75/25 split of sliding windows from the same "
        "trajectories leaks mechanically: a test window that shares "
        "raw timesteps with a train window has effectively been seen."
    )
    cols = st.columns(4)
    cols[0].metric("Test windows leaked",
                   "%.1f%%" % (100.0 * result["leaked_fraction"]))
    cols[1].metric("Mean leakage mass (leaked)",
                   "%.3f" % result["mean_mass_leaked"])
    cols[2].metric("Straddling components", "%d" % result["n_straddling"])
    cols[3].metric("Baseline catches", "%d" % result["baseline_hits"])
    cols = st.columns(4)
    cols[0].metric("Leaky RMSE", "%.2f" % result["rmse_full"])
    cols[1].metric("Purged RMSE", "%.2f" % result["rmse_purged"])
    cols[2].metric("Leave-own-unit-out RMSE", "%.2f" % result["rmse_louo"])
    cols[3].metric("Optimism (RMSE)", "%.2f" % result["inflation_rmse"])
    buckets = {"0": 0, "0-0.25": 0, "0.25-0.5": 0, "0.5-0.75": 0,
               "0.75-1": 0}
    for value in result["mass"].values():
        if value == 0.0:
            buckets["0"] += 1
        elif value <= 0.25:
            buckets["0-0.25"] += 1
        elif value <= 0.5:
            buckets["0.25-0.5"] += 1
        elif value <= 0.75:
            buckets["0.5-0.75"] += 1
        else:
            buckets["0.75-1"] += 1
    st.write("Leakage mass per test window:")
    st.bar_chart(buckets)
    rows = []
    for wid in sorted(result["mass"]):
        rows.append(
            {
                "window": wid,
                "leaked (truth)": result["truth_leaked"][wid],
                "leakage mass": round(result["mass"][wid], 3),
            }
        )
    st.dataframe(rows, height=240)


def tab_components():
    st.subheader("Per-component inflation (removal replay)")
    length, stride = _geometry_picker("comp_geometry")
    seed = st.selectbox("Fleet seed", [7, 21, 42], index=0, key="comp_seed")
    result = load_audit(int(seed), int(length), int(stride))
    st.write(
        "Each straddling component is replayed alone: remove only "
        "its leaked train windows and re-score its test windows. "
        "Replay recovers %.1f%% of the total MSE optimism in this "
        "split; the rest is the same-unit adjacency premium."
        % (100.0 * result["recovered_share"])
    )
    rows = []
    for comp in sorted(result["components"],
                       key=lambda c: c["mse_gain"], reverse=True):
        rows.append(
            {
                "units": ",".join(str(u) for u in comp["unit_ids"]),
                "train windows": len(comp["train_ids"]),
                "test windows": len(comp["test_ids"]),
                "mean mass": round(comp["mean_mass"], 3),
                "test RMSE before": round(comp["rmse_before"], 2),
                "test RMSE after removal": round(comp["rmse_after"], 2),
                "attributed MSE inflation": round(comp["mse_gain"], 2),
            }
        )
    st.dataframe(rows, height=320)
    if rows:
        chart = dict(
            (row["units"], row["attributed MSE inflation"]) for row in rows[:10]
        )
        st.write("Top components by attributed MSE inflation:")
        st.bar_chart(chart)


def tab_evaluation():
    st.subheader("Stored evaluation (4 seeds x 5 geometries)")
    payload = load_evaluation()
    if payload is None:
        st.write("No stored results found. Run: "
                 "python -m window_leakage evaluate --out results")
        return
    headline = payload["headline"]
    cols = st.columns(4)
    cols[0].metric("Test windows leaked",
                   "%.1f%%" % (100.0 * headline["mean_leaked_fraction"]))
    cols[1].metric("Graph detection P / R",
                   "%.3f / %.3f" % (headline["detection_precision"],
                                     headline["detection_recall"]))
    cols[2].metric("Baseline recall", "%.3f" % headline["baseline_recall"])
    cols[3].metric("Optimism (RMSE)", "%.2f" % headline["mean_inflation_rmse"])
    cols = st.columns(3)
    cols[0].metric("Leaky RMSE", "%.2f" % headline["mean_rmse_full"])
    cols[1].metric("Leave-own-unit-out RMSE",
                   "%.2f" % headline["mean_rmse_louo"])
    cols[2].metric("Replay recovery",
                   "%.1f%%" % (100.0 * headline["mean_recovered_share"]))
    rows = []
    for key in payload["configs"]:
        row = payload["per_config"][key]
        rows.append(
            {
                "geometry": key,
                "leaked %": round(100.0 * row["mean_leaked_fraction"], 1),
                "leaky RMSE": round(row["mean_rmse_full"], 2),
                "LOUO RMSE": round(row["mean_rmse_louo"], 2),
                "optimism": round(row["mean_inflation_rmse"], 2),
                "recovery %": round(100.0 * row["recovered_share"], 1),
            }
        )
    st.dataframe(rows)
    st.write(
        "The 20x20 geometry is the control: windows tile without "
        "overlapping, shared-timestep leakage is zero by construction, "
        "and the remaining optimism (%.2f RMSE) is pure same-unit "
        "adjacency - which the replay correctly attributes to no "
        "component." % headline["inflation_control"]
    )


def main():
    st.title("Window Leakage Cartographer")
    st.write(
        "Maps the sliding-window leakage that random splits create "
        "in time-series ML, and prices the score inflation component "
        "by component."
    )
    tab1, tab2, tab3 = st.tabs(["Leakage map", "Components", "Evaluation"])
    with tab1:
        tab_map()
    with tab2:
        tab_components()
    with tab3:
        tab_evaluation()


if __name__ == "__main__":
    main()
