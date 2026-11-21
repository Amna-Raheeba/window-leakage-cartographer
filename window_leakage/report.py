"""Human-readable reports for one audit and for the full evaluation."""


def render_single_seed_report(result):
    lines = []
    lines.append(
        "Window leakage audit - seed %d, windows %d long, stride %d"
        % (result["seed"], result["length"], result["stride"])
    )
    lines.append(
        "%d windows (%d train / %d test), %d graph edges, %d components, "
        "%d straddling the split."
        % (
            result["n_windows"],
            result["n_train"],
            result["n_test"],
            result["n_edges"],
            result["n_components"],
            result["n_straddling"],
        )
    )
    lines.append("")
    lines.append(
        "Leaked test windows: %.1f%% of the test set (mean leakage mass "
        "%.3f among them)."
        % (100.0 * result["leaked_fraction"], result["mean_mass_leaked"])
    )
    lines.append(
        "Graph detection vs ground truth: precision %.3f, recall %.3f."
        % (result["detection_precision"], result["detection_recall"])
    )
    lines.append(
        "Exact-duplicate baseline caught %d of the leaked test windows."
        % result["baseline_hits"]
    )
    lines.append("")
    lines.append(
        "kNN RUL score (RMSE, lower is better): leaky split %.2f, "
        "purged %.2f, leave-own-unit-out %.2f, unit-disjoint split "
        "reference %.2f."
        % (
            result["rmse_full"],
            result["rmse_purged"],
            result["rmse_louo"],
            result["reference_rmse"],
        )
    )
    lines.append(
        "Total optimism vs leave-own-unit-out: %.2f RMSE. Per-component "
        "removal replay recovers %.1f%% of the MSE optimism; the rest "
        "is the same-unit adjacency premium."
        % (result["inflation_rmse"], 100.0 * result["recovered_share"])
    )
    lines.append("")
    lines.append("Largest leakage components by attributed MSE inflation:")
    comps = sorted(
        result["components"], key=lambda c: c["mse_gain"], reverse=True
    )
    for comp in comps[:5]:
        lines.append(
            "  units %s: %d train / %d test windows, mean mass %.2f, "
            "test RMSE %.2f -> %.2f without its leaked train windows"
            % (
                ",".join(str(u) for u in comp["unit_ids"]),
                len(comp["train_ids"]),
                len(comp["test_ids"]),
                comp["mean_mass"],
                comp["rmse_before"],
                comp["rmse_after"],
            )
        )
    return "\n".join(lines) + "\n"


def render_evaluation_summary(evaluation, headline):
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append(
        "Pooled over %d runs (%d seeds x %d window geometries, %d units "
        "per fleet, random 75/25 splits, kNN k=3 RUL regression):"
        % (
            evaluation["pooled"]["n_runs"],
            len(evaluation["seeds"]),
            len(evaluation["configs"]),
            evaluation["n_units"],
        )
    )
    lines.append("")
    lines.append(
        "- Test windows leaked by the random split: %.1f%% on average."
        % (100.0 * headline["mean_leaked_fraction"])
    )
    lines.append(
        "- Overlap-graph detection: precision %.3f / recall %.3f; the "
        "exact-duplicate baseline's recall is %.3f."
        % (
            headline["detection_precision"],
            headline["detection_recall"],
            headline["baseline_recall"],
        )
    )
    lines.append(
        "- Score optimism: leaky RMSE %.2f vs leave-own-unit-out %.2f "
        "(purged %.2f, unit-disjoint split reference %.2f) - %.2f "
        "RMSE, %.1f%% of the honest score."
        % (
            headline["mean_rmse_full"],
            headline["mean_rmse_louo"],
            headline["mean_rmse_purged"],
            headline["mean_reference_rmse"],
            headline["mean_inflation_rmse"],
            100.0 * headline["inflation_pct"],
        )
    )
    lines.append(
        "- Per-component removal replay recovers %.1f%% of the MSE "
        "optimism; the remainder is the same-unit adjacency premium."
        % (100.0 * headline["mean_recovered_share"])
    )
    lines.append(
        "- Mean per-window MSE optimism: %.2f for leaked test windows "
        "vs %.2f for unleaked ones (the adjacency premium)."
        % (headline["mean_optimism_leaked"], headline["mean_optimism_clean"])
    )
    lines.append("")
    lines.append("Per geometry (window length x stride):")
    for key in evaluation["configs"]:
        row = evaluation["per_config"][key]
        lines.append(
            "- %s: leaked %.1f%%, optimism %.2f RMSE, replay recovery "
            "%.1f%%."
            % (
                key,
                100.0 * row["mean_leaked_fraction"],
                row["mean_inflation_rmse"],
                100.0 * row["recovered_share"],
            )
        )
    return "\n".join(lines) + "\n"
