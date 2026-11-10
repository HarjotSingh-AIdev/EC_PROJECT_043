"""Create and verify the report figures from saved experiment results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ec101-26bcs043-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import r2_score

from config import DE_EVALUATIONS, FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, SCHEMA_JSON, ensure_directories, log, record_script_time, write_json

sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a figure and reopen it to confirm the PNG is valid."""
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(path) as image:
        image.verify()


def _plot_surrogate_parity(outputs: list[str], test: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Plot all test output parity panels for the final ANN."""
    fig, axes = plt.subplots(5, 7, figsize=(17, 13), constrained_layout=True)
    prediction_values = predictions[outputs].to_numpy(float)
    truth_values = test[outputs].to_numpy(float)
    for index, axis in enumerate(axes.flat):
        truth, predicted = truth_values[:, index], prediction_values[:, index]
        axis.scatter(truth, predicted, s=20, alpha=0.78, color="#1877a3", edgecolors="white", linewidths=0.4)
        lower = float(min(truth.min(), predicted.min()))
        upper = float(max(truth.max(), predicted.max()))
        span = max(upper - lower, 1e-9)
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#b23a48", linewidth=1)
        axis.set_xlim(lower - span * 0.05, upper + span * 0.05)
        axis.set_ylim(lower - span * 0.05, upper + span * 0.05)
        r2 = float(r2_score(truth, predicted)) if np.std(truth) else float("nan")
        axis.set_title(f"{outputs[index]}\nR²={r2:.2f}", fontsize=8)
        axis.tick_params(labelsize=6)
    fig.suptitle("ANN test parity by output (7 held-out designs)", fontsize=15, weight="bold")
    _save(fig, "surrogate_parity_grid.png")


def _plot_convergence() -> None:
    """Plot median and interquartile best-objective traces across targets and seeds."""
    traces = pd.read_csv(RESULTS_DIR / "convergence_traces.csv")
    fig, axis = plt.subplots(figsize=(8.5, 5.3), constrained_layout=True)
    palette = {"ANN + DE": "#176b87", "ExtraTrees + DE": "#e07a32", "Random search (ANN)": "#59a14f", "Random search (ExtraTrees)": "#b07aa1"}
    for method, subset in traces.groupby("method", sort=False):
        grouped = subset.groupby("evaluation_count")["best_objective"]
        stats = grouped.agg(["median", lambda values: values.quantile(0.25), lambda values: values.quantile(0.75)]).reset_index()
        x = stats["evaluation_count"].to_numpy()
        median = np.maximum(stats["median"].to_numpy(), 1e-12)
        q25, q75 = np.maximum(stats["<lambda_0>"].to_numpy(), 1e-12), np.maximum(stats["<lambda_1>"].to_numpy(), 1e-12)
        color = palette.get(method, None)
        axis.plot(x, median, label=method, color=color, linewidth=1.7)
        axis.fill_between(x, q25, q75, color=color, alpha=0.14)
    axis.set_yscale("log")
    axis.set_xlabel("Surrogate evaluations")
    axis.set_ylabel("Best objective (log scale)")
    axis.set_title("Inverse-search convergence across targets and seeds", weight="bold")
    axis.legend(frameon=True, fontsize=8)
    _save(fig, "convergence_curves.png")


def _plot_inverse_parity(outputs: list[str], predictions: pd.DataFrame) -> None:
    """Plot per-output target-versus-achieved parity for both DE surrogates."""
    fig, axes = plt.subplots(5, 7, figsize=(17, 13), constrained_layout=True)
    subset = predictions[predictions["method"].isin(["ANN + DE", "ExtraTrees + DE"])]
    colors = {"ANN + DE": "#176b87", "ExtraTrees + DE": "#e07a32"}
    for axis, output in zip(axes.flat, outputs):
        part = subset[subset["output"] == output]
        lower = float(min(part["target_value"].min(), part["achieved_value"].min()))
        upper = float(max(part["target_value"].max(), part["achieved_value"].max()))
        for method, rows in part.groupby("method", sort=False):
            axis.scatter(rows["target_value"], rows["achieved_value"], s=12, alpha=0.65, label=method, color=colors[method])
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#b23a48", linewidth=1)
        axis.set_title(output, fontsize=8)
        axis.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6, loc="best")
    fig.suptitle("Surrogate-achieved versus requested performance", fontsize=15, weight="bold")
    _save(fig, "achieved_vs_target.png")


def _plot_dv_target_tradeoff(solutions: pd.DataFrame) -> None:
    """Show how original-DV error relates to target-performance error."""
    fig, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for method, part in solutions.groupby("method", sort=False):
        axis.scatter(part["target_nrmse"], part["dv_rmse_physical"], s=28, alpha=0.72, label=method)
    axis.set_xlabel("Target normalized RMSE (train-range normalized)")
    axis.set_ylabel("DV RMSE in physical units (mixed-scale macro)")
    axis.set_title("Inverse performance and design-variable error", weight="bold")
    axis.legend(fontsize=7, ncol=2)
    _save(fig, "dv_vs_target_error.png")


def _plot_repeatability(solutions: pd.DataFrame) -> None:
    """Plot objective distributions over fixed targets and optimizer seeds."""
    stochastic = solutions[solutions["seed"] >= 0].copy()
    fig, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    sns.boxplot(data=stochastic, x="method", y="objective", ax=axis, color="#75aadb", fliersize=2)
    axis.set_yscale("log")
    axis.set_xlabel("")
    axis.set_ylabel("Final standardized objective (log scale)")
    axis.set_title("Variation across five fixed seeds and test targets", weight="bold")
    axis.tick_params(axis="x", rotation=18, labelsize=8)
    _save(fig, "repeatability.png")


def _plot_baselines(solutions: pd.DataFrame) -> None:
    """Compare target error and physical DV error across all baselines."""
    summary = solutions.groupby("method", sort=False).agg(
        target_nrmse=("target_nrmse", "mean"), dv_rmse=("dv_rmse_physical", "mean")
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    palette = sns.color_palette("colorblind", len(summary))
    sns.barplot(data=summary, x="target_nrmse", y="method", hue="method", legend=False, ax=axes[0], palette=palette)
    sns.barplot(data=summary, x="dv_rmse", y="method", hue="method", legend=False, ax=axes[1], palette=palette)
    axes[0].set_xlabel("Mean target normalized RMSE")
    axes[1].set_xlabel("Mean physical DV RMSE")
    for axis in axes:
        axis.set_ylabel("")
    axes[0].set_title("Requested performance mismatch")
    axes[1].set_title("Distance from the known test design")
    fig.suptitle("Hybrid, optimizer, and retrieval baselines", fontsize=14, weight="bold")
    _save(fig, "baseline_comparison.png")


def _plot_benchmark_sanity() -> None:
    """Plot optimizer sanity-check traces on three known-optimum functions."""
    traces = pd.read_csv(RESULTS_DIR / "sanity_convergence.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for axis, (function, part) in zip(axes, traces.groupby("function", sort=False)):
        grouped = part.groupby("evaluation_count")["best_objective"]
        med = grouped.median()
        axis.plot(med.index, np.maximum(med.to_numpy(), 1e-14), color="#176b87", linewidth=1.7)
        axis.set_yscale("log")
        axis.set_title(function)
        axis.set_xlabel("Evaluations")
    axes[0].set_ylabel("Median best-so-far objective (log scale)")
    fig.suptitle("DE implementation sanity check (10 dimensions, known optimum 0)", fontsize=13, weight="bold")
    _save(fig, "benchmark_convergence.png")


def _plot_sensitivity() -> None:
    """Plot the fixed-budget population-size sensitivity results."""
    study = pd.read_csv(RESULTS_DIR / "sensitivity.csv")
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    sns.boxplot(data=study, x="population_size", y="final_objective", ax=axis, color="#81b29a", fliersize=3)
    sns.stripplot(data=study, x="population_size", y="final_objective", ax=axis, color="#253237", size=5)
    axis.set_yscale("log")
    axis.set_xlabel("DE population size")
    axis.set_ylabel("Final ANN objective (log scale)")
    axis.set_title(f"Population size under a fixed {DE_EVALUATIONS}-evaluation budget", weight="bold")
    _save(fig, "sensitivity.png")


def _plot_surrogate_ablation() -> None:
    """Separate test surrogate error from inverse residual error in a diagnostic view."""
    data = pd.read_csv(RESULTS_DIR / "surrogate_ablation.csv")
    data = data[data["surrogate_test_nrmse_macro"].notna()].copy()
    long = data.melt(
        id_vars=["surrogate", "optimizer"],
        value_vars=["surrogate_test_nrmse_macro", "inverse_predicted_target_nrmse"],
        var_name="component",
        value_name="normalized_error",
    )
    long["component"] = long["component"].map(
        {
            "surrogate_test_nrmse_macro": "Forward surrogate test error",
            "inverse_predicted_target_nrmse": "Inverse target residual",
        }
    )
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    sns.barplot(data=long, x="optimizer", y="normalized_error", hue="component", ax=axis, palette=["#4c78a8", "#f28e2b"])
    axis.set_xlabel("")
    axis.set_ylabel("Mean normalized error")
    axis.set_title("Surrogate and optimizer error components", weight="bold")
    axis.tick_params(axis="x", rotation=12, labelsize=8)
    axis.legend(fontsize=8)
    _save(fig, "surrogate_ablation.png")


def _plot_feature_responsibility(outputs: list[str]) -> None:
    """Visualize per-output permutation effects and the handout's grouped map."""
    importance = pd.read_csv(RESULTS_DIR / "feature_importance_per_output.csv")
    matrix = importance.pivot(index="feature", columns="output", values="importance_mean").reindex(
        index=[f"DV{i}" for i in range(1, 11)], columns=outputs
    )
    fig, axes = plt.subplots(2, 1, figsize=(19, 11), gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    sns.heatmap(matrix, cmap="vlag", center=0, ax=axes[0], cbar_kws={"label": "Test R² decrease after permutation"})
    axes[0].set_title("ANN permutation importance by input and performance output", weight="bold")
    axes[0].set_xlabel("Performance output")
    axes[0].set_ylabel("Design variable")
    axes[0].tick_params(axis="x", labelrotation=55, labelsize=7)
    for tick in axes[0].get_xticklabels():
        tick.set_horizontalalignment("right")

    responsibility = {
        "DV1-DV3 -> P3/P4": (["DV1", "DV2", "DV3"], [name for name in outputs if name.split("_")[0] in {"P3", "P4"}]),
        "DV4 -> P1/P2": (["DV4"], [name for name in outputs if name.split("_")[0] in {"P1", "P2"}]),
        "DV5-DV10 -> P5-P7": ([f"DV{i}" for i in range(5, 11)], [name for name in outputs if name.split("_")[0] in {"P5", "P6", "P7"}]),
    }
    names, expected_mean, other_mean = [], [], []
    for label, (features, expected_outputs) in responsibility.items():
        expected_mean.append(float(matrix.loc[features, expected_outputs].abs().to_numpy().mean()))
        other_outputs = [name for name in outputs if name not in expected_outputs]
        other_mean.append(float(matrix.loc[features, other_outputs].abs().to_numpy().mean()))
        names.append(label)
    x = np.arange(len(names))
    width = 0.34
    axes[1].bar(x - width / 2, expected_mean, width, label="Listed responsibility outputs", color="#176b87")
    axes[1].bar(x + width / 2, other_mean, width, label="Other outputs", color="#e07a32")
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_ylabel("Mean absolute permutation effect")
    axes[1].legend(fontsize=8)
    _save(fig, "responsibility_map_comparison.png")


def _plot_timing() -> None:
    """Compare training, average optimization-run, and per-sample prediction time."""
    timing = json.loads((RESULTS_DIR / "timing.json").read_text(encoding="utf-8"))
    methods = ["ANN (MLPRegressor)", "Reference ExtraTrees (500)"]
    training = [timing["ann_training_seconds"], timing["reference_extratrees_training_seconds"]]
    latency_ms = [timing["per_sample_latency_seconds_mean"][name] * 1000 for name in methods]
    runs_per_method = int(pd.read_csv(RESULTS_DIR / "benchmark_targets.csv").shape[0] * 5)
    opt = timing["inverse_optimization_seconds_by_method"]
    avg_de_run = [opt.get("ANN + DE", 0) / runs_per_method, opt.get("ExtraTrees + DE", 0) / runs_per_method]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    labels = ["ANN", "ExtraTrees"]
    axes[0].bar(labels, training, color=["#176b87", "#e07a32"])
    axes[0].set_title("Surrogate fit")
    axes[0].set_ylabel("Seconds")
    axes[1].bar(labels, avg_de_run, color=["#176b87", "#e07a32"])
    axes[1].set_title("Mean DE run")
    axes[1].set_ylabel("Seconds / target-seed run")
    axes[2].bar(labels, latency_ms, color=["#176b87", "#e07a32"])
    axes[2].set_title("Warm single-sample prediction")
    axes[2].set_ylabel("Milliseconds / sample")
    fig.suptitle("Compute time for hybrid comparison", fontsize=13, weight="bold")
    _save(fig, "timing_comparison.png")


def main() -> None:
    """Generate every report figure and verify each PNG opens cleanly."""
    started = time.perf_counter()
    ensure_directories()
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    outputs = schema["target_columns"]
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    ann_predictions = pd.read_csv(RESULTS_DIR / "ann_test_predictions.csv")
    predictions = pd.read_csv(RESULTS_DIR / "inverse_predictions.csv")
    solutions = pd.read_csv(RESULTS_DIR / "inverse_solutions.csv")
    _plot_surrogate_parity(outputs, test, ann_predictions)
    _plot_convergence()
    _plot_inverse_parity(outputs, predictions)
    _plot_dv_target_tradeoff(solutions)
    _plot_repeatability(solutions)
    _plot_baselines(solutions)
    _plot_benchmark_sanity()
    _plot_sensitivity()
    _plot_surrogate_ablation()
    _plot_feature_responsibility(outputs)
    _plot_timing()
    files = sorted(FIGURES_DIR.glob("*.png"))
    log(f"Verified {len(files)} PNG figures under {FIGURES_DIR}")
    for path in files:
        with Image.open(path) as image:
            log(f"FIGURE | {path.name} | {image.width}x{image.height}")
    pipeline_times = record_script_time("plot_rendering_seconds", time.perf_counter() - started)
    timing_path = RESULTS_DIR / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["pipeline_script_seconds"] = pipeline_times
    timing["total_pipeline_seconds"] = float(sum(pipeline_times.values()))
    write_json(timing_path, timing)
    report_path = PROJECT_ROOT / "report" / "report.md"
    report = report_path.read_text(encoding="utf-8")
    runtime_line = f"Full pipeline runtime: **{timing['total_pipeline_seconds']:.2f} seconds**."
    if "Full pipeline runtime:" in report:
        report = re.sub(r"Full pipeline runtime: \*\*[0-9.]+ seconds\*\*\.", runtime_line, report)
    else:
        report = report.replace("Full configuration", runtime_line + " Full configuration", 1)
    report_path.write_text(report, encoding="utf-8")
    log(f"TOTAL PIPELINE RUNTIME | {timing['total_pipeline_seconds']:.2f}s")


if __name__ == "__main__":
    main()
