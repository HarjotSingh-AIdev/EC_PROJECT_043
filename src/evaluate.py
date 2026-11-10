"""Shared forward and inverse evaluation helpers plus final artifact assembly."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    ALGORITHM,
    DE_EVALUATIONS,
    DE_GENERATIONS,
    DE_MUTATION,
    DE_POPULATION,
    DE_CROSSOVER,
    DE_SEEDS,
    FAMILY,
    NONUNIQUENESS_RADIUS,
    PROJECT_ROOT,
    RESULTS_DIR,
    ROLL,
    SCHEMA_JSON,
    SEED,
    SUCCESS_NRMSE_THRESHOLD,
    ensure_directories,
    log,
    record_script_time,
    write_json,
)


def per_output_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: list[str],
    train_y: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Compute physical and training-range normalized metrics per output."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        truth, prediction = y_true[:, index], y_pred[:, index]
        rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
        train_range = float(np.max(train_y[:, index]) - np.min(train_y[:, index]))
        rows.append(
            {
                "metric_type": "surrogate_test",
                "model": model_name,
                "output": name,
                "mae": float(mean_absolute_error(truth, prediction)),
                "rmse": rmse,
                "r2": float(r2_score(truth, prediction)) if len(truth) > 1 else float("nan"),
                "nrmse": rmse / train_range if train_range > 0 else float("nan"),
                "n": len(truth),
            }
        )
    return pd.DataFrame(rows)


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_y: np.ndarray) -> dict[str, float]:
    """Return macro MAE/RMSE/R2 and mean per-output normalized RMSE."""
    yt, yp, tr = map(lambda value: np.asarray(value, dtype=float), (y_true, y_pred, train_y))
    per_rmse = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
    ranges = np.ptp(tr, axis=0)
    normalized = np.divide(per_rmse, ranges, out=np.zeros_like(per_rmse), where=ranges > 0)
    return {
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.mean(per_rmse)),
        "r2": float(r2_score(yt, yp, multioutput="uniform_average")),
        "nrmse_macro": float(np.mean(normalized)),
    }


def main() -> None:
    """Assemble per-output metrics, inverse summaries, README, and report."""
    import time

    started = time.perf_counter()
    ensure_directories()
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    per_output_path = RESULTS_DIR / "surrogate_test_metrics.csv"
    if not per_output_path.exists():
        raise FileNotFoundError("Run src/surrogate.py before src/evaluate.py")
    metrics = [pd.read_csv(per_output_path)]
    inverse_predictions_path = RESULTS_DIR / "inverse_predictions.csv"
    solutions_path = RESULTS_DIR / "inverse_solutions.csv"
    if not inverse_predictions_path.exists() or not solutions_path.exists():
        raise FileNotFoundError("Run src/optimize_inverse.py before src/evaluate.py")

    inverse_predictions = pd.read_csv(inverse_predictions_path)
    target_names = schema["target_columns"]
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    train_y = train[target_names].to_numpy(dtype=float)
    for method, subset in inverse_predictions.groupby("method", sort=False):
        rows = []
        for name in target_names:
            part = subset.loc[subset["output"] == name]
            actual = part["target_value"].to_numpy(dtype=float)
            achieved = part["achieved_value"].to_numpy(dtype=float)
            rmse = float(np.sqrt(np.mean((achieved - actual) ** 2)))
            train_range = float(np.ptp(train_y[:, target_names.index(name)]))
            rows.append(
                {
                    "metric_type": "inverse_achievement",
                    "model": method,
                    "output": name,
                    "mae": float(np.mean(np.abs(achieved - actual))),
                    "rmse": rmse,
                    "r2": float("nan"),
                    "nrmse": rmse / train_range if train_range > 0 else float("nan"),
                    "n": int(part[["target_id", "seed"]].drop_duplicates().shape[0]),
                }
            )
        metrics.append(pd.DataFrame(rows))
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_csv(RESULTS_DIR / "metrics_per_output.csv", index=False, float_format="%.10g")

    solutions = pd.read_csv(solutions_path)
    solution_rows: list[dict[str, Any]] = []
    for (method, target_id), subset in solutions.groupby(["method", "target_id"], sort=False):
        solution_rows.append(
            {
                "method": method,
                "target_id": int(target_id),
                "n_runs": len(subset),
                "target_nrmse_mean": float(subset["target_nrmse"].mean()),
                "target_nrmse_std": float(subset["target_nrmse"].std(ddof=0)),
                "target_mae_mean": float(subset["target_mae"].mean()),
                "dv_mae_mean": float(subset["dv_mae_physical"].mean()),
                "dv_rmse_mean": float(subset["dv_rmse_physical"].mean()),
                "dv_mae_standardized_mean": float(subset["dv_mae_standardized"].mean()),
                "dv_rmse_standardized_mean": float(subset["dv_rmse_standardized"].mean()),
                "feasibility_rate": float(subset["feasible"].mean()),
                "objective_mean": float(subset["objective"].mean()),
                "evaluation_count_mean": float(subset["evaluation_count"].mean()),
                "wallclock_seconds_mean": float(subset["wallclock_seconds"].mean()),
                "success_rate": float((subset["target_nrmse"] <= SUCCESS_NRMSE_THRESHOLD).mean()),
                "nearest_neighbor_distance_mean": float(subset["nearest_neighbor_distance"].mean()),
                "nonunique_reference_count_mean": float(subset["nonunique_reference_count"].mean()),
                "dv_reference_spread_mean": float(subset["dv_reference_spread"].mean()),
            }
        )
    by_target = pd.DataFrame(solution_rows)
    by_target.to_csv(RESULTS_DIR / "inverse_metrics_by_target.csv", index=False, float_format="%.10g")
    summary = by_target.groupby("method", sort=False).agg(
        targets=("target_id", "nunique"),
        target_nrmse_mean=("target_nrmse_mean", "mean"),
        target_nrmse_across_targets_std=("target_nrmse_mean", "std"),
        target_mae_mean=("target_mae_mean", "mean"),
        dv_mae_physical_mean=("dv_mae_mean", "mean"),
        dv_rmse_physical_mean=("dv_rmse_mean", "mean"),
        dv_mae_standardized_mean=("dv_mae_standardized_mean", "mean"),
        dv_rmse_standardized_mean=("dv_rmse_standardized_mean", "mean"),
        feasibility_rate=("feasibility_rate", "mean"),
        objective_mean=("objective_mean", "mean"),
        evaluation_count_mean=("evaluation_count_mean", "mean"),
        wallclock_seconds_mean=("wallclock_seconds_mean", "mean"),
        success_rate=("success_rate", "mean"),
        nearest_neighbor_distance_mean=("nearest_neighbor_distance_mean", "mean"),
        nonunique_reference_count_mean=("nonunique_reference_count_mean", "mean"),
        dv_reference_spread_mean=("dv_reference_spread_mean", "mean"),
    ).reset_index()
    run_summary_rows = []
    for method, subset in solutions.groupby("method", sort=False):
        stochastic = subset.loc[subset["seed"] >= 0]
        if stochastic.empty:
            run_stds = {"target_nrmse_seed_run_std": 0.0, "objective_seed_run_std": 0.0, "dv_rmse_seed_run_std": 0.0, "wallclock_seed_run_std": 0.0}
        else:
            run_stds = {
                "target_nrmse_seed_run_std": float(stochastic.groupby("target_id")["target_nrmse"].std(ddof=0).mean()),
                "objective_seed_run_std": float(stochastic.groupby("target_id")["objective"].std(ddof=0).mean()),
                "dv_rmse_seed_run_std": float(stochastic.groupby("target_id")["dv_rmse_physical"].std(ddof=0).mean()),
                "wallclock_seed_run_std": float(stochastic.groupby("target_id")["wallclock_seconds"].std(ddof=0).mean()),
            }
        run_summary_rows.append(
            {
                "method": method,
                "target_nrmse_seed_run_mean": float(subset["target_nrmse"].mean()),
                "objective_seed_run_mean": float(subset["objective"].mean()),
                **run_stds,
            }
        )
    run_summary = pd.DataFrame(run_summary_rows)
    summary = summary.merge(run_summary, on="method", how="left")
    summary.to_csv(RESULTS_DIR / "inverse_metrics_summary.csv", index=False, float_format="%.10g")

    surrogate_rows = pd.read_csv(per_output_path)
    surrogate_summary = surrogate_rows.groupby("model").agg(
        macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"), macro_r2=("r2", "mean"), macro_nrmse=("nrmse", "mean")
    ).reset_index()
    top_hard = (
        surrogate_rows.loc[surrogate_rows["model"] == "ANN (MLPRegressor)"]
        .sort_values("rmse", ascending=False)
        .head(3)[["output", "rmse", "r2", "nrmse"]]
        .to_dict(orient="records")
    )
    diagnostics = json.loads((RESULTS_DIR / "surrogate_diagnostics.json").read_text(encoding="utf-8"))
    sanity = pd.read_csv(RESULTS_DIR / "optimizer_sanity_check.csv")
    sanity_summary = sanity.groupby("function", sort=False).agg(
        median_final_gap=("gap_to_known_optimum", "median"), median_relative_improvement=("relative_improvement_fraction", "median")
    ).reset_index().to_markdown(index=False, floatfmt=".4f")
    hardest_rows = (
        pd.read_csv(RESULTS_DIR / "surrogate_test_metrics.csv")
        .query("model == 'ANN (MLPRegressor)'")
        .nlargest(3, "rmse")
    )
    hardest_text = "\n".join(
        f"- **{row.output}**: test RMSE {row.rmse:.4f}, nRMSE {row.nrmse:.4f}, R2 {row.r2:.4f}; "
        + ("negative R2 on seven cases means the model did worse than the test mean; the sample is too small for a stable interpretation." if row.r2 < 0 else "largest absolute error is partly scale-dependent; use nRMSE to compare across outputs.")
        for row in hardest_rows.itertuples(index=False)
    )
    importance = pd.read_csv(RESULTS_DIR / "feature_importance_per_output.csv").pivot(
        index="feature", columns="output", values="importance_mean"
    )
    responsibility_groups = [
        ("DV1-DV3 -> P3/P4", ["DV1", "DV2", "DV3"], {"P3", "P4"}),
        ("DV4 -> P1/P2", ["DV4"], {"P1", "P2"}),
        ("DV5-DV10 -> P5-P7", [f"DV{i}" for i in range(5, 11)], {"P5", "P6", "P7"}),
    ]
    responsibility_lines = []
    for group_name, features, indicators in responsibility_groups:
        expected_outputs = [name for name in schema["target_columns"] if name.split("_")[0] in indicators]
        other_outputs = [name for name in schema["target_columns"] if name not in expected_outputs]
        expected_effect = float(importance.loc[features, expected_outputs].abs().to_numpy().mean())
        other_effect = float(importance.loc[features, other_outputs].abs().to_numpy().mean())
        comparison = "higher for listed outputs" if expected_effect > other_effect else "not higher for listed outputs"
        responsibility_lines.append(
            f"- {group_name}: mean absolute permutation importance {expected_effect:.4f} for listed outputs versus {other_effect:.4f} for other outputs ({comparison})."
        )
    responsibility_text = "\n".join(responsibility_lines)
    summary_json = {
        "roll": ROLL,
        "assignment": {
            "algorithm": ALGORITHM,
            "family": FAMILY,
            "inverse_input": f"Target performance vector ({schema['n_outputs']} outputs)",
            "inverse_output": "Candidate design DV1-DV10",
            "surrogate_input_count": schema["n_dv"],
            "surrogate_output_count": schema["n_outputs"],
        },
        "schema": {"rows": schema["n_rows"], "indicators": schema["indicator_ids"], "frequencies": schema["frequency_ids"]},
        "protocol": {
            "seed": SEED,
            "split": json.loads((RESULTS_DIR / "split_indices.json").read_text(encoding="utf-8")),
            "de_population": DE_POPULATION,
            "de_generations": DE_GENERATIONS,
            "de_evaluations_per_run": DE_EVALUATIONS,
            "de_seeds": list(DE_SEEDS),
            "de_F": DE_MUTATION,
            "de_CR": DE_CROSSOVER,
            "success_threshold_nrmse": SUCCESS_NRMSE_THRESHOLD,
        },
        "surrogate_test_summary": surrogate_summary.to_dict(orient="records"),
        "surrogate_diagnostics": diagnostics,
        "inverse_summary": summary.to_dict(orient="records"),
        "three_hardest_outputs_by_ann_rmse": top_hard,
    }
    write_json(RESULTS_DIR / "metrics_summary.json", summary_json)

    pipeline_times = record_script_time("evaluation_seconds", time.perf_counter() - started)
    timing_path = RESULTS_DIR / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["pipeline_script_seconds"] = pipeline_times
    timing["total_pipeline_seconds_so_far"] = float(sum(pipeline_times.values()))
    write_json(timing_path, timing)

    readme = _build_readme(schema, summary, surrogate_summary)
    (PROJECT_ROOT / "README.md").write_text(readme, encoding="utf-8")
    report = _build_report(schema, summary, surrogate_summary, top_hard, sanity_summary, hardest_text, responsibility_text)
    (PROJECT_ROOT / "report" / "report.md").write_text(report, encoding="utf-8")

    log("Evaluation artifacts complete | metrics_per_output.csv, inverse metrics, summary JSON, README, report")
    for row in summary.itertuples(index=False):
        log(f"INVERSE SUMMARY | {row.method} | target_nRMSE={row.target_nrmse_mean:.5f} | DV RMSE={row.dv_rmse_physical_mean:.5f}")


def _build_readme(schema: dict[str, Any], inverse: pd.DataFrame, surrogate: pd.DataFrame) -> str:
    """Build a reproducibility-first README populated with actual results."""
    headline_columns = [
        "method", "target_nrmse_mean", "target_nrmse_seed_run_std", "dv_rmse_physical_mean",
        "dv_rmse_standardized_mean", "feasibility_rate", "success_rate", "evaluation_count_mean",
    ]
    inverse_table = inverse[[column for column in headline_columns if column in inverse.columns]].to_markdown(index=False, floatfmt=".4f")
    surrogate_table = surrogate.to_markdown(index=False, floatfmt=".4f")
    test_size = len(pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv"))
    diagnostics = json.loads((RESULTS_DIR / "surrogate_diagnostics.json").read_text(encoding="utf-8"))
    return fr"""# EC-101 Project — {ROLL}: ANN + Differential Evolution

## Problem

This project studies hybrid inverse antenna design for roll {ROLL}. The main inverse input is a 35-value target performance vector and the requested output is a candidate design DV1-DV10. A feed-forward artificial neural network (implemented as scikit-learn's `MLPRegressor`) supplies the forward mapping from design variables to performance; Differential Evolution searches the training-set design bounds for candidates matching each held-out target. The optimizer's result is a surrogate-validated candidate; no CST or experimental evaluation is available here.

## Dataset

The source is `data/raw/DataSet for Project-EC 101.pdf` (50 records). Extraction expands newline-paired cells from the PDF table into individual records, then keeps DV1-DV10 and P1_F1-P7_F5. The table also contains Design_ID and frequency-coordinate columns F1-F5; those are excluded because the allotment defines the model inputs as DV1-DV10 and outputs as the P_F performance vector. The detected dataset has {schema['n_dv']} inputs, {schema['n_outputs']} outputs, {len(schema['indicator_ids'])} indicators, and {len(schema['frequency_ids'])} frequency positions. This resolves the handout's P1-P7 versus P8-P10 inconsistency in favor of the actual table headers.

## Headline results

### Forward surrogate test metrics

{surrogate_table}

### Inverse benchmark comparison

{inverse_table}

The ANN has negative test R2 on {diagnostics['ann_negative_r2_count']} outputs: {', '.join(diagnostics['ann_negative_r2_outputs']) if diagnostics['ann_negative_r2_outputs'] else 'none'}. See `results/surrogate_diagnostics.json` for the best and worst five outputs by RMSE.

The benchmark uses {test_size} test rows from the saved 70/15/15 split after the train/validation design has been established. The test portion is small, so the result is a pilot estimate and should not be read as a stable generalization claim.

## Preprocessing and protocol

- Exact duplicates are removed, then a single NumPy permutation with seed 42 creates a 70/15/15 split (largest-remainder allocation). Original source row indices are recorded in `results/split_indices.json`.
- Rows with more than 2% missing cells are dropped. Remaining gaps are filled with medians computed on train only. The processed split CSVs retain original units.
- Separate `StandardScaler` objects for X and y are fitted on train only. Hyperparameter selection uses validation only; the chosen ANN is refit on train+validation. Test metrics are computed once by the final pipeline.
- The named hybrid is ANN + Differential Evolution. The reference ExtraTrees surrogate is fixed at 500 trees and fitted on train only. DE uses DE/rand/1/bin, population {DE_POPULATION}, F=0.7, CR=0.9, {DE_GENERATIONS} generations, and {DE_EVALUATIONS} objective evaluations per run across seeds 42-46. Random search uses the same number of candidates. Retrieval returns the nearest training performance vector.
- The inverse target is the known test performance vector. The objective is the mean squared train-standard-deviation-scaled difference between ANN predictions and the target, plus a 1000x normalized squared bound-violation penalty. Feasibility is separately measured against training min/max bounds.
- Success is target normalized RMSE <= 0.10, where each output error is divided by its train range. Nearest-neighbor distances use train-standardized performance space.

## Reproduce from a clean Python 3.12 environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/extract_data.py
python src/preprocess.py
python src/surrogate.py
python src/optimize_inverse.py
python src/evaluate.py
python src/plots.py
pytest tests/
```

The scripts resolve paths from their own files, so they can also be run from the project directory without editing source. The raw source PDFs are kept unchanged under `data/raw/`.

## Project files

- `src/extract_data.py`: PDF parsing, paired-cell expansion, schema validation.
- `src/preprocess.py`: deterministic split, train-only imputation/scalers, dataset card.
- `src/surrogate.py`: MLP validation study, fixed ExtraTrees reference, test metrics and timing.
- `src/optimize_inverse.py`: DE, random search, retrieval, sanity functions, and inverse metrics.
- `src/evaluate.py`: reusable metric functions and report/README assembly.
- `src/plots.py`: all report figures.
- `results/`: tables, traces, split indices, timings, and machine-readable summary.
- `report/report.md`: full methods and results report.

## Limitations

Only {schema['n_rows']} data records are available; sample size is the primary limitation. The {schema['n_outputs']}-dimensional target has far more dimensions than records, so both the ANN fit and the inverse optimum have substantial uncertainty. Differential Evolution is stochastic, and five seeds on a small benchmark give only a rough repeatability estimate. The inverse map is many-to-one, the optimizer relies on the ANN as its evaluator, train-derived bounds do not guarantee physical feasibility, and no electromagnetic solver or physical prototype is available to confirm the candidates. The handout's variable responsibility map is descriptive only; with these data it cannot establish causality.
"""


def _build_report(
    schema: dict[str, Any],
    inverse: pd.DataFrame,
    surrogate: pd.DataFrame,
    hard_outputs: list[dict[str, Any]],
    sanity_summary: str,
    hardest_text: str,
    responsibility_text: str,
) -> str:
    """Build the academic report from saved result tables."""
    surrogate_table = surrogate.to_markdown(index=False, floatfmt=".5f")
    headline_columns = [
        "method", "target_nrmse_mean", "target_nrmse_seed_run_std", "dv_rmse_physical_mean",
        "dv_rmse_standardized_mean", "feasibility_rate", "success_rate", "evaluation_count_mean",
    ]
    inverse_table = inverse[[column for column in headline_columns if column in inverse.columns]].to_markdown(index=False, floatfmt=".5f")
    hard_table = pd.DataFrame(hard_outputs).to_markdown(index=False, floatfmt=".5f")
    reference_row = surrogate.loc[surrogate["model"] == "Reference ExtraTrees (500)"].iloc[0]
    ann_row = surrogate.loc[surrogate["model"] == "ANN (MLPRegressor)"].iloc[0]
    linear_row = surrogate.loc[surrogate["model"] == "LinearRegression"].iloc[0]
    inverse_rows = inverse.set_index("method")
    ann_de_row = inverse_rows.loc["ANN + DE"]
    et_de_row = inverse_rows.loc["ExtraTrees + DE"]
    ann_random_row = inverse_rows.loc["Random search (ANN)"]
    retrieval_row = inverse_rows.loc["Nearest-neighbor retrieval"]
    source_columns = "P1-P7 measured at F1-F5"
    return f"""# Hybrid ANN + Differential Evolution for EC-101 Antenna Design

**Student roll:** {ROLL}  
**Assignment:** {ALGORITHM}  
**Family:** {FAMILY}  
**Inverse input:** target performance vector ({schema['n_outputs']} values: {source_columns})  
**Inverse output:** candidate design DV1-DV10  
**Surrogate mapping:** DV1-DV10 -> {schema['n_outputs']} performance outputs  
**Dataset:** {schema['n_rows']} source records

## 1. Introduction

Forward regression estimates antenna performance from design variables, while inverse design searches for a design that may achieve a requested performance vector. This project implements the hybrid assigned to roll {ROLL}: an artificial neural network (ANN) is used as a fixed forward surrogate and Differential Evolution (DE) searches the bounded design space. The PDF allotment names “ANN + Differential Evolution,” consistent with the fallback map. No discrepancy was found. The project evaluates both the surrogate and the end-to-end inverse pipeline, while treating the computed designs as candidates rather than physically validated designs.

## 2. Dataset and preprocessing

The sole numeric source is `data/raw/DataSet for Project-EC 101.pdf`. The table's 38 extracted table rows contain 50 actual designs because 13 rows combine two newline-separated records in every cell; the parser aligns and expands those pairs. It found {schema['n_rows']} records, {schema['n_dv']} inputs, and {schema['n_outputs']} outputs. The real headers contain indicators {schema['indicator_ids']} at positions {schema['frequency_ids']}, yielding a complete {len(schema['indicator_ids'])} x {len(schema['frequency_ids'])} grid. The assignment prose mentions both P1-P7 and P5-P10; the PDF table has P1-P7 only, so the experiment follows the observed 35-column schema. Source Design_ID and F1-F5 coordinate columns are not model inputs under the allotment and are excluded.

Exact duplicate records are removed (count in `results/preprocessing_summary.json`). Rows exceeding 2% missingness are dropped; remaining gaps use medians learned only from train. One seeded permutation allocates 70/15/15 with largest-remainder rounding. Given 50 rows, the split is 35/8/7. The exact original row indices are stored in `results/split_indices.json`. The processed CSVs remain in physical units. X and y each use a StandardScaler fitted on train only, stored in `models/`. No outliers are clipped; 1.5-IQR counts and constant-column checks are recorded in the preprocessing summary. The complete output ranges are in `report/dataset_card.md`.

## 3. Algorithm

The ANN uses `MLPRegressor` with ReLU hidden layers, Adam optimization, and internal early stopping. A small validation grid compares three hidden-layer/regularization settings; model selection is based on validation macro nRMSE, so the small test set does not guide tuning. The selected configuration is refit on train plus validation, using the train-fitted scalers.

DE uses the DE/rand/1/bin strategy. For each population member, three distinct members are sampled, and a scaled difference vector is added to a base member. Binomial crossover mixes the mutant and target vectors, with at least one mutant coordinate. A greedy one-to-one selection retains a trial when it lowers the objective. This implementation follows Storn and Price's original Differential Evolution formulation ([1997 paper](https://doi.org/10.1023/A:1008202821328)). Candidates are clipped to train-derived variable bounds after mutation; the objective also contains a normalized squared bound-violation penalty (lambda = 1000), so feasibility is recorded explicitly.

## 4. Methodology

The fixed reference measuring instrument is `ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, random_state=42)`, fitted on train only. It achieved macro test R2 {reference_row['macro_r2']:.4f} and macro nRMSE {reference_row['macro_nrmse']:.4f}; these scores are the empirical accuracy ceiling for the reference-surrogate inverse ablation, since DE cannot overcome its evaluator's error. The named ANN is tuned on validation and refit on train+validation. The test subset provides K={len(pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'test.csv'))} fixed targets with their known DVs. The inverse map is non-unique; a target match is therefore not treated as proof of recovering the original design.

For target vector T and surrogate P-hat(d), the cost is

`J(d) = mean_i[((P-hat_i(d) - T_i) / s_i)^2] + 1000 * sum_j[normalized bound violation_j^2]`,

where s_i is the train standard deviation for output i. DE uses population size {DE_POPULATION}, {DE_GENERATIONS} generations, F=0.7, CR=0.9 and {DE_EVALUATIONS} evaluations per run (initial {DE_POPULATION} plus {DE_GENERATIONS} x {DE_POPULATION} trials). Five fixed seeds (42-46) measure stochastic variability. Uniform random search uses the same candidate-evaluation budget. Nearest-neighbor retrieval chooses a train row with the closest train-standardized target performance. The success threshold is normalized target RMSE <= 0.10; this threshold is declared before comparison. All methods report DV MAE/RMSE in physical and train-standardized units, target-vs-achieved error per output, feasibility, objective, evaluation count, runtime, success, and nearest train-reference distance.

DE's code is checked on 10-dimensional Sphere, Rosenbrock, and Rastrigin functions with known zero minima, using the same evaluation budget. The traces and final gaps are saved under `results/`.

Median final gaps and median relative improvements from each run's initial population are:

{sanity_summary}

## 5. Results

### Surrogate accuracy on the reserved test rows

{surrogate_table}

The ANN had lower macro MAE/RMSE than LinearRegression ({ann_row['macro_mae']:.4f}/{ann_row['macro_rmse']:.4f} versus {linear_row['macro_mae']:.4f}/{linear_row['macro_rmse']:.4f}), while LinearRegression had slightly higher macro R2 ({linear_row['macro_r2']:.4f} versus {ann_row['macro_r2']:.4f}). Model ranking therefore depends on metric on this small test set.

The three ANN outputs with largest test RMSE are:

{hard_table}

The three hardest outputs by physical RMSE are:

{hardest_text}

The held-out test R2 is negative for the outputs listed in `results/surrogate_diagnostics.json`; that file also ranks the best and worst five ANN outputs by RMSE. The test set contains only seven designs, so these rankings are exploratory.

Table values are generated from `results/surrogate_test_metrics.csv` and `results/metrics_summary.json`. Per-output R2 includes negative values rather than hiding them. Test R2 is based on only {len(pd.read_csv(PROJECT_ROOT / 'data' / 'processed' / 'test.csv'))} rows and is particularly variable for outputs with low test variance.

### Inverse methods

{inverse_table}

ANN+DE's mean target nRMSE was {ann_de_row['target_nrmse_mean']:.4f}, versus {ann_random_row['target_nrmse_mean']:.4f} for equal-budget random search under the same ANN. The fixed ExtraTrees + DE result was {et_de_row['target_nrmse_mean']:.4f}. Retrieval had the highest mean target nRMSE ({retrieval_row['target_nrmse_mean']:.4f}) but the lowest physical DV RMSE ({retrieval_row['dv_rmse_physical_mean']:.2f}); this difference is consistent with the non-unique inverse map and reflects distinct evaluation goals.

`target_nrmse_mean` is the average per-output RMSE divided by that output's train range. Physical DV errors are evaluated against the known true test DVs. Standardized DV errors divide each design-variable error by its train standard deviation. These metrics answer different questions: matching requested performance is the inverse objective, while matching the original DV is an imperfect measure under a many-to-one mapping. The method-by-target breakdown and per-output target errors are in `results/inverse_metrics_by_target.csv` and `results/metrics_per_output.csv`.

![ANN surrogate parity by output](../results/figures/surrogate_parity_grid.png)

Figure 1 compares the ANN predictions against measured test outputs. The inverse convergence plot in Figure 2 shows median best-so-far objective across fixed targets and seeds; reference ET and ANN traces correspond to their own surrogates.

![DE convergence by evaluation count](../results/figures/convergence_curves.png)

Figure 3 plots target-versus-achieved outputs for ANN+DE and exposes per-output fit quality. Figure 4 compares design-variable error with target mismatch. Figure 5 shows variation across seeds; the nearest-neighbor baseline is deterministic.

![Achieved versus target performance](../results/figures/achieved_vs_target.png)

![DV error versus target performance error](../results/figures/dv_vs_target_error.png)

![Repeatability across seeds](../results/figures/repeatability.png)

Figure 6 compares the ANN+DE hybrid against the fixed ExtraTrees ablation, equal-budget random search, and train-row retrieval. Figure 7 is the optimizer sanity check. Figure 8 varies population size under approximately fixed objective-evaluation budget on the first benchmark target. Figure 9 places surrogate test error alongside inverse residual error as a diagnostic decomposition; the components are measured on different evaluation questions and are not an additive causal attribution.

![Inverse baseline comparison](../results/figures/baseline_comparison.png)

![DE benchmark-function sanity check](../results/figures/benchmark_convergence.png)

![Population sensitivity](../results/figures/sensitivity.png)

![Surrogate and optimizer error decomposition](../results/figures/surrogate_ablation.png)

The ANN permutation importance and the handout's grouped responsibility map are compared in Figure 10. Importance is descriptive and correlated design variables can share or exchange it, so the comparison is exploratory.

![Permutation effects and the handout responsibility map](../results/figures/responsibility_map_comparison.png)

Figure 11 reports measured training, prediction, and mean optimization-run times; exact values are in `results/timing.json`.

![Training and inference timing](../results/figures/timing_comparison.png)

## 6. Discussion

The responsibility map in the assignment is a domain prior, not a causal result. The ANN's input effects are studied with permutation importance on the held-out test rows in `results/feature_importance_per_output.csv`; importance is descriptive and correlated design variables can share or exchange it. The report therefore does not hard-code physical mappings into the model. Figure 10 compares listed feature/output groups with other outputs. Because the data are small, the ranking should be read as exploratory.

{responsibility_text}

The benchmark is intentionally limited to the seven held-out targets. Nearest-reference distances indicate how far each requested vector lies from observed training performance in standardized space. Non-uniqueness counts and DV spreads among train designs within RMS radius {NONUNIQUENESS_RADIUS:.2f} in standardized performance space quantify how several recorded designs may explain similar performance. Candidates remain within the rectangular train min/max bounds, but this does not impose antenna geometry or manufacturability constraints. The optimizer is also optimizing the ANN, not the true electromagnetic response; surrogate error can produce optimistic predicted matches.

## 7. Limitations

1. **Sample size:** 50 rows are very few for 10 inputs and 35 outputs. The test set contains only seven examples, making R2, parameter rankings, and average inverse scores highly uncertain.
2. The ANN and all inverse optimization rely on a learned forward mapping with limited support in the design space; no CST or experimental confirmation exists.
3. A target may correspond to multiple valid DV combinations, so DV error against one known design is not a unique correctness criterion.
4. Train min/max bounds allow combinations that may not be physically feasible, and the table provides no engineering constraints beyond the observed ranges.
5. Five optimizer seeds give a compact repeatability study rather than a broad stochastic characterization.
6. The extracted file includes only the seven indicators shown in the dataset; the P8-P10 prose is unresolved by the source data and is excluded.

## 8. Conclusion and future work

The project completes the assigned ANN + Differential Evolution pipeline with a PDF-derived, schema-checked 50-row dataset, a deterministic train-only preprocessing protocol, a tuned ANN, fixed-budget inverse comparisons, optimizer sanity checks, and auditable results. The benchmark supports a cautious pilot comparison, not a strong performance claim. A larger set of independently simulated antenna designs, explicit geometry and manufacturability constraints, repeated external solver evaluations, and nested or repeated splits would provide the next evidence needed for dependable inverse design.

## Reproducibility and artifacts

Run in this order: `python src/extract_data.py`, `python src/preprocess.py`, `python src/surrogate.py`, `python src/optimize_inverse.py`, `python src/evaluate.py`, `python src/plots.py`, then `pytest tests/`. Full configuration, elapsed times, software versions, split identities, and result tables are saved in `results/`; installed versions are pinned in `requirements.txt`. No raw PDF has been edited.
"""


if __name__ == "__main__":
    main()
