"""Run the named ANN + Differential Evolution inverse-design experiment."""

from __future__ import annotations

import json
import time
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from config import (
    BOUND_PENALTY,
    DE_CROSSOVER,
    DE_EVALUATIONS,
    DE_GENERATIONS,
    DE_MUTATION,
    DE_POPULATION,
    DE_SEEDS,
    MODELS_DIR,
    NONUNIQUENESS_RADIUS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SCHEMA_JSON,
    SUCCESS_NRMSE_THRESHOLD,
    ensure_directories,
    log,
    record_script_time,
    write_json,
)

Array = np.ndarray


def _evaluate_many(objective: Callable[[Array], float], candidates: Array) -> Array:
    """Use a vectorized objective when available, otherwise score row by row."""
    candidates = np.asarray(candidates, dtype=float)
    batch_function = getattr(objective, "evaluate_batch", None)
    if batch_function is not None:
        return np.asarray(batch_function(candidates), dtype=float)
    return np.asarray([objective(candidate) for candidate in candidates], dtype=float)


def differential_evolution(
    objective: Callable[[Array], float],
    bounds: Array,
    seed: int,
    budget: int,
    population_size: int = DE_POPULATION,
    mutation: float = DE_MUTATION,
    crossover: float = DE_CROSSOVER,
) -> tuple[Array, float, list[tuple[int, float]]]:
    """Minimize with DE/rand/1/bin and return best vector plus evaluation trace."""
    if population_size < 4:
        raise ValueError("DE/rand/1 needs at least four population members")
    rng = np.random.default_rng(seed)
    lower, upper = bounds[:, 0], bounds[:, 1]
    population = rng.uniform(lower, upper, size=(population_size, len(bounds)))
    fitness = _evaluate_many(objective, population)
    evaluations = population_size
    best_index = int(np.argmin(fitness))
    best, best_cost = population[best_index].copy(), float(fitness[best_index])
    trace = [(evaluations, best_cost)]
    while evaluations < budget:
        snapshot = population.copy()
        remaining = min(population_size, budget - evaluations)
        trials = []
        indices = list(range(remaining))
        for index in indices:
            choices = np.delete(np.arange(population_size), index)
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            mutant = snapshot[r1] + mutation * (snapshot[r2] - snapshot[r3])
            mutant = np.clip(mutant, lower, upper)
            cross_mask = rng.random(len(bounds)) < crossover
            cross_mask[int(rng.integers(len(bounds)))] = True
            trials.append(np.where(cross_mask, mutant, snapshot[index]))
        trial_costs = _evaluate_many(objective, np.asarray(trials))
        for index, trial, trial_cost in zip(indices, trials, trial_costs):
            trial_cost = float(trial_cost)
            if trial_cost <= fitness[index]:
                population[index] = trial
                fitness[index] = trial_cost
                if trial_cost <= best_cost:
                    best = trial.copy()
                    best_cost = trial_cost
            evaluations += 1
            trace.append((evaluations, best_cost))
    return best, best_cost, trace


def _objective_function(model: object, target: Array, x_scaler: object, y_scaler: object, y_std: Array, bounds: Array) -> Callable[[Array], float]:
    """Create the train-standard-deviation objective with a bound penalty."""
    lower, upper = bounds[:, 0], bounds[:, 1]

    def objective(candidate: Array) -> float:
        return float(evaluate_batch(np.asarray(candidate, dtype=float).reshape(1, -1))[0])

    def evaluate_batch(candidates: Array) -> Array:
        """Score a matrix of candidates with one batched surrogate call."""
        candidates = np.atleast_2d(np.asarray(candidates, dtype=float))
        scaled_x = x_scaler.transform(candidates)
        predicted = y_scaler.inverse_transform(model.predict(scaled_x))
        normalized_error = (predicted - target.reshape(1, -1)) / y_std.reshape(1, -1)
        below = np.minimum(candidates - lower, 0.0) / np.maximum(upper - lower, 1e-12)
        above = np.maximum(candidates - upper, 0.0) / np.maximum(upper - lower, 1e-12)
        penalty = BOUND_PENALTY * np.sum(below**2 + above**2, axis=1)
        return np.mean(normalized_error**2, axis=1) + penalty

    setattr(objective, "evaluate_batch", evaluate_batch)

    return objective


def _predict_physical(model: object, candidate: Array, x_scaler: object, y_scaler: object) -> Array:
    """Predict one candidate in original performance units."""
    return y_scaler.inverse_transform(model.predict(x_scaler.transform(candidate.reshape(1, -1))))[0]


def _target_nrmse(prediction: Array, target: Array, train_y: Array) -> float:
    """Compute RMSE normalized by each output's train range."""
    ranges = np.ptp(train_y, axis=0)
    normalized = np.divide(prediction - target, ranges, out=np.zeros_like(prediction), where=ranges > 0)
    return float(np.sqrt(np.mean(normalized**2)))


def _run_random_search(
    objective: Callable[[Array], float], bounds: Array, seed: int, budget: int
) -> tuple[Array, float, list[tuple[int, float]]]:
    """Return the best of exactly budget uniformly sampled candidates and its trace."""
    rng = np.random.default_rng(seed)
    lower, upper = bounds[:, 0], bounds[:, 1]
    candidates = rng.uniform(lower, upper, size=(budget, len(bounds)))
    costs = _evaluate_many(objective, candidates)
    best_index = int(np.argmin(costs))
    best = candidates[best_index].copy()
    best_so_far = np.minimum.accumulate(costs)
    trace = [(index + 1, float(value)) for index, value in enumerate(best_so_far)]
    return best, float(costs[best_index]), trace


def _benchmark_functions(dimension: int) -> dict[str, tuple[Callable[[Array], float], Array]]:
    """Return standard benchmark objectives and symmetric bounds."""
    def sphere(x: Array) -> float:
        return float(np.sum(x**2))

    def rosenbrock(x: Array) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def rastrigin(x: Array) -> float:
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    return {
        "Sphere": (sphere, np.tile(np.array([[-5.12, 5.12]], dtype=float), (dimension, 1))),
        "Rosenbrock": (rosenbrock, np.tile(np.array([[-2.048, 2.048]], dtype=float), (dimension, 1))),
        "Rastrigin": (rastrigin, np.tile(np.array([[-5.12, 5.12]], dtype=float), (dimension, 1))),
    }


def _dv_statistics(candidate: Array, true_dv: Array, dv_std: Array) -> tuple[float, float, float, float]:
    """Return physical and standardized DV MAE/RMSE."""
    delta = np.asarray(candidate) - np.asarray(true_dv)
    standardized = delta / np.where(dv_std > 0, dv_std, 1.0)
    return (
        float(np.mean(np.abs(delta))),
        float(np.sqrt(np.mean(delta**2))),
        float(np.mean(np.abs(standardized))),
        float(np.sqrt(np.mean(standardized**2))),
    )


def main() -> None:
    """Create inverse targets, optimize them, and save all audit artifacts."""
    ensure_directories()
    started = time.perf_counter()
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    input_names, output_names = schema["input_columns"], schema["target_columns"]
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    split = json.loads((RESULTS_DIR / "split_indices.json").read_text(encoding="utf-8"))
    x_scaler = joblib.load(MODELS_DIR / "x_scaler.joblib")
    y_scaler = joblib.load(MODELS_DIR / "y_scaler.joblib")
    ann = joblib.load(MODELS_DIR / "ann_surrogate.joblib")
    reference = joblib.load(MODELS_DIR / "reference_extratrees.joblib")
    train_x, train_y = train[input_names].to_numpy(float), train[output_names].to_numpy(float)
    benchmark_x = test[input_names].to_numpy(float)
    benchmark_y = test[output_names].to_numpy(float)
    bounds = np.column_stack([train_x.min(axis=0), train_x.max(axis=0)])
    y_std = np.asarray(y_scaler.scale_, dtype=float)
    dv_std = np.asarray(x_scaler.scale_, dtype=float)
    train_y_scaled = y_scaler.transform(train_y)
    write_json(
        MODELS_DIR / "dv_bounds.json",
        {name: {"min": float(bounds[i, 0]), "max": float(bounds[i, 1])} for i, name in enumerate(input_names)},
    )
    benchmark_frame = pd.DataFrame(benchmark_y, columns=output_names)
    benchmark_frame.insert(0, "target_id", np.arange(len(test), dtype=int))
    benchmark_frame.insert(1, "source_row_index", [int(value) for value in split["test"]])
    for index, name in enumerate(input_names):
        benchmark_frame[f"true_{name}"] = benchmark_x[:, index]
    benchmark_frame.to_csv(RESULTS_DIR / "benchmark_targets.csv", index=False, float_format="%.10g")

    solution_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    model_specs = [
        ("ANN + DE", ann),
        ("ExtraTrees + DE", reference),
        ("Random search (ANN)", ann),
        ("Random search (ExtraTrees)", reference),
    ]
    model_eval_seconds: dict[str, float] = {}
    for target_id, (target, true_dv) in enumerate(zip(benchmark_y, benchmark_x)):
        nearest_distances = np.sqrt(np.mean((train_y_scaled - y_scaler.transform(target.reshape(1, -1))[0]) ** 2, axis=1))
        nearest_index = int(np.argmin(nearest_distances))
        nearest_distance = float(nearest_distances[nearest_index])
        near_mask = nearest_distances <= NONUNIQUENESS_RADIUS
        near_count = int(near_mask.sum())
        dv_spread = float(np.mean(np.std(train_x[near_mask], axis=0))) if near_count else float("nan")
        for method, model in model_specs:
            objective = _objective_function(model, target, x_scaler, y_scaler, y_std, bounds)
            for run_seed in DE_SEEDS:
                run_start = time.perf_counter()
                if method.endswith("+ DE"):
                    candidate, best_cost, trace = differential_evolution(
                        objective, bounds, run_seed, DE_EVALUATIONS, DE_POPULATION, DE_MUTATION, DE_CROSSOVER
                    )
                    for evaluations, cost in trace:
                        trace_rows.append(
                            {"method": method, "target_id": target_id, "seed": run_seed, "evaluation_count": evaluations, "best_objective": cost}
                        )
                    used_evaluations = DE_EVALUATIONS
                else:
                    candidate, best_cost, trace = _run_random_search(objective, bounds, run_seed, DE_EVALUATIONS)
                    for evaluations, cost in trace:
                        trace_rows.append(
                            {"method": method, "target_id": target_id, "seed": run_seed, "evaluation_count": evaluations, "best_objective": cost}
                        )
                    used_evaluations = DE_EVALUATIONS
                run_seconds = time.perf_counter() - run_start
                model_eval_seconds[method] = model_eval_seconds.get(method, 0.0) + run_seconds
                achieved = _predict_physical(model, candidate, x_scaler, y_scaler)
                target_nrmse = _target_nrmse(achieved, target, train_y)
                dv_mae, dv_rmse, dv_mae_std, dv_rmse_std = _dv_statistics(candidate, true_dv, dv_std)
                feasible = bool(np.all(candidate >= bounds[:, 0] - 1e-10) and np.all(candidate <= bounds[:, 1] + 1e-10))
                solution_rows.append(
                    {
                        "method": method,
                        "target_id": target_id,
                        "seed": run_seed,
                        "objective": float(best_cost),
                        "target_mae": float(np.mean(np.abs(achieved - target))),
                        "target_rmse": float(np.sqrt(np.mean((achieved - target) ** 2))),
                        "target_nrmse": target_nrmse,
                        "dv_mae_physical": dv_mae,
                        "dv_rmse_physical": dv_rmse,
                        "dv_mae_standardized": dv_mae_std,
                        "dv_rmse_standardized": dv_rmse_std,
                        "feasible": feasible,
                        "evaluation_count": used_evaluations,
                        "wallclock_seconds": run_seconds,
                        "success": target_nrmse <= SUCCESS_NRMSE_THRESHOLD,
                        "nearest_neighbor_distance": nearest_distance,
                        "nonunique_reference_count": near_count,
                        "dv_reference_spread": dv_spread,
                        **{name: float(candidate[index]) for index, name in enumerate(input_names)},
                    }
                )
                for output_index, output_name in enumerate(output_names):
                    prediction_rows.append(
                        {
                            "method": method,
                            "target_id": target_id,
                            "seed": run_seed,
                            "output": output_name,
                            "target_value": float(target[output_index]),
                            "achieved_value": float(achieved[output_index]),
                        }
                    )

        # Data retrieval is a deterministic, zero-optimizer baseline: return the nearest observed training pair.
        candidate = train_x[nearest_index].copy()
        achieved = train_y[nearest_index].copy()
        objective = _objective_function(ann, target, x_scaler, y_scaler, y_std, bounds)
        retrieval_cost = float(objective(candidate))
        target_nrmse = _target_nrmse(achieved, target, train_y)
        dv_mae, dv_rmse, dv_mae_std, dv_rmse_std = _dv_statistics(candidate, true_dv, dv_std)
        solution_rows.append(
            {
                "method": "Nearest-neighbor retrieval",
                "target_id": target_id,
                "seed": -1,
                "objective": retrieval_cost,
                "target_mae": float(np.mean(np.abs(achieved - target))),
                "target_rmse": float(np.sqrt(np.mean((achieved - target) ** 2))),
                "target_nrmse": target_nrmse,
                "dv_mae_physical": dv_mae,
                "dv_rmse_physical": dv_rmse,
                "dv_mae_standardized": dv_mae_std,
                "dv_rmse_standardized": dv_rmse_std,
                "feasible": True,
                "evaluation_count": 0,
                "wallclock_seconds": 0.0,
                "success": target_nrmse <= SUCCESS_NRMSE_THRESHOLD,
                "nearest_neighbor_distance": nearest_distance,
                "nonunique_reference_count": near_count,
                "dv_reference_spread": dv_spread,
                **{name: float(candidate[index]) for index, name in enumerate(input_names)},
            }
        )
        for output_index, output_name in enumerate(output_names):
            prediction_rows.append(
                {
                    "method": "Nearest-neighbor retrieval",
                    "target_id": target_id,
                    "seed": -1,
                    "output": output_name,
                    "target_value": float(target[output_index]),
                    "achieved_value": float(achieved[output_index]),
                }
            )
        log(f"Inverse target {target_id + 1}/{len(test)} complete | nearest train distance={nearest_distance:.4f}")

    solutions = pd.DataFrame(solution_rows)
    predictions = pd.DataFrame(prediction_rows)
    solutions.to_csv(RESULTS_DIR / "inverse_solutions.csv", index=False, float_format="%.10g")
    predictions.to_csv(RESULTS_DIR / "inverse_predictions.csv", index=False, float_format="%.10g")
    traces = pd.DataFrame(trace_rows)
    traces.to_csv(RESULTS_DIR / "convergence_traces.csv", index=False, float_format="%.10g")

    sanity_rows: list[dict[str, object]] = []
    sanity_trace: list[dict[str, object]] = []
    for function_name, (function, function_bounds) in _benchmark_functions(10).items():
        gaps = []
        for run_seed in DE_SEEDS:
            best, cost, trace = differential_evolution(function, function_bounds, run_seed, DE_EVALUATIONS)
            gaps.append(cost)
            sanity_rows.append(
                {
                    "function": function_name,
                    "seed": run_seed,
                    "initial_best_objective": trace[0][1],
                    "best_objective": cost,
                    "gap_to_known_optimum": cost,
                    "relative_improvement_fraction": 1 - cost / trace[0][1] if trace[0][1] else 0.0,
                    "evaluations": DE_EVALUATIONS,
                }
            )
            for evaluations, objective_value in trace:
                sanity_trace.append(
                    {"function": function_name, "seed": run_seed, "evaluation_count": evaluations, "best_objective": objective_value}
                )
        log(f"DE sanity | {function_name} | median optimum gap={float(np.median(gaps)):.6g}")
    pd.DataFrame(sanity_rows).to_csv(RESULTS_DIR / "optimizer_sanity_check.csv", index=False, float_format="%.10g")
    pd.DataFrame(sanity_trace).to_csv(RESULTS_DIR / "sanity_convergence.csv", index=False, float_format="%.10g")

    # Sensitivity: first test target, fixed total budget, population size varies; three independent seeds.
    sensitivity_rows: list[dict[str, object]] = []
    target = benchmark_y[0]
    sensitivity_objective = _objective_function(ann, target, x_scaler, y_scaler, y_std, bounds)
    for population_size in (25, 50, 75):
        for run_seed in DE_SEEDS[:3]:
            _, cost, trace = differential_evolution(
                sensitivity_objective, bounds, run_seed, DE_EVALUATIONS, population_size, DE_MUTATION, DE_CROSSOVER
            )
            sensitivity_rows.append(
                {"target_id": 0, "population_size": population_size, "seed": run_seed, "budget": DE_EVALUATIONS, "final_objective": cost, "evaluations": trace[-1][0]}
            )
    pd.DataFrame(sensitivity_rows).to_csv(RESULTS_DIR / "sensitivity.csv", index=False, float_format="%.10g")

    surrogate_metrics = pd.read_csv(RESULTS_DIR / "surrogate_test_metrics.csv")
    inverse_means = solutions.groupby("method")["target_nrmse"].mean()
    ablation_rows = []
    for model_name, method_name in (
        ("ANN (MLPRegressor)", "ANN + DE"),
        ("Reference ExtraTrees (500)", "ExtraTrees + DE"),
    ):
        surrogate_error = float(surrogate_metrics.loc[surrogate_metrics["model"] == model_name, "nrmse"].mean())
        ablation_rows.append(
            {
                "surrogate": model_name,
                "optimizer": "Differential Evolution",
                "surrogate_test_nrmse_macro": surrogate_error,
                "inverse_predicted_target_nrmse": float(inverse_means[method_name]),
                "interpretation": "Separate surrogate test error and inverse residual; not additive because they answer different questions.",
            }
        )
    for method_name, model_name in (("Random search (ANN)", "ANN (MLPRegressor)"), ("Random search (ExtraTrees)", "Reference ExtraTrees (500)")):
        ablation_rows.append(
            {
                "surrogate": model_name,
                "optimizer": "Equal-budget random search",
                "surrogate_test_nrmse_macro": float(surrogate_metrics.loc[surrogate_metrics["model"] == model_name, "nrmse"].mean()),
                "inverse_predicted_target_nrmse": float(inverse_means[method_name]),
                "interpretation": "Optimizer comparison under the same surrogate and candidate-evaluation budget.",
            }
        )
    ablation_rows.append(
        {
            "surrogate": "Observed train-set performance",
            "optimizer": "Nearest-neighbor retrieval",
            "surrogate_test_nrmse_macro": float("nan"),
            "inverse_predicted_target_nrmse": float(inverse_means["Nearest-neighbor retrieval"]),
            "interpretation": "Data-retrieval baseline, no surrogate prediction or optimization loop.",
        }
    )
    pd.DataFrame(ablation_rows).to_csv(RESULTS_DIR / "surrogate_ablation.csv", index=False, float_format="%.10g")

    timing_path = RESULTS_DIR / "timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing["inverse_optimization_seconds_by_method"] = model_eval_seconds
    timing["inverse_script_seconds"] = time.perf_counter() - started
    timing["total_pipeline_seconds_so_far"] = float(timing.get("total_surrogate_script_seconds", 0.0)) + timing["inverse_script_seconds"]
    timing["benchmark_targets"] = int(len(test))
    timing["evaluation_budget_per_stochastic_run"] = DE_EVALUATIONS
    timing["seed_count"] = len(DE_SEEDS)
    pipeline_times = record_script_time("inverse_optimization_seconds", timing["inverse_script_seconds"])
    timing["pipeline_script_seconds"] = pipeline_times
    timing["total_pipeline_seconds_so_far"] = float(sum(pipeline_times.values()))
    write_json(timing_path, timing)
    write_json(
        RESULTS_DIR / "inverse_protocol.json",
        {
            "objective": "mean squared output error standardized by train target standard deviation plus normalized squared bound violation",
            "bound_penalty_lambda": BOUND_PENALTY,
            "population_size": DE_POPULATION,
            "generations": DE_GENERATIONS,
            "evaluation_budget": DE_EVALUATIONS,
            "mutation_F": DE_MUTATION,
            "crossover_CR": DE_CROSSOVER,
            "seeds": list(DE_SEEDS),
            "random_search_candidates_per_run": DE_EVALUATIONS,
            "success_threshold_target_nrmse": SUCCESS_NRMSE_THRESHOLD,
            "target_nrmse_denominator": "train-set output range",
            "nearest_neighbor_distance": "RMS distance in train-standardized performance space",
            "nonuniqueness_radius": NONUNIQUENESS_RADIUS,
            "nonuniqueness_dv_spread": "mean DV standard deviation in raw units among train rows within the target-space radius",
        },
    )
    log(
        f"Inverse optimization complete | targets={len(test)} | budget={DE_EVALUATIONS} evaluations/run | "
        f"stochastic seeds={len(DE_SEEDS)} | elapsed={timing['inverse_script_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
