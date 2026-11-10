"""Design a candidate DV1-DV10 vector for a user-supplied performance target."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import DE_EVALUATIONS, DE_POPULATION, MODELS_DIR, PROJECT_ROOT, SCHEMA_JSON, write_json
from optimize_inverse import _objective_function, _predict_physical, _target_nrmse, differential_evolution


def load_target_csv(path: Path, output_names: list[str]) -> np.ndarray:
    """Read one target row, requiring every performance column exactly once."""
    frame = pd.read_csv(path)
    missing = [name for name in output_names if name not in frame.columns]
    extra = [name for name in frame.columns if name not in output_names]
    if missing or extra:
        raise ValueError(f"Target CSV columns must match the project schema; missing={missing}, extra={extra}")
    if len(frame) != 1:
        raise ValueError(f"Target CSV must contain exactly one data row; found {len(frame)}")
    target = frame[output_names].to_numpy(dtype=float)[0]
    if not np.isfinite(target).all():
        raise ValueError("Target performance values must all be finite numbers")
    return target


def design_target(
    target: np.ndarray,
    seed: int = 42,
    budget: int = DE_EVALUATIONS,
    population_size: int = DE_POPULATION,
) -> dict[str, object]:
    """Optimize one target with the saved ANN and return auditable outputs."""
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    input_names, output_names = schema["input_columns"], schema["target_columns"]
    target = np.asarray(target, dtype=float).reshape(-1)
    if target.shape != (len(output_names),) or not np.isfinite(target).all():
        raise ValueError(f"target must contain {len(output_names)} finite performance values")
    if population_size < 4 or budget < population_size:
        raise ValueError("DE needs population_size >= 4 and budget >= population_size")

    model = joblib.load(MODELS_DIR / "ann_surrogate.joblib")
    x_scaler = joblib.load(MODELS_DIR / "x_scaler.joblib")
    y_scaler = joblib.load(MODELS_DIR / "y_scaler.joblib")
    bounds_payload = json.loads((MODELS_DIR / "dv_bounds.json").read_text(encoding="utf-8"))
    bounds = np.asarray([[bounds_payload[name]["min"], bounds_payload[name]["max"]] for name in input_names])
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    train_y = train[output_names].to_numpy(dtype=float)
    objective = _objective_function(model, target, x_scaler, y_scaler, y_scaler.scale_, bounds)

    started = time.perf_counter()
    candidate, objective_value, trace = differential_evolution(
        objective, bounds, seed=seed, budget=budget, population_size=population_size
    )
    optimization_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    achieved = _predict_physical(model, candidate, x_scaler, y_scaler)
    prediction_seconds = time.perf_counter() - prediction_started

    error = achieved - target
    ranges = np.ptp(train_y, axis=0)
    normalized_error = np.divide(error, ranges, out=np.zeros_like(error), where=ranges > 0)
    performance_errors = pd.DataFrame(
        {
            "output": output_names,
            "target": target,
            "achieved_by_ann": achieved,
            "signed_error": error,
            "absolute_error": np.abs(error),
            "range_normalized_error": normalized_error,
        }
    )
    return {
        "input_names": input_names,
        "output_names": output_names,
        "candidate": candidate,
        "achieved": achieved,
        "performance_errors": performance_errors,
        "trace": trace,
        "summary": {
            "model": "saved ANN forward surrogate",
            "optimizer": "Differential Evolution rand/1/bin",
            "seed": int(seed),
            "population_size": int(population_size),
            "generations_equivalent": int(max(0, (budget - population_size) // population_size)),
            "evaluation_budget": int(budget),
            "evaluations_used": int(trace[-1][0]),
            "objective": float(objective_value),
            "target_mae": float(np.mean(np.abs(error))),
            "target_rmse": float(np.sqrt(np.mean(error**2))),
            "target_nrmse_by_train_range": _target_nrmse(achieved, target, train_y),
            "optimization_seconds": float(optimization_seconds),
            "forward_prediction_seconds": float(prediction_seconds),
            "validation_note": "Achieved performance is predicted by the held-out-validated ANN surrogate; CST/physical validation is not available.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", required=True, type=Path, help="CSV with one row and all 35 P*_F* columns")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "custom_inverse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=DE_EVALUATIONS)
    parser.add_argument("--population", type=int, default=DE_POPULATION)
    args = parser.parse_args()

    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    target = load_target_csv(args.target_csv, schema["target_columns"])
    result = design_target(target, args.seed, args.budget, args.population)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result["candidate"]], columns=result["input_names"]).to_csv(
        args.output_dir / "candidate_design.csv", index=False, float_format="%.10g"
    )
    result["performance_errors"].to_csv(
        args.output_dir / "target_vs_achieved.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(result["trace"], columns=["evaluations", "best_objective"]).to_csv(
        args.output_dir / "convergence_trace.csv", index=False, float_format="%.10g"
    )
    write_json(args.output_dir / "summary.json", result["summary"])
    print(f"Candidate design and target-versus-achieved results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
