"""Tune the ANN surrogate, fit the fixed reference model, and evaluate once."""

from __future__ import annotations

import json
import importlib.metadata
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from config import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, SCHEMA_JSON, SEED, ensure_directories, log, record_script_time, write_json
from evaluate import macro_metrics, per_output_metrics


def _scaled_xy(frame: pd.DataFrame, x_scaler: object, y_scaler: object, inputs: list[str], outputs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Transform one split with scalers fitted by preprocess.py."""
    return x_scaler.transform(frame[inputs].to_numpy(dtype=float)), y_scaler.transform(frame[outputs].to_numpy(dtype=float))


def main() -> None:
    """Run a small ANN validation study and save both hybrid surrogates."""
    ensure_directories()
    started = time.perf_counter()
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    inputs, outputs = schema["input_columns"], schema["target_columns"]
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val.csv")
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    x_scaler = joblib.load(MODELS_DIR / "x_scaler.joblib")
    y_scaler = joblib.load(MODELS_DIR / "y_scaler.joblib")
    x_train, y_train = _scaled_xy(train, x_scaler, y_scaler, inputs, outputs)
    x_val, y_val = _scaled_xy(val, x_scaler, y_scaler, inputs, outputs)
    x_test, y_test = _scaled_xy(test, x_scaler, y_scaler, inputs, outputs)
    _, y_train_raw = x_train, train[outputs].to_numpy(dtype=float)
    y_val_raw = val[outputs].to_numpy(dtype=float)
    y_test_raw = test[outputs].to_numpy(dtype=float)
    candidates = [
        {"hidden_layer_sizes": (8,), "alpha": 0.01},
        {"hidden_layer_sizes": (16, 8), "alpha": 0.01},
        {"hidden_layer_sizes": (24, 12), "alpha": 0.1},
    ]
    trial_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for trial, params in enumerate(candidates, start=1):
            model = MLPRegressor(
                **params,
                activation="relu",
                solver="adam",
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                max_iter=1800,
                tol=1e-4,
                random_state=SEED,
            )
            fit_start = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_start
            pred_scaled = model.predict(x_val)
            pred_raw = y_scaler.inverse_transform(pred_scaled)
            score = macro_metrics(y_val_raw, pred_raw, y_train_raw)
            trial_rows.append(
                {
                    "trial": trial,
                    "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
                    "alpha": params["alpha"],
                    "activation": "relu",
                    "solver": "adam",
                    "early_stopping": True,
                    "max_iter": 1800,
                    "n_iter": int(model.n_iter_),
                    "val_mae": score["mae"],
                    "val_rmse": score["rmse"],
                    "val_r2": score["r2"],
                    "val_nrmse_macro": score["nrmse_macro"],
                    "fit_seconds": fit_seconds,
                }
            )
    search = pd.DataFrame(trial_rows).sort_values(["val_nrmse_macro", "val_mae"], ascending=True)
    search.to_csv(RESULTS_DIR / "surrogate_hyperparameter_search.csv", index=False, float_format="%.10g")
    best = search.iloc[0]
    best_params = {
        "hidden_layer_sizes": tuple(int(value) for value in best["hidden_layer_sizes"].strip("() ").split(",") if value.strip()),
        "alpha": float(best["alpha"]),
    }

    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.vstack([y_train, y_val])
    ann = MLPRegressor(
        **best_params,
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        max_iter=1800,
        tol=1e-4,
        random_state=SEED,
    )
    ann_fit_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        ann.fit(x_trainval, y_trainval)
    ann_train_seconds = time.perf_counter() - ann_fit_start

    reference = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, random_state=SEED, n_jobs=1)
    ref_fit_start = time.perf_counter()
    reference.fit(x_train, y_train)
    reference_train_seconds = time.perf_counter() - ref_fit_start
    joblib.dump(ann, MODELS_DIR / "ann_surrogate.joblib")
    joblib.dump(reference, MODELS_DIR / "reference_extratrees.joblib")

    forward_baselines = {
        "DummyRegressor (mean)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
    }
    baseline_training_seconds = {}
    for baseline_name, baseline_model in forward_baselines.items():
        baseline_start = time.perf_counter()
        baseline_model.fit(x_trainval, y_trainval)
        baseline_training_seconds[baseline_name] = time.perf_counter() - baseline_start
        filename = "dummy_forward_baseline.joblib" if baseline_name.startswith("Dummy") else "linear_forward_baseline.joblib"
        joblib.dump(baseline_model, MODELS_DIR / filename)

    ann_test_prediction = y_scaler.inverse_transform(ann.predict(x_test))
    ref_test_prediction = y_scaler.inverse_transform(reference.predict(x_test))
    test_truth = y_test_raw
    ann_metrics = per_output_metrics(test_truth, ann_test_prediction, outputs, y_train_raw, "ANN (MLPRegressor)")
    ref_metrics = per_output_metrics(test_truth, ref_test_prediction, outputs, y_train_raw, "Reference ExtraTrees (500)")
    baseline_metric_frames = []
    for baseline_name, baseline_model in forward_baselines.items():
        prediction = y_scaler.inverse_transform(baseline_model.predict(x_test))
        baseline_metric_frames.append(per_output_metrics(test_truth, prediction, outputs, y_train_raw, baseline_name))
    pd.concat([ann_metrics, ref_metrics, *baseline_metric_frames], ignore_index=True).to_csv(
        RESULTS_DIR / "surrogate_test_metrics.csv", index=False, float_format="%.10g"
    )
    pd.concat([ann_metrics, ref_metrics, *baseline_metric_frames], ignore_index=True).groupby("model").agg(
        macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"), macro_r2=("r2", "mean"), macro_nrmse=("nrmse", "mean")
    ).reset_index().to_csv(RESULTS_DIR / "forward_baseline_comparison.csv", index=False, float_format="%.10g")
    # Post-fit, descriptive permutation importance; it does not alter model selection.
    rng = np.random.default_rng(SEED)
    importance_rows = []
    ann_test_scaled_prediction = ann.predict(x_test)
    baseline_r2 = np.asarray(
        [r2_score(y_test[:, index], ann_test_scaled_prediction[:, index]) for index in range(len(outputs))],
        dtype=float,
    )
    for feature_index, feature_name in enumerate(inputs):
        drops = []
        for _ in range(5):
            permuted_x = x_test.copy()
            permuted_x[:, feature_index] = permuted_x[rng.permutation(len(permuted_x)), feature_index]
            permuted_prediction = ann.predict(permuted_x)
            drops.append(
                np.asarray(
                    [
                        baseline_r2[index] - r2_score(y_test[:, index], permuted_prediction[:, index])
                        for index in range(len(outputs))
                    ],
                    dtype=float,
                )
            )
        drops_array = np.vstack(drops)
        for output_index, output_name in enumerate(outputs):
            importance_rows.append(
                {
                    "model": "ANN (MLPRegressor)",
                    "feature": feature_name,
                    "output": output_name,
                    "importance_mean": float(drops_array[:, output_index].mean()),
                    "importance_std": float(drops_array[:, output_index].std(ddof=0)),
                    "scoring": "test R2 decrease after one-feature permutation (5 repeats)",
                }
            )
    pd.DataFrame(importance_rows).to_csv(RESULTS_DIR / "feature_importance_per_output.csv", index=False, float_format="%.10g")
    prediction_frame = pd.DataFrame(ann_test_prediction, columns=outputs)
    prediction_frame.insert(0, "test_row", np.arange(len(test), dtype=int))
    prediction_frame.to_csv(RESULTS_DIR / "ann_test_predictions.csv", index=False, float_format="%.10g")

    timings: dict[str, object] = {
        "ann_training_seconds": ann_train_seconds,
        "reference_extratrees_training_seconds": reference_train_seconds,
        "baseline_training_seconds": baseline_training_seconds,
        "test_samples": int(len(test)),
        "test_batch_prediction_seconds": {},
        "per_sample_latency_seconds_mean": {},
        "per_sample_latency_seconds_std": {},
        "latency_repetitions_after_warmup": 100,
    }
    timing_models = {"ANN (MLPRegressor)": ann, "Reference ExtraTrees (500)": reference, **forward_baselines}
    for name, model in timing_models.items():
        x_for_batch = x_test
        batch_start = time.perf_counter()
        y_scaler.inverse_transform(model.predict(x_for_batch))
        timings["test_batch_prediction_seconds"][name] = time.perf_counter() - batch_start
        for _ in range(5):
            model.predict(x_for_batch[:1])
        samples = []
        for _ in range(100):
            latency_start = time.perf_counter()
            model.predict(x_for_batch[:1])
            samples.append(time.perf_counter() - latency_start)
        timings["per_sample_latency_seconds_mean"][name] = float(np.mean(samples))
        timings["per_sample_latency_seconds_std"][name] = float(np.std(samples, ddof=0))
    timings["total_surrogate_script_seconds"] = time.perf_counter() - started
    write_json(RESULTS_DIR / "timing.json", timings)
    write_json(
        MODELS_DIR / "surrogate_metadata.json",
        {
            "model": "sklearn.neural_network.MLPRegressor",
            "selected_parameters": {
                "hidden_layer_sizes": list(best_params["hidden_layer_sizes"]),
                "alpha": best_params["alpha"],
                "activation": "relu",
                "solver": "adam",
                "early_stopping": True,
                "random_state": SEED,
            },
            "selection_metric": "validation macro nRMSE",
            "selection_score": float(best["val_nrmse_macro"]),
            "refit_rows": int(len(train) + len(val)),
            "reference": "ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, random_state=42, n_jobs=1), fit on train only",
            "reference_test_macro_r2": float(r2_score(test_truth, ref_test_prediction, multioutput="uniform_average")),
            "ann_test_macro_r2": float(r2_score(test_truth, ann_test_prediction, multioutput="uniform_average")),
        },
    )
    ann_negative = ann_metrics.loc[ann_metrics["r2"] < 0, "output"].tolist()
    best_outputs = ann_metrics.nsmallest(5, "rmse")[["output", "rmse"]].to_dict(orient="records")
    worst_outputs = ann_metrics.nlargest(5, "rmse")[["output", "rmse"]].to_dict(orient="records")
    write_json(
        RESULTS_DIR / "surrogate_diagnostics.json",
        {
            "ann_negative_r2_count": len(ann_negative),
            "ann_negative_r2_outputs": ann_negative,
            "ann_best_5_outputs_by_rmse": best_outputs,
            "ann_worst_5_outputs_by_rmse": worst_outputs,
            "feature_importance_method": "Post-fit test permutation importance, five repeats; descriptive only and not used for tuning.",
        },
    )
    ann_macro = macro_metrics(test_truth, ann_test_prediction, y_train_raw)
    ref_macro = macro_metrics(test_truth, ref_test_prediction, y_train_raw)
    log(f"ANN selected | params={best_params} | validation nRMSE={best['val_nrmse_macro']:.5f} | test macro R2={ann_macro['r2']:.5f}")
    log(f"Reference ExtraTrees | 500 trees | train-only | test macro R2={ref_macro['r2']:.5f}")
    log(f"Surrogate fit and timing complete | elapsed={timings['total_surrogate_script_seconds']:.2f}s")
    pipeline_times = record_script_time("surrogate_seconds", timings["total_surrogate_script_seconds"])
    timings["pipeline_script_seconds"] = pipeline_times
    write_json(RESULTS_DIR / "timing.json", timings)
    write_json(
        RESULTS_DIR / "software_versions.json",
        {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "matplotlib": importlib.metadata.version("matplotlib"),
            "seaborn": importlib.metadata.version("seaborn"),
            "pdfplumber": importlib.metadata.version("pdfplumber"),
            "pytest": importlib.metadata.version("pytest"),
        },
    )


if __name__ == "__main__":
    main()
