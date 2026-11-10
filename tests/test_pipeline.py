"""Fast schema, split, prediction, and deterministic-smoke checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, SCHEMA_JSON
from extract_data import extract_pdf_table
from optimize_inverse import differential_evolution
from predict_inverse import design_target, load_target_csv


def test_schema_columns_and_count() -> None:
    """The PDF extraction must resolve the actual 10-by-35 dataset schema."""
    frame, metadata = extract_pdf_table()
    assert len(frame) == 50
    assert metadata["n_rows"] == 50
    assert len(metadata["input_columns"]) == 10
    assert len(metadata["target_columns"]) == 35
    assert frame.columns.tolist() == metadata["input_columns"] + metadata["target_columns"]


def test_split_sizes_disjointness_and_no_nans() -> None:
    """The saved 70/15/15 split is complete, disjoint, and fully imputed."""
    split = json.loads((RESULTS_DIR / "split_indices.json").read_text(encoding="utf-8"))
    assert [len(split[name]) for name in ("train", "val", "test")] == [35, 8, 7]
    groups = [set(split[name]) for name in ("train", "val", "test")]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    for name in ("train", "val", "test"):
        frame = pd.read_csv(PROJECT_ROOT / "data" / "processed" / f"{name}.csv")
        assert not frame.isna().any().any()


def test_final_surrogate_prediction_shape() -> None:
    """The fitted hybrid surrogate predicts all schema outputs."""
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    model = joblib.load(MODELS_DIR / "ann_surrogate.joblib")
    x_scaler = joblib.load(MODELS_DIR / "x_scaler.joblib")
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    sample = x_scaler.transform(train[schema["input_columns"]].iloc[:3].to_numpy(float))
    assert model.predict(sample).shape == (3, 35)


def test_smoke_run_repeats_with_same_seed() -> None:
    """The named DE implementation returns the same objective and trace for a fixed seed."""
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    bounds = np.tile(np.array([[-5.0, 5.0]]), (10, 1))
    first = differential_evolution(sphere, bounds, seed=123, budget=80, population_size=8)
    second = differential_evolution(sphere, bounds, seed=123, budget=80, population_size=8)
    assert first[0].shape == (10,)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[2][-1][0] == 80


def test_custom_inverse_target_contract_and_prediction() -> None:
    """A supplied performance target returns a bounded design and achieved outputs."""
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    output_names = schema["target_columns"]
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    target_frame = train[output_names].iloc[[0]]
    target_path = PROJECT_ROOT / "results" / "test_custom_target.csv"
    target_frame.to_csv(target_path, index=False)
    try:
        target = load_target_csv(target_path, output_names)
    finally:
        target_path.unlink(missing_ok=True)
    result = design_target(target, seed=7, budget=16, population_size=8)
    candidate = np.asarray(result["candidate"])
    bounds = json.loads((MODELS_DIR / "dv_bounds.json").read_text(encoding="utf-8"))
    for index, name in enumerate(schema["input_columns"]):
        assert bounds[name]["min"] <= candidate[index] <= bounds[name]["max"]
    assert np.asarray(result["achieved"]).shape == (35,)
    assert len(result["performance_errors"]) == 35
    assert result["summary"]["evaluations_used"] == 16
