"""Prepare auditable train/validation/test splits and train-fitted scalers."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    EXTRACTED_CSV,
    MAX_MISSING_FRACTION,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    SCHEMA_JSON,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    ensure_directories,
    log,
    record_script_time,
    write_json,
)


def _split_sizes(n_rows: int) -> tuple[int, int, int]:
    """Allocate integer split sizes using largest remainders."""
    ratios = (TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO)
    quotas = np.asarray(ratios) * n_rows
    sizes = np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - sizes), kind="stable")[: n_rows - int(sizes.sum())]:
        sizes[index] += 1
    return int(sizes[0]), int(sizes[1]), int(sizes[2])


def main() -> None:
    """Deduplicate, split once, impute from training medians, and scale."""
    started = time.perf_counter()
    ensure_directories()
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    frame = pd.read_csv(EXTRACTED_CSV)
    original_rows = frame.index.to_numpy(dtype=int)
    duplicate_mask = frame.duplicated(keep="first")
    n_duplicates = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].copy()
    source_index = original_rows[~duplicate_mask.to_numpy()]
    missing_fraction = frame.isna().mean(axis=1)
    drop_mask = missing_fraction > MAX_MISSING_FRACTION
    dropped_source = source_index[drop_mask.to_numpy()].tolist()
    frame = frame.loc[~drop_mask].copy()
    source_index = source_index[~drop_mask.to_numpy()]
    if frame.empty:
        raise ValueError("No rows remain after the missing-data threshold")

    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(frame))
    n_train, n_val, n_test = _split_sizes(len(frame))
    index_train = permutation[:n_train]
    index_val = permutation[n_train : n_train + n_val]
    index_test = permutation[n_train + n_val :]
    row_groups = {"train": index_train, "val": index_val, "test": index_test}
    input_columns = schema["input_columns"]
    target_columns = schema["target_columns"]
    write_json(MODELS_DIR / "input_names.json", input_columns)
    write_json(MODELS_DIR / "output_names.json", target_columns)
    train_frame = frame.iloc[index_train]
    train_medians = train_frame.median(axis=0)
    imputed: dict[str, pd.DataFrame] = {}
    for split_name, positions in row_groups.items():
        imputed[split_name] = frame.iloc[positions].fillna(train_medians)
        imputed[split_name].to_csv(PROCESSED_DIR / f"{split_name}.csv", index=False, float_format="%.10g")

    x_scaler = StandardScaler().fit(imputed["train"][input_columns].to_numpy(dtype=float))
    y_scaler = StandardScaler().fit(imputed["train"][target_columns].to_numpy(dtype=float))
    joblib.dump(x_scaler, MODELS_DIR / "x_scaler.joblib")
    joblib.dump(y_scaler, MODELS_DIR / "y_scaler.joblib")
    write_json(
        MODELS_DIR / "imputation.json",
        {"strategy": "training-column-median", "training_medians": {k: float(v) for k, v in train_medians.items()}},
    )
    split_indices = {
        split_name: [int(source_index[position]) for position in positions]
        for split_name, positions in row_groups.items()
    }
    split_indices["dropped_source_rows"] = [int(index) for index in dropped_source]
    split_indices["original_index_base"] = 0
    write_json(RESULTS_DIR / "split_indices.json", split_indices)

    full = frame.copy()
    outlier_counts: dict[str, int] = {}
    constant_columns: list[str] = []
    for column in full.columns:
        values = full[column].dropna()
        if values.nunique() <= 1:
            constant_columns.append(column)
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_counts[column] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if iqr else 0

    for split_name in ("train", "val", "test"):
        processed = pd.read_csv(PROCESSED_DIR / f"{split_name}.csv")
        if processed.isna().any().any():
            raise AssertionError(f"NaNs remain in {split_name}.csv")
    all_saved = [set(split_indices[name]) for name in ("train", "val", "test")]
    if any(all_saved[a].intersection(all_saved[b]) for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Split indices are not disjoint")

    ranges = {
        column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
        for column in target_columns
    }
    card = {
        "source_rows": int(len(pd.read_csv(EXTRACTED_CSV))),
        "deduplicated_rows": int(len(frame) + len(dropped_source)),
        "rows_dropped_for_missingness": int(len(dropped_source)),
        "duplicates_removed": n_duplicates,
        "n_inputs": len(input_columns),
        "n_outputs": len(target_columns),
        "split_sizes": {name: len(values) for name, values in split_indices.items() if name in row_groups},
        "target_ranges": ranges,
        "missing_data_treatment": "Rows with >2% missing cells are dropped; remaining missing cells are filled with each training-column median.",
        "split_protocol": "One NumPy Generator permutation (seed 42); largest-remainder 70/15/15 allocation.",
        "scaling": "Separate StandardScaler instances fitted only on train for X and y; processed CSVs remain in original units.",
        "constant_columns": constant_columns,
        "outlier_counts_iqr_1_5": outlier_counts,
    }
    write_json(RESULTS_DIR / "preprocessing_summary.json", card)
    lines = [
        "# Dataset card",
        "",
        f"- Source: `{schema['source_pdf']}` ({schema['source_pages']} page).",
        f"- Rows after cleaning: {len(frame)}; duplicate rows removed: {n_duplicates}; rows removed for missingness: {len(dropped_source)}.",
        f"- Inputs: {len(input_columns)} design variables; outputs: {len(target_columns)} performance values.",
        f"- Splits: train {n_train}, validation {n_val}, test {n_test}.",
        "- Missing-data treatment: drop rows above 2% missingness, then use training medians for remaining cells.",
        "- Scalers: X and y StandardScaler fitted on train only; split CSV files remain in original units.",
        f"- Constant columns: {constant_columns or 'none'}.",
        "",
        "## Performance output ranges",
        "",
        "| Output | Min | Max |",
        "|---|---:|---:|",
    ]
    lines.extend(f"| {column} | {values['min']:.6g} | {values['max']:.6g} |" for column, values in ranges.items())
    (schema_json_path := (PROCESSED_DIR.parent.parent / "report" / "dataset_card.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(
        f"Preprocessing complete | rows={len(frame)} | splits={n_train}/{n_val}/{n_test} | "
        f"duplicates_removed={n_duplicates} | missing_rows_dropped={len(dropped_source)}"
    )
    log(f"Train-only scalers saved to {MODELS_DIR}; split audit saved to {RESULTS_DIR / 'split_indices.json'}")
    record_script_time("preprocessing_seconds", time.perf_counter() - started)


if __name__ == "__main__":
    main()
