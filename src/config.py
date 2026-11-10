"""Central configuration and project paths for EC-101 roll 26BCS043."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ROLL = "26BCS043"
EMAIL = "26BCS043@nith.ac.in"
ALGORITHM = "ANN + Differential Evolution"
FAMILY = "HYBRID"
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
MAX_MISSING_FRACTION = 0.02
DE_POPULATION = 50
DE_GENERATIONS = 100
DE_EVALUATIONS = DE_POPULATION * (DE_GENERATIONS + 1)
DE_SEEDS = (42, 43, 44, 45, 46)
DE_MUTATION = 0.7
DE_CROSSOVER = 0.9
BOUND_PENALTY = 1000.0
SUCCESS_NRMSE_THRESHOLD = 0.10
NONUNIQUENESS_RADIUS = 1.00

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "DataSet for Project-EC 101.pdf"
ALLOTMENT_PDF = PROJECT_ROOT / "data" / "raw" / "EC-101 Projects.pdf"
EXTRACTED_CSV = PROJECT_ROOT / "data" / "interim" / "extracted_raw.csv"
SCHEMA_JSON = PROJECT_ROOT / "data" / "processed" / "schema.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"


def ensure_directories() -> None:
    """Create all generated-artifact directories."""
    for path in (EXTRACTED_CSV.parent, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    """Print a progress line and append it to the shared run log."""
    print(message, flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "run_log.txt").open("a", encoding="utf-8") as stream:
        stream.write(message.rstrip() + "\n")


def write_json(path: Path, value: Any) -> None:
    """Write JSON with stable formatting and a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_script_time(script_name: str, seconds: float) -> dict[str, float]:
    """Record one script duration in the shared pipeline timing ledger."""
    path = RESULTS_DIR / "pipeline_times.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    values[script_name] = float(seconds)
    write_json(path, values)
    return values
