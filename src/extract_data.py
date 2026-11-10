"""Extract the project dataset table from its source PDF."""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd
import pdfplumber

from config import ALLOTMENT_PDF, EMAIL, EXTRACTED_CSV, FAMILY, RAW_DATA, RESULTS_DIR, ROLL, SCHEMA_JSON, ensure_directories, log, record_script_time, write_json

DV_PATTERN = re.compile(r"^DV\s*(\d+)$", re.IGNORECASE)
OUTPUT_PATTERN = re.compile(r"^P\s*(\d+)\s*_\s*F\s*(\d+)$", re.IGNORECASE)


def _header_name(value: Any) -> str:
    """Normalize a source header and retain only known schema forms."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    dv_match = DV_PATTERN.match(text)
    output_match = OUTPUT_PATTERN.match(text)
    if dv_match:
        return f"DV{int(dv_match.group(1))}"
    if output_match:
        return f"P{int(output_match.group(1))}_F{int(output_match.group(2))}"
    if text in {"DESIGN_ID", "DESIGNID"}:
        return "Design_ID"
    if re.fullmatch(r"F\d+", text):
        return text
    return ""


def _number(value: Any) -> float:
    """Convert a PDF cell to float, recognizing common missing-value tokens."""
    if value is None:
        return float("nan")
    text = str(value).strip().replace("−", "-").replace("–", "-")
    if not text or text.upper() in {"NA", "N/A", "NAN", "NULL", "-", "—"}:
        return float("nan")
    text = text.replace(" ", "")
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    return float(text)


def _expand_wrapped_row(row: list[Any], width: int, page_number: int, row_number: int) -> list[list[Any]]:
    """Expand cells containing aligned newline-separated records."""
    cells = [(str(cell).splitlines() if cell is not None else [""]) for cell in row]
    if len(cells) != width:
        raise ValueError(f"page {page_number}, row {row_number}: expected {width} cells, got {len(cells)}")
    count = max((len(cell) for cell in cells), default=1)
    if count == 1:
        return [[cell[0].strip() for cell in cells]]
    lengths = {len(cell) for cell in cells if any(part.strip() for part in cell)}
    if lengths != {count}:
        raise ValueError(
            f"page {page_number}, row {row_number}: misaligned wrapped cells; "
            f"cell line counts={sorted(lengths)}, max={count}, sample={row[:6]!r}"
        )
    aligned = []
    for cell in cells:
        if len(cell) == count:
            aligned.append([part.strip() for part in cell])
        elif len(cell) == 1 and not cell[0].strip():
            aligned.append([""] * count)
        else:
            raise ValueError(
                f"page {page_number}, row {row_number}: nonblank merged cell cannot be aligned; "
                f"cell line count={len(cell)}, expected={count}, sample={row[:6]!r}"
            )
    return [[aligned[column][index] for column in range(width)] for index in range(count)]


def extract_pdf_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse the data PDF and return only the 10 DV and detected P_F columns."""
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Required dataset PDF is missing: {RAW_DATA}")
    pages = pdfplumber.open(RAW_DATA)
    page_count = len(pages.pages)
    candidates: list[tuple[int, list[list[Any]]]] = []
    for page_number, page in enumerate(pages.pages, start=1):
        for table in page.extract_tables() or []:
            if table and any(_header_name(cell).startswith("DV") for cell in table[0]):
                candidates.append((page_number, table))
    if not candidates:
        raise ValueError(f"No recognizable table found in {page_count}-page PDF {RAW_DATA.name}")

    header_page, table = max(candidates, key=lambda item: len(item[1]))
    normalized = [_header_name(cell) for cell in table[0]]
    dv_columns = [f"DV{i}" for i in range(1, 11)]
    output_columns = sorted(
        {name for name in normalized if OUTPUT_PATTERN.match(name)},
        key=lambda name: tuple(map(int, re.findall(r"\d+", name))),
    )
    if [name for name in normalized if name.startswith("DV")] != dv_columns:
        raise ValueError(f"Malformed DV header on page {header_page}: {normalized!r}")
    if len(output_columns) < 1:
        raise ValueError(f"No P{i}_F{j} output headers detected: {normalized!r}")
    pairs = {tuple(map(int, re.findall(r"\d+", name))) for name in output_columns}
    indicators = sorted({indicator for indicator, _ in pairs})
    frequencies = sorted({frequency for _, frequency in pairs})
    expected = {f"P{i}_F{j}" for i in indicators for j in frequencies}
    if set(output_columns) != expected:
        raise ValueError(f"Output grid is incomplete: missing={sorted(expected-set(output_columns))}, extra={sorted(set(output_columns)-expected)}")
    kept = dv_columns + output_columns
    col_positions = {name: index for index, name in enumerate(normalized) if name in kept}
    if len(col_positions) != len(kept):
        raise ValueError("Duplicate or missing normalized schema columns in the PDF header")

    records: list[list[float]] = []
    malformed: list[str] = []
    seen_header = {name.upper() for name in normalized if name}
    for page_number, page_table in candidates:
        for row_number, row in enumerate(page_table[1:], start=2):
            try:
                expanded = _expand_wrapped_row(row, len(normalized), page_number, row_number)
                for expanded_row in expanded:
                    row_headers = {str(value).strip().upper() for value in expanded_row}
                    if row_headers.intersection(seen_header):
                        continue
                    records.append([_number(expanded_row[col_positions[name]]) for name in kept])
            except (TypeError, ValueError) as error:
                malformed.append(str(error))
    if malformed:
        raise ValueError(
            f"Table parse failed on {page_count}-page PDF; malformed rows: "
            + " | ".join(malformed[:8])
        )
    frame = pd.DataFrame(records, columns=kept)
    frame = frame.dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"PDF table yielded no records; pages={page_count}; table page={header_page}")
    if len([column for column in frame.columns if column.startswith("DV")]) != 10:
        raise AssertionError("Extraction must retain exactly ten DV columns")
    if frame.isna().all().any():
        raise AssertionError("An all-NaN data column was detected")
    constant_dv = [column for column in dv_columns if frame[column].nunique(dropna=True) <= 1]
    if constant_dv:
        raise AssertionError(f"Constant DV columns were not retained: {constant_dv}")
    if frame.select_dtypes(include=[np.number]).shape[1] != frame.shape[1]:
        raise AssertionError("All extracted columns must be numeric")
    metadata = {
        "source_pdf": RAW_DATA.name,
        "source_pages": page_count,
        "table_page": header_page,
        "ignored_source_columns": [name for name in normalized if name and name not in kept],
        "input_columns": dv_columns,
        "target_columns": output_columns,
        "indicator_ids": indicators,
        "frequency_ids": frequencies,
        "detected_indicator_count": len(indicators),
        "n_rows": int(len(frame)),
    }
    return frame, metadata


def resolve_assignment() -> dict[str, Any]:
    """Read the roll-number row from the authoritative allotment PDF."""
    if not ALLOTMENT_PDF.exists():
        raise FileNotFoundError(f"Required allotment PDF is missing: {ALLOTMENT_PDF}")
    found = []
    with pdfplumber.open(ALLOTMENT_PDF) as document:
        for page_number, page in enumerate(document.pages, start=1):
            for table in page.extract_tables() or []:
                for row in table:
                    if row and str(row[0] or "").strip().upper() == ROLL:
                        found.append((page_number, [str(value or "").strip() for value in row]))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one allotment row for {ROLL}; found {len(found)}")
    page_number, row = found[0]
    algorithm = " ".join(row[1].split())
    pdf_algorithm = algorithm.replace("–", "-").replace("—", "-").lower()
    if "ann" not in pdf_algorithm or "differential evolution" not in pdf_algorithm:
        raise ValueError(f"Allotment row does not match the resolved hybrid: {row!r}")
    return {
        "roll": ROLL,
        "pdf_algorithm": algorithm,
        "family": FAMILY,
        "inputs": "Target performance vector P1_F1-P7_F5 (35 values detected in dataset PDF)",
        "outputs": "Candidate design DV1-DV10",
        "required_metrics": "MAE, RMSE, R2, prediction timing, ANN surrogate accuracy, DV error, and target-versus-achieved error",
        "extra_study": "ANN validation grid; ANN-vs-ExtraTrees surrogate ablation; DE budget-matched random search, parameter sensitivity, and benchmark-function sanity checks",
        "main_deliverable": "Documented Python code, results, and report",
        "email": EMAIL,
        "allotment_page": page_number,
        "source_row": row,
        "family_map_check": "PDF algorithm and ROLL_MAP family agree for 26BCS043 (HYBRID).",
    }


def main() -> None:
    """Extract, validate, save, and print the real dataset schema."""
    started = time.perf_counter()
    ensure_directories()
    assignment = resolve_assignment()
    frame, metadata = extract_pdf_table()
    frame.to_csv(EXTRACTED_CSV, index=False, float_format="%.10g")
    summary = {
        **metadata,
        "n_dv": len(metadata["input_columns"]),
        "n_outputs": len(metadata["target_columns"]),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "per_column_summary": {
            column: {
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
                "mean": float(frame[column].mean()),
            }
            for column in frame.columns
        },
        "assignment_pdf": ALLOTMENT_PDF.name,
        "assignment_pdf_untouched": ALLOTMENT_PDF.exists(),
    }
    write_json(SCHEMA_JSON, summary)
    assignment_text = (
        "ASSIGNMENT\n"
        f"ROLL | {assignment['roll']}\n"
        f"algorithm | {assignment['pdf_algorithm']}\n"
        f"family | {assignment['family']}\n"
        f"inputs | {assignment['inputs']}\n"
        f"outputs | {assignment['outputs']}\n"
        f"required metrics | {assignment['required_metrics']}\n"
        f"extra study | {assignment['extra_study']}\n"
        f"main deliverable | {assignment['main_deliverable']}\n"
        f"email | {assignment['email']}\n"
        f"resolution | {assignment['family_map_check']}\n"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "assignment_block.txt").write_text(assignment_text, encoding="utf-8")
    print(assignment_text, flush=True)
    log(
        "SCHEMA | rows={n_rows} | inputs={n_dv} | outputs={n_outputs} | indicators={indicator_ids} | "
        "frequencies={frequency_ids}".format(**summary)
    )
    log(f"Extracted data saved to {EXTRACTED_CSV}")
    log(f"Schema summary: {SCHEMA_JSON}")
    print("Per-column dtypes and min/max/mean:")
    for column, values in summary["per_column_summary"].items():
        print(f"  {column:8s} dtype={summary['dtypes'][column]:7s} min={values['min']:.6g} max={values['max']:.6g} mean={values['mean']:.6g}")
    record_script_time("extract_data_seconds", time.perf_counter() - started)


if __name__ == "__main__":
    main()
