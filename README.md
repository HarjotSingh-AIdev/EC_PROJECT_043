# EC-101 Project — 26BCS043: ANN + Differential Evolution

## Problem

This project studies hybrid inverse antenna design for roll 26BCS043. The main inverse input is a 35-value target performance vector and the requested output is a candidate design DV1-DV10. A feed-forward artificial neural network (implemented as scikit-learn's `MLPRegressor`) supplies the forward mapping from design variables to performance; Differential Evolution searches the training-set design bounds for candidates matching each held-out target. The optimizer's result is a surrogate-validated candidate; no CST or experimental evaluation is available here.

## Dataset

The source is `data/raw/DataSet for Project-EC 101.pdf` (50 records). Extraction expands newline-paired cells from the PDF table into individual records, then keeps DV1-DV10 and P1_F1-P7_F5. The table also contains Design_ID and frequency-coordinate columns F1-F5; those are excluded because the allotment defines the model inputs as DV1-DV10 and outputs as the P_F performance vector. The detected dataset has 10 inputs, 35 outputs, 7 indicators, and 5 frequency positions. This resolves the handout's P1-P7 versus P8-P10 inconsistency in favor of the actual table headers.

## Headline results

### Forward surrogate test metrics

| model                      |   macro_mae |   macro_rmse |   macro_r2 |   macro_nrmse |
|:---------------------------|------------:|-------------:|-----------:|--------------:|
| ANN (MLPRegressor)         |      1.3836 |       1.7240 |     0.1889 |        0.2223 |
| DummyRegressor (mean)      |      1.7290 |       2.0394 |    -0.1484 |        0.2794 |
| LinearRegression           |      1.5671 |       1.8015 |     0.1972 |        0.2135 |
| Reference ExtraTrees (500) |      1.1983 |       1.4974 |     0.1671 |        0.2133 |

### Inverse benchmark comparison

| method                     |   target_nrmse_mean |   target_nrmse_seed_run_std |   dv_rmse_physical_mean |   dv_rmse_standardized_mean |   feasibility_rate |   success_rate |   evaluation_count_mean |
|:---------------------------|--------------------:|----------------------------:|------------------------:|----------------------------:|-------------------:|---------------:|------------------------:|
| ANN + DE                   |              0.1916 |                      0.0010 |                605.9004 |                      1.2993 |             1.0000 |         0.0000 |               5050.0000 |
| ExtraTrees + DE            |              0.1898 |                      0.0010 |                360.5765 |                      1.0618 |             1.0000 |         0.0000 |               5050.0000 |
| Random search (ANN)        |              0.2081 |                      0.0035 |                479.3077 |                      0.9888 |             1.0000 |         0.0000 |               5050.0000 |
| Random search (ExtraTrees) |              0.2030 |                      0.0028 |                364.8102 |                      1.0536 |             1.0000 |         0.0000 |               5050.0000 |
| Nearest-neighbor retrieval |              0.2266 |                      0.0000 |                213.3249 |                      0.7477 |             1.0000 |         0.0000 |                  0.0000 |

The ANN has negative test R2 on 11 outputs: P1_F4, P3_F5, P5_F1, P5_F2, P5_F3, P5_F4, P5_F5, P6_F2, P7_F1, P7_F2, P7_F4. See `results/surrogate_diagnostics.json` for the best and worst five outputs by RMSE.

The benchmark uses 7 test rows from the saved 70/15/15 split after the train/validation design has been established. The test portion is small, so the result is a pilot estimate and should not be read as a stable generalization claim.

## Preprocessing and protocol

- Exact duplicates are removed, then a single NumPy permutation with seed 42 creates a 70/15/15 split (largest-remainder allocation). Original source row indices are recorded in `results/split_indices.json`.
- Rows with more than 2% missing cells are dropped. Remaining gaps are filled with medians computed on train only. The processed split CSVs retain original units.
- Separate `StandardScaler` objects for X and y are fitted on train only. Hyperparameter selection uses validation only; the chosen ANN is refit on train+validation. Test metrics are computed once by the final pipeline.
- The named hybrid is ANN + Differential Evolution. The reference ExtraTrees surrogate is fixed at 500 trees and fitted on train only. DE uses DE/rand/1/bin, population 50, F=0.7, CR=0.9, 100 generations, and 5050 objective evaluations per run across seeds 42-46. Random search uses the same number of candidates. Retrieval returns the nearest training performance vector.
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

### Design for a new target

After running the pipeline above, create a one-row CSV with all 35 performance columns (`P1_F1` through `P7_F5`) using the same column names as `data/processed/train.csv`. Then run:

```powershell
python src/predict_inverse.py --target-csv examples\sample_target.csv --output-dir results\custom_inverse --seed 42
```

The script returns `candidate_design.csv`, a per-output `target_vs_achieved.csv`, a convergence trace, and a summary with target MAE/RMSE/nRMSE and timing. It evaluates achieved performance by passing the candidate DV vector through the saved, held-out-validated ANN surrogate. CST or physical validation remains a separate step. `examples/sample_target.csv` is a ready-to-run example; replace its values with the requested target vector for a new design task.

## Guideline coverage

- **Clean source and run instructions:** modular scripts under `src/`; installation and end-to-end commands above.
- **Preprocessing and normalization:** `src/preprocess.py` makes one saved split, fits median imputation and separate X/y scalers on train only, and applies those transforms to validation/test data.
- **Training and testing:** `src/surrogate.py` selects ANN settings using validation only and reports final test metrics; exact split membership is in `results/split_indices.json`.
- **Prediction/inverse design:** `src/predict_inverse.py` accepts a user target and writes a candidate, convergence trace, and target-versus-achieved errors.
- **Metrics and timing:** MAE, RMSE, R2, and inference timing are in `results/surrogate_test_metrics.csv` and `results/timing.json`; inverse target errors are in `results/inverse_metrics_by_target.csv` and `results/metrics_per_output.csv`.
- **Plots and report:** predicted-versus-actual and convergence figures are in `results/figures/`; methods, results, limitations, and future work are in `report/report.md`.
- **Reproducibility and validation:** seeds, hyperparameters, elapsed times, and software versions are recorded under `results/`. Inverse candidates are passed through the held-out-validated ANN; CST/physical differences are unavailable because those evaluations were not provided.

## Project files

- `src/extract_data.py`: PDF parsing, paired-cell expansion, schema validation.
- `src/preprocess.py`: deterministic split, train-only imputation/scalers, dataset card.
- `src/surrogate.py`: MLP validation study, fixed ExtraTrees reference, test metrics and timing.
- `src/optimize_inverse.py`: DE, random search, retrieval, sanity functions, and inverse metrics.
- `src/predict_inverse.py`: command-line inverse design for a user-supplied 35-value target.
- `examples/sample_target.csv`: example target input for the custom inverse command.
- `src/predict_inverse.py`: command-line inverse design for one user-supplied 35-value target.
- `src/evaluate.py`: reusable metric functions and report/README assembly.
- `src/plots.py`: all report figures.
- `results/`: tables, traces, split indices, timings, and machine-readable summary.
- `report/report.md`: full methods and results report.
- `report/report.pdf`: printable version of the full report.

## Limitations

Only 50 data records are available; sample size is the primary limitation. The 35-dimensional target has far more dimensions than records, so both the ANN fit and the inverse optimum have substantial uncertainty. Differential Evolution is stochastic, and five seeds on a small benchmark give only a rough repeatability estimate. The inverse map is many-to-one, the optimizer relies on the ANN as its evaluator, train-derived bounds do not guarantee physical feasibility, and no electromagnetic solver or physical prototype is available to confirm the candidates. The handout's variable responsibility map is descriptive only; with these data it cannot establish causality.
