# Hybrid ANN + Differential Evolution for EC-101 Antenna Design

**Student roll:** 26BCS043  
**Assignment:** ANN + Differential Evolution  
**Family:** HYBRID  
**Inverse input:** target performance vector (35 values: P1-P7 measured at F1-F5)  
**Inverse output:** candidate design DV1-DV10  
**Surrogate mapping:** DV1-DV10 -> 35 performance outputs  
**Dataset:** 50 source records

## 1. Introduction

Forward regression estimates antenna performance from design variables, while inverse design searches for a design that may achieve a requested performance vector. This project implements the hybrid assigned to roll 26BCS043: an artificial neural network (ANN) is used as a fixed forward surrogate and Differential Evolution (DE) searches the bounded design space. The PDF allotment names “ANN + Differential Evolution,” consistent with the fallback map. No discrepancy was found. The project evaluates both the surrogate and the end-to-end inverse pipeline, while treating the computed designs as candidates rather than physically validated designs.

## 2. Dataset and preprocessing

The sole numeric source is `data/raw/DataSet for Project-EC 101.pdf`. The table's 38 extracted table rows contain 50 actual designs because 13 rows combine two newline-separated records in every cell; the parser aligns and expands those pairs. It found 50 records, 10 inputs, and 35 outputs. The real headers contain indicators [1, 2, 3, 4, 5, 6, 7] at positions [1, 2, 3, 4, 5], yielding a complete 7 x 5 grid. The assignment prose mentions both P1-P7 and P5-P10; the PDF table has P1-P7 only, so the experiment follows the observed 35-column schema. Source Design_ID and F1-F5 coordinate columns are not model inputs under the allotment and are excluded.

Exact duplicate records are removed (count in `results/preprocessing_summary.json`). Rows exceeding 2% missingness are dropped; remaining gaps use medians learned only from train. One seeded permutation allocates 70/15/15 with largest-remainder rounding. Given 50 rows, the split is 35/8/7. The exact original row indices are stored in `results/split_indices.json`. The processed CSVs remain in physical units. X and y each use a StandardScaler fitted on train only, stored in `models/`. No outliers are clipped; 1.5-IQR counts and constant-column checks are recorded in the preprocessing summary. The complete output ranges are in `report/dataset_card.md`.

## 3. Algorithm

The ANN uses `MLPRegressor` with ReLU hidden layers, Adam optimization, and internal early stopping. A small validation grid compares three hidden-layer/regularization settings; model selection is based on validation macro nRMSE, so the small test set does not guide tuning. The selected configuration is refit on train plus validation, using the train-fitted scalers.

DE uses the DE/rand/1/bin strategy. For each population member, three distinct members are sampled, and a scaled difference vector is added to a base member. Binomial crossover mixes the mutant and target vectors, with at least one mutant coordinate. A greedy one-to-one selection retains a trial when it lowers the objective. This implementation follows Storn and Price's original Differential Evolution formulation ([1997 paper](https://doi.org/10.1023/A:1008202821328)). Candidates are clipped to train-derived variable bounds after mutation; the objective also contains a normalized squared bound-violation penalty (lambda = 1000), so feasibility is recorded explicitly.

## 4. Methodology

The fixed reference measuring instrument is `ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, random_state=42)`, fitted on train only. It achieved macro test R2 0.1671 and macro nRMSE 0.2133; these scores are the empirical accuracy ceiling for the reference-surrogate inverse ablation, since DE cannot overcome its evaluator's error. The named ANN is tuned on validation and refit on train+validation. The test subset provides K=7 fixed targets with their known DVs. The inverse map is non-unique; a target match is therefore not treated as proof of recovering the original design.

For target vector T and surrogate P-hat(d), the cost is

`J(d) = mean_i[((P-hat_i(d) - T_i) / s_i)^2] + 1000 * sum_j[normalized bound violation_j^2]`,

where s_i is the train standard deviation for output i. DE uses population size 50, 100 generations, F=0.7, CR=0.9 and 5050 evaluations per run (initial 50 plus 100 x 50 trials). Five fixed seeds (42-46) measure stochastic variability. Uniform random search uses the same candidate-evaluation budget. Nearest-neighbor retrieval chooses a train row with the closest train-standardized target performance. The success threshold is normalized target RMSE <= 0.10; this threshold is declared before comparison. All methods report DV MAE/RMSE in physical and train-standardized units, target-vs-achieved error per output, feasibility, objective, evaluation count, runtime, success, and nearest train-reference distance.

DE's code is checked on 10-dimensional Sphere, Rosenbrock, and Rastrigin functions with known zero minima, using the same evaluation budget. The traces and final gaps are saved under `results/`.

Median final gaps and median relative improvements from each run's initial population are:

| function   |   median_final_gap |   median_relative_improvement |
|:-----------|-------------------:|------------------------------:|
| Sphere     |             0.3934 |                        0.9873 |
| Rosenbrock |            16.9483 |                        0.9767 |
| Rastrigin  |            52.5180 |                        0.5466 |

## 5. Results

### Surrogate accuracy on the reserved test rows

| model                      |   macro_mae |   macro_rmse |   macro_r2 |   macro_nrmse |
|:---------------------------|------------:|-------------:|-----------:|--------------:|
| ANN (MLPRegressor)         |     1.38359 |      1.72399 |    0.18892 |       0.22229 |
| DummyRegressor (mean)      |     1.72904 |      2.03942 |   -0.14839 |       0.27942 |
| LinearRegression           |     1.56713 |      1.80146 |    0.19716 |       0.21347 |
| Reference ExtraTrees (500) |     1.19832 |      1.49739 |    0.16708 |       0.21331 |

The ANN had lower macro MAE/RMSE than LinearRegression (1.3836/1.7240 versus 1.5671/1.8015), while LinearRegression had slightly higher macro R2 (0.1972 versus 0.1889). Model ranking therefore depends on metric on this small test set.

The three ANN outputs with largest test RMSE are:

| output   |    rmse |       r2 |   nrmse |
|:---------|--------:|---------:|--------:|
| P1_F4    | 8.07596 | -0.09403 | 0.38228 |
| P1_F1    | 7.51298 |  0.31272 | 0.34085 |
| P1_F3    | 6.96814 |  0.27798 | 0.35292 |

The three hardest outputs by physical RMSE are:

- **P1_F4**: test RMSE 8.0760, nRMSE 0.3823, R2 -0.0940; negative R2 on seven cases means the model did worse than the test mean; the sample is too small for a stable interpretation.
- **P1_F1**: test RMSE 7.5130, nRMSE 0.3408, R2 0.3127; largest absolute error is partly scale-dependent; use nRMSE to compare across outputs.
- **P1_F3**: test RMSE 6.9681, nRMSE 0.3529, R2 0.2780; largest absolute error is partly scale-dependent; use nRMSE to compare across outputs.

The held-out test R2 is negative for the outputs listed in `results/surrogate_diagnostics.json`; that file also ranks the best and worst five ANN outputs by RMSE. The test set contains only seven designs, so these rankings are exploratory.

Table values are generated from `results/surrogate_test_metrics.csv` and `results/metrics_summary.json`. Per-output R2 includes negative values rather than hiding them. Test R2 is based on only 7 rows and is particularly variable for outputs with low test variance.

### Inverse methods

| method                     |   target_nrmse_mean |   target_nrmse_seed_run_std |   dv_rmse_physical_mean |   dv_rmse_standardized_mean |   feasibility_rate |   success_rate |   evaluation_count_mean |
|:---------------------------|--------------------:|----------------------------:|------------------------:|----------------------------:|-------------------:|---------------:|------------------------:|
| ANN + DE                   |             0.19160 |                     0.00098 |               605.90038 |                     1.29929 |            1.00000 |        0.00000 |              5050.00000 |
| ExtraTrees + DE            |             0.18981 |                     0.00095 |               360.57648 |                     1.06176 |            1.00000 |        0.00000 |              5050.00000 |
| Random search (ANN)        |             0.20808 |                     0.00351 |               479.30766 |                     0.98880 |            1.00000 |        0.00000 |              5050.00000 |
| Random search (ExtraTrees) |             0.20301 |                     0.00284 |               364.81016 |                     1.05359 |            1.00000 |        0.00000 |              5050.00000 |
| Nearest-neighbor retrieval |             0.22659 |                     0.00000 |               213.32486 |                     0.74774 |            1.00000 |        0.00000 |                 0.00000 |

ANN+DE's mean target nRMSE was 0.1916, versus 0.2081 for equal-budget random search under the same ANN. The fixed ExtraTrees + DE result was 0.1898. Retrieval had the highest mean target nRMSE (0.2266) but the lowest physical DV RMSE (213.32); this difference is consistent with the non-unique inverse map and reflects distinct evaluation goals.

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

- DV1-DV3 -> P3/P4: mean absolute permutation importance 0.3092 for listed outputs versus 0.2129 for other outputs (higher for listed outputs).
- DV4 -> P1/P2: mean absolute permutation importance 0.0371 for listed outputs versus 0.0471 for other outputs (not higher for listed outputs).
- DV5-DV10 -> P5-P7: mean absolute permutation importance 0.1028 for listed outputs versus 0.0817 for other outputs (higher for listed outputs).

The benchmark is intentionally limited to the seven held-out targets. Nearest-reference distances indicate how far each requested vector lies from observed training performance in standardized space. Non-uniqueness counts and DV spreads among train designs within RMS radius 1.00 in standardized performance space quantify how several recorded designs may explain similar performance. Candidates remain within the rectangular train min/max bounds, but this does not impose antenna geometry or manufacturability constraints. The optimizer is also optimizing the ANN, not the true electromagnetic response; surrogate error can produce optimistic predicted matches.

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

Run in this order: `python src/extract_data.py`, `python src/preprocess.py`, `python src/surrogate.py`, `python src/optimize_inverse.py`, `python src/evaluate.py`, `python src/plots.py`, then `pytest tests/`. The separate `src/predict_inverse.py` command accepts a new one-row, 35-column performance target CSV (see `examples/sample_target.csv`) and produces a candidate DV vector plus per-output target-versus-achieved errors using the held-out-validated ANN surrogate. The expanded test suite passes all five tests. Full benchmark pipeline runtime: **130.47 seconds**. Full configuration, elapsed times, software versions, split identities, and result tables are saved in `results/`; installed versions are pinned in `requirements.txt`. No raw PDF has been edited.
