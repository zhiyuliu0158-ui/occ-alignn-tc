# SC-Tc Experiment Record

Last updated: 2026-06-05

This document records the current development and experiment state for the
`occ-alignn-tc` project. It intentionally excludes server passwords and private
login secrets.

## Repository And Runtime

- Repository: `zhiyuliu0158-ui/occ-alignn-tc`
- Local workspace: `D:\SC-Tc\occ-alignn-tc`
- Remote workspace: `/data/home/intern002/ziyang/sc-tc`
- Remote conda environment: `sc-tc`
- Main baseline configuration: `configs/occ_alignn_comp_formula_huber_tc.yaml`
- Main baseline definition: real formula + CIF graph + original pressure/field
  condition input + Huber loss.
- Original condition encoding:
  - Pressure: `[P_raw, log1p(P), unknown_flag]`
  - Magnetic field: `[H_raw, log1p(H), unknown_flag]`

Slurm submissions must keep:

```bash
#SBATCH -A hmt03
```

## Model Type

The project is a neural network model for superconducting critical temperature
regression. The main model is an occupancy-aware ALIGNN-style graph neural
network using CIF-derived crystal graphs, formula features, and pressure/field
condition inputs.

Traditional ML references discussed during comparison used
`MAGPIE + DSOAP + XGB`.

## Implemented Code Changes

### Data And Features

- Added formula descriptor support in `src/occ_alignn/featurizers/formula.py`.
- Added composition/formula features into graph construction and batching.
- Added optional confidence metadata features from:
  - `formula_level_cif_composition_l1`
  - `match_type`
  - `fidelity`
  - `formula_level_cif_status`
- Added sample weights computed from configuration for high-Tc/high-pressure
  experiments.
- Improved grouped split support and added cross-validation split generation.

### Model

- Added formula branch support to `OccAlignnFullTc`.
- Added optional confidence metadata branch.
- Kept the original A baseline available without confidence metadata.
- Added experimental configs for:
  - A baseline with formula features
  - pressure log1p variants
  - field log1p variants
  - pressure branch / FiLM-style variant
  - weighted loss
  - confidence metadata
  - virtual doping / approximation metadata experiments

### Training And Evaluation

- Added weighted loss support for Huber/Gaussian paths.
- Added evaluator diagnostics:
  - subset metrics
  - grouped metrics by family/match/fidelity/source/Tc/P/H bins
  - worst-error tables
- Added scripts:
  - `scripts/diagnose_results.py`
  - `scripts/make_cv_splits.py`
  - `scripts/summarize_cv_results.py`

### Tests

Added or updated tests for:

- CIF parsing
- formula featurizer
- model forward pass
- weighted training step
- grouped split logic
- cross-validation split generation

Local verification after these changes:

```text
pytest tests
23 passed
```

Remote lightweight verification:

```text
pytest tests/test_model_forward.py tests/test_training_step.py
5 passed
```

## External Traditional ML Reference Metrics

These were used as external comparison points.

| Method / Split | MAE K | R2 | MSLE |
|---|---:|---:|---:|
| 3DSC train -> 3DSC test | 5.499 | 0.485 | 0.773 |
| 3DSC train -> our test | 9.588 | 0.522 | 0.850 |
| our train -> our test | 8.702 | 0.500 | 0.575 |
| our train -> 3DSC test | 6.659 | 0.456 | 1.694 |

## Neural Network Experiment Summary

### Original A Baseline

- Config: `configs/occ_alignn_comp_formula_huber_tc.yaml`
- Run: `/data/home/intern002/ziyang/sc-tc/runs/real_occ_alignn_comp_formula_only_4g_v2_lr3e4`

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 9.919 | 15.298 | 0.739 | 0.547 |
| test | 5.637 | 15.555 | 0.520 | 0.237 |
| test, excluding H3S | 4.317 | 7.173 | 0.813 | 0.216 |
| test, P > 50 GPa | 78.450 | 103.790 | -0.454 | 1.512 |

Interpretation: fixed test performance is relatively optimistic overall, but
high-pressure/H3S behavior is unstable.

### A With P-log1p Only

- Config: `configs/occ_alignn_comp_formula_pressure_log1p_huber_tc.yaml`
- Run: `/data/home/intern002/ziyang/sc-tc/runs/real_occ_alignn_comp_formula_pressure_log1p_4g_lr3e4`

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 9.291 | 14.771 | 0.757 | 0.481 |
| test | 5.602 | 15.889 | 0.499 | 0.241 |

Interpretation: slightly lower MAE than A on the fixed test split, but R2 and
MSLE do not clearly improve. Keep as a candidate, not the main baseline.

### A With P/H Both log1p

- Run: `/data/home/intern002/ziyang/sc-tc/runs/real_occ_alignn_comp_formula_log1p_4g_lr3e4`

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 9.234 | 15.613 | 0.728 | 0.493 |
| test | 5.591 | 16.803 | 0.440 | 0.236 |

Interpretation: test MAE/MSLE are close to A, but R2 is worse.

### A With H-log1p Only

- Run: `/data/home/intern002/ziyang/sc-tc/runs/real_occ_alignn_comp_formula_field_log1p_4g_lr3e4`

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 10.824 | 17.587 | 0.655 | 0.459 |
| test | 14.958 | 21.840 | 0.054 | 0.619 |

Interpretation: harmful. Do not prioritize `log1p(H)` as a replacement.

### A-pressure / FiLM-style Variant

- Run: `/data/home/intern002/ziyang/sc-tc/runs/real_occ_alignn_comp_formula_pressure_4g_lr3e4`

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 9.532 | 16.621 | 0.692 | 0.495 |
| test | 7.270 | 15.766 | 0.507 | 0.249 |

Interpretation: not better than A.

### A With High-Tc/High-Pressure Sample Weighting

- Config: `configs/occ_alignn_comp_formula_weighted_huber_tc.yaml`
- Run: `/data/home/intern002/ziyang/sc-tc/runs/real_occ_alignn_comp_formula_weighted_4g_lr3e4`

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 10.365 | 17.028 | 0.677 | 0.507 |
| test | 7.106 | 18.156 | 0.346 | 0.282 |

Interpretation: worse than A. Reject as the current main direction.

### A With Lightweight Confidence Metadata

- Config: `configs/occ_alignn_comp_formula_confidence_huber_tc.yaml`
- Run: `/data/home/intern002/ziyang/sc-tc/runs/real_occ_alignn_comp_formula_confidence_4g_lr3e4`

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 10.415 | 16.693 | 0.690 | 0.545 |
| test | 6.112 | 16.359 | 0.469 | 0.279 |

Interpretation: overall worse than A. There may be a small sub-signal for
`formula_similarity`, but it is not strong enough to replace A.

## 5-fold Cross Validation

The 5-fold split was generated by grouping on `parent_cif_id`:

```bash
python scripts/make_cv_splits.py \
  --data_csv data/real/tc_data.csv \
  --out_dir data/real/cv5_parent_cif_seed42 \
  --n_folds 5 \
  --group_column parent_cif_id \
  --val_group_frac 0.1 \
  --seed 42
```

Important path fix: fold CSVs live in
`data/real/cv5_parent_cif_seed42`, so relative CIF paths must be rewritten from
`cifs/...` to `../cifs/...`.

Slurm array:

- Job ID: `131654`
- Partition: `gpu4090_128`
- Nodes used: `gpu40904`
- Output root: `/data/home/intern002/ziyang/sc-tc/runs/cv5_a_original_4090`
- Status: completed

### 5-fold Results

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val mean +/- std | 6.868 +/- 2.003 | 11.333 +/- 1.905 | 0.573 +/- 0.183 | 0.474 +/- 0.144 |
| test mean +/- std | 8.809 +/- 1.999 | 16.112 +/- 4.911 | 0.447 +/- 0.211 | 0.618 +/- 0.246 |

Per-fold test metrics:

| Fold | MAE K | RMSE K | R2 | MSLE |
|---:|---:|---:|---:|---:|
| 0 | 10.019 | 21.485 | 0.325 | 0.912 |
| 1 | 7.146 | 11.259 | 0.678 | 0.593 |
| 2 | 11.251 | 20.974 | 0.146 | 0.796 |
| 3 | 9.194 | 15.141 | 0.563 | 0.499 |
| 4 | 6.435 | 11.701 | 0.521 | 0.287 |

Interpretation: the original fixed test split is likely optimistic relative to
parent-CIF grouped CV. The 5-fold average is close to the traditional ML
`our train -> our test` result.

## Data Diagnostics

Diagnostics root:

```text
/data/home/intern002/ziyang/sc-tc/runs/data_diagnostics_cv5_a
```

### Overall Dataset

- Rows: 11846
- Unique formulas: 1759
- Unique parent CIFs: 708

| Condition | Count |
|---|---:|
| P > 0 | 1118 |
| P > 10 GPa | 452 |
| P > 50 GPa | 150 |
| Tc > 40 K | 1195 |
| Tc > 90 K | 169 |
| Tc > 120 K | 34 |
| H3S | 25 |

### Fixed Split Distribution

| Split | n | Tc mean | Tc median | Tc max | P > 50 | Tc > 120 | formulas | parent CIFs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 8844 | 17.276 | 9.400 | 133.000 | 60 | 10 | 1402 | 547 |
| val | 632 | 24.634 | 11.430 | 123.000 | 48 | 1 | 135 | 61 |
| test | 2370 | 25.144 | 32.000 | 203.000 | 42 | 23 | 222 | 141 |

The fixed split is not a random same-distribution split. For example:

| family | test | train | val |
|---|---:|---:|---:|
| magnesium_boride | 1356 | 0 | 0 |
| other | 840 | 4391 | 345 |
| iron_based_or_fe_chalcogenide | 144 | 2821 | 71 |
| cuprate_or_cu_oxide | 23 | 1291 | 163 |

This is not necessarily wrong if the goal is out-of-family generalization, but
the metric must be interpreted as a structured extrapolation test.

### Family-level CV Error

| Family | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| cuprate_or_cu_oxide | 1477 | 19.436 | 27.295 | 0.152 | 0.634 |
| iron_based_or_fe_chalcogenide | 3036 | 9.498 | 12.981 | 0.091 | 0.435 |
| magnesium_boride | 1356 | 6.631 | 7.627 | -0.283 | 0.090 |
| other | 5576 | 5.659 | 15.367 | -0.023 | 0.770 |
| carbon_based | 401 | 4.835 | 7.292 | 0.579 | 0.507 |

The cuprate family is the hardest broad family. MgB2 has moderate MAE but poor
R2 because its internal Tc range is relatively narrow.

### Match/Fidelity CV Error

| Group | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| formula_exact | 8301 | 7.661 | 16.514 | 0.460 | 0.617 |
| formula_similarity | 3539 | 10.287 | 14.703 | 0.362 | 0.496 |
| exact | 8304 | 7.660 | 16.511 | 0.460 | 0.617 |
| synthetic_doped | 3542 | 10.281 | 14.698 | 0.363 | 0.495 |

Synthetic-doped/formula-similarity samples are harder than exact formula
matches, supporting separate reporting for these subsets.

### Tc-bin CV Error

| Tc bin | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| 0-1 | 823 | 3.730 | 6.787 | -697.855 | 1.396 |
| 1-5 | 2821 | 4.048 | 7.887 | -49.691 | 0.703 |
| 5-10 | 2080 | 4.368 | 7.166 | -25.783 | 0.374 |
| 10-20 | 2048 | 7.642 | 9.639 | -10.261 | 0.660 |
| 20-40 | 2879 | 7.777 | 9.927 | -1.525 | 0.171 |
| 40-90 | 1026 | 26.611 | 32.809 | -3.195 | 0.866 |
| 90-120 | 135 | 51.724 | 63.405 | -94.409 | 1.006 |
| >120 | 34 | 121.293 | 131.459 | -19.589 | 2.851 |

The model is much weaker in high-Tc regions, especially above 90 K.

### Pressure-bin CV Error

| Pressure bin | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| 0 | 10728 | 8.125 | 13.888 | 0.540 | 0.576 |
| 0-1 | 163 | 5.563 | 8.157 | 0.857 | 0.350 |
| 1-10 | 503 | 6.781 | 10.951 | 0.654 | 0.479 |
| 10-50 | 302 | 9.897 | 18.519 | 0.184 | 0.716 |
| 50-100 | 69 | 6.692 | 9.648 | -0.086 | 0.629 |
| >100 | 81 | 62.855 | 98.049 | -0.805 | 1.717 |

The `P > 100 GPa` samples are a distinct extrapolation regime and should not be
hidden inside one aggregate metric.

### Worst-error Pattern

Worst CV errors are dominated by high-pressure hydrides and a few high-Tc
outliers:

- PH3 at 200-207 GPa
- H3S at 140-170 GPa
- SbH4 at 184 GPa
- isolated high-Tc entries such as Ca2RuO4 and Bi2212

Worst-error concentration:

| Top-k worst samples | Mean abs error K | Share of total absolute error |
|---:|---:|---:|
| 10 | 196.970 | 0.020 |
| 25 | 165.985 | 0.041 |
| 50 | 126.001 | 0.063 |
| 100 | 102.016 | 0.102 |
| 200 | 83.430 | 0.167 |

The error is not only a handful of samples, but the extreme high-pressure/high-Tc
tail has a large effect on RMSE and R2.

## Exact-first Real-CIF Experiment

This experiment was added to test whether 3DSC-style synthetic/approximate CIFs
were the main reason for weak model performance.

### Dataset

- Local source: `D:\SC-Tc\exact_first_all_current`
- Remote source: `/data/home/intern002/ziyang/sc-tc/data/exact_first_all_current`
- Raw CSV: `tc_data_exact_first_no_leak.csv`
- Cleaned training CSV: `tc_data_exact_first_no_leak_tc_valid.csv`
- CIF files: 918
- Missing referenced CIF paths after upload fix: 0
- Artificial doping included: false
- Rows before filtering missing `Tc_K`: 14823
- Rows after filtering missing `Tc_K`: 14798

The raw no-leak CSV contained 25 rows with missing `Tc_K`:

| Split | Missing Tc_K |
|---|---:|
| train | 11 |
| val | 2 |
| test | 12 |

The cleaned split sizes were:

| Split | Rows |
|---|---:|
| train | 9603 |
| val | 1477 |
| test | 3718 |

### Training Run

- Config: `configs/occ_alignn_comp_formula_huber_tc.yaml`
- Slurm job: `133413`
- Partition: `gpu4090_128`
- Node: `gpu40903`
- Runtime: `06:00:13`
- Early stopping: epoch 25
- Output root:
  `/data/home/intern002/ziyang/sc-tc/runs/exact_first_a_no_leak_tc_valid_4090_lr3e4`

The first attempt, job `133223`, failed after `00:05:55` because the raw CSV
contained missing `Tc_K` values. The cleaned CSV above was used for the final
run.

### Overall Metrics

| Split | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|
| val | 15.079 | 31.352 | -0.087 | 1.109 |
| test | 24.435 | 52.052 | 0.100 | 1.293 |

Comparison with previous neural baselines:

| Version | Test MAE K | Test R2 | Test MSLE |
|---|---:|---:|---:|
| Original A fixed split | 5.637 | 0.520 | 0.237 |
| Original A 5-fold mean | 8.809 | 0.447 | 0.618 |
| Exact-first real-CIF no-leak | 24.435 | 0.100 | 1.293 |

### Test Subset Metrics

| Subset | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| all | 3718 | 24.435 | 52.052 | 0.100 | 1.293 |
| exclude H3S | 3669 | 23.518 | 51.204 | 0.055 | 1.302 |
| exclude Tc > 120 K | 3623 | 20.047 | 32.663 | 0.207 | 1.175 |
| high pressure > 50 GPa | 73 | 61.473 | 74.443 | 0.064 | 1.160 |
| formula_exact | 3718 | 24.435 | 52.052 | 0.100 | 1.293 |
| formula_similarity | 0 | NaN | NaN | NaN | NaN |

### Test Family Metrics

| Family | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| other | 3002 | 19.100 | 52.086 | 0.106 | 1.456 |
| cuprate_or_cu_oxide | 713 | 46.978 | 52.019 | -4.141 | 0.611 |
| carbon_based | 3 | 4.803 | 4.814 | -302.093 | 0.118 |

### Test Tc-bin Metrics

| Tc bin | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| <=1 | 411 | 5.817 | 14.532 | -2709.997 | 1.849 |
| 1-5 | 1261 | 4.803 | 12.168 | -116.061 | 0.877 |
| 5-10 | 444 | 8.365 | 18.172 | -159.970 | 0.677 |
| 10-20 | 325 | 17.345 | 28.838 | -150.672 | 1.146 |
| 20-40 | 120 | 44.924 | 54.015 | -79.317 | 2.419 |
| 40-80 | 356 | 57.059 | 58.367 | -30.749 | 1.550 |
| 80-120 | 706 | 41.257 | 47.782 | -34.757 | 1.242 |
| >120 | 95 | 191.764 | 255.641 | -2.046 | 5.764 |

### Test Pressure-bin Metrics

| Pressure bin | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| 0 | 3266 | 24.880 | 53.720 | 0.058 | 1.283 |
| 0-1 | 98 | 5.740 | 9.818 | 0.673 | 0.553 |
| 1-10 | 182 | 4.852 | 11.422 | 0.529 | 0.644 |
| 10-50 | 97 | 37.349 | 46.549 | -1.519 | 3.615 |
| 50-100 | 18 | 19.839 | 25.089 | -0.377 | 1.641 |
| >100 | 55 | 75.099 | 84.554 | -0.500 | 1.002 |

### Interpretation

This run does not support the hypothesis that 3DSC synthetic CIFs are the main
reason for the weak performance. The exact-first real-CIF no-leak split is much
harder than the previous fixed split and parent-CIF CV baseline, with especially
large errors in high-Tc cuprates and `Tc > 120 K` samples.

The likely issue is still dominated by data distribution and extrapolation
regime:

- the exact-first test split has many high-Tc samples;
- all samples are exact formula matches, so the bad result is not caused by
  synthetic-doped CIFs;
- cuprates and high-pressure/high-Tc cases remain the most difficult subsets;
- real CIFs alone do not solve the high-Tc extrapolation problem.

## Final-clean Exact-CIF Experiment

This experiment used the cleaned real-CIF archive received on 2026-06-02:

```text
C:\Users\Administrator\Desktop\final_clean.rar
```

Only the `01_exact_cif_only/3dsc_formula_group_train_test_split` subset was
used for this run. The `02_exact_plus_3dsc` subset was not mixed into the
exact-only benchmark.

### Data Update

Local data path:

```text
D:\SC-Tc\exact_first_all_current
```

Remote data path:

```text
/data/home/intern002/ziyang/sc-tc/data/exact_first_all_current
```

The previous local and remote data directories were backed up before
replacement:

```text
D:\SC-Tc\exact_first_all_current_backup_20260602_141401
/data/home/intern002/ziyang/sc-tc/data/exact_first_all_current_backup_20260602_141531
```

The archive CSV columns were adapted to the training schema:

| Source column | Training column |
|---|---|
| `primary_cif_relpath` | `cif_path` |
| `tc_k_standardized` | `Tc_K` |
| `pressure_gpa_standardized` | `pressure_GPa` |
| `magnetic_field_t_standardized` | `magnetic_field_T` |

The cleaned package only provided train/test splits. A validation split was
generated from the provided train split using `formula_reduced` groups and
random seed 42. The provided test split was left unchanged.

Validation after local and remote sync:

| Item | Count |
|---|---:|
| total rows | 5877 |
| rows with valid `Tc_K` | 5877 |
| train rows | 4294 |
| val rows | 664 |
| test rows | 919 |
| unique referenced CIF paths | 522 |
| copied CIF files | 1753 |
| missing referenced CIF paths | 0 |

The server upload initially produced Windows-style backslash paths after ZIP
extraction. These were normalized on the server before training, and the final
remote missing-CIF count was 0.

### Training Run

- Config: `configs/occ_alignn_comp_formula_huber_tc.yaml`
- Slurm job: `136307`
- Partition: `gpu4090_128`
- Node: `gpu40903`
- Runtime: `01:44:52`
- Best epoch: 2
- Output root:
  `/data/home/intern002/ziyang/sc-tc/runs/exact_first_a_final_clean_4090_lr3e4`

### Overall Metrics

| Version | Test n | Best epoch | Val MAE K | Val RMSE K | Val R2 | Val MSLE | Test MAE K | Test RMSE K | Test R2 | Test MSLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old exact-first A | 3718 | 13 | 15.079 | 31.352 | -0.087 | 1.109 | 24.435 | 52.052 | 0.100 | 1.293 |
| Final-clean exact A | 919 | 2 | 10.020 | 17.216 | 0.499 | 0.560 | 20.965 | 38.522 | -0.528 | 0.897 |

The final-clean run has lower MAE, RMSE, and MSLE than the old exact-first run,
but the test split is different and smaller. The negative test R2 means it
should not be interpreted as a clean solution to the modeling problem.

### Final-clean Test Subset Metrics

| Subset | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| all | 919 | 20.965 | 38.522 | -0.528 | 0.897 |
| exclude H3S | 919 | 20.965 | 38.522 | -0.528 | 0.897 |
| exclude PH3/SbH4 | 919 | 20.965 | 38.522 | -0.528 | 0.897 |
| exclude hydride high pressure | 919 | 20.965 | 38.522 | -0.528 | 0.897 |
| exclude Tc > 120 K | 919 | 20.965 | 38.522 | -0.528 | 0.897 |
| P > 50 GPa | 2 | 69.720 | 96.495 | -4.406 | 0.531 |

The final-clean test split has no H3S, PH3/SbH4, hydride-high-pressure, MgB2,
or `Tc > 120 K` samples under the current evaluator masks. Its main failure is
therefore not the same hydride tail seen in the older exact-first run.

### Final-clean Tc-bin Metrics

| Tc bin | n | MAE K | RMSE K |
|---|---:|---:|---:|
| <=1 | 43 | 3.066 | 6.447 |
| 1-5 | 282 | 3.879 | 6.294 |
| 5-10 | 123 | 6.172 | 9.506 |
| 10-20 | 103 | 8.670 | 22.469 |
| 20-40 | 157 | 14.547 | 25.008 |
| 40-80 | 82 | 63.733 | 76.201 |
| 80-120 | 129 | 68.827 | 74.363 |

The low-Tc region is much better than the high-Tc region. The model still fails
strongly in `40-80 K` and `80-120 K`.

### Final-clean Pressure-bin Metrics

| Pressure bin | n | MAE K | Max error K |
|---|---:|---:|---:|
| 0 | 748 | 23.303 | 156.244 |
| 0-1 | 43 | 15.346 | 85.344 |
| 1-10 | 103 | 6.834 | 76.259 |
| 10-50 | 23 | 14.479 | 122.507 |
| 50-100 | 2 | 69.720 | 136.432 |

The largest total error is not only a high-pressure issue. Ambient-pressure
cuprate samples also dominate the worst-error table.

### Final-clean Worst-error Pattern

Worst-error concentration:

| Top-k worst samples | Share of total absolute error |
|---:|---:|
| 10 | 0.070 |
| 25 | 0.152 |
| 50 | 0.272 |
| 100 | 0.498 |

Largest chemical-system error groups:

| Chemical system | n | MAE K | Max error K |
|---|---:|---:|---:|
| `Sr2CaCu2(BiO4)2` | 128 | 91.860 | 156.244 |
| `Ag` | 30 | 56.516 | 117.478 |
| `MgO` | 8 | 41.328 | 82.401 |
| `Au` | 11 | 40.498 | 94.967 |
| `FeReAsO` | 3 | 35.734 | 39.468 |

The worst errors are dominated by `Sr2CaCu2(BiO4)2` records. Many of these share
the same reduced formula and CIF but have different `doping_record_original`,
`tc_raw`, pressure, or field values. The current A input does not fully encode
the experimental doping context, so different Tc values can collapse to very
similar model inputs.

Representative worst samples:

| Sample | Chemical system | True Tc K | Pred Tc K | Abs error K | Pressure GPa | Field T | Doping/context |
|---|---|---:|---:|---:|---:|---:|---|
| `record_004639` | `Sr2CaCu2(BiO4)2` | 13.700 | 169.944 | 156.244 | 0.0 | 0.0 | `p=0.06` |
| `record_004640` | `Sr2CaCu2(BiO4)2` | 20.000 | 169.944 | 149.944 | 0.0 | 0.0 | multiple low-p entries |
| `record_000247` | `Sr2CaCu2(BiO4)2` | 20.250 | 166.809 | 146.559 | 0.0 | 3.0 | `x=0` |
| `record_002222` | `Sr2CaCu2(BiO4)2` | 90.000 | 226.432 | 136.432 | 56.0 | 0.0 | high pressure |
| `record_004281` | `Ag` | 120.000 | 2.522 | 117.478 | 0.0 | 0.0 | raw `120 K` |

### Interpretation

The final-clean exact-CIF run suggests that data cleaning helps some aggregate
metrics, but it does not solve the main problem. The remaining dominant issues
are:

- high-Tc cuprates still have large errors;
- the current formula/CIF inputs do not sufficiently distinguish doping level,
  sample preparation, and measurement context;
- a single CIF can correspond to many experimental Tc values, and those values
  can span a large range;
- final-clean real CIFs remove one source of noise, but the model still needs
  better high-Tc/cuprate/doping-aware evaluation and representation.

Current conclusion: CIF cleaning is helpful but not sufficient. The next model
or data direction should focus on high-Tc cuprate regimes and doping/context
features before claiming that CIF quality was the main bottleneck.

## Final-cleanned Exact-CIF Experiment

This experiment used the second cleaned real-CIF archive received on
2026-06-04:

```text
C:\Users\Administrator\Desktop\final_cleanned.rar
```

Only the `final_cleanned/exact_cif_only` subset was used. This run is the
current best exact-CIF neural-network result.

### Data Update

Local data path:

```text
D:\SC-Tc\exact_first_all_current
```

Remote data path:

```text
/data/home/intern002/ziyang/sc-tc/data/exact_first_all_current
```

The previous local data directory was backed up before replacement:

```text
D:\SC-Tc\exact_first_all_current_backup_20260604_203727
```

The second cleaned package already contained the core training columns
`Tc_K`, `pressure_GPa`, `magnetic_field_T`, `formula_standardized`,
`formula_reduced`, `cif_match_type`, and `primary_cif_relpath`. The training
adapter still normalized the schema so that `primary_cif_relpath` became
`cif_path`, and it added compatibility columns expected by the existing
evaluator.

The package provided train/test splits. A validation split was generated from
the provided train split using `formula_reduced` groups and random seed 42. The
provided test split was left unchanged.

Validation after local and remote sync:

| Item | Count |
|---|---:|
| total rows | 5428 |
| rows with valid `Tc_K` | 5428 |
| train rows | 3894 |
| val rows | 501 |
| test rows | 1033 |
| unique referenced CIF paths | 988 |
| copied CIF files | 988 |
| missing referenced CIF paths | 0 |

### Training Run

- Config: `configs/occ_alignn_comp_formula_huber_tc.yaml`
- Slurm job: `138115`
- Partition: `gpu4090_128`
- Node: `gpu40903`
- Output root:
  `/data/home/intern002/ziyang/sc-tc/runs/exact_first_a_final_cleanned_4090_lr3e4`

### Overall Metrics

| Version | Test n | Best epoch | Val MAE K | Val RMSE K | Val R2 | Val MSLE | Test MAE K | Test RMSE K | Test R2 | Test MSLE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Old exact-first A | 3718 | 13 | 15.079 | 31.352 | -0.087 | 1.109 | 24.435 | 52.052 | 0.100 | 1.293 |
| Final-clean exact A | 919 | 2 | 10.020 | 17.216 | 0.499 | 0.560 | 20.965 | 38.522 | -0.528 | 0.897 |
| Final-cleanned exact A | 1033 | 3 | 14.272 | 22.891 | 0.239 | 0.346 | 9.172 | 13.782 | 0.288 | 0.678 |

The second cleaned dataset produced a large improvement over the previous
exact-CIF runs. It also brings the neural model much closer to the traditional
ML baselines, although the test R2 still lags behind.

### Comparison With Traditional ML Reference

| Method / Split | MAE K | R2 | MSLE |
|---|---:|---:|---:|
| 3DSC train -> 3DSC test | 5.499 | 0.485 | 0.773 |
| 3DSC train -> our test | 9.588 | 0.522 | 0.850 |
| our train -> our test | 8.702 | 0.500 | 0.575 |
| our train -> 3DSC test | 6.659 | 0.456 | 1.694 |
| Final-cleanned exact A | 9.172 | 0.288 | 0.678 |

Interpretation:

- MAE is now close to the `3DSC train -> our test` and `our train -> our test`
  traditional ML results.
- MSLE is better than `3DSC train -> our test` but still worse than
  `our train -> our test`.
- R2 remains clearly worse than the traditional ML references, indicating that
  the neural model still does not capture the full trend across regimes.

### Final-cleanned Test Subset Metrics

| Subset | n | MAE K | RMSE K | R2 | MSLE |
|---|---:|---:|---:|---:|---:|
| all | 1033 | 9.172 | 13.782 | 0.288 | 0.678 |
| exclude H3S | 1033 | 9.172 | 13.782 | 0.288 | 0.678 |
| exclude PH3/SbH4 | 1033 | 9.172 | 13.782 | 0.288 | 0.678 |
| exclude hydride high pressure | 1020 | 8.759 | 12.993 | 0.298 | 0.674 |
| exclude Tc > 120 K | 1031 | 9.182 | 13.794 | 0.209 | 0.679 |
| P > 50 GPa | 34 | 21.737 | 29.765 | -0.379 | 0.754 |
| P > 100 GPa | 16 | 33.474 | 38.139 | -1.639 | 0.789 |
| hydride high pressure | 13 | 41.574 | 42.977 | -36.891 | 0.969 |
| exact fidelity | 1033 | 9.172 | 13.782 | 0.288 | 0.678 |

### Final-cleanned Tc-bin Metrics

| Tc bin | n | MAE K | RMSE K |
|---|---:|---:|---:|
| <=1 | 93 | 2.825 | 4.560 |
| 1-5 | 348 | 4.292 | 7.724 |
| 5-10 | 170 | 6.174 | 9.160 |
| 10-20 | 150 | 10.814 | 14.576 |
| 20-40 | 230 | 15.751 | 17.509 |
| 40-80 | 27 | 34.754 | 37.796 |
| 80-120 | 13 | 36.761 | 37.473 |
| >120 | 2 | 3.935 | 4.414 |

The low-Tc region is now substantially better, but the `40-80 K` and `80-120 K`
bins remain difficult.

### Final-cleanned Pressure-bin Metrics

| Pressure bin | n | MAE K | Max error K |
|---|---:|---:|---:|
| 0 | 786 | 9.054 | 46.617 |
| 0-1 | 45 | 7.769 | 28.189 |
| 1-10 | 116 | 7.351 | 48.680 |
| 10-50 | 52 | 8.026 | 47.379 |
| 50-100 | 18 | 11.304 | 63.000 |
| >100 | 16 | 33.474 | 75.288 |

The high-pressure tail is still a distinct weak regime, especially above
100 GPa.

### Final-cleanned Worst-error Pattern

Worst-error concentration:

| Top-k worst samples | Share of total absolute error |
|---:|---:|
| 10 | 0.054 |
| 25 | 0.118 |
| 50 | 0.210 |
| 100 | 0.353 |

Largest chemical-system error groups:

| Chemical system | n | MAE K | Max error K |
|---|---:|---:|---:|
| `SnH4` | 1 | 75.288 | 75.288 |
| `Nd1Fe1As1O0.8F0.2` | 6 | 39.545 | 41.967 |
| `Ba2Y1Cu3O6.8` | 14 | 39.139 | 48.680 |
| `K0.1Ba0.9Fe2As2` | 1 | 38.878 | 38.878 |
| `BiH2` | 13 | 36.370 | 47.801 |
| `Ba1Fe1.9As2Pt0.1` | 4 | 36.201 | 36.704 |
| `La4Ni3O10` | 12 | 34.777 | 63.000 |

Representative worst samples:

| Sample | Chemical system | True Tc K | Pred Tc K | Abs error K | Pressure GPa | Field T |
|---|---|---:|---:|---:|---:|---:|
| `record_008153` | `SnH4` | 72.000 | 147.288 | 75.288 | 180.0 | 0.0 |
| `record_006319` | `La4Ni3O10` | 4.000 | 67.000 | 63.000 | 72.0 | 0.0 |
| `record_003511` | `Ba2Y1Cu3O6.8` | 82.300 | 130.980 | 48.680 | 1.1 | 0.5 |
| `record_002474` | `BiH2` | 70.000 | 22.199 | 47.801 | 159.0 | 0.0 |
| `record_006318` | `La4Ni3O10` | 3.000 | 50.379 | 47.379 | 25.3 | 0.0 |

### Interpretation

The latest cleaned dataset changes the main conclusion:

- data cleaning and dataset quality were a major source of the previously poor
  exact-CIF results;
- after cleaning, the neural model is close to traditional ML on MAE and MSLE;
- however, R2 still lags, so the neural model has not learned the global trend
  as well as `MAGPIE + DSOAP + XGB`;
- high-pressure hydrides and high-Tc cuprate/oxide-like materials remain the
  main weak subsets.

Current conclusion: the poor earlier exact-CIF runs were partly caused by data
quality and split construction. The remaining bottleneck is now more specific:
high-pressure/high-Tc regimes and condition-sensitive extrapolation.

## Implemented Distribution-First Tooling

The following tooling was added to support the next optimization stage without
changing the original A baseline by default.

### Standard Regime Definitions

Shared helpers now define the same bins and subsets for evaluator, diagnostics,
split generation, and sampling:

- Tc bins: `<=1`, `1-5`, `5-10`, `10-20`, `20-40`, `40-80`, `80-120`, `>120`
- pressure bins: `0`, `0-1`, `1-10`, `10-50`, `50-100`, `>100`
- field bins: `0`, `0-0.01`, `0.01-1`, `1-10`, `>10`
- key regimes: MgB2, cuprate, high-Tc, very-high-Tc, high-pressure,
  hydride-high-pressure, formula-exact, formula-similarity, exact fidelity,
  synthetic-doped

### Data Distribution Diagnostics

New script:

```bash
python scripts/diagnose_data_distribution.py \
  --data_csv data/real/tc_data.csv \
  --out_dir runs/data_distribution_real
```

It writes:

- `distribution_summary.csv`
- `family_by_split.csv`
- `tc_bin_by_split.csv`
- `pressure_bin_by_split.csv`
- `regime_by_split.csv`
- `parent_cif_reuse.csv`
- `formula_parent_cif_span.csv`
- `high_pressure_rows.csv`
- `high_pressure_concentration.csv`

### Split Generation

`scripts/make_cv_splits.py` now supports:

- `grouped_cv`: parent-CIF grouped CV, preserving the previous behavior
- `stratified`: one group-preserving split balanced by family/Tc/P bins
- `family_holdout`: hold out any group containing a selected family
- `regime_holdout`: hold out any group containing a selected regime

Examples:

```bash
python scripts/make_cv_splits.py \
  --data_csv data/real/tc_data.csv \
  --out_dir data/real/cv5_parent_cif_seed42 \
  --mode grouped_cv \
  --n_folds 5 \
  --group_column parent_cif_id

python scripts/make_cv_splits.py \
  --data_csv data/real/tc_data.csv \
  --out_dir data/real/split_family_mgb2_holdout \
  --mode family_holdout \
  --holdout_value magnesium_boride \
  --group_column parent_cif_id

python scripts/make_cv_splits.py \
  --data_csv data/real/tc_data.csv \
  --out_dir data/real/split_hydride_hp_holdout \
  --mode regime_holdout \
  --holdout_value hydride_high_pressure \
  --group_column parent_cif_id
```

### Low-risk Training Strategy Configs

New configs:

- `configs/occ_alignn_comp_formula_balanced_sampler_huber_tc.yaml`
- `configs/occ_alignn_comp_formula_mild_weighted_huber_tc.yaml`

The balanced sampler oversamples high-risk regimes while leaving the model and
loss unchanged. Mild weighting uses lower capped sample weights than the earlier
coarse weighted experiment.

Both are opt-in. Original A remains unchanged.

## Current Interpretation

1. The dataset is usable, but a single aggregate score is misleading.
2. The fixed split is a structured extrapolation split rather than a simple
   same-distribution random split.
3. Parent-CIF grouped CV is more realistic for generalization claims.
4. High-pressure hydrides and high-Tc cuprates should be reported separately.
5. `formula_similarity` / `synthetic_doped` samples should be separately
   reported because they are harder and involve approximate structures.
6. MgB2 all in test is not inherently wrong if the goal is out-of-family
   extrapolation, but that must be stated explicitly.
7. Cleaned exact CIFs improve some aggregate metrics but do not remove the
   high-Tc cuprate failure mode.
8. Doping/sample context is likely a first-order missing signal for cuprates:
   multiple records can share a reduced formula and CIF while having very
   different Tc values.
9. The second cleaned exact-CIF dataset substantially improves the neural
   baseline, so data quality was a major factor in the poor earlier exact-CIF
   runs.
10. After the second cleaning, the remaining gap is more concentrated in
    high-pressure hydrides, high-Tc cuprate/oxide-like samples, and global trend
    fitting as reflected by R2.

## Recommended Next Steps

1. Keep original A as the main neural baseline for now.
2. Report fixed split and parent-CIF grouped CV separately.
3. Add mandatory subset tables for:
   - all samples
   - excluding H3S
   - excluding `Tc > 120 K`
   - `P > 50 GPa`
   - `P > 100 GPa`
   - formula_exact vs formula_similarity
   - exact vs synthetic_doped
   - family-level metrics
4. Consider a two-regime evaluation:
   - ambient/low-pressure superconductors
   - high-pressure hydrides
5. Do not prioritize complex CIF-correction models unless reliable paired
   approximate-CIF/true-CIF data is available.
6. For future optimization, focus on data splits and regime-aware reporting
   before making large architecture changes.
7. Add a cuprate/doping-focused diagnostic:
   - group by reduced formula plus CIF;
   - measure Tc span inside each group;
   - identify groups where the input is nearly identical but Tc varies widely.
8. For high-Tc cuprates, test lightweight context features before larger
   architecture changes:
   - parsed doping level from `doping_record_original`;
   - raw formula expansion and composition deltas;
   - measurement criterion and field/pressure context;
   - synthesis/substrate metadata only as a controlled ablation.
9. Treat `final_cleanned` as the current exact-CIF benchmark and compare future
   neural experiments against it, not against the older uncleaned exact-first
   result.
10. Continue reporting traditional ML references in the same table because the
    latest neural MAE is close to those baselines while R2 remains lower.
