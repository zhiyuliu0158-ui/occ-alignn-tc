# XGBoost Tc regression: 19,894-record SC-Atlas result

This directory contains the code snapshot and result files for the paper-scale
XGBoost Tc regression experiment on 19,894 SC-Atlas records.

## Scope

- Input public core data: `../data/sc-atlas-data.csv` (20,119 records).
- Final analysis data: 19,894 records after excluding 225 high-pressure
  hydride records using `code/filter_high_pressure_hydrides.py`.
- Features: 131 MAGPIE composition descriptors and 46 experimental-condition
  descriptors (pressure, magnetic field, missingness indicators, Tc criterion,
  and Physical Substrate), for 177 features in total.
- No CIF, crystal-structure, eDOS, phDOS, DSOAP, or ALIGNN-DOS output is used
  for this result.
- Validation: 10 repeated 80/20 GroupShuffleSplit partitions grouped by
  complete chemical system, with zero train/test chemical-system overlap.

## Main result

Unweighted test metrics reported as the mean +/- sample standard deviation over
the ten group-held-out partitions:

| Metric | Result |
|---|---:|
| MAE | 5.739 +/- 1.524 K |
| RMSE | 10.248 +/- 2.639 K |
| Raw-Tc R2 | 0.785 +/- 0.042 |
| log1p(Tc) R2 | 0.784 +/- 0.024 |
| MSLE | 0.259 +/- 0.042 |

The model is trained on a restricted `arcsinh(Tc)` target, not directly on
`log1p(Tc)`. The `log1p(Tc)` R2 is an evaluation metric.

## Contents

- `code/filter_high_pressure_hydrides.py`: creates the 19,894-record study
  scope from the core table.
- `code/prepare_sc_atlas_current_model_input.py`: prepares model input fields.
- `code/run_xgb_3dsc_with_edos_current_split.py`: XGBoost training and
  evaluation script. The historical filename is retained for traceability;
  this result uses its `MAGPIE` feature set only.
- `code/evaluate_final_cleanned_3dsc_official.py` and
  `code/download_mp_exact_cifs.py`: exact helper-code snapshots required by
  the training script.
- `results/`: parameters, feature manifest, 177-feature list, per-split
  metrics, split audit, aggregate metrics, paper statistics, paper-ready
  methods, and all held-out predictions.

The full reproducibility package originally also included a derived 19,894-row
CSV and the removed-hydride audit tables. They are intentionally not duplicated
here: the uploaded core table and the included filter script regenerate the
study-scope data.
