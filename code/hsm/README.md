# Habitat Suitability Index (HSI) Modeling Scripts

This folder contains twi habitat suitability modeling pipelines. All two scripts share the same
input/output conventions and are evaluated on a consistent set of five metrics:
**ROC-AUC, PR-AUC, TSS, Specificity, and ECE**.

## Scripts

| Script | Model | Notes |
|---|---|---|
| `Maxent.py` | Maximum Entropy (`elapid.MaxentModel`) | Classical SDM benchmark |
| `Simclr.py` | SimCLR encoder + estimator + calibration (+ optional PPM-lite) | Self-supervised representation learning |

## Inputs

Each script expects the same three input artifacts (paths configured at the top
of each script):

1. **Presence points** — CSV with `Longitude` and `Latitude` columns (or a Shapefile).
2. **Background / pseudo-absence points** — CSV with the same columns.
3. **Environmental rasters** — a directory of `.asc` files (one variable per file).
4. **Study area shapefile** — used to clip points and the prediction map.

## Outputs

All scripts write results to `OUTPUT_DIR`:

- `<model>_*.pkl` / `*.pth` — trained model
- `cross_validation_results_*.csv` (or `spatial_cv_results_*.csv`) — per-fold metrics
- `cv_summary_*.csv` — mean ± std over folds (where available)
- `final_model_evaluation_*.csv` (or `all_points_evaluation_*.xlsx`) — full-dataset metrics
- `hsi_map_*.tif` — raw probability map (raster aligned to the env rasters)
- `hsi_map_*_normalized.tif` — min-max normalized version (0–1)

The SimCLR script additionally produces:
- `hsi_prediction_*_ppm_lite.tif` — PPM-lite probability map (if `USE_PPM_LITE`)

## Evaluation metrics

All two scripts report only the following five metrics (plus the TSS-maximizing
threshold used to compute TSS and Specificity):

| Metric | Description | Range | Best |
|---|---|---|---|
| **ROC-AUC** | Area under the ROC curve | 0–1 | 1 |
| **PR-AUC** | Area under the precision-recall curve | 0–1 | 1 |
| **TSS** | True Skill Statistic (sensitivity + specificity − 1) | −1 to 1 | 1 |
| **Specificity** | True negative rate (1 − false positive rate) | 0–1 | 1 |
| **ECE** | Expected Calibration Error | 0–1 | 0 |

## Cross-validation

All two scripts use **K-fold cross-validation** (default `K_FOLDS = 5`). When
`USE_SPATIAL_CV = True`, points are grouped into 50 km × 50 km spatial blocks and
the folds are produced via `GroupKFold`, mitigating spatial autocorrelation
between train and validation sets. Background points are also split per fold to
avoid data leakage.

## Quick start

1. Edit the configuration block at the top of the chosen script:
   - `PRESENCE_POINTS_FILE`
   - `BACKGROUND_POINTS_FILE`
   - `ENV_VAR_RASTER_FILES`
   - `STUDY_AREA_SHP_FILE`
   - `OUTPUT_DIR`
2. Run the script:
   ```bash
   python "Maxent.py"
   python "Simclr.py"
   ```

## Dependencies

Common to all three scripts:

```
numpy
pandas
geopandas
rasterio
scikit-learn
joblib
tqdm
matplotlib
seaborn
openpyxl
```

Maxent only:
```
elapid
statsmodels
```

SimCLR only:
```
torch
scipy
statsmodels
```

Install via:
```bash
pip install numpy pandas geopandas rasterio scikit-learn joblib tqdm matplotlib seaborn openpyxl
pip install elapid statsmodels        # for Maxent
pip install torch scipy               # for SimCLR
```

## Reproducibility
All three scripts set fixed random seeds (`RANDOM_STATE` in each file) to keep
results reproducible across runs.
