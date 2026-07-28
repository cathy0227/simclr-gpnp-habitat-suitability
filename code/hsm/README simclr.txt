SimCLR-HSI: Habitat Suitability Index Estimation for GPNP
==========================================================

Overview
--------
This script estimates habitat suitability for the Giant Panda National Park
(GPNP) using environmental raster data and a SimCLR-based self-supervised
learning framework.

The pipeline:
1. Pre-trains a neural encoder via SimCLR contrastive learning with SCARF
   feature corruption (Bahri et al. 2021) on all presence + background points.
2. Freezes the encoder and trains a Random Forest classifier on the learned
   representations under spatial-block cross-validation.
3. Applies Platt-scaling (sigmoid) calibration to RF outputs.
4. Produces a continuous HSI map via PPM-lite (Point Process Model lite):
   the calibrated logit is used as relative intensity, then normalized to
   [0, 1] via quantile ranking.

Inputs
------
- Presence point CSV (columns: Longitude, Latitude)
- Background point CSV (MCMC-sampled; columns: Longitude, Latitude)
- Environmental variable rasters (.asc files in a directory)
- Study area boundary shapefile (.shp)

Outputs
-------
- *_ppm_lite.tif        PPM-lite habitat suitability GeoTIFF
- best_model_*.pth      Model checkpoint (SimCLR encoder + RF + calibrator)
- rf_classifier_*.joblib   Trained Random Forest classifier
- spatial_cv_results_*.csv   Per-fold cross-validation metrics
- cv_summary_*.csv      Mean/std summary of CV metrics
- all_points_evaluation_*.xlsx/csv   Final evaluation on full dataset

Evaluation Metrics
------------------
- ROC-AUC: Area Under ROC Curve
- PR-AUC:  Area Under Precision-Recall Curve
- CBI:     Continuous Boyce Index (Spearman correlation of P/E ratio)
- Sensitivity: True Positive Rate at max-TSS threshold

Usage
-----
  python "simclr_hsi_model3_v2_rf_calibrated shaixuan_buchongfu.py"

Optional CLI arguments:
  no_filter              Disable variable selection
  method=<m>            Variable selection method (correlation|vif|combined)
  corr_threshold=<v>    Correlation threshold (default: 0.8)
  vif_threshold=<v>     VIF threshold (default: 10)

For batch runs across multiple background-point ratios:
  python batch_run_simclr.py
  (configure ratios and paths in batch_config_simclr.py)

Requirements
------------
See requirements.txt. Tested with Python 3.9, PyTorch 1.12+, CUDA 11.x.

References
----------
- Chen et al. (2020) "A Simple Framework for Contrastive Learning of
  Visual Representations" (SimCLR)
- Bahri et al. (2021) "SCARF: Self-Supervised Contrastive Learning using
  Random Feature Corruption"
- Renner et al. (2015) Point Process Models for presence-only analysis
