Maxent Habitat Suitability Modeling for Giant Panda National Park (GPNP)
=========================================================================

Overview
--------
This directory contains a MaxEnt (Maximum Entropy) habitat suitability
modeling pipeline for the Giant Panda National Park. It uses the elapid
Python library's MaxentModel implementation with spatial-block cross-
validation to produce habitat suitability index (HSI) maps.

The pipeline supports batch runs across multiple background-point sampling
ratios (0.1x to 10x of presence count) with both MCMC-diversity and simple-
random background sampling strategies.


Files
-----
  Maxent.py                       Main training + HSI mapping script
  batch_config_maxent_mcmc.py     Config for MCMC-diversity background points
  batch_config_maxent_random.py   Config for simple-random background points
  batch_run_maxent_mcmc.py        Batch runner (MCMC backgrounds)
  batch_run_maxent random.py      Batch runner (random backgrounds)


Inputs
------
  - Presence point CSV with Longitude/Latitude columns
  - Background point CSV (MCMC or random sampled)
  - Environmental variable rasters (.asc) in a single directory
  - Study area boundary shapefile (.shp)


Outputs (per run)
-----------------
  - HSI GeoTIFF map (hsi_map_*.tif)
  - Saved model (.pkl via joblib)
  - Cross-validation results CSV (per-fold metrics)
  - CV summary CSV (mean/std/min/max/median of key metrics)
  - Final model evaluation CSV


Usage
-----
Single run (edit paths in Maxent.py):
    python Maxent.py

Batch run across background-point ratios:
    python batch_run_maxent_mcmc.py
    python "batch_run_maxent random.py"

Edit the corresponding batch_config_*.py to set data paths, enable/disable
specific ratios, and adjust model hyperparameters.


Model Configuration
-------------------
Key MaxEnt hyperparameters (set in Maxent.py or overridden by batch config):
  - feature_types: ['linear', 'hinge']
  - beta_multiplier: 6.0 (regularization strength)
  - beta_hinge: 5.0
  - n_hinge_features: 3
  - scorer: 'aicc'
  - spatial_block_size: 50 km (for GroupKFold CV)
  - k_folds: 5


Evaluation Metrics
------------------
  - ROC-AUC
  - PR-AUC (and normalized PR-AUC)
  - Continuous Boyce Index (CBI)
  - TSS (True Skill Statistic)
  - Sensitivity / Specificity
  - ECE (Expected Calibration Error)


Requirements
------------
See requirements.txt in this directory. Key dependencies:
  - Python >= 3.9
  - elapid >= 0.5
  - scikit-learn >= 1.3
  - rasterio >= 1.3
  - geopandas >= 0.13


References
----------
  - Phillips et al. (2006). Maximum entropy modeling of species geographic
    distributions. Ecological Modelling 190:231-259.
  - Elith et al. (2011). A statistical explanation of MaxEnt for ecologists.
    Diversity and Distributions 17:43-57.
  - elapid library: https://github.com/earth-chris/elapid
