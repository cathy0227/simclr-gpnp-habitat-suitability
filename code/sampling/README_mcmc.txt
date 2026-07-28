MCMC Background Point Sampler
==============================

Generates spatially balanced, environmentally diverse pseudo-absence
(background) points for species distribution modeling using Metropolis-Hastings
MCMC sampling.

Designed for the Giant Panda National Park (GPNP) habitat suitability study.

Algorithm
---------
The sampler runs multiple parallel Markov chains with stratified initialization.
Each proposal is accepted based on a weighted combination of:
  1. Environmental diversity (cosine dissimilarity to current position)
  2. Spatial uniformity (grid-density penalty favoring under-sampled cells)
  3. Distance decay (preference for points far from known presences)

Quality evaluation reports grid coverage, coefficient of variation,
nearest-presence distances, and a k-NN clustering index.

Inputs
------
  - Presence points CSV with columns: Longitude, Latitude
  - Study area boundary shapefile (.shp)
  - (Optional) Environmental rasters directory (.tif or .asc files)

Outputs
-------
  - Background points CSV: Longitude, Latitude
  - Console quality metrics (coverage, clustering index, distance stats)

Usage
-----
  # Single run (edit __main__ config section):
  python MCMC_points_generate_revised.py

  # Programmatic usage:
  from MCMC_points_generate_revised import generate_background_points

  generate_background_points(
      presence_csv="path/to/presence.csv",
      shapefile_path="path/to/study_area.shp",
      num_background=1715,
      env_dir="path/to/env_rasters/",
      output_csv="output/background_points.csv",
      env_weight=0.7,
      spatial_balance_weight=0.2,
      distance_weight=0.1,
  )

Key Parameters
--------------
  env_weight             Weight of environmental diversity term (0-1)
  spatial_balance_weight Weight of spatial uniformity term (0-1)
  distance_weight        Weight of distance-from-presence term (0-1)
  step_size              Proposal step size in degrees (None = auto)
  num_chains             Number of parallel Markov chains
  num_steps              Steps per chain
  burn_in                Initial samples to discard per chain
  grid_cells             Grid resolution for density tracking

The three weights should sum to <= 1.0; the residual is a uniform baseline.

Dependencies
------------
See requirements.txt in this directory.

References
----------
- Metropolis-Hastings algorithm for spatial sampling
- Phillips et al. (2009) - Sample selection bias in presence-only SDMs
- Warton & Shepherd (2010) - Poisson point process models for presence-only data
