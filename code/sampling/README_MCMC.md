# MCMC Background Point Sampler

`MCMC_points_generate.py` generates spatially balanced and environmentally
diverse background (pseudo-absence) points for species distribution modeling
using a Metropolis-Hastings random walk over a study-area polygon.

This module also exposes the shared quality-evaluation toolkit used by the
companion `Random_points_generate.py` script.

---

## Inputs

| Argument          | Format     | Description                                                              |
|-------------------|------------|--------------------------------------------------------------------------|
| `presence_csv`    | CSV        | Presence points with `Longitude` and `Latitude` columns.                 |
| `shapefile_path`  | Shapefile  | Study area polygon.                                                      |
| `env_dir`         | Directory  | Optional. Folder of `.tif` / `.asc` environmental rasters (same extent). |
| `num_background`  | int        | Number of background points to generate.                                 |

## Outputs

A single CSV with `Longitude` / `Latitude` columns is written to the path
given by `output_csv`. The console additionally prints the four quality
metrics summarized at the end of this document.

---

## Method

Each Metropolis-Hastings proposal is evaluated against four terms, combined
into a linear weighted acceptance probability:

```
acceptance = env_weight              * env_score
           + spatial_balance_weight  * spatial_balance_score
           + distance_weight         * distance_score
```

1. **Spatial validity** — the proposal must lie inside the study-area polygon
   (hard constraint; out-of-area proposals are rejected outright).
2. **Environmental dissimilarity** (`env_weight`) — points whose environmental
   conditions differ from the current sample are preferred. Computed as
   `1 - cosine_similarity` (cosine distance) over the environmental raster
   stack.
3. **Spatial balance** (`spatial_balance_weight`) — a regular grid tracks
   density, and proposals in low-density cells are favored via
   `exp(-2 * density / max_density)`.
4. **Distance to presence points** (`distance_weight`) — proposals far from
   any presence point are favored via `1 - exp(-distance_scale * min_dist)`.

Multiple chains run in parallel with stratified initial points. Burn-in
samples are discarded and the remaining samples are thinned to reduce
autocorrelation.

---

## Usage

```python
from MCMC_points_generate import generate_background_points

generate_background_points(
    presence_csv="presence.csv",
    shapefile_path="study_area.shp",
    num_background="number of background here",
    env_dir="env_rasters/",
    output_csv="mcmc_background_points.csv",
    num_chains="number of chains", # 12
    n_workers="number of workers", # 4
)
```

Running the script directly (`python MCMC_points_generate.py`) executes the
example pipeline at the bottom of the file, generating background-point sets
at several multipliers of the presence count.

---

## Quality metrics

After sampling, the script prints a console summary of:

| Metric                          | Meaning                                                       |
|---------------------------------|---------------------------------------------------------------|
| Coverage ratio                  | Fraction of grid cells containing at least one point.         |
| Coefficient of variation        | Per-cell density spread (smaller = more uniform).             |
| Distance to nearest presence    | Mean / median / min distance to the nearest presence point.   |
| Clustering index (k-NN based)   | <1 clustered, ~1 random, >1 dispersed.                        |
| KS test per environmental layer | Distribution difference between background and presence sets. |

---

## Dependencies

`geopandas`, `pandas`, `numpy`, `rasterio`, `shapely`, `scipy`,
`scikit-learn`, `tqdm`.
