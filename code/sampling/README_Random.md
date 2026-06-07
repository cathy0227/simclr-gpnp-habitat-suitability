# Random Background Point Sampler

`Random_points_generate.py` generates background (pseudo-absence) points for
species distribution modeling using simple uniform random sampling inside a
study-area polygon. It reuses the shared quality-evaluation toolkit from
`MCMC_points_generate.py`, so the two files must stay in the same directory.

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

Each candidate is drawn uniformly inside the bounding box of the study-area
polygon and rejected if it falls outside the polygon. An optional
`min_distance_to_presence` enforces a minimum distance (in degrees) to every
presence point.

---

## Usage

```python
from Random_points_generate import generate_background_points

generate_background_points(
    presence_csv="presence.csv",
    shapefile_path="study_area.shp",
    num_background="number of background here",
    env_dir="env_rasters/",
    output_csv="random_background_points.csv",
    min_distance_to_presence=None,
)
```

Running the script directly (`python Random_points_generate.py`) executes
the example pipeline at the bottom of the file, generating background-point
sets at several multipliers of the presence count.

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

The quality evaluator is imported from `MCMC_points_generate.py`, which must
be present in the same directory.
