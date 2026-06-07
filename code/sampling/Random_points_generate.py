"""
Random background point sampler for species distribution modeling.

Generates background (pseudo-absence) points by uniform random sampling
inside the study-area polygon. Optionally enforces a minimum-distance
constraint to known presence points. Reuses the quality-evaluation toolkit
from `MCMC_points_generate`.
"""

import os
import random
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point
from tqdm import tqdm

# Reuse the evaluator from the MCMC module instead of duplicating ~200 lines
from MCMC_points_generate import evaluate_background_quality

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class RandomSampler:
    """
    Simple random background point sampler.

    Samples uniformly inside the study-area polygon, optionally with a
    minimum-distance constraint to presence points.
    """

    def __init__(self, study_area_shp, presence_points=None,
                 min_distance_to_presence=None):
        """
        Args:
            study_area_shp: Path to the study area shapefile.
            presence_points: Optional DataFrame with Longitude/Latitude.
            min_distance_to_presence: Optional minimum distance (in degrees)
                that every background point must keep from any presence point.
        """
        self.area_gdf = gpd.read_file(study_area_shp)
        # Merge all geometries into a single (multi)polygon
        self.region = self.area_gdf.unary_union

        # Bounding box of the study area
        self.bounds = self.region.bounds  # (minx, miny, maxx, maxy)

        # Presence point coordinates (used for distance constraint)
        if presence_points is not None:
            self.presence_coords = np.array([
                presence_points['Longitude'].values,
                presence_points['Latitude'].values
            ]).T
        else:
            self.presence_coords = None

        self.min_distance_to_presence = min_distance_to_presence

    def sample(self, num_samples, max_attempts=100000):
        """Simple random sampling inside the study area polygon."""
        samples = []
        attempts = 0

        print(f"Starting simple random sampling, target: {num_samples} samples")

        with tqdm(total=num_samples, desc="Random sampling") as pbar:
            while len(samples) < num_samples and attempts < max_attempts:
                # Sample uniformly inside the bounding box
                x = np.random.uniform(self.bounds[0], self.bounds[2])
                y = np.random.uniform(self.bounds[1], self.bounds[3])

                point = Point(x, y)

                # Reject points outside the study area
                if self.region.contains(point):
                    if self._check_distance_constraint([x, y]):
                        samples.append([x, y])
                        pbar.update(1)

                attempts += 1

        if len(samples) < num_samples:
            print(f"Warning: only {len(samples)} samples generated "
                  f"(target was {num_samples})")

        return pd.DataFrame(samples, columns=['Longitude', 'Latitude'])

    def _check_distance_constraint(self, point):
        """Return True if `point` satisfies the minimum-distance constraint."""
        if self.presence_coords is None or self.min_distance_to_presence is None:
            return True

        distances = np.sqrt(
            np.sum((self.presence_coords - point) ** 2, axis=1)
        )
        return np.min(distances) >= self.min_distance_to_presence


def generate_background_points(
    presence_csv,
    shapefile_path,
    num_background=5000,
    env_dir=None,
    output_csv='random_background_points.csv',
    min_distance_to_presence=None,      # Min. distance constraint (degrees)
    evaluate_quality=True,
    random_seed=42
):
    """
    Generate background points using simple random sampling.

    Args:
        presence_csv: Path to presence-points CSV (Longitude/Latitude columns).
        shapefile_path: Path to study-area shapefile.
        num_background: Number of background points to generate.
        env_dir: Optional directory containing environmental rasters
            (used only for quality evaluation).
        output_csv: Output CSV path.
        min_distance_to_presence: Optional minimum distance to presence points.
        evaluate_quality: Whether to run the numeric quality evaluation.
        random_seed: Random seed for reproducibility.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    presence_df = pd.read_csv(presence_csv)
    print(f"Loaded {len(presence_df)} presence points")

    sampler = RandomSampler(
        study_area_shp=shapefile_path,
        presence_points=presence_df,
        min_distance_to_presence=min_distance_to_presence
    )

    print(f"Generating {num_background} background points (simple random)...")
    background_df = sampler.sample(num_samples=num_background)

    background_df.to_csv(output_csv, index=False)
    print(f"Generated {len(background_df)} background points -> {output_csv}")

    if evaluate_quality:
        print("\n" + "=" * 50)
        print("Starting background point quality evaluation...")
        print("=" * 50)

        quality_results = evaluate_background_quality(
            shapefile_path=shapefile_path,
            presence_csv=presence_csv,
            background_csv=output_csv,
            env_dir=env_dir
        )

        print("\n" + "=" * 50)
        print("Quality evaluation summary:")
        print("=" * 50)

        uniformity = quality_results['spatial_uniformity']
        print("Spatial uniformity:")
        print(f"   - Grid coverage: {uniformity['coverage_ratio']:.2%}")
        print(f"   - Coefficient of variation: {uniformity['coefficient_of_variation']:.3f}")

        distance_stats = quality_results['distance_analysis']
        print("\nDistance to presence points:")
        print(f"   - Mean: {distance_stats['mean_distance']:.3f}")
        print(f"   - Median: {distance_stats['median_distance']:.3f}")
        print(f"   - Min: {distance_stats['min_distance']:.3f}")

        clustering = quality_results['spatial_clustering']
        print("\nSpatial clustering:")
        print(f"   - Clustering index: {clustering['clustering_index']:.3f}")
        if clustering['clustering_index'] < 0.8:
            print("   - Interpretation: clustered distribution")
        elif clustering['clustering_index'] > 1.2:
            print("   - Interpretation: dispersed distribution")
        else:
            print("   - Interpretation: approximately random distribution")

        if quality_results.get('environmental_difference'):
            env_diff = quality_results['environmental_difference']
            ks_results = env_diff['ks_results']
            significant_count = sum(1 for r in ks_results if r['p_value'] < 0.05)

            print("\nEnvironmental distribution differences:")
            print(f"   - Total environmental variables: {len(ks_results)}")
            print(f"   - Significant (p<0.05): {significant_count}")
            print(f"   - Significant ratio: {significant_count / len(ks_results):.1%}")

            if significant_count > len(ks_results) * 0.7:
                print("   - Verdict: large environmental differences (good)")
            elif significant_count > len(ks_results) * 0.3:
                print("   - Verdict: moderate environmental differences")
            else:
                print("   - Verdict: small environmental differences (warning)")

        print("\n" + "=" * 50)
        print("Quality evaluation complete.")
        print("=" * 50)

    return background_df


if __name__ == "__main__":
    # Configuration (same inputs as the MCMC script)
    presence_csv = r".\data\occruence\Rarefy GP occurence example.csv"
    shapefile_path = r"shapfile here"
    env_dir = r".\data\environmental\Ascii"

    presence_df = pd.read_csv(presence_csv)
    num_presence = len(presence_df)
    print(f"Number of presence points: {num_presence}")

    # Background-point counts to generate, expressed as multipliers of presences
    multipliers = [0.1, 0.2, 0.5,1,2,5,10]
    date = datetime.now().strftime('%Y%m%d-%H%M')

    for multiplier in multipliers:
        points_num = num_presence * multiplier
        output_name = f'{points_num}p_{multiplier}x_{date}'
        output_csv = (
            fr".\data\background\random samples"
            fr"\background_points_random_{output_name}"
            fr"\random_background_points_{output_name}.csv"
        )
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Generating {multiplier}x background points ({points_num} points)")
        print(f"{'=' * 60}")

        generate_background_points(
            presence_csv=presence_csv,
            shapefile_path=shapefile_path,
            num_background=points_num,
            env_dir=env_dir,
            output_csv=output_csv,
            min_distance_to_presence=None,  # No distance constraint
            evaluate_quality=True,
            random_seed=42
        )

        print(f"{multiplier}x background points completed.")
        print(f"Output file: {output_csv}")

    print(f"\n{'=' * 80}")
    print("All random-sampling background point generation tasks completed.")
    print(f"Generated {len(multipliers)} multiplier-based sets.")
    print(r"All files saved under: G:\Boshi_Suitability_Model\B_Suitablity Model\1.background_points_random\\")
    print(f"{'=' * 80}")
