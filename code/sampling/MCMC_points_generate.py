"""
MCMC-based background point sampler for species distribution modeling.

This module generates spatially balanced and environmentally diverse background
(pseudo-absence) points within a study area using a Metropolis-Hastings random
walk.
 The algorithm ensures that the generated points are:
 1. Spatially balanced (avoid clustering)
 2. Environmentally diverse (based on raster layers)
 3. Uniformly distributed across the study area
 4. Distant from known presence points (if provided)
"""

import os
import random
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import ks_2samp
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class EnhancedMCMCSampler:
    """
    Enhanced MCMC background point sampler with the following features:
    1. Adaptive step size based on study area extent
    2. Environmental constraint sampling (cosine dissimilarity)
    3. Multi-start strategy with stratified initialization
    4. Parallel chain execution
    5. Spatial uniformity constraint via grid-density penalty
    """

    def __init__(self, study_area_shp, env_rasters=None,
                 num_steps=10000, burn_in=2000,
                 step_size_degrees=0.01,
                 env_weight=0.3,
                 spatial_balance_weight=0.4,
                 grid_cells=20,
                 presence_points=None,
                 distance_weight=0.3,
                 distance_scale=0.1):
        """
        Initialize the MCMC sampler.

        Args:
            study_area_shp: Path to the study area shapefile.
            env_rasters: Optional list of environmental raster paths.
            num_steps: Number of steps per Markov chain.
            burn_in: Number of initial samples to discard.
            step_size_degrees: Proposal step size in degrees.
            env_weight: Weight of the environmental term in acceptance prob.
            spatial_balance_weight: Weight of the spatial balance term.
            grid_cells: Number of grid divisions per axis for density tracking.
            presence_points: Optional DataFrame with presence coordinates.
            distance_weight: Weight of the distance-decay term.
            distance_scale: Scale parameter for distance decay.
        """
        self.area_gdf = gpd.read_file(study_area_shp)
        # Merge all geometries into a single (multi)polygon
        self.region = self.area_gdf.unary_union

        # Bounds and center of the study area
        self.bounds = self.region.bounds  # (minx, miny, maxx, maxy)
        self.center = (
            (self.bounds[0] + self.bounds[2]) / 2,
            (self.bounds[1] + self.bounds[3]) / 2
        )

        # Adaptive step size: 0.5% of the smaller side of the bounding box
        region_width = self.bounds[2] - self.bounds[0]
        region_height = self.bounds[3] - self.bounds[1]
        region_size = min(region_width, region_height)
        self.step_size = step_size_degrees or (region_size * 0.005)

        self.num_steps = num_steps
        self.burn_in = burn_in
        self.env_rasters = env_rasters
        self.env_weight = env_weight
        self.spatial_balance_weight = spatial_balance_weight
        self.distance_weight = distance_weight
        self.distance_scale = distance_scale

        # Store presence point coordinates
        if presence_points is not None:
            self.presence_coords = np.array([
                presence_points['Longitude'].values,
                presence_points['Latitude'].values
            ]).T
        else:
            self.presence_coords = None

        # Load environmental raster data
        self.env_data = []
        self.env_transforms = []

        if env_rasters:
            print("Loading environmental rasters...")
            for raster_path in tqdm(env_rasters):
                with rasterio.open(raster_path) as src:
                    self.env_data.append(src.read(1))
                    self.env_transforms.append(src.transform)

        # Spatial grid used to track point density
        self.grid_cells = grid_cells
        self.init_spatial_grid()

        # History of accepted samples (used for spatial-balance evaluation)
        self.sampled_points = []

    def init_spatial_grid(self):
        """Initialize a regular grid for tracking point density."""
        minx, miny, maxx, maxy = self.bounds

        self.grid_x = np.linspace(minx, maxx, self.grid_cells + 1)
        self.grid_y = np.linspace(miny, maxy, self.grid_cells + 1)

        # Per-cell point counter
        self.grid_counts = np.zeros((self.grid_cells, self.grid_cells))

    def _update_grid_counts(self, point):
        """Increment the count of the grid cell containing `point`."""
        x, y = point
        i = np.searchsorted(self.grid_x, x) - 1
        j = np.searchsorted(self.grid_y, y) - 1

        if 0 <= i < self.grid_cells and 0 <= j < self.grid_cells:
            self.grid_counts[i, j] += 1

    def _get_grid_density(self, point):
        """Return the number of accepted points in the cell of `point`."""
        x, y = point
        i = np.searchsorted(self.grid_x, x) - 1
        j = np.searchsorted(self.grid_y, y) - 1

        if 0 <= i < self.grid_cells and 0 <= j < self.grid_cells:
            return self.grid_counts[i, j]
        return 0

    def _spatial_balance_score(self, point):
        """Score for spatial uniformity (low-density cells score higher)."""
        density = self._get_grid_density(point)

        if density == 0:
            return 1.0

        max_density = np.max(self.grid_counts)
        if max_density == 0:
            return 1.0

        # Exponential decay so that high-density cells are strongly penalized
        return np.exp(-2.0 * density / max_density)

    def _propose(self, current):
        """Propose a new sample using a Gaussian random walk."""
        dx = np.random.normal(0, self.step_size)
        dy = np.random.normal(0, self.step_size)
        return np.array([current[0] + dx, current[1] + dy])

    def _get_env_values(self, coords):
        """Return environmental variable values at the given coordinates."""
        if not self.env_rasters:
            return None

        env_values = []
        for env_data, transform in zip(self.env_data, self.env_transforms):
            try:
                row, col = rasterio.transform.rowcol(transform, coords[0], coords[1])
                if 0 <= row < env_data.shape[0] and 0 <= col < env_data.shape[1]:
                    value = env_data[row, col]
                    if np.isnan(value):
                        value = 0
                else:
                    value = 0
            except Exception:
                value = 0
            env_values.append(value)

        return np.array(env_values)

    def _calculate_env_acceptance(self, current_env, proposed_env):
        """Acceptance probability based on environmental dissimilarity.

        Uses cosine distance (`1 - cosine_similarity`) between the current and
        proposed environments. Dissimilar (more diverse) points yield a higher
        acceptance.
        """
        if current_env is None or proposed_env is None:
            return 1.0

        current_env = np.nan_to_num(current_env)
        proposed_env = np.nan_to_num(proposed_env)

        if np.all(current_env == 0) or np.all(proposed_env == 0):
            return 1.0

        similarity = np.dot(current_env, proposed_env) / (
            max(np.linalg.norm(current_env), 1e-10) *
            max(np.linalg.norm(proposed_env), 1e-10)
        )

        # Cosine distance: dissimilar points get a higher score.
        # Clip to [0, 1] (negative similarities -> 1).
        acceptance = 1.0 - similarity
        return max(0.0, min(1.0, acceptance))

    def _calculate_distance_score(self, point):
        """Distance-decay score with respect to presence points.

        Larger distance to the nearest presence point yields a higher score.
        """
        if self.presence_coords is None:
            return 1.0

        distances = np.sqrt(
            np.sum((self.presence_coords - point) ** 2, axis=1)
        )
        min_distance = np.min(distances)
        score = 1 - np.exp(-self.distance_scale * min_distance)
        return score

    def _sample_chain(self, init_point, num_samples, chain_id=0):
        """Run a single Markov chain starting at `init_point`."""
        samples = []
        current = np.array(init_point, dtype=float)
        accepted = 0

        current_env = self._get_env_values(current) if self.env_rasters else None

        desc = f"Chain {chain_id}" if chain_id else "MCMC sampling"
        for i in tqdm(range(self.num_steps), desc=desc, disable=chain_id != 0):
            proposed = self._propose(current)

            # Reject points outside the study area
            p = Point(proposed[0], proposed[1])
            in_region = self.region.contains(p)

            if in_region:
                proposed_env = self._get_env_values(proposed) if self.env_rasters else None

                env_acceptance = self._calculate_env_acceptance(current_env, proposed_env)
                spatial_balance = self._spatial_balance_score(proposed)
                distance_score = self._calculate_distance_score(proposed)

                # Combined acceptance probability
                acceptance_prob = (
                    (1 - self.env_weight - self.spatial_balance_weight - self.distance_weight)
                    + (self.env_weight * env_acceptance)
                    + (self.spatial_balance_weight * spatial_balance)
                    + (self.distance_weight * distance_score)
                )
                acceptance_prob = max(0, min(1, acceptance_prob))

                # Metropolis decision
                if np.random.rand() < acceptance_prob:
                    current = proposed
                    current_env = proposed_env
                    accepted += 1

            # Record samples after burn-in, with thinning to reduce autocorrelation
            if i >= self.burn_in and len(samples) < num_samples:
                thinning = max(1, (self.num_steps - self.burn_in) // (num_samples * 2))
                if i % thinning == 0:
                    samples.append(current.copy())
                    self._update_grid_counts(current)

            if len(samples) >= num_samples:
                break

        acceptance_rate = accepted / self.num_steps
        return np.array(samples), acceptance_rate

    def run_multiple_chains(self, start_points, num_samples, n_workers=4):
        """Run multiple Markov chains, optionally in parallel."""
        samples_per_chain = num_samples // len(start_points)
        all_samples = []

        print(f"Running {len(start_points)} Markov chains, "
              f"{samples_per_chain} samples per chain...")

        if n_workers > 1 and len(start_points) > 1:
            with ProcessPoolExecutor(max_workers=min(n_workers, len(start_points))) as executor:
                futures = []
                for i, start_point in enumerate(start_points):
                    futures.append(
                        executor.submit(
                            self._sample_chain,
                            start_point,
                            samples_per_chain,
                            i + 1
                        )
                    )

                for future in tqdm(futures, desc="Collecting results"):
                    chain_samples, acc_rate = future.result()
                    all_samples.append(chain_samples)
                    print(f"Chain acceptance rate: {acc_rate:.2f}")
        else:
            for i, start_point in enumerate(start_points):
                chain_samples, acc_rate = self._sample_chain(
                    start_point,
                    samples_per_chain,
                    i + 1
                )
                all_samples.append(chain_samples)
                print(f"Chain {i + 1} acceptance rate: {acc_rate:.2f}")

        combined_samples = np.vstack(all_samples)
        if len(combined_samples) > num_samples:
            combined_samples = combined_samples[:num_samples]

        return combined_samples

    def sample(self, num_samples, presence_points=None, n_chains=5, n_workers=4):
        """Generate `num_samples` background points.

        Args:
            num_samples: Number of background points to generate.
            presence_points: Optional DataFrame; used for chain initialization
                and distance-decay scoring.
            n_chains: Number of Markov chains to run.
            n_workers: Number of parallel workers.

        Returns:
            DataFrame with 'Longitude' and 'Latitude' columns.
        """
        if presence_points is not None:
            self.presence_coords = np.array([
                presence_points['Longitude'].values,
                presence_points['Latitude'].values
            ]).T
        else:
            self.presence_coords = None

        self.init_spatial_grid()

        # Stratified start points: split the bounding box into sub-regions
        start_points = []
        minx, miny, maxx, maxy = self.bounds
        n_sections = max(2, int(np.sqrt(n_chains)))
        x_sections = np.linspace(minx, maxx, n_sections + 1)
        y_sections = np.linspace(miny, maxy, n_sections + 1)

        section_indices = [(i, j) for i in range(n_sections) for j in range(n_sections)]
        random.shuffle(section_indices)

        for i, j in section_indices[:n_chains]:
            x_min, x_max = x_sections[i], x_sections[i + 1]
            y_min, y_max = y_sections[j], y_sections[j + 1]

            for _ in range(10):  # Up to 10 attempts per sub-region
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
                p = Point(x, y)
                if self.region.contains(p):
                    start_points.append([x, y])
                    break

        # Optionally seed remaining chains from presence points
        if presence_points is not None and len(presence_points) > 0 and len(start_points) < n_chains:
            n_presence = min(n_chains - len(start_points), len(presence_points))
            presence_indices = np.random.choice(
                len(presence_points),
                size=n_presence,
                replace=False
            )
            for idx in presence_indices:
                start_points.append([
                    presence_points.iloc[idx]['Longitude'],
                    presence_points.iloc[idx]['Latitude']
                ])

        # Fall back to random valid points if still short
        while len(start_points) < n_chains:
            x = np.random.uniform(self.bounds[0], self.bounds[2])
            y = np.random.uniform(self.bounds[1], self.bounds[3])
            p = Point(x, y)
            if self.region.contains(p):
                start_points.append([x, y])

        combined_samples = self.run_multiple_chains(
            start_points,
            num_samples,
            n_workers
        )

        return pd.DataFrame(combined_samples, columns=['Longitude', 'Latitude'])


class BackgroundPointsQualityEvaluator:
    """
    Quality evaluator for background points.

    Reported metrics:
    1. Spatial uniformity (grid coverage, coefficient of variation)
    2. Distance distribution to nearest presence point
    3. Environmental distribution differences (Kolmogorov-Smirnov test)
    4. Spatial clustering index (k-NN based)
    """

    def __init__(self, study_area_shp, env_rasters=None):
        self.area_gdf = gpd.read_file(study_area_shp)
        self.region = self.area_gdf.unary_union
        self.bounds = self.region.bounds

        self.env_data = []
        self.env_transforms = []
        self.env_names = []

        if env_rasters:
            print("Loading environmental rasters for quality evaluation...")
            for raster_path in tqdm(env_rasters):
                try:
                    with rasterio.open(raster_path) as src:
                        self.env_data.append(src.read(1))
                        self.env_transforms.append(src.transform)
                        self.env_names.append(os.path.basename(raster_path).split('.')[0])
                except Exception as e:
                    print(f"Failed to load environmental raster {raster_path}: {e}")

    def _get_env_values_for_points(self, points_df):
        """Sample environmental values at every point in `points_df`."""
        if not self.env_data:
            return None

        env_values = []
        for _, row in points_df.iterrows():
            point_env = []
            for env_data, transform in zip(self.env_data, self.env_transforms):
                try:
                    row_idx, col_idx = rasterio.transform.rowcol(
                        transform, row['Longitude'], row['Latitude']
                    )
                    if 0 <= row_idx < env_data.shape[0] and 0 <= col_idx < env_data.shape[1]:
                        value = env_data[row_idx, col_idx]
                        if np.isnan(value):
                            value = 0
                    else:
                        value = 0
                except Exception:
                    value = 0
                point_env.append(value)
            env_values.append(point_env)

        return np.array(env_values)

    def evaluate_spatial_uniformity(self, background_df, grid_size=20):
        """Coverage ratio and coefficient of variation across a regular grid."""
        minx, miny, maxx, maxy = self.bounds

        x_edges = np.linspace(minx, maxx, grid_size + 1)
        y_edges = np.linspace(miny, maxy, grid_size + 1)
        grid_counts = np.zeros((grid_size, grid_size))

        for _, row in background_df.iterrows():
            x, y = row['Longitude'], row['Latitude']
            i = np.searchsorted(x_edges, x) - 1
            j = np.searchsorted(y_edges, y) - 1
            if 0 <= i < grid_size and 0 <= j < grid_size:
                grid_counts[i, j] += 1

        non_zero_cells = np.sum(grid_counts > 0)
        total_cells = grid_size * grid_size
        coverage_ratio = non_zero_cells / total_cells

        mean_count = np.mean(grid_counts[grid_counts > 0]) if non_zero_cells > 0 else 0
        std_count = np.std(grid_counts[grid_counts > 0]) if non_zero_cells > 0 else 0
        cv = std_count / mean_count if mean_count > 0 else 0

        return {
            'coverage_ratio': coverage_ratio,
            'coefficient_of_variation': cv,
            'grid_counts': grid_counts,
            'x_edges': x_edges,
            'y_edges': y_edges
        }

    def evaluate_distance_to_presence(self, background_df, presence_df):
        """Distance from each background point to its nearest presence point."""
        bg_coords = background_df[['Longitude', 'Latitude']].values
        pr_coords = presence_df[['Longitude', 'Latitude']].values

        nbrs = NearestNeighbors(n_neighbors=1).fit(pr_coords)
        distances, _ = nbrs.kneighbors(bg_coords)
        distances = distances.flatten()

        return {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'distances': distances
        }

    def evaluate_environmental_difference(self, background_df, presence_df):
        """KS-test of environmental distributions between background and presence."""
        if not self.env_data:
            return None

        bg_env = self._get_env_values_for_points(background_df)
        pr_env = self._get_env_values_for_points(presence_df)

        if bg_env is None or pr_env is None:
            return None

        ks_results = []
        for i in range(len(self.env_names)):
            bg_values = bg_env[:, i]
            pr_values = pr_env[:, i]

            bg_valid = bg_values[~np.isnan(bg_values) & (bg_values != 0)]
            pr_valid = pr_values[~np.isnan(pr_values) & (pr_values != 0)]

            if len(bg_valid) > 0 and len(pr_valid) > 0:
                ks_stat, p_value = ks_2samp(bg_valid, pr_valid)
                ks_results.append({
                    'env_name': self.env_names[i],
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'bg_mean': np.mean(bg_valid),
                    'pr_mean': np.mean(pr_valid),
                    'bg_std': np.std(bg_valid),
                    'pr_std': np.std(pr_valid)
                })

        return {
            'ks_results': ks_results,
            'bg_env_values': bg_env,
            'pr_env_values': pr_env
        }

    def evaluate_spatial_clustering(self, background_df, k=5):
        """k-NN clustering index (<1 clustered, >1 dispersed, ~1 random)."""
        coords = background_df[['Longitude', 'Latitude']].values

        if len(coords) < k:
            return {'clustering_index': 0, 'mean_nn_distance': 0}

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        # Drop the self-distance (column 0)
        nn_distances = np.mean(distances[:, 1:], axis=1)
        mean_nn_distance = np.mean(nn_distances)

        # Expected nearest-neighbor distance under a Poisson process
        area = (self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1])
        density = len(coords) / area
        expected_distance = 1 / (2 * np.sqrt(density)) if density > 0 else 0
        clustering_index = mean_nn_distance / expected_distance if expected_distance > 0 else 1

        return {
            'clustering_index': clustering_index,
            'mean_nn_distance': mean_nn_distance,
            'expected_distance': expected_distance,
            'nn_distances': nn_distances
        }

    def comprehensive_evaluation(self, background_df, presence_df):
        """Run the full quality evaluation suite."""
        print("Starting background-point quality evaluation...")

        results = {}

        print("Evaluating spatial uniformity...")
        results['spatial_uniformity'] = self.evaluate_spatial_uniformity(background_df)

        print("Evaluating distance to presence points...")
        results['distance_analysis'] = self.evaluate_distance_to_presence(background_df, presence_df)

        if self.env_data:
            print("Evaluating environmental distribution differences...")
            results['environmental_difference'] = self.evaluate_environmental_difference(
                background_df, presence_df
            )

        print("Evaluating spatial clustering...")
        results['spatial_clustering'] = self.evaluate_spatial_clustering(background_df)

        return results


def evaluate_background_quality(
    shapefile_path,
    presence_csv,
    background_csv,
    env_dir=None
):
    """Run the quality evaluator and return numeric metrics (no plotting)."""
    presence_df = pd.read_csv(presence_csv)
    background_df = pd.read_csv(background_csv)

    env_rasters = None
    if env_dir and os.path.exists(env_dir):
        env_rasters = [
            os.path.join(env_dir, f)
            for f in os.listdir(env_dir)
            if f.endswith(('.tif', '.asc'))
        ][:6]  # Use at most 6 environmental layers for evaluation

    evaluator = BackgroundPointsQualityEvaluator(shapefile_path, env_rasters)
    return evaluator.comprehensive_evaluation(background_df, presence_df)


def generate_background_points(
    presence_csv,
    shapefile_path,
    num_background=5000,
    env_dir=None,
    output_csv='mcmc_background_points.csv',
    num_chains=12,
    step_size=0.02,
    env_weight=0.6,
    spatial_balance_weight=0.6,
    distance_weight=0.3,
    distance_scale=0.05,
    grid_cells=100,
    num_steps=15000,
    burn_in=2000,
    n_workers=4,
    evaluate_quality=True,
    random_seed=42
):
    """
    Generate background points using MCMC sampling.

    Args:
        presence_csv: Path to presence-points CSV (must contain Longitude/Latitude).
        shapefile_path: Path to study-area shapefile.
        num_background: Number of background points to generate.
        env_dir: Optional directory containing environmental rasters.
        output_csv: Output CSV path.
        num_chains: Number of Markov chains.
        step_size: Proposal step size in degrees (None for auto).
        env_weight: Weight of the environmental term (0-1).
        spatial_balance_weight: Weight of the spatial-balance term (0-1).
        distance_weight: Weight of the distance-decay term (0-1).
        distance_scale: Scale parameter for distance decay.
        grid_cells: Grid resolution for density tracking.
        num_steps: Steps per chain.
        burn_in: Number of initial samples to discard.
        n_workers: Number of parallel workers.
        evaluate_quality: Whether to run the numeric quality evaluation.
        random_seed: Random seed for reproducibility.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    presence_df = pd.read_csv(presence_csv)
    print(f"Loaded {len(presence_df)} presence points")

    env_rasters = None
    if env_dir and os.path.exists(env_dir):
        env_rasters = [
            os.path.join(env_dir, f)
            for f in os.listdir(env_dir)
            if f.endswith(('.tif', '.asc'))
        ]
        print(f"Found {len(env_rasters)} environmental raster files")

    sampler = EnhancedMCMCSampler(
        study_area_shp=shapefile_path,
        env_rasters=env_rasters,
        num_steps=num_steps,
        burn_in=burn_in,
        step_size_degrees=step_size,
        env_weight=env_weight,
        spatial_balance_weight=spatial_balance_weight,
        distance_weight=distance_weight,
        distance_scale=distance_scale,
        grid_cells=grid_cells,
        presence_points=presence_df
    )

    background_df = sampler.sample(
        num_samples=num_background,
        presence_points=presence_df,
        n_chains=num_chains,
        n_workers=n_workers
    )

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
    # Configuration
    presence_csv = r".\data\occruence\Rarefy GP occurence example.csv"
    shapefile_path = r"shapfile here"
    env_dir = r".\data\environmental\Ascii"

    presence_df = pd.read_csv(presence_csv)
    num_presence = len(presence_df)
    print(f"Number of presence points: {num_presence}")

    # Background-point counts to generate, expressed as multipliers of presences
    multipliers = [0.1, 0.2, 0.5,1,2,5,10]
    data = '20250505'

    for multiplier in multipliers:
        points_num = num_presence * multiplier
        output_name = f'{points_num}p_{multiplier}x_{data}'
        output_csv = (fr".\data\background\MCMC samples\"
            fr"\background_points_duoyangxing_{output_name}"
            fr"\mcmc_background_points_duoyangxing_{output_name}.csv"
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
            num_chains=30,
            step_size=0.04,
            env_weight=0.7,
            spatial_balance_weight=0.2,
            distance_weight=0.1,
            distance_scale=0.05,
            grid_cells=150,
            num_steps=20000,
            burn_in=2000,
            n_workers=4,
            evaluate_quality=True,
            random_seed=42
        )

        print(f"{multiplier}x background points completed.")
        print(f"Output file: {output_csv}")

    print(f"\n{'=' * 80}")
    print("All background point generation tasks completed.")
    print(f"Generated {len(multipliers)} multiplier-based background point sets.")
    print(f"{'=' * 80}")
