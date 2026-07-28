"""
MCMC background point sampler for species distribution modeling.

Generates spatially balanced, environmentally diverse pseudo-absence points
within a study area using Metropolis-Hastings random walk with:
  - Spatial uniformity (grid-density penalty)
  - Environmental diversity (cosine dissimilarity)
  - Distance decay from presence points

Inputs:
  - Presence point CSV (Longitude, Latitude columns)
  - Study area shapefile (.shp)
  - Optional: directory of environmental rasters (.tif/.asc)

Outputs:
  - Background points CSV (Longitude, Latitude)
  - Quality evaluation metrics printed to console
"""

import os
import random
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class EnhancedMCMCSampler:
    """MCMC background point sampler with adaptive step size, environmental
    constraints, multi-chain stratified starts, and spatial uniformity tracking."""

    def __init__(self, study_area_shp, env_rasters=None,
                 num_steps=10000, burn_in=2000,
                 step_size_degrees=0.01,
                 env_weight=0.3,
                 spatial_balance_weight=0.4,
                 grid_cells=20,
                 presence_points=None,
                 distance_weight=0.3,
                 distance_scale=0.1):
        self.area_gdf = gpd.read_file(study_area_shp)
        self.region = self.area_gdf.unary_union
        self.bounds = self.region.bounds

        # Adaptive step: 0.5% of the smaller bounding-box side
        region_size = min(self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1])
        self.step_size = step_size_degrees or (region_size * 0.005)

        self.num_steps = num_steps
        self.burn_in = burn_in
        self.env_rasters = env_rasters
        self.env_weight = env_weight
        self.spatial_balance_weight = spatial_balance_weight
        self.distance_weight = distance_weight
        self.distance_scale = distance_scale

        if presence_points is not None:
            self.presence_coords = np.column_stack([
                presence_points['Longitude'].values,
                presence_points['Latitude'].values
            ])
        else:
            self.presence_coords = None

        # Load environmental rasters
        self.env_data = []
        self.env_transforms = []
        if env_rasters:
            print("Loading environmental rasters...")
            for raster_path in tqdm(env_rasters):
                with rasterio.open(raster_path) as src:
                    self.env_data.append(src.read(1))
                    self.env_transforms.append(src.transform)

        self.grid_cells = grid_cells
        self.init_spatial_grid()
        self.sampled_points = []

    def init_spatial_grid(self):
        """Initialize grid for tracking point density."""
        minx, miny, maxx, maxy = self.bounds
        self.grid_x = np.linspace(minx, maxx, self.grid_cells + 1)
        self.grid_y = np.linspace(miny, maxy, self.grid_cells + 1)
        self.grid_counts = np.zeros((self.grid_cells, self.grid_cells))

    def _update_grid_counts(self, point):
        x, y = point
        i = np.searchsorted(self.grid_x, x) - 1
        j = np.searchsorted(self.grid_y, y) - 1
        if 0 <= i < self.grid_cells and 0 <= j < self.grid_cells:
            self.grid_counts[i, j] += 1

    def _get_grid_density(self, point):
        x, y = point
        i = np.searchsorted(self.grid_x, x) - 1
        j = np.searchsorted(self.grid_y, y) - 1
        if 0 <= i < self.grid_cells and 0 <= j < self.grid_cells:
            return self.grid_counts[i, j]
        return 0

    def _spatial_balance_score(self, point):
        """Low-density cells score higher (exponential decay penalty)."""
        density = self._get_grid_density(point)
        if density == 0:
            return 1.0
        max_density = np.max(self.grid_counts)
        if max_density == 0:
            return 1.0
        return np.exp(-2.0 * density / max_density)

    def _propose(self, current):
        """Gaussian random-walk proposal."""
        dx = np.random.normal(0, self.step_size)
        dy = np.random.normal(0, self.step_size)
        return np.array([current[0] + dx, current[1] + dy])

    def _get_env_values(self, coords):
        """Extract environmental values at coordinates."""
        if not self.env_rasters:
            return None
        env_values = []
        for env_data, transform in zip(self.env_data, self.env_transforms):
            try:
                row, col = rasterio.transform.rowcol(transform, coords[0], coords[1])
                if 0 <= row < env_data.shape[0] and 0 <= col < env_data.shape[1]:
                    value = env_data[row, col]
                    value = 0 if np.isnan(value) else value
                else:
                    value = 0
            except Exception:
                value = 0
            env_values.append(value)
        return np.array(env_values)

    def _calculate_env_acceptance(self, current_env, proposed_env):
        """Cosine dissimilarity: diverse proposals get higher acceptance."""
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
        return max(0.0, min(1.0, 1.0 - similarity))

    def _calculate_distance_score(self, point):
        """Distance-decay: farther from presence points scores higher."""
        if self.presence_coords is None:
            return 1.0
        distances = np.sqrt(np.sum((self.presence_coords - point) ** 2, axis=1))
        min_distance = np.min(distances)
        return 1 - np.exp(-self.distance_scale * min_distance)

    def _sample_chain(self, init_point, num_samples, chain_id=0):
        """Run a single Markov chain."""
        samples = []
        current = np.array(init_point, dtype=float)
        accepted = 0
        current_env = self._get_env_values(current) if self.env_rasters else None

        for i in tqdm(range(self.num_steps), desc=f"Chain {chain_id}", disable=chain_id != 0):
            proposed = self._propose(current)
            if self.region.contains(Point(proposed[0], proposed[1])):
                proposed_env = self._get_env_values(proposed) if self.env_rasters else None
                acceptance_prob = (
                    (1 - self.env_weight - self.spatial_balance_weight - self.distance_weight)
                    + self.env_weight * self._calculate_env_acceptance(current_env, proposed_env)
                    + self.spatial_balance_weight * self._spatial_balance_score(proposed)
                    + self.distance_weight * self._calculate_distance_score(proposed)
                )
                acceptance_prob = max(0, min(1, acceptance_prob))

                if np.random.rand() < acceptance_prob:
                    current = proposed
                    current_env = proposed_env
                    accepted += 1

            # Record after burn-in with thinning
            if i >= self.burn_in and len(samples) < num_samples:
                thinning = max(1, (self.num_steps - self.burn_in) // (num_samples * 2))
                if i % thinning == 0:
                    samples.append(current.copy())
                    self._update_grid_counts(current)

            if len(samples) >= num_samples:
                break

        return np.array(samples), accepted / self.num_steps

    def trace_chain(self, init_point, seed=None):
        """Run a chain returning the full trace (for convergence diagnostics)."""
        if seed is not None:
            np.random.seed(seed)
        self.init_spatial_grid()

        current = np.array(init_point, dtype=float)
        current_env = self._get_env_values(current) if self.env_rasters else None
        trace = np.empty((self.num_steps, 2), dtype=float)
        accepted = 0
        in_region_count = 0

        for i in range(self.num_steps):
            proposed = self._propose(current)
            if self.region.contains(Point(proposed[0], proposed[1])):
                in_region_count += 1
                proposed_env = self._get_env_values(proposed) if self.env_rasters else None
                acceptance_prob = (
                    (1 - self.env_weight - self.spatial_balance_weight - self.distance_weight)
                    + self.env_weight * self._calculate_env_acceptance(current_env, proposed_env)
                    + self.spatial_balance_weight * self._spatial_balance_score(proposed)
                    + self.distance_weight * self._calculate_distance_score(proposed)
                )
                acceptance_prob = max(0, min(1, acceptance_prob))
                if np.random.rand() < acceptance_prob:
                    current = proposed
                    current_env = proposed_env
                    accepted += 1
                    self._update_grid_counts(current)
            trace[i] = current

        overall_acceptance = accepted / self.num_steps
        in_region_acceptance = accepted / in_region_count if in_region_count else 0.0
        return trace, overall_acceptance, in_region_acceptance

    def run_multiple_chains(self, start_points, num_samples, n_workers=4):
        """Run multiple chains, optionally in parallel."""
        samples_per_chain = num_samples // len(start_points)
        all_samples = []

        print(f"Running {len(start_points)} chains, {samples_per_chain} samples each...")

        if n_workers > 1 and len(start_points) > 1:
            with ProcessPoolExecutor(max_workers=min(n_workers, len(start_points))) as executor:
                futures = [
                    executor.submit(self._sample_chain, sp, samples_per_chain, i + 1)
                    for i, sp in enumerate(start_points)
                ]
                for future in tqdm(futures, desc="Collecting chains"):
                    chain_samples, acc_rate = future.result()
                    all_samples.append(chain_samples)
                    print(f"  Chain acceptance rate: {acc_rate:.2f}")
        else:
            for i, sp in enumerate(start_points):
                chain_samples, acc_rate = self._sample_chain(sp, samples_per_chain, i + 1)
                all_samples.append(chain_samples)
                print(f"  Chain {i + 1} acceptance rate: {acc_rate:.2f}")

        all_samples = [s for s in all_samples if s.ndim == 2 and s.shape[0] > 0]
        if not all_samples:
            raise ValueError("No samples collected. Increase num_steps or reduce burn_in.")

        combined = np.vstack(all_samples)
        return combined[:num_samples] if len(combined) > num_samples else combined

    def sample(self, num_samples, presence_points=None, n_chains=5, n_workers=4):
        """Generate background points. Returns DataFrame (Longitude, Latitude)."""
        if presence_points is not None:
            self.presence_coords = np.column_stack([
                presence_points['Longitude'].values,
                presence_points['Latitude'].values
            ])
        else:
            self.presence_coords = None

        self.init_spatial_grid()

        # Stratified start points from bounding-box sub-regions
        start_points = []
        minx, miny, maxx, maxy = self.bounds
        n_sec = max(2, int(np.sqrt(n_chains)))
        x_sec = np.linspace(minx, maxx, n_sec + 1)
        y_sec = np.linspace(miny, maxy, n_sec + 1)

        indices = [(i, j) for i in range(n_sec) for j in range(n_sec)]
        random.shuffle(indices)

        for i, j in indices[:n_chains]:
            for _ in range(10):
                x = np.random.uniform(x_sec[i], x_sec[i + 1])
                y = np.random.uniform(y_sec[j], y_sec[j + 1])
                if self.region.contains(Point(x, y)):
                    start_points.append([x, y])
                    break

        # Seed remaining from presence points
        if presence_points is not None and len(start_points) < n_chains:
            n_extra = min(n_chains - len(start_points), len(presence_points))
            for idx in np.random.choice(len(presence_points), size=n_extra, replace=False):
                start_points.append([
                    presence_points.iloc[idx]['Longitude'],
                    presence_points.iloc[idx]['Latitude']
                ])

        # Random fallback
        while len(start_points) < n_chains:
            x = np.random.uniform(self.bounds[0], self.bounds[2])
            y = np.random.uniform(self.bounds[1], self.bounds[3])
            if self.region.contains(Point(x, y)):
                start_points.append([x, y])

        combined = self.run_multiple_chains(start_points, num_samples, n_workers)
        return pd.DataFrame(combined, columns=['Longitude', 'Latitude'])


class BackgroundPointsQualityEvaluator:
    """Evaluates spatial uniformity, presence-distance, and clustering."""

    def __init__(self, study_area_shp):
        self.area_gdf = gpd.read_file(study_area_shp)
        self.region = self.area_gdf.unary_union
        self.bounds = self.region.bounds

    def evaluate_spatial_uniformity(self, background_df, grid_size=20):
        minx, miny, maxx, maxy = self.bounds
        x_edges = np.linspace(minx, maxx, grid_size + 1)
        y_edges = np.linspace(miny, maxy, grid_size + 1)
        grid_counts = np.zeros((grid_size, grid_size))

        for _, row in background_df.iterrows():
            i = np.searchsorted(x_edges, row['Longitude']) - 1
            j = np.searchsorted(y_edges, row['Latitude']) - 1
            if 0 <= i < grid_size and 0 <= j < grid_size:
                grid_counts[i, j] += 1

        non_zero = np.sum(grid_counts > 0)
        coverage = non_zero / (grid_size * grid_size)
        mean_c = np.mean(grid_counts[grid_counts > 0]) if non_zero > 0 else 0
        std_c = np.std(grid_counts[grid_counts > 0]) if non_zero > 0 else 0
        cv = std_c / mean_c if mean_c > 0 else 0

        return {'coverage_ratio': coverage, 'coefficient_of_variation': cv}

    def evaluate_distance_to_presence(self, background_df, presence_df):
        bg_coords = background_df[['Longitude', 'Latitude']].values
        pr_coords = presence_df[['Longitude', 'Latitude']].values
        nbrs = NearestNeighbors(n_neighbors=1).fit(pr_coords)
        distances = nbrs.kneighbors(bg_coords)[0].flatten()
        return {
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
        }

    def evaluate_spatial_clustering(self, background_df, k=5):
        coords = background_df[['Longitude', 'Latitude']].values
        if len(coords) < k:
            return {'clustering_index': 0, 'mean_nn_distance': 0}

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
        distances = nbrs.kneighbors(coords)[0]
        mean_nn = np.mean(distances[:, 1:])

        area = (self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1])
        density = len(coords) / area
        expected = 1 / (2 * np.sqrt(density)) if density > 0 else 0
        clustering_index = mean_nn / expected if expected > 0 else 1

        return {'clustering_index': clustering_index, 'mean_nn_distance': mean_nn}

    def comprehensive_evaluation(self, background_df, presence_df):
        results = {}
        results['spatial_uniformity'] = self.evaluate_spatial_uniformity(background_df)
        results['distance_analysis'] = self.evaluate_distance_to_presence(background_df, presence_df)
        results['spatial_clustering'] = self.evaluate_spatial_clustering(background_df)
        return results


def evaluate_background_quality(shapefile_path, presence_csv, background_csv, env_dir=None):
    """Run quality evaluation and return metrics dict."""
    presence_df = pd.read_csv(presence_csv)
    background_df = pd.read_csv(background_csv)
    evaluator = BackgroundPointsQualityEvaluator(shapefile_path)
    return evaluator.comprehensive_evaluation(background_df, presence_df)


def generate_background_points(
    presence_csv, shapefile_path, num_background=5000,
    env_dir=None, output_csv='mcmc_background_points.csv',
    num_chains=12, step_size=0.02,
    env_weight=0.6, spatial_balance_weight=0.6,
    distance_weight=0.3, distance_scale=0.05,
    grid_cells=100, num_steps=15000, burn_in=2000,
    n_workers=4, evaluate_quality=True, random_seed=42
):
    """Generate MCMC background points and optionally evaluate quality."""
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
        print(f"Found {len(env_rasters)} environmental rasters")

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
        quality = evaluate_background_quality(shapefile_path, presence_csv, output_csv, env_dir)

        uniformity = quality['spatial_uniformity']
        dist = quality['distance_analysis']
        clust = quality['spatial_clustering']

        print(f"\nQuality summary:")
        print(f"  Coverage: {uniformity['coverage_ratio']:.2%}, CV: {uniformity['coefficient_of_variation']:.3f}")
        print(f"  Distance to presence: mean={dist['mean_distance']:.3f}, min={dist['min_distance']:.3f}")
        print(f"  Clustering index: {clust['clustering_index']:.3f} "
              f"({'clustered' if clust['clustering_index'] < 0.8 else 'dispersed' if clust['clustering_index'] > 1.2 else 'random'})")

    return background_df


if __name__ == "__main__":
    # --- Configuration (edit paths to match your local layout) ---
    PRESENCE_CSV = r"G:\GIANT PANDA Presidences\maxent filter\GP_spatially rarefy occurence data\Spatially rarefy occurence data GPNP_rarefied_points.csv"
    SHAPEFILE_PATH = r"G:\##GPNP_Shapfile\GPNP_Shapfile.shp"
    ENV_DIR = r"G:\Boshi_Suitability_Model\A_DuiqiDataset\Ascii2015"
    OUTPUT_ROOT = r"G:\Boshi_Suitability_Model\B_Suitablity Model\1.background_points_duoyangxing"

    # Best weights (selected via ROC analysis)
    ENV_WEIGHT = 0.7
    SPATIAL_BALANCE_WEIGHT = 0.2
    DISTANCE_WEIGHT = 0.1

    # Sampler parameters
    DATE_TAG = "20260720"
    NUM_CHAINS = 30
    STEP_SIZE = 0.20936
    DISTANCE_SCALE = 0.05
    GRID_CELLS = 150
    NUM_STEPS = 20000
    BURN_IN = 2000
    N_WORKERS = 4
    RANDOM_SEED = 42

    # Multiplier sweep (ratio of background to presence points)
    MULTIPLIERS = [0.1, 0.2, 0.5, 1, 2, 5, 10]

    presence_df = pd.read_csv(PRESENCE_CSV)
    num_presence = len(presence_df)
    print(f"Presence points: {num_presence}")
    print(f"Weights: env={ENV_WEIGHT}, spatial={SPATIAL_BALANCE_WEIGHT}, distance={DISTANCE_WEIGHT}")

    for mult in MULTIPLIERS:
        points_num = int(num_presence * mult)
        tag = f"{points_num}p_{mult}x_{DATE_TAG}"
        out_dir = os.path.join(OUTPUT_ROOT, f"background_points_duoyangxing_{tag}")
        output_csv = os.path.join(out_dir, f"mcmc_background_points_duoyangxing_{tag}.csv")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Generating {mult}x background points ({points_num})")
        print(f"{'=' * 60}")

        generate_background_points(
            presence_csv=PRESENCE_CSV,
            shapefile_path=SHAPEFILE_PATH,
            num_background=points_num,
            env_dir=ENV_DIR,
            output_csv=output_csv,
            num_chains=NUM_CHAINS,
            step_size=STEP_SIZE,
            env_weight=ENV_WEIGHT,
            spatial_balance_weight=SPATIAL_BALANCE_WEIGHT,
            distance_weight=DISTANCE_WEIGHT,
            distance_scale=DISTANCE_SCALE,
            grid_cells=GRID_CELLS,
            num_steps=NUM_STEPS,
            burn_in=BURN_IN,
            n_workers=N_WORKERS,
            evaluate_quality=True,
            random_seed=RANDOM_SEED,
        )

    print(f"\nAll {len(MULTIPLIERS)} background point sets generated.")
    print(f"Output: {OUTPUT_ROOT}")
