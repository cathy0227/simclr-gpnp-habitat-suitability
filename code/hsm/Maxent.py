# Maxent habitat suitability modeling script - based on elapid.MaxentModel
import os
import sys
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import glob
from tqdm import tqdm
import elapid as ela
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# --- Configuration (edit paths to match your local layout) ---
date = datetime.now().strftime('%Y%m%d')
points_num = 1715
try_times = str(points_num) + 'p_' + date

PRESENCE_POINTS_FILE = r"data/presence_points.csv"
BACKGROUND_POINTS_FILE = r"data/background_points.csv"

PRESENCE_POINTS_LONGITUDE_COL = 'Longitude'
PRESENCE_POINTS_LATITUDE_COL = 'Latitude'

SKIP_TRAINING = False

ENV_VAR_RASTER_FILES = r"data/env_rasters"
STUDY_AREA_SHP_FILE = r"data/study_area.shp"
OUTPUT_DIR = f"output/Maxent_{try_times}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [Data splitting parameters]
K_FOLDS = 5
RANDOM_STATE = 42
# Spatial block size (km) for GroupKFold
SPATIAL_BLOCK_SIZE_KM = 50

# [Maxent model parameters]
FEATURE_TYPES = ['linear', 'hinge']
TAU = 0.5
CLAMP = True

# Only the global regularization multiplier (RM) is increased above the default
# (elapid default = 1.5); it was selected by spatial-block CV over RM in {1,2,3,4,6}.
BETA_MULTIPLIER = 6.0
# Per-feature-class regularization coefficients kept at the elapid defaults (1.0).
BETA_LQP = 1.0
BETA_HINGE = 1.0
BETA_THRESHOLD = 1.0
BETA_CATEGORICAL = 1.0

N_HINGE_FEATURES = 3
N_THRESHOLD_FEATURES = 0

CONVERGENCE_TOLERANCE = 1e-06
USE_LAMBDAS = 'best'
SCORER = 'aicc'
N_CPUS = -1
MAX_ITERATIONS = 5000

# [HSI mapping parameters]
HSI_BATCH_SIZE_INFERENCE = 1024

# [Auto-generated output file paths]
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, f"maxent_model_kFold_{try_times}.pkl")
HSI_MAP_FILE = os.path.join(OUTPUT_DIR, f"hsi_map_{try_times}.tif")

print(f"Output will be saved to: {os.path.abspath(OUTPUT_DIR)}")


# --- Helper functions ---

def _compute_spatial_blocks(points_gdf: gpd.GeoDataFrame, block_size_km: float, crs_epsg_for_meter: int = 3857):
    """
    Generate spatial block IDs for each point, used for GroupKFold.
    Projects to a metric CRS (default Web Mercator) and assigns grid cell IDs.
    Returns a numpy array of block ID strings (length = number of points).
    """
    if points_gdf is None or points_gdf.empty:
        return None
    try:
        gdf = points_gdf.to_crs(epsg=crs_epsg_for_meter)
    except Exception:
        gdf = points_gdf
    coords = np.stack([gdf.geometry.x.values, gdf.geometry.y.values], axis=1)
    block_m = max(1.0, float(block_size_km) * 1000.0)
    bx = np.floor(coords[:, 0] / block_m).astype(np.int64)
    by = np.floor(coords[:, 1] / block_m).astype(np.int64)
    groups = np.char.add(bx.astype(str), '_')
    groups = np.char.add(groups, by.astype(str))
    return groups


def extract_raster_values_at_points(points_gdf, raster_files):
    """
    Extract raster values at given point locations.
    Returns a DataFrame with raster values indexed to match points_gdf.
    Points outside the raster extent or on NoData cells get NaN.
    """
    all_raster_data = []
    print(f"Extracting values from {len(raster_files)} rasters for {len(points_gdf)} points...")
    for raster_path in raster_files:
        try:
            with rasterio.open(raster_path) as src:
                raster_values = []
                coords = [(p.x, p.y) for p in points_gdf.geometry]
                nodata_val = src.nodatavals[0] if src.nodatavals else None
                for val_array in src.sample(coords):
                    val = val_array[0]
                    if (nodata_val is not None and val == nodata_val) or np.isnan(val):
                        raster_values.append(np.nan)
                    else:
                        raster_values.append(val)
                col_name = os.path.basename(raster_path).split('.')[0]
                all_raster_data.append(pd.Series(raster_values, index=points_gdf.index, name=col_name))
        except Exception as e:
            print(f"Error processing raster {raster_path}: {e}")
            col_name = os.path.basename(raster_path).split('.')[0]
            all_raster_data.append(pd.Series([np.nan] * len(points_gdf), index=points_gdf.index, name=col_name))

    if not all_raster_data:
        return pd.DataFrame(index=points_gdf.index)
    return pd.concat(all_raster_data, axis=1)


def filter_env_rasters_by_model(env_var_raster_filepaths, model_output_path):
    """
    Filter environmental raster files to match the variables used by the saved model.
    """
    model_vars_path = os.path.join(
        os.path.dirname(model_output_path),
        f"model_variables_{os.path.basename(model_output_path).replace('.pkl', '.txt')}"
    )
    if os.path.exists(model_vars_path):
        try:
            with open(model_vars_path, 'r') as f:
                lines = f.readlines()
            model_env_vars = [l.strip().replace('- ', '') for l in lines if l.startswith('- ')]
            if model_env_vars:
                print(f"Loaded model variable list from file: {len(model_env_vars)} variables")
                filtered = [fp for fp in env_var_raster_filepaths
                            if os.path.basename(fp).split('.')[0] in model_env_vars]
                if filtered:
                    print(f"Filtered to {len(filtered)} raster files matching the model variables")
                    return filtered
                else:
                    print("Warning: No matching raster files found for model variables; using all available")
        except Exception as e:
            print(f"Error reading model variables file: {e}")
    return env_var_raster_filepaths


def prepare_and_split_data(presence_points_file, env_var_raster_files, study_area_shp_file, output_dir):
    """
    1. Load presence points and study area.
    2. Clip points to the study area.
    3. Extract environmental variable values at point locations.
    2. Save the final environmental data CSV.
    Returns: (env_data_numpy, valid_points_gdf, env_var_names_list)
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Step 1: Data preparation")

    # Load presence points
    try:
        if presence_points_file.lower().endswith('.csv'):
            df = pd.read_csv(presence_points_file)
            if PRESENCE_POINTS_LONGITUDE_COL not in df.columns or PRESENCE_POINTS_LATITUDE_COL not in df.columns:
                print(f"Error: Longitude column '{PRESENCE_POINTS_LONGITUDE_COL}' or latitude column "
                      f"'{PRESENCE_POINTS_LATITUDE_COL}' not found in CSV.")
                print(f"Available columns: {df.columns.tolist()}")
                return None, None, None
            points_gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[PRESENCE_POINTS_LONGITUDE_COL], df[PRESENCE_POINTS_LATITUDE_COL]),
                crs="EPSG:4326"
            )
            print(f"Loaded presence points from CSV '{presence_points_file}' with CRS EPSG:4326.")
        else:
            points_gdf = gpd.read_file(presence_points_file)
            if points_gdf.crs is None:
                print(f"Warning: CRS not set for '{presence_points_file}'; assuming EPSG:4326.")
                points_gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)

        study_area_gdf = gpd.read_file(study_area_shp_file)
        if study_area_gdf.crs is None:
            print(f"Warning: CRS not set for '{study_area_shp_file}'; assuming EPSG:4326.")
            study_area_gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading input files: {e}")
        return None, None, None

    # Reproject study area to match presence points CRS if needed
    if not study_area_gdf.crs.equals(points_gdf.crs):
        print(f"Reprojecting study area from {study_area_gdf.crs} to {points_gdf.crs}")
        try:
            study_area_gdf = study_area_gdf.to_crs(points_gdf.crs)
        except Exception as e:
            print(f"Error reprojecting study area: {e}")
            return None, None, None

    # Clip presence points to study area
    print("Clipping presence points to study area...")
    points_in_study_area_gdf = gpd.sjoin(points_gdf, study_area_gdf, how="inner", predicate="within")
    if points_in_study_area_gdf.empty:
        print("Error: No presence points found within study area.")
        return None, None, None

    original_cols = [c for c in points_gdf.columns if c in points_in_study_area_gdf.columns]
    points_in_study_area_gdf = points_in_study_area_gdf[list(set(original_cols))].copy()
    points_in_study_area_gdf = points_in_study_area_gdf.drop_duplicates(
        subset=[points_in_study_area_gdf.geometry.name])
    print(f"Found {len(points_in_study_area_gdf)} presence points within study area.")

    # Extract environmental variable values
    print("Extracting environmental variable values at point locations...")
    env_data_df = extract_raster_values_at_points(points_in_study_area_gdf, env_var_raster_files)

    points_in_study_area_gdf = points_in_study_area_gdf.reset_index(drop=True)
    env_data_df = env_data_df.reset_index(drop=True)
    all_points_with_env_gdf = points_in_study_area_gdf.join(env_data_df)

    original_count = len(all_points_with_env_gdf)
    all_points_with_env_gdf.dropna(subset=env_data_df.columns, inplace=True)
    if len(all_points_with_env_gdf) < original_count:
        print(f"Dropped {original_count - len(all_points_with_env_gdf)} points due to missing env values.")

    if len(all_points_with_env_gdf) < 2:
        print("Error: Insufficient valid data points after removing NaN values.")
        return None, None, None

    env_var_names = env_data_df.columns.tolist()

    print(f"\nUsing all {len(env_var_names)} environmental variables for training and prediction.")

    X_env_data = all_points_with_env_gdf[env_var_names].values
    return X_env_data, all_points_with_env_gdf, env_var_names


def _find_max_tss_threshold(presence_predictions, background_predictions):
    """Find threshold maximizing TSS (used internally for sensitivity calculation)."""
    all_preds = np.concatenate([presence_predictions, background_predictions])
    thresholds = np.unique(all_preds)
    if len(thresholds) > 100:
        thresholds = np.percentile(all_preds, np.linspace(0, 100, 101))

    best_tss, best_threshold = -np.inf, 0.5
    for t in thresholds:
        tp = np.sum(presence_predictions >= t)
        tn = np.sum(background_predictions < t)
        sens = tp / len(presence_predictions) if len(presence_predictions) > 0 else 0
        spec = tn / len(background_predictions) if len(background_predictions) > 0 else 0
        tss = sens + spec - 1
        if tss > best_tss:
            best_tss, best_threshold = tss, t
    return best_threshold


def calculate_boyce_index(presence_predictions, background_predictions, num_bins=None):
    """Compute moving-window Continuous Boyce Index following Hirzel et al. (2006).

    The window width is 10% of the background prediction range. One hundred
    equally spaced windows span the pooled presence/background prediction range.
    Windows without background predictions are excluded, and consecutive
    duplicate P/E ratios are collapsed before calculating Spearman correlation.
    ``num_bins`` is retained only for backward-compatible calls.
    """
    _ = num_bins
    presence_predictions = np.asarray(presence_predictions, dtype=float).ravel()
    background_predictions = np.asarray(background_predictions, dtype=float).ravel()

    if (not np.isfinite(presence_predictions).all()
            or not np.isfinite(background_predictions).all()):
        raise ValueError("CBI inputs must be finite; no records were removed silently.")
    if len(presence_predictions) == 0 or len(background_predictions) == 0:
        return np.nan

    score_min = min(presence_predictions.min(), background_predictions.min())
    score_max = max(presence_predictions.max(), background_predictions.max())
    window_width = 0.1 * np.ptp(background_predictions)
    if window_width <= 0 or score_max <= score_min:
        return np.nan

    n_windows = 100
    window_starts = np.linspace(score_min, score_max - window_width, n_windows)
    window_ends = window_starts + window_width
    window_ends[-1] = score_max
    window_centers = (window_starts + window_ends) / 2

    sorted_presence = np.sort(presence_predictions)
    sorted_background = np.sort(background_predictions)
    presence_counts = (
        np.searchsorted(sorted_presence, window_ends, side="right")
        - np.searchsorted(sorted_presence, window_starts, side="left")
    )
    background_counts = (
        np.searchsorted(sorted_background, window_ends, side="right")
        - np.searchsorted(sorted_background, window_starts, side="left")
    )

    predicted_to_expected = np.divide(
        presence_counts / len(presence_predictions),
        background_counts / len(background_predictions),
        out=np.full(n_windows, np.nan),
        where=background_counts > 0,
    )
    predicted_to_expected = np.round(predicted_to_expected, 10)

    valid = np.flatnonzero(np.isfinite(predicted_to_expected))
    if len(valid):
        keep = valid[np.r_[
            True,
            predicted_to_expected[valid][1:] != predicted_to_expected[valid][:-1],
        ]]
    else:
        keep = valid

    if len(keep) < 2 or np.ptp(predicted_to_expected[keep]) <= 0:
        return np.nan

    correlation, _ = spearmanr(
        window_centers[keep], predicted_to_expected[keep]
    )
    return float(correlation) if np.isfinite(correlation) else np.nan



def calculate_metrics(presence_predictions, background_predictions):
    """Compute: ROC-AUC, PR-AUC, Normalized PR-AUC, CBI, Sensitivity (at max-TSS threshold)."""
    presence_predictions = np.array(presence_predictions)
    background_predictions = np.array(background_predictions)
    all_predictions = np.concatenate([presence_predictions, background_predictions])
    all_labels = np.concatenate([np.ones(len(presence_predictions)), np.zeros(len(background_predictions))])

    threshold = _find_max_tss_threshold(presence_predictions, background_predictions)
    # Sensitivity (true positive rate): fraction of presence points correctly
    # classified as suitable at the max-TSS threshold.
    tp = np.sum(presence_predictions >= threshold)
    sensitivity = tp / len(presence_predictions) if len(presence_predictions) > 0 else 0

    auc = roc_auc_score(all_labels, all_predictions)
    pr_auc = average_precision_score(all_labels, all_predictions)

    pi = np.mean(all_labels)
    normalized_pr_auc = (pr_auc - pi) / (1 - pi) if (1 - pi) > 0 else 0.0

    boyce_index = calculate_boyce_index(presence_predictions, background_predictions)

    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'normalized_pr_auc': normalized_pr_auc,
        'boyce_index': boyce_index,
        'sensitivity': sensitivity,
    }


def train_with_cross_validation(all_env_data, env_var_names_list, model_output_path, points_gdf):
    """
    Train MaxEnt with spatial block GroupKFold cross-validation.
    Selects the best fold model based on validation AUC penalized for overfitting,
    then trains a final model on all data.
    Returns the final trained model.
    """
    print("\nStep 2: Spatial block cross-validation training")

    # Load background (pseudo-absence) points
    print("Loading background points...")
    try:
        background_df = pd.read_csv(BACKGROUND_POINTS_FILE)
        background_gdf = gpd.GeoDataFrame(
            background_df,
            geometry=gpd.points_from_xy(background_df[PRESENCE_POINTS_LONGITUDE_COL],
                                        background_df[PRESENCE_POINTS_LATITUDE_COL]),
            crs="EPSG:4326"
        )
        background_env_data = extract_raster_values_at_points(
            background_gdf,
            sorted(glob.glob(os.path.join(ENV_VAR_RASTER_FILES, "*.asc")))
        )
        background_env_data = background_env_data.dropna()

        # Align background variables to training variable set
        if env_var_names_list and set(env_var_names_list) != set(background_env_data.columns.tolist()):
            valid_vars = [v for v in env_var_names_list if v in background_env_data.columns]
            missing = [v for v in env_var_names_list if v not in background_env_data.columns]
            if missing:
                print(f"Warning: Background data missing variables: {missing}")
            background_env_data = background_env_data[valid_vars]

        background_env_array = background_env_data.values
        print(f"Loaded {len(background_env_array)} valid background points")
    except Exception as e:
        print(f"Error loading background points: {e}")
        return None

    # Set up spatial block GroupKFold or fall back to standard KFold
    try:
        groups = _compute_spatial_blocks(points_gdf.reset_index(drop=True), SPATIAL_BLOCK_SIZE_KM)
        unique_groups = np.unique(groups) if groups is not None else []
        if groups is not None and len(unique_groups) >= K_FOLDS:
            print(f"Using spatial block GroupKFold: {len(unique_groups)} groups, {K_FOLDS} folds, "
                  f"block size={SPATIAL_BLOCK_SIZE_KM}km")
            splitter = GroupKFold(n_splits=K_FOLDS)
            split_iter = list(splitter.split(all_env_data, groups=groups))
            use_spatial = True
        else:
            raise ValueError("Insufficient groups for GroupKFold")
    except Exception as e:
        print(f"Spatial block CV failed, falling back to standard KFold: {e}")
        splitter = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = list(splitter.split(all_env_data))
        use_spatial = False

    num_folds = len(split_iter)
    best_score = -float('inf')
    best_model = None
    best_fold = -1
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\nTraining fold {fold + 1}/{num_folds} "
              f"({'spatial block' if use_spatial else 'standard KFold'})...")

        train_data = all_env_data[train_idx]
        val_data = all_env_data[val_idx]

        X_train = np.vstack([train_data, background_env_array])
        y_train = np.concatenate([np.ones(len(train_data)), np.zeros(len(background_env_array))])

        model = ela.MaxentModel(
            feature_types=FEATURE_TYPES, tau=TAU, clamp=CLAMP, scorer=SCORER,
            beta_multiplier=BETA_MULTIPLIER, beta_lqp=BETA_LQP, beta_hinge=BETA_HINGE,
            beta_threshold=BETA_THRESHOLD, beta_categorical=BETA_CATEGORICAL,
            n_hinge_features=N_HINGE_FEATURES, n_threshold_features=N_THRESHOLD_FEATURES,
            convergence_tolerance=CONVERGENCE_TOLERANCE, use_lambdas=USE_LAMBDAS, n_cpus=N_CPUS
        )

        try:
            model.fit(X_train, y_train)

            def predict(m, X):
                return m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') else m.predict(X)

            train_scores = predict(model, train_data)
            val_scores = predict(model, val_data)
            background_scores = predict(model, background_env_array)

            train_m = calculate_metrics(train_scores, background_scores)
            val_m = calculate_metrics(val_scores, background_scores)

            print(f"Fold {fold + 1} results:")
            print(f"  Train - AUC: {train_m['auc']:.4f}, PR-AUC: {train_m['pr_auc']:.4f}, "
                  f"CBI: {train_m['boyce_index']:.4f}, Sensitivity: {train_m['sensitivity']:.4f}")
            print(f"  Val   - AUC: {val_m['auc']:.4f}, PR-AUC: {val_m['pr_auc']:.4f}, "
                  f"CBI: {val_m['boyce_index']:.4f}, Sensitivity: {val_m['sensitivity']:.4f}")

            fold_metrics.append({
                'fold': fold + 1,
                'train_auc': float(train_m['auc']),
                'train_pr_auc': float(train_m['pr_auc']),
                'train_normalized_pr_auc': float(train_m['normalized_pr_auc']),
                'train_boyce_index': float(train_m['boyce_index']),
                'train_sensitivity': float(train_m['sensitivity']),
                'val_auc': float(val_m['auc']),
                'val_pr_auc': float(val_m['pr_auc']),
                'val_normalized_pr_auc': float(val_m['normalized_pr_auc']),
                'val_boyce_index': float(val_m['boyce_index']),
                'val_sensitivity': float(val_m['sensitivity']),
            })

            # Select best model: validation AUC minus overfitting penalty
            overfitting_gap = train_m['auc'] - val_m['auc']
            composite_score = val_m['auc'] - 0.1 * overfitting_gap
            if composite_score > best_score:
                best_score = composite_score
                best_model = model
                best_fold = fold
                print(f"  -> New best model at fold {fold + 1} "
                      f"(val AUC={val_m['auc']:.4f}, gap={overfitting_gap:.4f}, score={composite_score:.4f})")

        except Exception as e:
            print(f"Fold {fold + 1} training failed: {e}")
            continue

    # Save cross-validation results
    if fold_metrics:
        results_df = pd.DataFrame(fold_metrics)
        results_df.to_csv(os.path.join(OUTPUT_DIR, f"cross_validation_results_{try_times}.csv"), index=False)

        print("\nCross-validation summary (validation set):")
        for metric in ['auc', 'pr_auc', 'normalized_pr_auc', 'boyce_index', 'sensitivity']:
            vals = [m[f'val_{metric}'] for m in fold_metrics]
            print(f"  {metric.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Generate cv_summary CSV
        summary_metrics = ['auc', 'pr_auc', 'normalized_pr_auc', 'boyce_index', 'sensitivity']
        metric_display_names = {
            'auc': 'ROC_AUC',
            'pr_auc': 'PR_AUC',
            'normalized_pr_auc': 'NORMALIZED_PR_AUC',
            'boyce_index': 'BOYCE_INDEX',
            'sensitivity': 'SENSITIVITY',
        }
        summary_rows = []
        for metric in summary_metrics:
            vals = np.array([m[f'val_{metric}'] for m in fold_metrics])
            summary_rows.append({
                'Metric': metric_display_names.get(metric, metric.upper()),
                'Mean': round(float(np.mean(vals)), 4),
                'Std_Dev': round(float(np.std(vals)), 4),
                'Min': round(float(np.min(vals)), 4),
                'Max': round(float(np.max(vals)), 4),
                'Median': round(float(np.median(vals)), 4),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUTPUT_DIR, f"cv_summary_{try_times}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nCV summary saved to {summary_path}")

        avg_train_auc = np.mean([m['train_auc'] for m in fold_metrics])
        avg_val_auc = np.mean([m['val_auc'] for m in fold_metrics])
        print(f"\nOverfitting check - Train AUC: {avg_train_auc:.4f}, Val AUC: {avg_val_auc:.4f}, "
              f"Gap: {avg_train_auc - avg_val_auc:.4f}")

    if best_model is not None:
        import joblib
        joblib.dump(best_model, model_output_path)
        print(f"Best model (fold {best_fold + 1}) saved to {model_output_path}")

    # Train final model on all data
    if best_model is not None and background_env_array is not None:
        print("\nTraining final model on all data...")
        final_model = train_final_model_and_evaluate(
            all_env_data, background_env_array, env_var_names_list, model_output_path)
        if final_model is not None:
            return final_model

    return best_model


def train_final_model_and_evaluate(all_env_data, background_env_array, env_var_names_list, model_output_path):
    """
    Train a final MaxEnt model on all data and evaluate on training data (for reference only).
    Saves the model and evaluation metrics to disk.
    Returns the trained model, or None on failure.
    """
    print("\nTraining final MaxEnt model on all data...")

    if background_env_array is None:
        print("Error: Background points required for MaxEnt training.")
        return None

    X_train = np.vstack([all_env_data, background_env_array])
    y_train = np.concatenate([np.ones(len(all_env_data)), np.zeros(len(background_env_array))])
    print(f"Training data: {len(all_env_data)} presence points, {len(background_env_array)} background points")

    final_model = ela.MaxentModel(
        feature_types=FEATURE_TYPES, tau=TAU, clamp=CLAMP, scorer=SCORER,
        beta_multiplier=BETA_MULTIPLIER, beta_lqp=BETA_LQP, beta_hinge=BETA_HINGE,
        beta_threshold=BETA_THRESHOLD, beta_categorical=BETA_CATEGORICAL,
        n_hinge_features=N_HINGE_FEATURES, n_threshold_features=N_THRESHOLD_FEATURES,
        convergence_tolerance=CONVERGENCE_TOLERANCE, use_lambdas=USE_LAMBDAS, n_cpus=N_CPUS
    )

    try:
        final_model.fit(X_train, y_train)
        print("Final MaxEnt model training complete.")

        def predict(m, X):
            return m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') else m.predict(X)

        presence_scores = predict(final_model, all_env_data)
        background_scores = predict(final_model, background_env_array)
        final_m = calculate_metrics(presence_scores, background_scores)

        print("\nFinal model evaluation (training data - for reference only, may be optimistic):")
        print(f"  AUC: {final_m['auc']:.4f}")
        print(f"  PR-AUC: {final_m['pr_auc']:.4f}")
        print(f"  Normalized PR-AUC: {final_m['normalized_pr_auc']:.4f}")
        print(f"  CBI: {final_m['boyce_index']:.4f}")
        print(f"  Sensitivity: {final_m['sensitivity']:.4f}")
        print("Note: Use cross-validation results for unbiased performance estimates.")

        # Save evaluation metrics
        final_evaluation_df = pd.DataFrame({
            'Metric': ['ROC-AUC', 'PR-AUC', 'Normalized_PR-AUC', 'CBI', 'Sensitivity'],
            'Value': [final_m['auc'], final_m['pr_auc'], final_m['normalized_pr_auc'],
                      final_m['boyce_index'], final_m['sensitivity']],
            'Note': ['Area Under ROC Curve', 'Area Under Precision-Recall Curve',
                     'Normalized PR-AUC: (PR-AUC - pi) / (1 - pi)',
                     'Continuous Boyce Index', 'True Positive Rate at max-TSS threshold']
        })
        eval_path = os.path.join(OUTPUT_DIR, f"final_model_evaluation_{try_times}.csv")
        final_evaluation_df.to_csv(eval_path, index=False)
        print(f"Final model evaluation saved to {eval_path}")

        # Save final model
        import joblib
        final_model_path = model_output_path.replace('.pkl', '_final.pkl')
        joblib.dump(final_model, final_model_path)
        print(f"Final model saved to {final_model_path}")

        return final_model

    except Exception as e:
        print(f"Final model training failed: {e}")
        return None


def map_habitat_suitability_maxent(study_area_shp_filepath, env_var_raster_filepaths,
                                   trained_maxent_model, output_hsi_raster_filepath,
                                   env_var_names_ordered_list, model_path=None):
    """
    Generate a habitat suitability index (HSI) map using a trained MaxEnt model.
    Outputs a GeoTIFF raster covering the study area.
    """
    print(f"\nStep 3: Habitat suitability mapping (MaxEnt)")

    if model_path:
        env_var_raster_filepaths = filter_env_rasters_by_model(env_var_raster_filepaths, model_path)

    with rasterio.open(env_var_raster_filepaths[0]) as src_template:
        output_meta = src_template.meta.copy()
        base_transform_affine = src_template.transform
        base_crs_obj = src_template.crs
        height_pixels = src_template.height
        width_pixels = src_template.width

    study_area_geodata = gpd.read_file(study_area_shp_filepath)
    if study_area_geodata.crs != base_crs_obj:
        study_area_geodata = study_area_geodata.to_crs(base_crs_obj)

    # Align raster file list to ordered variable list
    if env_var_names_ordered_list and len(env_var_raster_filepaths) != len(env_var_names_ordered_list):
        print(f"Aligning raster files to {len(env_var_names_ordered_list)} model variables...")
        filtered_raster_files = []
        for var_name in env_var_names_ordered_list:
            match = next((fp for fp in env_var_raster_filepaths
                          if os.path.basename(fp).split('.')[0] == var_name), None)
            if match:
                filtered_raster_files.append(match)
            else:
                print(f"Warning: No raster file found for variable '{var_name}'")
        if len(filtered_raster_files) == len(env_var_names_ordered_list):
            env_var_raster_filepaths = filtered_raster_files
        else:
            print(f"Error: Only found {len(filtered_raster_files)} matching raster files; "
                  f"need {len(env_var_names_ordered_list)}. Aborting mapping.")
            return

    num_vars = len(env_var_raster_filepaths)

    # Create study area mask
    study_area_mask = geometry_mask(
        study_area_geodata.geometry,
        out_shape=(height_pixels, width_pixels),
        transform=base_transform_affine,
        invert=True,
        all_touched=True
    )

    output_meta.update(count=1, dtype='float32', nodata=np.nan)

    # Stack all env rasters into a single array
    stacked_env = np.zeros((num_vars, height_pixels, width_pixels), dtype=np.float32)
    for i, raster_path in enumerate(tqdm(env_var_raster_filepaths, desc="Loading env rasters")):
        with rasterio.open(raster_path) as src:
            img = src.read(1).astype(np.float32)
            nodata_val = src.nodatavals[0] if src.nodatavals else None
            if nodata_val is not None:
                img[img == nodata_val] = np.nan
            img[img == -9999] = np.nan
            stacked_env[i] = img

    study_area_rows, study_area_cols = np.where(study_area_mask)
    print(f"Study area contains {len(study_area_rows)} pixels")
    if len(study_area_rows) == 0:
        print("Warning: Study area mask is empty; cannot compute HSI.")
        return

    # Fill NaN values within study area with 0 before prediction
    filled_env = np.copy(stacked_env)
    for var_idx in range(num_vars):
        nan_mask = np.isnan(filled_env[var_idx])
        filled_env[var_idx, study_area_rows, study_area_cols] = np.where(
            nan_mask[study_area_rows, study_area_cols],
            0.0,
            filled_env[var_idx, study_area_rows, study_area_cols]
        )

    env_data_for_study_area = filled_env[:, study_area_rows, study_area_cols].T
    print(f"Computing MaxEnt HSI for {len(env_data_for_study_area)} study area pixels...")

    all_hsi = []
    for i in tqdm(range(0, len(env_data_for_study_area), HSI_BATCH_SIZE_INFERENCE), desc="HSI batches"):
        batch = env_data_for_study_area[i:i + HSI_BATCH_SIZE_INFERENCE]
        try:
            if hasattr(trained_maxent_model, 'predict_proba'):
                batch_hsi = trained_maxent_model.predict_proba(batch)[:, 1]
            else:
                batch_hsi = trained_maxent_model.predict(batch)
            all_hsi.extend(batch_hsi)
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            all_hsi.extend([0.0] * len(batch))

    hsi_map = np.full((height_pixels, width_pixels), np.nan, dtype=np.float32)
    hsi_map[study_area_rows, study_area_cols] = np.array(all_hsi, dtype=np.float32)

    raw_out = output_hsi_raster_filepath.replace('.tif', '_maxent_raw.tif')
    with rasterio.open(raw_out, 'w', **output_meta) as dst:
        dst.write(hsi_map, 1)
    print(f"MaxEnt raw HSI map saved to {raw_out}")


def main():
    print(f"Output will be saved to: {os.path.abspath(OUTPUT_DIR)}")

    # Discover env variable raster files
    env_raster_directory = ENV_VAR_RASTER_FILES
    actual_env_var_raster_files = sorted(glob.glob(os.path.join(env_raster_directory, "*.asc")))
    print(f"Found {len(actual_env_var_raster_files)} .asc files in '{env_raster_directory}':")
    for f_path in actual_env_var_raster_files:
        print(f"  - {f_path}")

    if SKIP_TRAINING:
        # Mode 1: Skip training, load saved model and run HSI mapping only
        print("\nSkipping training; loading saved model for HSI mapping...")
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"Error: Saved model not found at {MODEL_SAVE_PATH}")
            return

        env_var_names_list = [os.path.basename(fp).split('.')[0] for fp in actual_env_var_raster_files]
        import joblib
        try:
            trained_model = joblib.load(MODEL_SAVE_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        map_habitat_suitability_maxent(
            STUDY_AREA_SHP_FILE, actual_env_var_raster_files,
            trained_model, HSI_MAP_FILE, env_var_names_list,
            model_path=MODEL_SAVE_PATH
        )

    else:
        # Mode 2: Full training and mapping pipeline
        all_env_data_np, all_valid_points_gdf, env_var_names_list = prepare_and_split_data(
            PRESENCE_POINTS_FILE, actual_env_var_raster_files, STUDY_AREA_SHP_FILE, OUTPUT_DIR
        )

        if all_env_data_np is None or len(all_env_data_np) == 0:
            print("Error: Data preparation failed. Exiting.")
            return

        print(f"Number of environmental variables: {all_env_data_np.shape[1]}")
        print(f"Variable names: {env_var_names_list}")

        best_model = train_with_cross_validation(
            all_env_data_np, env_var_names_list, MODEL_SAVE_PATH, all_valid_points_gdf
        )

        if best_model is not None and all_valid_points_gdf is not None and not all_valid_points_gdf.empty:
            map_habitat_suitability_maxent(
                STUDY_AREA_SHP_FILE, actual_env_var_raster_files,
                best_model, HSI_MAP_FILE, env_var_names_list,
                model_path=MODEL_SAVE_PATH
            )
        else:
            print("Warning: No valid model or presence data for HSI mapping.")

    print("\nScript completed.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate required input files and directories
    missing = []
    for f_path in [PRESENCE_POINTS_FILE, STUDY_AREA_SHP_FILE]:
        if not os.path.exists(f_path):
            missing.append(f_path)
    if not os.path.isdir(ENV_VAR_RASTER_FILES):
        missing.append(f"{ENV_VAR_RASTER_FILES} (env raster directory)")
    else:
        if not glob.glob(os.path.join(ENV_VAR_RASTER_FILES, "*.asc")):
            print(f"Warning: No .asc files found in '{ENV_VAR_RASTER_FILES}'.")

    if missing:
        print("Error: The following required inputs were not found:")
        for item in missing:
            print(f"  - {item}")
        print("Please update the configuration paths at the top of this script.")
    else:
        main()
