"""
SimCLR-HSI: Habitat Suitability Index estimation for Giant Panda National Park (GPNP).

Uses SimCLR self-supervised contrastive learning (with SCARF feature corruption) to
pre-train an encoder on environmental raster variables, then trains a Random Forest
classifier with Platt-scaling calibration under spatial-block cross-validation.


Inputs:
    - Presence point CSV (Longitude, Latitude)
    - Background point CSV (MCMC-sampled)
    - Environmental variable rasters (.asc) in a directory
    - Study area shapefile (.shp)

Outputs:
    - HSI GeoTIFF (*_ppm_lite.tif)
    - Best model checkpoint (.pth)
    - RF classifier (.joblib)
    - Cross-validation results (CSV) and evaluation report (Excel/CSV)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import rasterio
from rasterio.features import geometry_mask
import os
from tqdm import tqdm
import glob
import geopandas as gpd
from scipy.stats import spearmanr
import random
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime


def print_usage():
    print("=" * 80)
    print("SimCLR-HSI Model Training Script")
    print("=" * 80)
    print("Command line arguments:")
    print("  no_filter                    - Disable variable filtering")
    print("  method=<method>              - correlation | vif | combined")
    print("  corr_threshold=<value>       - Correlation threshold (default: 0.8)")
    print("  vif_threshold=<value>        - VIF threshold (default: 10)")
    print("=" * 80)


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


def _calculate_vif(df, threshold=10):
    """Iteratively remove variables with VIF > threshold."""
    X = df.copy()
    removed_vars = []
    vif_data = []
    max_iter = 100
    iter_count = 0
    max_vif = float('inf')

    while max_vif > threshold and iter_count < max_iter and X.shape[1] > 1:
        iter_count += 1
        vif_current = []
        for i in range(X.shape[1]):
            try:
                y = X.iloc[:, i]
                X_i = sm.add_constant(X.drop(X.columns[i], axis=1))
                r_squared = sm.OLS(y, X_i).fit().rsquared
                vif_current.append({'Variable': X.columns[i], 'VIF': 1. / (1. - r_squared)})
            except Exception as e:
                print(f"Error calculating VIF for {X.columns[i]}: {e}")
                vif_current.append({'Variable': X.columns[i], 'VIF': float('nan')})

        vif_df = pd.DataFrame(vif_current)
        max_vif_row = vif_df.loc[vif_df['VIF'].idxmax()]
        max_vif = max_vif_row['VIF']
        max_vif_var = max_vif_row['Variable']

        if max_vif > threshold:
            print(f"VIF analysis removing variable: {max_vif_var} (VIF = {max_vif:.2f})")
            X = X.drop(max_vif_var, axis=1)
            removed_vars.append(max_vif_var)
            vif_data.append({'Variable': max_vif_var, 'VIF': max_vif, 'Status': 'Removed'})

    return X, removed_vars, vif_data


def _analyze_correlation(df, threshold=0.8):
    """Iteratively remove highly correlated variables, keeping the one with lower mean correlation."""
    X = df.copy()
    removed_vars = []
    high_corr_pairs = []

    while True:
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        max_corr = upper_tri.max().max()

        if max_corr < threshold:
            break

        var1, var2 = upper_tri.stack().idxmax()
        mean_corr_var1 = corr_matrix[var1].mean()
        mean_corr_var2 = corr_matrix[var2].mean()

        high_corr_pairs.append({
            'Variable1': var1,
            'Variable2': var2,
            'Correlation': corr_matrix.loc[var1, var2],
            'Mean_Corr_Var1': mean_corr_var1,
            'Mean_Corr_Var2': mean_corr_var2
        })

        var_to_remove = var1 if mean_corr_var1 > mean_corr_var2 else var2
        print(f"Correlation removing: {var_to_remove} (|r|={corr_matrix.loc[var1, var2]:.4f})")
        X = X.drop(var_to_remove, axis=1)
        removed_vars.append(var_to_remove)

    return X, removed_vars, high_corr_pairs


def _save_corr_heatmap(corr_matrix, title, path, annot=False):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=annot, fmt='.2f',
                cbar_kws={"shrink": .5})
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {path}")


def perform_variable_selection(df, method='combined', corr_threshold=0.8, vif_threshold=10, output_dir=None):
    """Perform variable selection; returns filtered DataFrame, removed vars, and step info."""
    print(f"\nStarting variable selection (method: {method})...")

    if output_dir:
        try:
            _save_corr_heatmap(df.corr(),
                               'Correlation Matrix of Environmental Variables (Before Selection)',
                               os.path.join(output_dir, "correlation_heatmap_before.png"))
        except Exception as e:
            print(f"Error generating heatmap: {e}")

    corr_filtered_df = df.copy()
    vif_filtered_df = df.copy()
    corr_removed_vars = []
    vif_removed_vars = []

    if method in ['correlation', 'combined']:
        print(f"\nCorrelation analysis (threshold: |r|>{corr_threshold})...")
        corr_filtered_df, corr_removed_vars, high_corr_pairs = _analyze_correlation(df, corr_threshold)
        if output_dir and high_corr_pairs:
            path = os.path.join(output_dir, "high_correlation_pairs.csv")
            pd.DataFrame(high_corr_pairs).to_csv(path, index=False)
            print(f"High-correlation pairs saved to {path}")

    if method in ['vif', 'combined']:
        print(f"\nVIF analysis (threshold: VIF>{vif_threshold})...")
        input_df = corr_filtered_df if method == 'combined' else df
        vif_filtered_df, vif_removed_vars, vif_data = _calculate_vif(input_df, vif_threshold)
        if output_dir and vif_data:
            path = os.path.join(output_dir, "vif_results.csv")
            pd.DataFrame(vif_data).to_csv(path, index=False)
            print(f"VIF results saved to {path}")

    if method == 'correlation':
        final_df = corr_filtered_df
        all_removed_vars = corr_removed_vars
    elif method == 'vif':
        final_df = vif_filtered_df
        all_removed_vars = vif_removed_vars
    else:
        final_df = vif_filtered_df
        all_removed_vars = corr_removed_vars + vif_removed_vars

    if output_dir and final_df.shape[1] > 1:
        try:
            _save_corr_heatmap(final_df.corr(),
                               'Correlation Matrix of Environmental Variables (After Selection)',
                               os.path.join(output_dir, "correlation_heatmap_after.png"),
                               annot=True)
        except Exception as e:
            print(f"Error generating post-selection heatmap: {e}")

    step_info = {
        "method": method,
        "original_vars_count": df.shape[1],
        "corr_filtered_vars_count": corr_filtered_df.shape[1] if method in ['correlation', 'combined'] else df.shape[1],
        "final_vars_count": final_df.shape[1],
        "corr_removed_vars": corr_removed_vars,
        "vif_removed_vars": vif_removed_vars,
        "corr_removed_count": len(corr_removed_vars),
        "vif_removed_count": len(vif_removed_vars),
        "total_removed_count": len(all_removed_vars),
        "corr_threshold": corr_threshold,
        "vif_threshold": vif_threshold
    }

    print(f"\nVariable selection done (method: {method}):")
    print(f"Original vars: {step_info['original_vars_count']}")
    if method in ['correlation', 'combined']:
        print(f"After correlation: {step_info['corr_filtered_vars_count']} retained, {step_info['corr_removed_count']} removed")
    if method in ['vif', 'combined']:
        print(f"After VIF: {step_info['final_vars_count']} retained, {step_info['vif_removed_count']} removed")
    print(f"Final: {step_info['final_vars_count']} retained, {step_info['total_removed_count']} removed total")

    return final_df, all_removed_vars, step_info


def extract_raster_values_at_points(points_gdf, raster_files):
    """Extract raster values at point locations; out-of-bounds or NoData become NaN."""
    all_raster_data = []
    print(f"Extracting values from {len(raster_files)} rasters for {len(points_gdf)} points...")
    for raster_path in raster_files:
        col_name = os.path.basename(raster_path).split('.')[0]
        try:
            with rasterio.open(raster_path) as src:
                coords = [(p.x, p.y) for p in points_gdf.geometry]
                nodata_val = src.nodatavals[0] if src.nodatavals else None
                raster_values = []
                for val_array in src.sample(coords):
                    val = val_array[0]
                    if (nodata_val is not None and val == nodata_val) or np.isnan(val):
                        raster_values.append(np.nan)
                    else:
                        raster_values.append(val)
                all_raster_data.append(pd.Series(raster_values, index=points_gdf.index, name=col_name))
        except Exception as e:
            print(f"Error processing raster {raster_path}: {e}")
            all_raster_data.append(pd.Series([np.nan] * len(points_gdf), index=points_gdf.index, name=col_name))

    if not all_raster_data:
        return pd.DataFrame(index=points_gdf.index)

    return pd.concat(all_raster_data, axis=1)


def prepare_and_split_data(presence_data, background_data, env_dir, output_dir,
                           study_area_shp_file=None, use_filtered_variables=True):
    """Clip presence points to study area, extract env variables, optionally filter."""
    os.makedirs(output_dir, exist_ok=True)
    print("Step 1: Data preparation")

    try:
        points_gdf = gpd.GeoDataFrame(
            presence_data,
            geometry=gpd.points_from_xy(presence_data['Longitude'], presence_data['Latitude']),
            crs="EPSG:4326"
        )
        study_area_gdf = gpd.read_file(study_area_shp_file)
        if study_area_gdf.crs is None:
            study_area_gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
    except Exception as e:
        print(f"Error: cannot load input files: {e}")
        return None, None, None

    if not (hasattr(study_area_gdf.crs, "equals") and study_area_gdf.crs.equals(points_gdf.crs)):
        print(f"Reprojecting study area to presence CRS: {points_gdf.crs}")
        try:
            study_area_gdf = study_area_gdf.to_crs(points_gdf.crs)
        except Exception as e:
            print(f"Error: reprojection failed: {e}")
            return None, None, None

    points_in_study_area_gdf = gpd.sjoin(points_gdf, study_area_gdf, how="inner", predicate="within")
    if points_in_study_area_gdf.empty:
        print("Error: no presence points within study area.")
        return None, None, None

    cols_to_keep = [c for c in points_gdf.columns if c in points_in_study_area_gdf.columns]
    points_in_study_area_gdf = points_in_study_area_gdf[list(set(cols_to_keep))].copy()
    points_in_study_area_gdf = points_in_study_area_gdf.drop_duplicates(
        subset=[points_in_study_area_gdf.geometry.name])
    print(f"Found {len(points_in_study_area_gdf)} presence points within study area.")

    env_files = sorted(glob.glob(os.path.join(env_dir, "*.asc")))
    if not env_files:
        raise ValueError(f"No .asc files found in {env_dir}")
    print(f"Found {len(env_files)} env raster files")

    env_data_df = extract_raster_values_at_points(points_in_study_area_gdf, env_files)
    points_in_study_area_gdf = points_in_study_area_gdf.reset_index(drop=True)
    env_data_df = env_data_df.reset_index(drop=True)

    all_points_with_env_gdf = points_in_study_area_gdf.join(env_data_df)
    original_count = len(all_points_with_env_gdf)
    all_points_with_env_gdf.dropna(subset=env_data_df.columns, inplace=True)
    if len(all_points_with_env_gdf) < original_count:
        print(f"Dropped {original_count - len(all_points_with_env_gdf)} points due to missing env data.")

    if len(all_points_with_env_gdf) < 2:
        print("Error: insufficient valid data points.")
        return None, None, None

    env_var_names = env_data_df.columns.tolist()

    original_vars_path = os.path.join(output_dir, "original_variables.txt")
    with open(original_vars_path, 'w') as f:
        f.write("# Original environmental variables\n\n")
        for var in env_var_names:
            f.write(f"- {var}\n")
    print(f"Original variable list saved to {original_vars_path}")

    if use_filtered_variables:
        combined_output_dir = os.path.join(output_dir, "variable_selection")
        os.makedirs(combined_output_dir, exist_ok=True)

        background_gdf = gpd.GeoDataFrame(
            background_data,
            geometry=gpd.points_from_xy(background_data['Longitude'], background_data['Latitude']),
            crs="EPSG:4326"
        )
        all_points_gdf = pd.concat([
            points_in_study_area_gdf.assign(label=1),
            background_gdf.assign(label=0)
        ], ignore_index=True)

        print("Extracting env data for all points for variable selection...")
        all_env_data_df = extract_raster_values_at_points(all_points_gdf, env_files)
        all_points_gdf = all_points_gdf.reset_index(drop=True)
        all_env_data_df = all_env_data_df.reset_index(drop=True)

        all_points_with_env_for_filter = all_points_gdf.join(all_env_data_df)
        all_points_with_env_for_filter.dropna(subset=all_env_data_df.columns, inplace=True)

        env_data_filtered_df, removed_vars, step_info = perform_variable_selection(
            all_points_with_env_for_filter[env_var_names],
            method=VARIABLE_SELECTION_METHOD,
            corr_threshold=CORRELATION_THRESHOLD,
            vif_threshold=VIF_THRESHOLD,
            output_dir=combined_output_dir
        )

        filtered_vars = env_data_filtered_df.columns.tolist()
        filtered_vars_path = os.path.join(combined_output_dir, "filtered_variables.txt")
        with open(filtered_vars_path, 'w') as f:
            f.write(f"# Variable selection result (method: {step_info['method']})\n")
            f.write(f"# corr_threshold: {step_info['corr_threshold']}, vif_threshold: {step_info['vif_threshold']}\n\n")
            f.write("Retained variables:\n")
            for var in filtered_vars:
                f.write(f"- {var}\n")
        print(f"Filtered variable list saved to {filtered_vars_path}")

        removed_vars_path = os.path.join(combined_output_dir, "removed_variables.txt")
        with open(removed_vars_path, 'w') as f:
            f.write(f"# Removed variables (method: {step_info['method']})\n\n")
            if step_info['corr_removed_vars']:
                f.write(f"Removed by correlation analysis (|r|>{step_info['corr_threshold']}):\n")
                for var in step_info['corr_removed_vars']:
                    f.write(f"- {var}\n")
                f.write("\n")
            if step_info['vif_removed_vars']:
                f.write(f"Variables removed by VIF (VIF>{step_info['vif_threshold']}):\n")
                for var in step_info['vif_removed_vars']:
                    f.write(f"- {var}\n")
        print(f"Removed variables list saved to {removed_vars_path}")

        env_var_names = filtered_vars
        print(f"\nUsing {len(env_var_names)} filtered variables for training")
    else:
        print(f"\nUsing all {len(env_var_names)} original variables for training")

    X_env_data = all_points_with_env_gdf[env_var_names].values
    csv_path = os.path.join(output_dir, "all_points_env_data.csv")
    pd.DataFrame(X_env_data, columns=env_var_names).to_csv(csv_path, index=False)
    print(f"All-points env data saved to {csv_path}")

    return X_env_data, all_points_with_env_gdf, env_var_names


SPATIAL_BLOCK_SIZE_KM = 50
K_FOLDS = 5
BACKGROUND_CV_SPLIT = True

CALIBRATION_METHOD = 'sigmoid'          # 'isotonic' | 'sigmoid' | 'none'

USE_PPM_LITE = True
PPM_INTENSITY_METHOD = 'platt_logit'    # 'platt_logit' | 'platt_prob' | 'rf_prob' | 'rf_logit'
PPM_NORMALIZATION_METHOD = 'quantile'   # 'study_area' | 'global' | 'quantile'

USE_SCARF_AUGMENTATION = True
SCARF_CORRUPTION_RATE = 0.6

SIMCLR_PRETRAIN_EPOCHS = 200
SIMCLR_WARMUP_EPOCHS = 10
SIMCLR_BASE_LR = 1e-5
SIMCLR_MIN_LR = 0.0

USE_FILTERED_VARIABLES = False
CORRELATION_THRESHOLD = 0.8
VIF_THRESHOLD = 10
VARIABLE_SELECTION_METHOD = 'combined'


class SimCLRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super(SimCLRModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        self.projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        self.rf_classifier = None
        self.calibrator = None

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

    def get_features(self, x):
        return self.encoder(x)


class RemoteSensingAugmentation:
    """Remote-sensing style augmentation; fallback when USE_SCARF_AUGMENTATION=False."""

    def __init__(self, noise_level=0.1, brightness_range=0.15, contrast_range=0.15):
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def add_gaussian_noise(self, data):
        return data + torch.randn_like(data) * self.noise_level

    def adjust_brightness(self, data):
        factor = 1.0 + (torch.rand(1, device=data.device) * 2 - 1) * self.brightness_range
        return data * factor

    def adjust_contrast(self, data):
        factor = 1.0 + (torch.rand(1, device=data.device) * 2 - 1) * self.contrast_range
        mean = torch.mean(data, dim=1, keepdim=True)
        return (data - mean) * factor + mean

    def apply_augmentation(self, data):
        aug_data = data.clone()
        if torch.rand(1, device=data.device) < 0.5:
            aug_data = self.add_gaussian_noise(aug_data)
        if torch.rand(1, device=data.device) < 0.5:
            aug_data = self.adjust_brightness(aug_data)
        if torch.rand(1, device=data.device) < 0.5:
            aug_data = self.adjust_contrast(aug_data)
        return aug_data


class ScarfCorruption:
    """SCARF-style feature corruption: replace features with empirical marginal samples."""

    def __init__(self, feature_pool, corruption_rate=0.6):
        self.feature_pool = feature_pool
        self.n_samples, self.n_features = feature_pool.shape
        self.corruption_rate = corruption_rate

    def corrupt(self, x):
        batch_size, n_features = x.shape
        rand_idx = torch.randint(0, self.n_samples, (batch_size, n_features), device=x.device)
        x_random = torch.gather(self.feature_pool, 0, rand_idx)
        corruption_mask = torch.rand(batch_size, n_features, device=x.device) < self.corruption_rate
        return torch.where(corruption_mask, x_random, x)


def _build_scarf_corruption(train_subset, corruption_rate=SCARF_CORRUPTION_RATE):
    base_dataset = train_subset.dataset
    indices = np.asarray(train_subset.indices)
    feats = np.asarray(base_dataset.features, dtype=np.float32)[indices]
    feats = np.nan_to_num(feats, nan=0.0)
    feature_pool = torch.tensor(feats, dtype=torch.float32).cuda()
    return ScarfCorruption(feature_pool, corruption_rate=corruption_rate)


class HabitatDataset(Dataset):
    def __init__(self, presence_data, background_data, env_dir, filtered_env_files=None):
        self.presence_data = presence_data
        self.background_data = background_data

        if filtered_env_files is not None:
            self.env_files = filtered_env_files
            print(f"Using {len(self.env_files)} filtered env files")
        else:
            self.env_files = glob.glob(os.path.join(env_dir, "*.asc"))
            print(f"Using all {len(self.env_files)} env files")

        self.env_data = []
        self.env_transforms = []

        print("Loading env raster data...")
        for env_file in tqdm(self.env_files):
            with rasterio.open(env_file) as src:
                self.env_data.append(src.read(1))
                self.env_transforms.append(src.transform)

        self.data = pd.concat([
            pd.DataFrame({'longitude': presence_data['Longitude'],
                          'latitude': presence_data['Latitude'],
                          'label': 1}),
            pd.DataFrame({'longitude': background_data['Longitude'],
                          'latitude': background_data['Latitude'],
                          'label': 0})
        ])

        self.points_gdf = gpd.GeoDataFrame(
            self.data,
            geometry=gpd.points_from_xy(self.data['longitude'], self.data['latitude']),
            crs="EPSG:4326"
        )

        print("Extracting env features...")
        self.features = self.extract_features()

    def extract_features(self):
        features = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            features.append(self.get_environmental_features(row['longitude'], row['latitude']))
        return np.array(features)

    def get_environmental_features(self, x, y):
        features = []
        for env_data, transform in zip(self.env_data, self.env_transforms):
            try:
                row, col = rasterio.transform.rowcol(transform, x, y)
                if 0 <= row < env_data.shape[0] and 0 <= col < env_data.shape[1]:
                    value = env_data[row, col]
                else:
                    value = np.nan
            except Exception:
                value = np.nan
            features.append(value)
        return features

    def get_presence_points_gdf(self):
        return self.points_gdf[self.data['label'] == 1].reset_index(drop=True)

    def get_background_points_gdf(self):
        return self.points_gdf[self.data['label'] == 0].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.FloatTensor([self.data.iloc[idx]['label']])
        return features, label


def train_simclr(model, train_loader, optimizer, temperature=0.5, corruption=None):
    """SimCLR contrastive learning. When corruption is not None: anchor=original, positive=corrupted."""
    model.train()
    total_loss = 0

    augmentation = None
    if corruption is None:
        augmentation = RemoteSensingAugmentation(noise_level=0.1, brightness_range=0.15,
                                                 contrast_range=0.15)

    for data, _ in train_loader:
        if len(data) < 2:
            continue
        data = data.cuda()

        if corruption is not None:
            data_aug1 = data
            data_aug2 = corruption.corrupt(data)
        else:
            data_aug1 = augmentation.apply_augmentation(data)
            data_aug2 = augmentation.apply_augmentation(data)

        _, z1 = model(data_aug1)
        _, z2 = model(data_aug2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        similarity_matrix = torch.matmul(z1, z2.T) / temperature
        contrast_labels = torch.arange(len(data)).cuda()
        loss = F.cross_entropy(similarity_matrix, contrast_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def train_rf_classifier(model, train_loader, val_loader, calibration_method='isotonic'):
    """Train RF classifier and calibrator on frozen SimCLR features."""
    print(f"  Training RF + calibrator (method: {calibration_method})...")

    model.eval()
    train_features, train_labels = [], []
    with torch.no_grad():
        for data, labels in train_loader:
            features = model.get_features(data.cuda()).cpu().numpy()
            train_features.append(features)
            train_labels.extend(labels.numpy().flatten())

    train_features = np.vstack(train_features)
    train_labels = np.array(train_labels)
    print(f"    Train features shape: {train_features.shape}")
    print(f"    Train label distribution: {np.bincount(train_labels.astype(int))}")

    rf_classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_classifier.fit(train_features, train_labels)

    val_features, val_labels = [], []
    with torch.no_grad():
        for data, labels in val_loader:
            features = model.get_features(data.cuda()).cpu().numpy()
            val_features.append(features)
            val_labels.extend(labels.numpy().flatten())

    val_features = np.vstack(val_features)
    val_labels = np.array(val_labels)
    val_rf_probs = rf_classifier.predict_proba(val_features)[:, 1]

    if calibration_method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_rf_probs, val_labels)
        print("    Using Isotonic regression calibration")
    elif calibration_method == 'sigmoid':
        calibrator = LogisticRegression(random_state=42)
        calibrator.fit(val_rf_probs.reshape(-1, 1), val_labels)
        print("    Using Sigmoid calibration (Platt scaling)")
    elif calibration_method == 'none':
        calibrator = None
        print("    No probability calibration")
    else:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")

    model.rf_classifier = rf_classifier
    model.calibrator = calibrator
    model.calibration_method = calibration_method

    return rf_classifier, calibrator


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



METRIC_KEYS = ['roc_auc', 'pr_auc', 'boyce_index', 'sensitivity']

METRIC_LABELS = {
    'roc_auc': 'ROC-AUC',
    'pr_auc': 'PR-AUC',
    'boyce_index': 'CBI',
    'sensitivity': 'Sensitivity',
}

METRIC_DESCRIPTIONS = {
    'roc_auc': 'Area Under ROC Curve',
    'pr_auc': 'Area Under Precision-Recall Curve',
    'boyce_index': 'Continuous Boyce Index',
    'sensitivity': 'True Positive Rate (at max-TSS threshold)',
}


def compute_metrics(y_true, y_prob, y_pred):
    """Compute four metrics: ROC-AUC, PR-AUC, CBI, Sensitivity."""
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        presence_predictions = y_prob[y_true == 1]
        background_predictions = y_prob[y_true == 0]
        if len(presence_predictions) > 0 and len(background_predictions) > 0:
            boyce_index = calculate_boyce_index(presence_predictions, background_predictions)
        else:
            boyce_index = 0.0
    except Exception as e:
        print(f"Warning: Boyce Index calculation failed: {e}")
        boyce_index = 0.0

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'boyce_index': boyce_index,
        'sensitivity': sensitivity,
    }


def evaluate_model(model, val_loader, threshold_method='max_tss', use_calibration=False):
    """Evaluate model using uncalibrated RF probs (calibrator only used for mapping)."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for data, labels in val_loader:
            features = model.get_features(data.cuda()).cpu().numpy()
            rf_probs = model.rf_classifier.predict_proba(features)[:, 1]

            if use_calibration and model.calibrator is not None:
                if getattr(model, 'calibration_method', '') == 'sigmoid':
                    eval_probs = model.calibrator.predict_proba(rf_probs.reshape(-1, 1))[:, 1]
                else:
                    eval_probs = model.calibrator.predict(rf_probs.reshape(-1, 1))
            else:
                eval_probs = rf_probs

            all_probs.extend(eval_probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    if all_labels.ndim > 1:
        all_labels = all_labels.flatten()

    best_threshold = 0.5
    if threshold_method == 'max_tss':
        best_tss = -1
        for threshold in np.linspace(0, 1, 101):
            y_pred = (all_probs >= threshold).astype(int)
            tn = np.sum((all_labels == 0) & (y_pred == 0))
            fp = np.sum((all_labels == 0) & (y_pred == 1))
            fn = np.sum((all_labels == 1) & (y_pred == 0))
            tp = np.sum((all_labels == 1) & (y_pred == 1))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = sensitivity + specificity - 1
            if tss > best_tss:
                best_tss = tss
                best_threshold = threshold

    y_pred = (all_probs >= best_threshold).astype(int)
    return compute_metrics(all_labels, all_probs, y_pred)


def compute_ppm_lite_intensity(model, features, method='platt_logit'):
    """Compute PPM-lite relative intensity."""
    if model.rf_classifier is None:
        raise ValueError("RF classifier not trained")

    rf_probs = model.rf_classifier.predict_proba(features)[:, 1]
    has_platt = (model.calibrator is not None
                 and getattr(model, 'calibration_method', '') == 'sigmoid')

    if method == 'platt_logit':
        if has_platt:
            intensity = np.exp(model.calibrator.decision_function(rf_probs.reshape(-1, 1)))
        else:
            print("Warning: no Sigmoid calibrator found, falling back to RF prob")
            intensity = rf_probs
    elif method == 'platt_prob':
        if has_platt:
            intensity = model.calibrator.predict_proba(rf_probs.reshape(-1, 1))[:, 1]
        else:
            print("Warning: no Sigmoid calibrator found, falling back to RF prob")
            intensity = rf_probs
    elif method == 'rf_prob':
        intensity = rf_probs
    elif method == 'rf_logit':
        rf_probs_clipped = np.clip(rf_probs, 1e-7, 1 - 1e-7)
        logit_values = np.log(rf_probs_clipped / (1 - rf_probs_clipped))
        min_logit = np.min(logit_values)
        intensity = logit_values - min_logit + 1e-6 if min_logit < 0 else logit_values
    else:
        raise ValueError(f"Unsupported intensity method: {method}")

    intensity = np.maximum(intensity, 1e-10)
    return np.nan_to_num(intensity, nan=1e-10, posinf=1e10, neginf=1e-10)


def normalize_intensity_to_probability(intensity, normalization_method='study_area'):
    """Normalize relative intensity to probability."""
    intensity = np.array(intensity)

    if normalization_method == 'study_area':
        total_intensity = np.sum(intensity)
        probabilities = intensity / total_intensity if total_intensity > 0 else np.zeros_like(intensity)
    elif normalization_method == 'global':
        min_i, max_i = np.min(intensity), np.max(intensity)
        probabilities = (intensity - min_i) / (max_i - min_i) if max_i > min_i else np.zeros_like(intensity)
    elif normalization_method == 'quantile':
        from scipy.stats import rankdata
        ranks = rankdata(intensity, method='average')
        probabilities = (ranks - 1) / (len(ranks) - 1)
    else:
        raise ValueError(f"Unsupported normalization method: {normalization_method}")

    probabilities = np.maximum(probabilities, 0)
    return np.nan_to_num(probabilities, nan=0, posinf=1, neginf=0)


def predict_hsi_with_ppm_lite(model, env_dir, output_path, mask_shp_path=None,
                              intensity_method='platt_logit', normalization_method='study_area',
                              env_files=None):
    """PPM-lite HSI mapping: Platt-calibrated logit as relative intensity, normalized to probability."""
    print("Starting PPM-lite HSI prediction...")
    print(f"Intensity: {intensity_method}, normalization: {normalization_method}")

    if model.rf_classifier is None:
        raise ValueError("RF classifier not trained, cannot predict")

    if intensity_method == 'platt_logit' and (
            model.calibrator is None or getattr(model, 'calibration_method', '') != 'sigmoid'):
        print("Warning: PPM-lite requires Sigmoid calibrator for logit; falling back to rf_prob")
        intensity_method = 'rf_prob'

    model.eval()

    if env_files is None:
        env_files = sorted(glob.glob(os.path.join(env_dir, "*.asc")))
    else:
        env_files = sorted(env_files)
    if not env_files:
        raise ValueError(f"No .asc files found in {env_dir}")
    print(f"Found {len(env_files)} env raster files")

    with rasterio.open(env_files[0]) as src_template:
        output_meta = src_template.meta.copy()
        base_transform_affine = src_template.transform
        base_crs_obj = src_template.crs
        height_pixels = src_template.height
        width_pixels = src_template.width

    if mask_shp_path and os.path.exists(mask_shp_path):
        print("Using study area mask...")
        study_area_geodata = gpd.read_file(mask_shp_path)
        if study_area_geodata.crs != base_crs_obj:
            study_area_geodata = study_area_geodata.to_crs(base_crs_obj)
        study_area_raster_mask = geometry_mask(study_area_geodata.geometry,
                                               out_shape=(height_pixels, width_pixels),
                                               transform=base_transform_affine,
                                               invert=True,
                                               all_touched=True)
    else:
        print("No mask provided, processing entire raster extent...")
        study_area_raster_mask = np.ones((height_pixels, width_pixels), dtype=bool)

    print("Loading env rasters...")
    num_env_vars = len(env_files)
    stacked_env_rasters_numpy = np.zeros((num_env_vars, height_pixels, width_pixels), dtype=np.float32)

    for i, raster_path_current in enumerate(tqdm(env_files, desc="Loading env rasters")):
        with rasterio.open(raster_path_current) as src_raster:
            img_pixel_data = src_raster.read(1).astype(np.float32)
            nodata_val_current = src_raster.nodatavals[0] if src_raster.nodatavals else None
            if nodata_val_current is not None:
                img_pixel_data[img_pixel_data == nodata_val_current] = np.nan
            img_pixel_data[img_pixel_data == -9999] = np.nan
            stacked_env_rasters_numpy[i, :, :] = img_pixel_data

    study_area_rows, study_area_cols = np.where(study_area_raster_mask)
    print(f"Study area: {len(study_area_rows)} pixels")
    if len(study_area_rows) == 0:
        print("Warning: study area mask is empty!")
        return

    filled_env_rasters = np.copy(stacked_env_rasters_numpy)
    for var_idx in range(num_env_vars):
        mask = np.isnan(filled_env_rasters[var_idx])
        filled_env_rasters[var_idx, study_area_rows, study_area_cols] = np.where(
            mask[study_area_rows, study_area_cols],
            0.0,
            filled_env_rasters[var_idx, study_area_rows, study_area_cols]
        )

    env_data_for_study_area = filled_env_rasters[:, study_area_rows, study_area_cols].T
    env_data_for_study_area_tensor = torch.tensor(env_data_for_study_area, dtype=torch.float32).cuda()
    print(f"Computing PPM-lite intensity for {len(env_data_for_study_area_tensor)} pixels...")

    batch_size = 5000
    all_intensities = []
    with torch.no_grad():
        for i in tqdm(range(0, len(env_data_for_study_area_tensor), batch_size), desc="PPM-lite intensity"):
            batch_cell_env_data = env_data_for_study_area_tensor[i:i + batch_size]
            batch_features = model.get_features(batch_cell_env_data).cpu().numpy()
            batch_intensity = compute_ppm_lite_intensity(model, batch_features, intensity_method)
            all_intensities.extend(batch_intensity.flatten())

    all_intensities = np.array(all_intensities)

    print(f"Normalizing intensity to probability (method: {normalization_method})...")
    ppm_probabilities = normalize_intensity_to_probability(all_intensities, normalization_method)

    ppm_map = np.full((height_pixels, width_pixels), np.nan, dtype=np.float32)
    ppm_map[study_area_rows, study_area_cols] = ppm_probabilities.astype(np.float32)

    output_meta.update(count=1, dtype='float32', nodata=np.nan)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(output_path, 'w', **output_meta) as dst_ppm_map:
        dst_ppm_map.write(ppm_map.astype(output_meta['dtype']), 1)
    print(f"PPM-lite HSI saved to: {output_path}")

    print("\nPPM-lite prediction statistics:")
    print(f"Intensity: min={np.min(all_intensities):.6f}, max={np.max(all_intensities):.6f}, "
          f"mean={np.mean(all_intensities):.6f}, std={np.std(all_intensities):.6f}")
    print(f"Probability: min={np.min(ppm_probabilities):.6f}, max={np.max(ppm_probabilities):.6f}, "
          f"mean={np.mean(ppm_probabilities):.6f}, std={np.std(ppm_probabilities):.6f}")

    print("\nNote: quantile-normalized HSI cannot be compared across years directly.")
    print("PPM-lite prediction complete!")

def _compute_spatial_blocks(points_gdf, block_size_km, crs_epsg_for_meter=3857):
    """Generate spatial block IDs for GroupKFold."""
    if points_gdf is None or points_gdf.empty:
        return None

    try:
        if points_gdf.crs.to_epsg() != crs_epsg_for_meter:
            gdf = points_gdf.to_crs(epsg=crs_epsg_for_meter)
            print(f"  CRS transform: {points_gdf.crs} -> EPSG:{crs_epsg_for_meter}")
        else:
            gdf = points_gdf.copy()
    except Exception as e:
        print(f"  Warning: CRS transform failed ({e}), using original")
        gdf = points_gdf.copy()

    coords = np.stack([gdf.geometry.x.values, gdf.geometry.y.values], axis=1)
    block_m = max(1000.0, float(block_size_km) * 1000.0)

    min_x, min_y = coords.min(axis=0)
    bx = np.floor((coords[:, 0] - min_x) / block_m).astype(np.int64)
    by = np.floor((coords[:, 1] - min_y) / block_m).astype(np.int64)

    groups = np.char.add(bx.astype(str), '_')
    groups = np.char.add(groups, by.astype(str))

    unique_groups, group_counts = np.unique(groups, return_counts=True)
    print(f"  Generated {len(unique_groups)} blocks, size={block_size_km}km")
    print(f"  Points/block: {group_counts.min()}-{group_counts.max()}, mean={group_counts.mean():.1f}")

    return groups


def _fallback_random_split(total_bg_points):
    bg_indices = np.arange(total_bg_points)
    np.random.shuffle(bg_indices)
    split_point = total_bg_points // 2
    return bg_indices[:split_point], bg_indices[split_point:]


def spatial_split_background_points(background_gdf, presence_train_idx, presence_val_idx,
                                    presence_gdf, block_size_km=50):
    """Split background points by presence spatial blocks for train/val consistency."""
    print(f"  Spatial split of background points (block={block_size_km}km)...")

    train_presence_gdf = presence_gdf.iloc[presence_train_idx].reset_index(drop=True)
    val_presence_gdf = presence_gdf.iloc[presence_val_idx].reset_index(drop=True)

    background_groups = _compute_spatial_blocks(background_gdf, block_size_km)
    if background_groups is None:
        print("  Warning: background spatial grouping failed, using random split")
        return _fallback_random_split(len(background_gdf))

    train_presence_groups = _compute_spatial_blocks(train_presence_gdf, block_size_km)
    val_presence_groups = _compute_spatial_blocks(val_presence_gdf, block_size_km)
    if train_presence_groups is None or val_presence_groups is None:
        print("  Warning: presence spatial grouping failed, using random split")
        return _fallback_random_split(len(background_gdf))

    train_blocks = set(train_presence_groups)
    val_blocks = set(val_presence_groups)
    print(f"  Train presence in {len(train_blocks)} blocks")
    print(f"  Val presence in {len(val_blocks)} blocks")

    train_bg_mask = np.array([bg_group in train_blocks for bg_group in background_groups])
    val_bg_mask = np.array([bg_group in val_blocks for bg_group in background_groups])

    overlap_blocks = train_blocks.intersection(val_blocks)
    if overlap_blocks:
        print(f"  {len(overlap_blocks)} overlapping blocks, randomly splitting")
        overlap_mask = np.array([bg_group in overlap_blocks for bg_group in background_groups])
        overlap_indices = np.where(overlap_mask)[0]
        if len(overlap_indices) > 0:
            np.random.shuffle(overlap_indices)
            split_point = len(overlap_indices) // 2
            train_bg_mask[overlap_mask] = False
            val_bg_mask[overlap_mask] = False
            train_bg_mask[overlap_indices[:split_point]] = True
            val_bg_mask[overlap_indices[split_point:]] = True

    train_bg_idx = np.where(train_bg_mask)[0]
    val_bg_idx = np.where(val_bg_mask)[0]

    unassigned_indices = np.where(~(train_bg_mask | val_bg_mask))[0]
    if len(unassigned_indices) > 0:
        print(f"  {len(unassigned_indices)} unassigned background points, randomly splitting")
        np.random.shuffle(unassigned_indices)
        split_point = len(unassigned_indices) // 2
        train_bg_idx = np.concatenate([train_bg_idx, unassigned_indices[:split_point]])
        val_bg_idx = np.concatenate([val_bg_idx, unassigned_indices[split_point:]])

    min_bg_points = 100
    if len(train_bg_idx) < min_bg_points or len(val_bg_idx) < min_bg_points:
        print(f"  Warning: insufficient background points after spatial split (train:{len(train_bg_idx)}, val:{len(val_bg_idx)}), using random")
        return _fallback_random_split(len(background_gdf))

    print(f"  Background spatial split done: train={len(train_bg_idx)}, val={len(val_bg_idx)}")
    return train_bg_idx, val_bg_idx


def random_split_background_points(background_gdf, fold_idx, total_folds, random_state=38):
    """Random background split when spatial grouping unavailable."""
    np.random.seed(random_state + fold_idx)

    bg_indices = np.arange(len(background_gdf))
    np.random.shuffle(bg_indices)

    fold_size = len(bg_indices) // total_folds
    val_start = fold_idx * fold_size
    val_end = (fold_idx + 1) * fold_size if fold_idx < total_folds - 1 else len(bg_indices)

    val_bg_idx = bg_indices[val_start:val_end]
    train_bg_idx = np.concatenate([bg_indices[:val_start], bg_indices[val_end:]])

    print(f"  Background random split done: train={len(train_bg_idx)}, val={len(val_bg_idx)}")
    return train_bg_idx, val_bg_idx


def _draw_block_grid(ax, x, y, block_size_km):
    block_m = block_size_km * 1000
    grid_x = np.arange(np.floor(x.min() / block_m) * block_m,
                       np.ceil(x.max() / block_m) * block_m + block_m, block_m)
    grid_y = np.arange(np.floor(y.min() / block_m) * block_m,
                       np.ceil(y.max() / block_m) * block_m + block_m, block_m)
    for gx in grid_x:
        ax.axvline(x=gx, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    for gy in grid_y:
        ax.axhline(y=gy, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)


def visualize_spatial_blocks(points_gdf, groups, block_size_km, output_path,
                             title="Spatial Blocks Distribution"):
    try:
        gdf_proj = points_gdf.to_crs(epsg=3857) if points_gdf.crs.to_epsg() != 3857 else points_gdf.copy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        x = gdf_proj.geometry.x.values
        y = gdf_proj.geometry.y.values

        unique_groups = np.unique(groups)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))

        for i, group in enumerate(unique_groups):
            mask = groups == group
            ax.scatter(x[mask], y[mask], c=[colors[i]], label=f'Block {group}', alpha=0.7, s=20)

        _draw_block_grid(ax, x, y, block_size_km)

        ax.set_xlabel('X Coordinate (m)', fontsize=12)
        ax.set_ylabel('Y Coordinate (m)', fontsize=12)
        ax.set_title(f'{title}\nBlock Size: {block_size_km}km, Total Blocks: {len(unique_groups)}',
                     fontsize=14)
        ax.grid(True, alpha=0.3)
        if len(unique_groups) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Spatial blocks plot saved to: {output_path}")
    except Exception as e:
        print(f"  Warning: cannot generate spatial blocks plot ({e})")


def visualize_cv_splits(points_gdf, groups, output_dir, block_size_km):
    try:
        gkf = GroupKFold(n_splits=K_FOLDS)
        gdf_proj = points_gdf.to_crs(epsg=3857) if points_gdf.crs.to_epsg() != 3857 else points_gdf.copy()

        x = gdf_proj.geometry.x.values
        y = gdf_proj.geometry.y.values

        for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(points_gdf)), groups=groups)):
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.scatter(x[train_idx], y[train_idx], c='blue', alpha=0.6, s=20,
                       label=f'Training Set ({len(train_idx)} points)')
            ax.scatter(x[val_idx], y[val_idx], c='red', alpha=0.8, s=30,
                       label=f'Validation Set ({len(val_idx)} points)')

            _draw_block_grid(ax, x, y, block_size_km)

            ax.set_xlabel('X Coordinate (m)', fontsize=12)
            ax.set_ylabel('Y Coordinate (m)', fontsize=12)
            ax.set_title(f'Spatial Block Cross-Validation - Fold {fold + 1}\n'
                         f'Block Size: {block_size_km}km', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"spatial_cv_fold_{fold + 1}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  CV split plots saved to: {output_dir}")
    except Exception as e:
        print(f"  Warning: cannot generate CV split plots ({e})")


def save_rf_classifier(model, output_dir, try_times):
    """Save RF classifier and parameters."""
    if model.rf_classifier is None:
        return

    rf_path = os.path.join(output_dir, f"rf_classifier_{try_times}.joblib")
    joblib.dump(model.rf_classifier, rf_path)
    print(f"RF classifier saved to: {rf_path}")

    rf_params_path = os.path.join(output_dir, f"rf_parameters_{try_times}.txt")
    with open(rf_params_path, 'w', encoding='utf-8') as f:
        f.write("Random Forest Classifier Parameters\n")
        f.write("=" * 50 + "\n\n")
        params = model.rf_classifier.get_params()
        for param_name in sorted(params.keys()):
            f.write(f"{param_name}: {params[param_name]}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Saved at: {try_times}\n")
    print(f"RF params saved to: {rf_params_path}")


def save_cv_summary_to_csv(fold_metrics, output_path):
    """Save CV summary to CSV."""
    import csv

    csv_data = [['Metric', 'Mean', 'Std_Dev', 'Min', 'Max', 'Median']]
    for metric in METRIC_KEYS:
        values = [m[metric] for m in fold_metrics]
        csv_data.append([
            METRIC_LABELS[metric],
            f"{np.mean(values):.4f}",
            f"{np.std(values):.4f}",
            f"{np.min(values):.4f}",
            f"{np.max(values):.4f}",
            f"{np.median(values):.4f}"
        ])

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv.writer(csvfile).writerows(csv_data)

    print(f"CV summary saved to: {output_path}")


def save_evaluation_results(metrics, output_path):
    """Save evaluation results to Excel and CSV."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Model Evaluation Results"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    center = Alignment(horizontal="center", vertical="center")

    ws.merge_cells('A1:C1')
    ws['A1'] = "SimCLR+RF Model Evaluation Results (Spatial Cross-Validation)"
    ws['A1'].font = Font(bold=True, size=16)
    ws['A1'].alignment = center

    for col, header in enumerate(["Metric", "Value", "Description"], 1):
        cell = ws.cell(row=3, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center

    row = 4
    for key in METRIC_KEYS:
        ws.cell(row=row, column=1, value=METRIC_LABELS[key]).alignment = center
        ws.cell(row=row, column=2, value=round(metrics[key], 4)).alignment = center
        ws.cell(row=row, column=3, value=METRIC_DESCRIPTIONS[key]).alignment = center
        row += 1

    ws.cell(row=row + 1, column=1, value="Model Parameters").font = Font(bold=True)
    ws.cell(row=row + 2, column=1, value="Model Architecture: SimCLR + RandomForest")
    ws.cell(row=row + 3, column=1, value=f"Spatial Block Size: {SPATIAL_BLOCK_SIZE_KM} km")
    ws.cell(row=row + 4, column=1, value=f"Cross-Validation Folds: {K_FOLDS}")
    ws.cell(row=row + 5, column=1, value=f"Random Seed: {RANDOM_STATE}")
    ws.cell(row=row + 6, column=1, value=f"Calibration Method: {CALIBRATION_METHOD}")
    ws.cell(row=row + 7, column=1,
            value=f"Variable Selection: {'Enabled' if USE_FILTERED_VARIABLES else 'Disabled'}")

    for col_idx in range(1, 4):
        max_length = 0
        column_letter = chr(64 + col_idx)
        for row_idx in range(1, row + 10):
            value = ws.cell(row=row_idx, column=col_idx).value
            if value is not None:
                max_length = max(max_length, len(str(value)))
        ws.column_dimensions[column_letter].width = min(max_length + 2, 40)

    wb.save(output_path)
    print(f"Evaluation results saved to: {output_path}")

    import csv
    csv_path = output_path.replace('.xlsx', '.csv')
    csv_data = [['Metric', 'Value', 'Description']]
    for key in METRIC_KEYS:
        csv_data.append([METRIC_LABELS[key], round(metrics[key], 4), METRIC_DESCRIPTIONS[key]])
    csv_data.append(['', '', ''])
    csv_data.append(['Model Parameters', '', ''])
    csv_data.append(['Model Architecture', 'SimCLR + RandomForest', ''])
    csv_data.append(['Spatial Block Size', f'{SPATIAL_BLOCK_SIZE_KM} km', ''])
    csv_data.append(['Cross-Validation Folds', str(K_FOLDS), ''])
    csv_data.append(['Random Seed', str(RANDOM_STATE), ''])
    csv_data.append(['Calibration Method', CALIBRATION_METHOD, ''])
    csv_data.append(['Variable Selection',
                     'Enabled' if USE_FILTERED_VARIABLES else 'Disabled', ''])

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv.writer(csvfile).writerows(csv_data)
    print(f"Evaluation CSV saved to: {csv_path}")


def _build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr=0.0):
    """Linear warmup + cosine decay LR schedule (Chen et al. 2020)."""
    init_lr = optimizer.param_groups[0]['lr']

    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            target = base_lr * float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            progress = min(1.0, max(0.0, progress))
            target = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
        return target / init_lr if init_lr > 0 else 0.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_simclr_all_data(model, dataset, optimizer,
                             epochs=SIMCLR_PRETRAIN_EPOCHS,
                             warmup_epochs=SIMCLR_WARMUP_EPOCHS,
                             base_lr=SIMCLR_BASE_LR,
                             min_lr=SIMCLR_MIN_LR):
    """One-shot unsupervised SimCLR pretraining; freezes encoder and projection head after."""
    print("\n" + "=" * 60)
    print("SimCLR unsupervised pretraining (all data)")
    print(f"  Schedule: warmup({warmup_epochs}) + cosine decay, {epochs} epochs, no early stop")
    print("=" * 60)

    all_subset = torch.utils.data.Subset(dataset, np.arange(len(dataset)))
    batch_size = min(256, max(2, len(all_subset) // 2))
    loader = DataLoader(all_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"  Pretrain samples: {len(all_subset)}, batch_size: {batch_size}")

    corruption = _build_scarf_corruption(all_subset) if USE_SCARF_AUGMENTATION else None
    if corruption is not None:
        print(f"  Using SCARF augmentation (corruption_rate={SCARF_CORRUPTION_RATE})")
    else:
        print("  Using remote-sensing augmentation (noise/brightness/contrast)")

    scheduler = _build_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs, base_lr, min_lr)

    for epoch in range(epochs):
        model.train()
        train_loss = train_simclr(model, loader, optimizer, corruption=corruption)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  SimCLR Pretrain Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss = {train_loss:.4f}, LR = {current_lr:.2e}")

    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.projection.parameters():
        param.requires_grad = False
    model.eval()
    print("  SimCLR encoder frozen, pretraining complete.\n")


def train_with_spatial_cv(dataset, model, output_dir):
    """Spatial CV: encoder frozen, only RF and calibrator trained per fold."""
    print("\nStarting spatial cross-validation training...")

    presence_gdf = dataset.get_presence_points_gdf()
    background_gdf = dataset.get_background_points_gdf()

    groups = None
    if presence_gdf is not None:
        print(f"Using spatial block CV, block size={SPATIAL_BLOCK_SIZE_KM} km")
        groups = _compute_spatial_blocks(presence_gdf, SPATIAL_BLOCK_SIZE_KM)

        if groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) < K_FOLDS:
                print(f"Warning: {len(unique_groups)} blocks < {K_FOLDS} folds, falling back to standard KFold")
                groups = None
            else:
                print(f"Spatial block CV ready: {len(unique_groups)} blocks, {K_FOLDS} folds")
                viz_dir = os.path.join(output_dir, "spatial_cv_visualization")
                os.makedirs(viz_dir, exist_ok=True)
                visualize_spatial_blocks(presence_gdf, groups, SPATIAL_BLOCK_SIZE_KM,
                                         os.path.join(viz_dir, "spatial_blocks_distribution.png"),
                                         "Presence Points Spatial Blocks")
                visualize_cv_splits(presence_gdf, groups, viz_dir, SPATIAL_BLOCK_SIZE_KM)

    presence_indices = np.where(dataset.data['label'] == 1)[0]
    background_indices = np.where(dataset.data['label'] == 0)[0]

    if groups is not None:
        split_iter = GroupKFold(n_splits=K_FOLDS).split(presence_indices, groups=groups)
        print("Using GroupKFold for spatial block CV")
    else:
        split_iter = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42).split(presence_indices)
        print("Using standard KFold (random CV)")

    best_val_metrics = None
    best_roc_auc = -float('inf')
    fold_metrics = []

    for fold, (train_presence_idx, val_presence_idx) in enumerate(split_iter):
        print(f"\nTraining fold {fold + 1}/{K_FOLDS}...")

        train_presence_global_idx = presence_indices[train_presence_idx]
        val_presence_global_idx = presence_indices[val_presence_idx]

        if BACKGROUND_CV_SPLIT and background_gdf is not None:
            if groups is not None:
                print("  Spatial split of background points...")
                train_bg_idx, val_bg_idx = spatial_split_background_points(
                    background_gdf, train_presence_idx, val_presence_idx,
                    presence_gdf, SPATIAL_BLOCK_SIZE_KM
                )
            else:
                print("  Random split of background points...")
                train_bg_idx, val_bg_idx = random_split_background_points(
                    background_gdf, fold, K_FOLDS, 42
                )
            train_bg_global_idx = background_indices[train_bg_idx]
            val_bg_global_idx = background_indices[val_bg_idx]
        else:
            print("  Using all background points (warning: potential data leakage)")
            train_bg_global_idx = background_indices
            val_bg_global_idx = background_indices

        train_idx = np.concatenate([train_presence_global_idx, train_bg_global_idx])
        val_idx = np.concatenate([val_presence_global_idx, val_bg_global_idx])

        print(f"  Train: {len(train_idx)} samples (presence: {len(train_presence_global_idx)}, "
              f"background: {len(train_bg_global_idx)})")
        print(f"  Val: {len(val_idx)} samples (presence: {len(val_presence_global_idx)}, "
              f"background: {len(val_bg_global_idx)})")

        print(f"  [FOLD-HASH] fold {fold + 1}: train_head = {sorted(train_idx.tolist())[:10]}, "
              f"val_head = {sorted(val_idx.tolist())[:10]}, "
              f"n_train = {len(train_idx)}, n_val = {len(val_idx)}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        batch_size = max(min(512, len(train_idx) // 2, len(val_idx) // 2), 2)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)

        print("  Phase 1: Using frozen pretrained SimCLR encoder")

        print("  Phase 2: Training RF + calibrator...")
        rf_classifier, calibrator = train_rf_classifier(model, train_loader, val_loader,
                                                        CALIBRATION_METHOD)

        print("  Phase 3: Evaluating current fold...")
        final_val_metrics = evaluate_model(model, val_loader, threshold_method='max_tss')

        fold_record = {'fold': fold + 1}
        fold_record.update({k: final_val_metrics[k] for k in METRIC_KEYS})
        fold_metrics.append(fold_record)

        print(f"Fold {fold + 1} results: " + ", ".join(
            f"{METRIC_LABELS[k]}={final_val_metrics[k]:.4f}" for k in METRIC_KEYS))

        if final_val_metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = final_val_metrics['roc_auc']
            best_val_metrics = final_val_metrics
            torch.save({
                'simclr_state_dict': model.state_dict(),
                'rf_classifier': rf_classifier,
                'calibrator': calibrator,
                'calibration_method': CALIBRATION_METHOD
            }, model_path)
            print(f"Saved best model (ROC-AUC: {final_val_metrics['roc_auc']:.4f})")

    if fold_metrics:
        print("\nSpatial CV summary:")
        print("=" * 60)
        for key in METRIC_KEYS:
            values = [m[key] for m in fold_metrics]
            print(f"  {METRIC_LABELS[key]:<12} {np.mean(values):.4f} ± {np.std(values):.4f}")
        print("=" * 60)

        results_path = os.path.join(output_dir, f"spatial_cv_results_{try_times}.csv")
        pd.DataFrame(fold_metrics).to_csv(results_path, index=False)
        print(f"CV results saved to: {results_path}")

        save_cv_summary_to_csv(fold_metrics, os.path.join(output_dir, f"cv_summary_{try_times}.csv"))

    return best_val_metrics


def _parse_cli_args():
    """Parse CLI args to override variable selection config."""
    import sys
    global USE_FILTERED_VARIABLES, VARIABLE_SELECTION_METHOD
    global CORRELATION_THRESHOLD, VIF_THRESHOLD

    if len(sys.argv) <= 1:
        return

    print("Parsing CLI args...")
    for arg in sys.argv[1:]:
        low = arg.lower()
        if low == "no_filter":
            USE_FILTERED_VARIABLES = False
            print("CLI: variable selection disabled")
        elif low.startswith("method="):
            method = low.split("=")[1]
            if method in ['correlation', 'vif', 'combined']:
                VARIABLE_SELECTION_METHOD = method
                print(f"CLI: selection method = {method}")
            else:
                print(f"Warning: invalid method {method}, using default {VARIABLE_SELECTION_METHOD}")
        elif low.startswith("corr_threshold="):
            try:
                CORRELATION_THRESHOLD = float(arg.split("=")[1])
                print(f"CLI: corr_threshold = {CORRELATION_THRESHOLD}")
            except ValueError:
                print(f"Warning: invalid corr_threshold {arg}, using default {CORRELATION_THRESHOLD}")
        elif low.startswith("vif_threshold="):
            try:
                VIF_THRESHOLD = float(arg.split("=")[1])
                print(f"CLI: vif_threshold = {VIF_THRESHOLD}")
            except ValueError:
                print(f"Warning: invalid vif_threshold {arg}, using default {VIF_THRESHOLD}")


def main(presence_path, background_path, env_dir, output_dir, model_path):
    print_usage()
    _parse_cli_args()

    print("\nCurrent config:")
    print(f"   Variable selection: {'enabled' if USE_FILTERED_VARIABLES else 'disabled'}")
    if USE_FILTERED_VARIABLES:
        print(f"   Method: {VARIABLE_SELECTION_METHOD}")
        print(f"   Corr threshold: {CORRELATION_THRESHOLD}")
        print(f"   VIF threshold: {VIF_THRESHOLD}")
    print()

    presence_data = pd.read_csv(presence_path)
    background_data = pd.read_csv(background_path)

    filtered_env_files = None
    if USE_FILTERED_VARIABLES:
        _, _, filtered_var_names = prepare_and_split_data(
            presence_data, background_data, env_dir, output_dir,
            study_area_shp_file=STUDY_AREA_SHP_FILE,
            use_filtered_variables=True
        )
        print(f"Variable selection done, {len(filtered_var_names)} vars retained")

        filtered_env_files = [
            f for f in sorted(glob.glob(os.path.join(env_dir, "*.asc")))
            if os.path.basename(f).split('.')[0] in filtered_var_names
        ]

        model_vars_path = os.path.join(
            output_dir, f"model_variables_{os.path.basename(model_path).replace('.pth', '.txt')}")
        with open(model_vars_path, 'w') as f:
            f.write("# Environmental variables used by the model\n")
            f.write(f"# selection_method: {VARIABLE_SELECTION_METHOD}\n")
            f.write(f"# corr_threshold: {CORRELATION_THRESHOLD}\n")
            f.write(f"# vif_threshold: {VIF_THRESHOLD}\n\n")
            for var in filtered_var_names:
                f.write(f"- {var}\n")
        print(f"Model variable list saved to: {model_vars_path}")
    else:
        print("Skipping variable selection, using all env vars")

    dataset = HabitatDataset(presence_data, background_data, env_dir, filtered_env_files)

    input_dim = len(dataset.env_files)
    model = SimCLRModel(input_dim, hidden_dim=256, output_dim=256).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    pretrain_simclr_all_data(model, dataset, optimizer)

    print("Training with spatial CV...")
    train_with_spatial_cv(dataset, model, output_dir)

    print("\nLoading best model...")
    model_save_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(model_save_dict['simclr_state_dict'])
    model.rf_classifier = model_save_dict['rf_classifier']
    model.calibrator = model_save_dict['calibrator']
    model.calibration_method = model_save_dict.get('calibration_method', 'isotonic')
    model.eval()

    print("\nFinal model evaluation:")
    full_loader = DataLoader(dataset, batch_size=2014, shuffle=False)
    final_metrics = evaluate_model(model, full_loader, threshold_method='max_tss')
    for key in METRIC_KEYS:
        print(f"{METRIC_LABELS[key]}: {final_metrics[key]:.4f}")

    save_evaluation_results(final_metrics,
                            os.path.join(output_dir, f"all_points_evaluation_{try_times}.xlsx"))
    save_rf_classifier(model, output_dir, try_times)

    if USE_PPM_LITE:
        print("\nStarting PPM-lite HSI prediction...")
        ppm_output_path = output_path.replace('.tif', '_ppm_lite.tif')
        predict_hsi_with_ppm_lite(
            model, env_dir, ppm_output_path, mask_shp_path,
            intensity_method=PPM_INTENSITY_METHOD,
            normalization_method=PPM_NORMALIZATION_METHOD,
            env_files=filtered_env_files
        )
    else:
        print("\nSkipping PPM-lite HSI prediction (disabled)")

    print("\nOutput files:")
    print("=" * 60)
    print(f"  {model_path}")
    print(f"  {os.path.join(output_dir, f'rf_classifier_{try_times}.joblib')}")
    print(f"  {os.path.join(output_dir, f'spatial_cv_results_{try_times}.csv')}")
    print(f"  {os.path.join(output_dir, f'cv_summary_{try_times}.csv')}")
    print(f"  {os.path.join(output_dir, f'all_points_evaluation_{try_times}.xlsx')}")
    print(f"  {os.path.join(output_dir, f'all_points_evaluation_{try_times}.csv')}")
    if USE_PPM_LITE:
        print(f"  {output_path.replace('.tif', '_ppm_lite.tif')} (PPM-lite HSI)")
    if USE_FILTERED_VARIABLES:
        print(f"  {os.path.join(output_dir, 'variable_selection/')}")
    print("=" * 60)
    print("\nAll tasks complete!")


# ===========================================================================
# User-configurable paths (edit these to match your local directory layout)
# ===========================================================================
date = datetime.now().strftime('%Y%m%d_%H%M%S')
points_num = '10x'
try_times = f"{points_num}p_{date}_simclr_rf_bushaixuan"

PRESENCE_POINTS_FILE = r"./data/presence_points.csv"
BACKGROUND_POINTS_FILE = r"./data/background_points.csv"
ENV_VAR_RASTER_FILES = r"./data/env_rasters"
STUDY_AREA_SHP_FILE = r"./data/study_area.shp"
OUTPUT_DIR = f"./output/hsi_map_result_{try_times}/"

RANDOM_STATE = 18

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["help", "-h", "--help"]:
        print_usage()
        sys.exit(0)

    set_random_seed(RANDOM_STATE)

    model_path = os.path.join(OUTPUT_DIR, f"best_model_{try_times}.pth")
    output_path = os.path.join(OUTPUT_DIR, f"hsi_prediction_{try_times}.tif")
    mask_shp_path = STUDY_AREA_SHP_FILE

    main(
        presence_path=PRESENCE_POINTS_FILE,
        background_path=BACKGROUND_POINTS_FILE,
        env_dir=ENV_VAR_RASTER_FILES,
        output_dir=OUTPUT_DIR,
        model_path=model_path
    )
