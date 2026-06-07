# SimCLR model for Habitat Suitability Index (HSI) mapping

# See README.md alongside this file for usage instructions.

import os
import glob
import random
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import rasterio
from rasterio.features import geometry_mask
import geopandas as gpd

from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

# Output naming
date = datetime.now().strftime('%Y%m%d_%H%M%S')
points_num = '1x'
try_times = str(points_num) + date + '_simclr'

# Input paths
PRESENCE_POINTS_FILE = r".\data\occruence\Rarefy GP occurence example.csv"
BACKGROUND_POINTS_FILE = r".\data\background\MCMC samples\background_points_duoyangxing_1715p_1x\mcmc_background_points_duoyangxing_1715p_1x.csv"
ENV_VAR_RASTER_FILES = r".\data\environmental\Ascii"
STUDY_AREA_SHP_FILE = r"shap file here"
OUTPUT_DIR = fr"./result/simclr_{try_times}/"   # .\results\Simclr_evaluation_mapping\


# Column names in the presence/background CSVs
LONGITUDE_COL = 'Longitude'
LATITUDE_COL = 'Latitude'

# Random seed
RANDOM_STATE = 18

# Cross-validation
USE_SPATIAL_CV = True
SPATIAL_BLOCK_SIZE_KM = 50
K_FOLDS = 5
BACKGROUND_CV_SPLIT = True   # Split background points per fold (avoids leakage)

# ============================================================================
# Utilities
# ============================================================================

def set_random_seed(seed=38):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


# ============================================================================
# SimCLR model and dataset
# ============================================================================

class SimCLRModel(nn.Module):
    """
    SimCLR encoder + projection head.
    The Random Forest classifier and probability calibrator are attached after
    self-supervised training (model.rf_classifier, model.calibrator).
    """

    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
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
            nn.Linear(hidden_dim, output_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
        self.rf_classifier = None
        self.calibrator = None

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return h, z

    def get_features(self, x):
        """Return encoder output for downstream RF training/inference."""
        return self.encoder(x)


class RemoteSensingAugmentation:
    """Lightweight augmentation for tabular remote-sensing features."""

    def __init__(self, noise_level=0.1, brightness_range=0.2, contrast_range=0.2, mixup_alpha=0.2):
        self.noise_level = noise_level
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.mixup_alpha = mixup_alpha

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
        out = data.clone()
        if torch.rand(1, device=data.device) < 0.5:
            out = self.add_gaussian_noise(out)
        if torch.rand(1, device=data.device) < 0.5:
            out = self.adjust_brightness(out)
        if torch.rand(1, device=data.device) < 0.5:
            out = self.adjust_contrast(out)
        return out


class HabitatDataset(Dataset):
    """
    Dataset combining presence (label=1) and background (label=0) points.
    Each sample yields the per-point environmental feature vector.
    """

    def __init__(self, presence_data, background_data, env_dir):
        self.presence_data = presence_data
        self.background_data = background_data

        self.env_files = sorted(glob.glob(os.path.join(env_dir, "*.asc")))
        print(f"Using {len(self.env_files)} env raster files")

        print("Loading env raster arrays...")
        self.env_data = []
        self.env_transforms = []
        for env_file in tqdm(self.env_files):
            with rasterio.open(env_file) as src:
                self.env_data.append(src.read(1))
                self.env_transforms.append(src.transform)

        self.data = pd.concat([
            pd.DataFrame({
                'longitude': presence_data['Longitude'],
                'latitude': presence_data['Latitude'],
                'label': 1
            }),
            pd.DataFrame({
                'longitude': background_data['Longitude'],
                'latitude': background_data['Latitude'],
                'label': 0
            })
        ])
        self.points_gdf = gpd.GeoDataFrame(
            self.data,
            geometry=gpd.points_from_xy(self.data['longitude'], self.data['latitude']),
            crs="EPSG:4326"
        )

        print("Extracting per-point environmental features...")
        self.features = self._extract_features()

    def _extract_features(self):
        features = []
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            x, y = row['longitude'], row['latitude']
            vec = []
            for env_arr, transform in zip(self.env_data, self.env_transforms):
                try:
                    r, c = rasterio.transform.rowcol(transform, x, y)
                    if 0 <= r < env_arr.shape[0] and 0 <= c < env_arr.shape[1]:
                        vec.append(env_arr[r, c])
                    else:
                        vec.append(np.nan)
                except Exception:
                    vec.append(np.nan)
            features.append(vec)
        return np.array(features)

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


# ============================================================================
# Training: SimCLR contrastive loss + RF classifier + calibration
# ============================================================================

def train_simclr(model, train_loader, optimizer, temperature=0.5):
    """One epoch of SimCLR contrastive training."""
    model.train()
    total_loss = 0
    augmentation = RemoteSensingAugmentation(
        noise_level=0.2, brightness_range=0.3, contrast_range=0.3, mixup_alpha=0.3)

    n_batches = 0
    for data, _ in train_loader:
        if len(data) < 2:
            continue   # BatchNorm requires batch size >= 2
        data = data.cuda()
        d1 = augmentation.apply_augmentation(data)
        d2 = augmentation.apply_augmentation(data)

        _, z1 = model(d1)
        _, z2 = model(d2)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        similarity = torch.matmul(z1, z2.T) / temperature
        targets = torch.arange(len(data)).cuda()
        loss = F.cross_entropy(similarity, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_rf_classifier(model, train_loader, val_loader):
    """Train the Random Forest classifier on SimCLR features and fit a sigmoid calibrator."""
    print("  Training RF classifier and sigmoid (Platt) calibrator...")
    model.eval()

    train_features, train_labels = [], []
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.cuda()
            train_features.append(model.get_features(data).cpu().numpy())
            train_labels.extend(labels.numpy().flatten())
    train_features = np.vstack(train_features)
    train_labels = np.array(train_labels)
    print(f"    Train features: {train_features.shape}, label counts: {np.bincount(train_labels.astype(int))}")

    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_classifier.fit(train_features, train_labels)

    val_features, val_labels = [], []
    with torch.no_grad():
        for data, labels in val_loader:
            data = data.cuda()
            val_features.append(model.get_features(data).cpu().numpy())
            val_labels.extend(labels.numpy().flatten())
    val_features = np.vstack(val_features)
    val_labels = np.array(val_labels)

    val_rf_probs = rf_classifier.predict_proba(val_features)[:, 1]

    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(val_rf_probs.reshape(-1, 1), val_labels)
    print("    Calibration: sigmoid (Platt scaling)")

    model.rf_classifier = rf_classifier
    model.calibrator = calibrator

    return rf_classifier, calibrator


# ============================================================================
# Evaluation metrics: ROC-AUC, PR-AUC, TSS, Specificity, ECE
# ============================================================================

def calculate_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error using equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        prop = in_bin.mean()
        if prop > 0:
            acc = y_true[in_bin].mean()
            conf = y_prob[in_bin].mean()
            ece += np.abs(conf - acc) * prop
    return ece


def calculate_metrics(y_true, y_prob, threshold=None):
    """
    Compute ROC-AUC, PR-AUC, TSS, Specificity, ECE.

    If threshold is None, picks the threshold maximizing TSS.
    Returns a dict with keys: auc, pr_auc, tss, specificity, ece, threshold.
    """
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()

    if threshold is None:
        thresholds = np.unique(y_prob)
        if len(thresholds) > 100:
            thresholds = np.percentile(y_prob, np.linspace(0, 100, 101))
        best_tss, best_threshold = -np.inf, 0.5
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            tss = sens + spec - 1
            if tss > best_tss:
                best_tss, best_threshold = tss, t
        threshold = best_threshold

    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = sens + spec - 1

    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        auc = 0.0
    try:
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        pr_auc = 0.0
    ece = calculate_ece(y_true, y_prob)

    return {
        'auc': auc,
        'pr_auc': pr_auc,
        'tss': tss,
        'specificity': spec,
        'ece': ece,
        'threshold': float(threshold)
    }


def evaluate_model(model, val_loader):
    """
    Evaluate the model (RF + sigmoid calibrator) on a validation loader.

    Returns the same dict as calculate_metrics: auc, pr_auc, tss, specificity, ece, threshold.
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.cuda()
            features = model.get_features(data).cpu().numpy()
            rf_probs = model.rf_classifier.predict_proba(features)[:, 1]

            if getattr(model, 'calibrator', None) is not None:
                cal = model.calibrator.predict_proba(rf_probs.reshape(-1, 1))[:, 1]
            else:
                cal = rf_probs

            all_probs.extend(cal)
            all_labels.extend(labels.cpu().numpy())

    return calculate_metrics(np.array(all_labels).flatten(), np.array(all_probs))


# ============================================================================
# Spatial cross-validation helpers
# ============================================================================

def _compute_spatial_blocks(points_gdf, block_size_km, crs_epsg_for_meter=3857):
    """
    Generate spatial block IDs for each point, used by GroupKFold.
    Projects to a metric CRS and assigns grid-cell IDs.
    """
    if points_gdf is None or points_gdf.empty:
        return None

    try:
        if points_gdf.crs.to_epsg() != crs_epsg_for_meter:
            gdf = points_gdf.to_crs(epsg=crs_epsg_for_meter)
            print(f"  CRS conversion: {points_gdf.crs} -> EPSG:{crs_epsg_for_meter}")
        else:
            gdf = points_gdf.copy()
    except Exception as e:
        print(f"  Warning: CRS conversion failed ({e}); using original CRS")
        gdf = points_gdf.copy()

    coords = np.stack([gdf.geometry.x.values, gdf.geometry.y.values], axis=1)
    block_m = max(1000.0, float(block_size_km) * 1000.0)
    min_x, min_y = coords.min(axis=0)
    bx = np.floor((coords[:, 0] - min_x) / block_m).astype(np.int64)
    by = np.floor((coords[:, 1] - min_y) / block_m).astype(np.int64)
    groups = np.char.add(bx.astype(str), '_')
    groups = np.char.add(groups, by.astype(str))

    unique_groups, group_counts = np.unique(groups, return_counts=True)
    print(f"  Generated {len(unique_groups)} spatial blocks (block size={block_size_km}km), "
          f"points per block: {group_counts.min()}-{group_counts.max()}, mean {group_counts.mean():.1f}")
    return groups


def spatial_split_background_points(background_gdf, presence_train_idx, presence_val_idx,
                                    presence_gdf, block_size_km=50):
    """Split background points spatially consistent with the presence fold split."""
    print(f"  Splitting background points spatially (block size={block_size_km}km)...")

    train_presence_gdf = presence_gdf.iloc[presence_train_idx].reset_index(drop=True)
    val_presence_gdf = presence_gdf.iloc[presence_val_idx].reset_index(drop=True)

    background_groups = _compute_spatial_blocks(background_gdf, block_size_km)
    if background_groups is None:
        print("  Warning: spatial grouping failed for background points; using random split")
        return _fallback_random_split(len(background_gdf))

    train_presence_groups = _compute_spatial_blocks(train_presence_gdf, block_size_km)
    val_presence_groups = _compute_spatial_blocks(val_presence_gdf, block_size_km)
    if train_presence_groups is None or val_presence_groups is None:
        print("  Warning: spatial grouping failed for presence points; using random split")
        return _fallback_random_split(len(background_gdf))

    train_blocks = set(train_presence_groups)
    val_blocks = set(val_presence_groups)

    train_bg_mask = np.array([g in train_blocks for g in background_groups])
    val_bg_mask = np.array([g in val_blocks for g in background_groups])

    overlap_blocks = train_blocks.intersection(val_blocks)
    if overlap_blocks:
        overlap_mask = np.array([g in overlap_blocks for g in background_groups])
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

    unassigned = np.where(~(train_bg_mask | val_bg_mask))[0]
    if len(unassigned) > 0:
        np.random.shuffle(unassigned)
        split_point = len(unassigned) // 2
        train_bg_idx = np.concatenate([train_bg_idx, unassigned[:split_point]])
        val_bg_idx = np.concatenate([val_bg_idx, unassigned[split_point:]])

    if len(train_bg_idx) < 100 or len(val_bg_idx) < 100:
        print(f"  Warning: too few background points after spatial split "
              f"(train:{len(train_bg_idx)}, val:{len(val_bg_idx)}); using random split")
        return _fallback_random_split(len(background_gdf))

    print(f"  Background spatial split complete: train={len(train_bg_idx)}, val={len(val_bg_idx)}")
    return train_bg_idx, val_bg_idx


def _fallback_random_split(total_bg_points):
    indices = np.arange(total_bg_points)
    np.random.shuffle(indices)
    split = total_bg_points // 2
    return indices[:split], indices[split:]


def random_split_background_points(background_gdf, fold_idx, total_folds, random_state=38):
    """Randomly split background points for the given fold."""
    np.random.seed(random_state + fold_idx)
    indices = np.arange(len(background_gdf))
    np.random.shuffle(indices)
    fold_size = len(indices) // total_folds
    val_start = fold_idx * fold_size
    val_end = (fold_idx + 1) * fold_size if fold_idx < total_folds - 1 else len(indices)
    val_bg_idx = indices[val_start:val_end]
    train_bg_idx = np.concatenate([indices[:val_start], indices[val_end:]])
    print(f"  Background random split complete: train={len(train_bg_idx)}, val={len(val_bg_idx)}")
    return train_bg_idx, val_bg_idx


# ============================================================================
# Visualization (spatial blocks and CV folds)
# ============================================================================

def visualize_spatial_blocks(points_gdf, groups, block_size_km, output_path,
                             title="Spatial Blocks Distribution"):
    """Plot spatial block membership for the given points."""
    try:
        if points_gdf.crs.to_epsg() != 3857:
            gdf_proj = points_gdf.to_crs(epsg=3857)
        else:
            gdf_proj = points_gdf.copy()

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        x = gdf_proj.geometry.x.values
        y = gdf_proj.geometry.y.values

        unique_groups = np.unique(groups)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
        group_colors = {g: colors[i] for i, g in enumerate(unique_groups)}
        for g in unique_groups:
            mask = groups == g
            ax.scatter(x[mask], y[mask], c=[group_colors[g]], label=f'Block {g}', alpha=0.7, s=20)

        block_m = block_size_km * 1000
        grid_x = np.arange(np.floor(x.min() / block_m) * block_m,
                           np.ceil(x.max() / block_m) * block_m + block_m, block_m)
        grid_y = np.arange(np.floor(y.min() / block_m) * block_m,
                           np.ceil(y.max() / block_m) * block_m + block_m, block_m)
        for gx in grid_x:
            ax.axvline(x=gx, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        for gy in grid_y:
            ax.axhline(y=gy, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('X Coordinate (m)', fontsize=12)
        ax.set_ylabel('Y Coordinate (m)', fontsize=12)
        ax.set_title(f'{title}\nBlock size: {block_size_km}km, total blocks: {len(unique_groups)}',
                     fontsize=14)
        ax.grid(True, alpha=0.3)
        if len(unique_groups) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Spatial block plot saved to: {output_path}")
    except Exception as e:
        print(f"  Warning: spatial block plot failed ({e})")


def visualize_cv_splits(points_gdf, groups, output_dir, block_size_km):
    """Plot the train/val split for each CV fold."""
    try:
        gkf = GroupKFold(n_splits=K_FOLDS)
        if points_gdf.crs.to_epsg() != 3857:
            gdf_proj = points_gdf.to_crs(epsg=3857)
        else:
            gdf_proj = points_gdf.copy()
        x = gdf_proj.geometry.x.values
        y = gdf_proj.geometry.y.values

        for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(points_gdf)), groups=groups)):
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.scatter(x[train_idx], y[train_idx], c='blue', alpha=0.6, s=20,
                       label=f'Training Set ({len(train_idx)} points)')
            ax.scatter(x[val_idx], y[val_idx], c='red', alpha=0.8, s=30,
                       label=f'Validation Set ({len(val_idx)} points)')

            block_m = block_size_km * 1000
            grid_x = np.arange(np.floor(x.min() / block_m) * block_m,
                               np.ceil(x.max() / block_m) * block_m + block_m, block_m)
            grid_y = np.arange(np.floor(y.min() / block_m) * block_m,
                               np.ceil(y.max() / block_m) * block_m + block_m, block_m)
            for gx in grid_x:
                ax.axvline(x=gx, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
            for gy in grid_y:
                ax.axhline(y=gy, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

            ax.set_xlabel('X Coordinate (m)', fontsize=12)
            ax.set_ylabel('Y Coordinate (m)', fontsize=12)
            ax.set_title(f'Spatial Block Cross-Validation - Fold {fold + 1}\n'
                         f'Block size: {block_size_km}km', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"spatial_cv_fold_{fold + 1}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
        print(f"  CV split plots saved to: {output_dir}")
    except Exception as e:
        print(f"  Warning: CV split plot failed ({e})")


# ============================================================================
# HSI map prediction (PPM-lite)
# ============================================================================

def _load_env_rasters(env_files):
    """Load env raster files into a stacked numpy array. Returns (stack, meta)."""
    with rasterio.open(env_files[0]) as src_template:
        meta = src_template.meta.copy()
        h, w = src_template.height, src_template.width
        crs = src_template.crs
        transform = src_template.transform

    stack = np.zeros((len(env_files), h, w), dtype=np.float32)
    for i, raster_path in enumerate(tqdm(env_files, desc="Loading env rasters")):
        with rasterio.open(raster_path) as src:
            img = src.read(1).astype(np.float32)
            nodata = src.nodatavals[0] if src.nodatavals else None
            if nodata is not None:
                img[img == nodata] = np.nan
            img[img == -9999] = np.nan
            stack[i] = img
    return stack, meta, transform, crs, h, w


def _study_area_mask(study_area_gdf, transform, crs, height, width):
    if study_area_gdf is None:
        return np.ones((height, width), dtype=bool)
    if study_area_gdf.crs != crs:
        study_area_gdf = study_area_gdf.to_crs(crs)
    return geometry_mask(study_area_gdf.geometry, out_shape=(height, width),
                         transform=transform, invert=True, all_touched=True)


def compute_ppm_lite_intensity(model, features):
    """
    Compute PPM-lite relative intensity from the Platt-calibrated logits
    of the RF classifier (i.e. the 'platt_logit' strategy).

    Args:
        model: Trained model with rf_classifier and a sigmoid calibrator attached.
        features: SimCLR feature matrix.
    """
    if model.rf_classifier is None:
        raise ValueError("RF classifier not trained")
    rf_probs = model.rf_classifier.predict_proba(features)[:, 1]

    if getattr(model, 'calibrator', None) is not None:
        logit_values = model.calibrator.decision_function(rf_probs.reshape(-1, 1))
        intensity = np.exp(logit_values)
    else:
        intensity = rf_probs

    intensity = np.maximum(intensity, 1e-10)
    intensity = np.nan_to_num(intensity, nan=1e-10, posinf=1e10, neginf=1e-10)
    return intensity


def normalize_intensity_to_probability(intensity):
    """
    Map relative intensity to a probability map using rank-based quantile
    normalization (output values lie in [0, 1]).
    """
    from scipy.stats import rankdata

    intensity = np.array(intensity)
    ranks = rankdata(intensity, method='average')
    probs = (ranks - 1) / (len(ranks) - 1)
    probs = np.maximum(probs, 0)
    return np.nan_to_num(probs, nan=0, posinf=1, neginf=0)


def predict_hsi_with_ppm_lite(model, env_dir, output_path, mask_shp_path=None,
                              env_files=None):
    """
    Generate the HSI map (PPM-lite: Platt-calibrated logits + quantile normalization).

    The output is a single GeoTIFF with values in [0, 1].
    """
    print("Generating HSI map (PPM-lite: platt_logit + quantile)...")

    if model.rf_classifier is None:
        raise ValueError("RF classifier not trained")
    if getattr(model, 'calibrator', None) is None:
        print("Warning: calibrator not found; falling back to RF probs")

    model.eval()

    if env_files is None:
        env_files = sorted(glob.glob(os.path.join(env_dir, "*.asc")))
    if not env_files:
        raise ValueError(f"No .asc files found in {env_dir}")
    print(f"Found {len(env_files)} env raster files")

    stack, meta, transform, crs, height, width = _load_env_rasters(env_files)

    if mask_shp_path and os.path.exists(mask_shp_path):
        print("Using study area mask...")
        study_area = gpd.read_file(mask_shp_path)
    else:
        print("No mask provided; processing entire raster extent")
        study_area = None
    mask = _study_area_mask(study_area, transform, crs, height, width)

    rows, cols = np.where(mask)
    print(f"Study area pixels: {len(rows)}")
    if len(rows) == 0:
        print("Warning: empty study area mask; aborting")
        return

    filled = np.copy(stack)
    for v in range(stack.shape[0]):
        nan_mask = np.isnan(filled[v])
        filled[v, rows, cols] = np.where(nan_mask[rows, cols], 0.0, filled[v, rows, cols])
    env_data = filled[:, rows, cols].T
    env_tensor = torch.tensor(env_data, dtype=torch.float32).cuda()

    print(f"Computing PPM-lite intensity for {len(env_tensor)} pixels...")
    batch_size = 5000
    intensities = []
    with torch.no_grad():
        for i in tqdm(range(0, len(env_tensor), batch_size), desc="PPM-lite batches"):
            batch = env_tensor[i:i + batch_size]
            features = model.get_features(batch).cpu().numpy()
            batch_intensity = compute_ppm_lite_intensity(model, features)
            intensities.extend(batch_intensity.flatten())
    intensities = np.array(intensities)

    print("Normalizing intensity to probability (quantile)...")
    probs = normalize_intensity_to_probability(intensities)

    ppm_map = np.full((height, width), np.nan, dtype=np.float32)
    ppm_map[rows, cols] = probs.astype(np.float32)

    meta.update(count=1, dtype='float32', nodata=np.nan)
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(ppm_map, 1)
    print(f"HSI map saved to {output_path}")

    print("\nHSI statistics:")
    print(f"  Probability: min={np.min(probs):.6f}, max={np.max(probs):.6f}, "
          f"mean={np.mean(probs):.6f}")


# ============================================================================
# Save / report helpers
# ============================================================================

def save_rf_and_calibrator(model, output_dir, try_times):
    """Save the RF classifier and sigmoid calibrator to joblib files."""
    if model.rf_classifier is None:
        return
    rf_path = os.path.join(output_dir, f"rf_classifier_{try_times}.joblib")
    joblib.dump(model.rf_classifier, rf_path)
    print(f"RF classifier saved to: {rf_path}")

    if model.calibrator is not None:
        cal_path = os.path.join(output_dir, f"calibrator_{try_times}.joblib")
        joblib.dump(model.calibrator, cal_path)
        print(f"Calibrator saved to: {cal_path}")


def save_cv_summary_to_csv(fold_metrics, output_path):
    """Save mean/std summary of CV fold metrics to a CSV file."""
    print("Saving cross-validation summary to CSV...")
    metrics_to_summarize = ['auc', 'pr_auc', 'tss', 'specificity', 'ece']

    rows = [['Metric', 'Mean', 'Std_Dev', 'Min', 'Max', 'Median']]
    for metric in metrics_to_summarize:
        if metric in fold_metrics[0]:
            values = [m[metric] for m in fold_metrics]
            rows.append([
                metric.upper(),
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{np.max(values):.4f}",
                f"{np.median(values):.4f}"
            ])

    rows.append(['', '', '', '', '', ''])
    rows.append(['Configuration', '', '', '', '', ''])
    rows.append(['Total_Folds', str(len(fold_metrics)), '', '', '', ''])
    rows.append(['Spatial_CV', 'True' if USE_SPATIAL_CV else 'False', '', '', '', ''])
    rows.append(['Block_Size_km', str(SPATIAL_BLOCK_SIZE_KM), '', '', '', ''])
    rows.append(['Calibration_Method', 'sigmoid', '', '', '', ''])

    import csv
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"CV summary saved to: {output_path}")


def save_evaluation_results(metrics, output_path):
    """
    Save evaluation results to both Excel and CSV.
    Only saves the five core metrics: AUC, PR-AUC, TSS, Specificity, ECE.
    """
    print("Saving evaluation results to Excel and CSV...")

    metric_descriptions = {
        'auc': 'Area Under ROC Curve',
        'pr_auc': 'Area Under Precision-Recall Curve',
        'tss': 'True Skill Statistic',
        'specificity': 'True Negative Rate',
        'ece': 'Expected Calibration Error',
        'threshold': 'TSS-maximizing classification threshold'
    }
    metric_order = ['auc', 'pr_auc', 'tss', 'specificity', 'ece', 'threshold']

    # Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Model Evaluation Results"
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    center = Alignment(horizontal="center", vertical="center")

    ws.merge_cells('A1:C1')
    ws['A1'] = "SimCLR + RF + Calibration Model Evaluation Results"
    ws['A1'].font = Font(bold=True, size=16)
    ws['A1'].alignment = center

    for col, header in enumerate(["Metric", "Value", "Description"], 1):
        cell = ws.cell(row=3, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center

    row = 4
    for key in metric_order:
        if key in metrics:
            ws.cell(row=row, column=1, value=key.upper()).alignment = center
            ws.cell(row=row, column=2, value=round(metrics[key], 4)).alignment = center
            ws.cell(row=row, column=3, value=metric_descriptions[key]).alignment = center
            row += 1

    ws.cell(row=row + 1, column=1, value="Model Configuration").font = Font(bold=True)
    config_lines = [
        "Architecture: SimCLR + RandomForest + sigmoid (Platt) calibration",
        f"Spatial CV: {'enabled' if USE_SPATIAL_CV else 'disabled'}, "
        f"block size {SPATIAL_BLOCK_SIZE_KM}km, {K_FOLDS} folds",
        "Threshold selection: TSS-maximizing"
    ]
    for i, line in enumerate(config_lines, start=row + 2):
        ws.cell(row=i, column=1, value=line)

    for col_letter in ['A', 'B', 'C']:
        ws.column_dimensions[col_letter].width = 35
    wb.save(output_path)
    print(f"Excel saved to: {output_path}")

    # CSV
    csv_path = output_path.replace('.xlsx', '.csv')
    rows = [['Metric', 'Value', 'Description']]
    for key in metric_order:
        if key in metrics:
            rows.append([key.upper(), round(metrics[key], 4), metric_descriptions[key]])
    rows.append(['', '', ''])
    rows.append(['Model Configuration', '', ''])
    for line in config_lines:
        rows.append([line, '', ''])

    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print(f"CSV saved to: {csv_path}")


# ============================================================================
# Spatial cross-validation training loop
# ============================================================================

def train_with_spatial_cv(dataset, model, optimizer, scheduler, output_dir, model_path):
    """
    Train using spatial-block (or random) K-fold cross-validation.
    Each fold trains the SimCLR encoder, then the RF + calibrator, then optionally
    the PU correction. The best fold (by validation AUC) is saved to disk.

    Returns the validation metrics of the best fold.
    """
    print("\nStarting spatial cross-validation training...")

    presence_gdf = dataset.get_presence_points_gdf()
    background_gdf = dataset.get_background_points_gdf()

    groups = None
    if USE_SPATIAL_CV and presence_gdf is not None:
        print(f"Setting up spatial block GroupKFold (block size={SPATIAL_BLOCK_SIZE_KM}km)...")
        groups = _compute_spatial_blocks(presence_gdf, SPATIAL_BLOCK_SIZE_KM)
        if groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) < K_FOLDS:
                print(f"Warning: only {len(unique_groups)} blocks for {K_FOLDS} folds; "
                      f"falling back to random KFold")
                groups = None
            else:
                print(f"Spatial block GroupKFold: {len(unique_groups)} blocks, {K_FOLDS} folds")
                viz_dir = os.path.join(output_dir, "spatial_cv_visualization")
                os.makedirs(viz_dir, exist_ok=True)
                visualize_spatial_blocks(presence_gdf, groups, SPATIAL_BLOCK_SIZE_KM,
                                         os.path.join(viz_dir, "spatial_blocks_distribution.png"),
                                         "Presence Points Spatial Blocks")
                visualize_cv_splits(presence_gdf, groups, viz_dir, SPATIAL_BLOCK_SIZE_KM)

    presence_indices = np.where(dataset.data['label'] == 1)[0]
    background_indices = np.where(dataset.data['label'] == 0)[0]

    if groups is not None:
        splitter = GroupKFold(n_splits=K_FOLDS)
        split_iter = splitter.split(presence_indices, groups=groups)
        print("Using spatial-block GroupKFold")
    else:
        splitter = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        split_iter = splitter.split(presence_indices)
        print("Using standard KFold")

    best_metrics = None
    best_score = -float('inf')
    best_fold = -1
    fold_metrics = []

    for fold, (train_pres_idx, val_pres_idx) in enumerate(split_iter):
        print(f"\nTraining fold {fold + 1}/{K_FOLDS}...")
        train_presence_global = presence_indices[train_pres_idx]
        val_presence_global = presence_indices[val_pres_idx]

        if BACKGROUND_CV_SPLIT and background_gdf is not None:
            if USE_SPATIAL_CV and groups is not None:
                train_bg_idx, val_bg_idx = spatial_split_background_points(
                    background_gdf, train_pres_idx, val_pres_idx,
                    presence_gdf, SPATIAL_BLOCK_SIZE_KM)
            else:
                train_bg_idx, val_bg_idx = random_split_background_points(
                    background_gdf, fold, K_FOLDS, 42)
            train_bg_global = background_indices[train_bg_idx]
            val_bg_global = background_indices[val_bg_idx]
        else:
            print("  Using full background set (warning: possible data leakage)")
            train_bg_global = background_indices
            val_bg_global = background_indices

        train_idx = np.concatenate([train_presence_global, train_bg_global])
        val_idx = np.concatenate([val_presence_global, val_bg_global])
        print(f"  Train: {len(train_idx)} ({len(train_presence_global)} pres + {len(train_bg_global)} bg)")
        print(f"  Val:   {len(val_idx)} ({len(val_presence_global)} pres + {len(val_bg_global)} bg)")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Class-balanced sampling
        train_labels = [dataset.data.iloc[i]['label'] for i in train_idx]
        class_counts = pd.Series(train_labels).value_counts()
        class_weights = len(train_labels) / (2 * class_counts)
        sample_weights = [class_weights[dataset.data.iloc[i]['label']] for i in train_idx]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_idx), replacement=True)

        batch_size = max(2, min(64, len(train_idx) // 2, len(val_idx) // 2))
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Phase 1: SimCLR encoder
        print("  Phase 1: Training SimCLR encoder...")
        fold_best_loss = float('inf')
        patience_counter = 0
        for epoch in range(100):
            train_loss = train_simclr(model, train_loader, optimizer)
            model.eval()
            val_loss, n = 0, 0
            with torch.no_grad():
                for data, _ in val_loader:
                    if len(data) < 2:
                        continue
                    data = data.cuda()
                    _, z1 = model(data)
                    _, z2 = model(data)
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)
                    similarity = torch.matmul(z1, z2.T) / 0.5
                    targets = torch.arange(len(data)).cuda()
                    val_loss += F.cross_entropy(similarity, targets).item()
                    n += 1
            val_loss = val_loss / max(n, 1)
            scheduler.step(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"    Fold {fold + 1}, SimCLR epoch {epoch + 1}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                if val_loss < fold_best_loss:
                    fold_best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= 3:
                    print(f"    SimCLR early stopping at epoch {epoch + 1}")
                    break

        # Phase 2: RF classifier + calibrator
        print("  Phase 2: Training RF classifier + calibrator...")
        rf_classifier, calibrator = train_rf_classifier(
            model, train_loader, val_loader)

        # Phase 3: Evaluate
        print("  Phase 3: Evaluating fold...")
        val_metrics = evaluate_model(model, val_loader)
        fold_record = {
            'fold': fold + 1,
            'auc': val_metrics['auc'],
            'pr_auc': val_metrics['pr_auc'],
            'tss': val_metrics['tss'],
            'specificity': val_metrics['specificity'],
            'ece': val_metrics['ece'],
            'threshold': val_metrics['threshold'],
        }
        fold_metrics.append(fold_record)
        print(f"  Fold {fold + 1} - AUC: {val_metrics['auc']:.4f}, "
              f"PR-AUC: {val_metrics['pr_auc']:.4f}, TSS: {val_metrics['tss']:.4f}, "
              f"Specificity: {val_metrics['specificity']:.4f}, ECE: {val_metrics['ece']:.4f}")

        if val_metrics['auc'] > best_score:
            best_score = val_metrics['auc']
            best_metrics = val_metrics
            best_fold = fold
            torch.save({
                'simclr_state_dict': model.state_dict(),
                'rf_classifier': rf_classifier,
                'calibrator': calibrator,
            }, model_path)
            print(f"  -> New best model at fold {fold + 1} (AUC={best_score:.4f})")

    if fold_metrics:
        print("\nCross-validation summary (validation set):")
        for metric in ['auc', 'pr_auc', 'tss', 'specificity', 'ece']:
            vals = [m[metric] for m in fold_metrics]
            print(f"  {metric.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        results_df = pd.DataFrame(fold_metrics)
        results_path = os.path.join(output_dir, f"spatial_cv_results_{try_times}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Detailed CV results saved to: {results_path}")

        summary_path = os.path.join(output_dir, f"cv_summary_{try_times}.csv")
        save_cv_summary_to_csv(fold_metrics, summary_path)

    print(f"\nBest fold: {best_fold + 1} (AUC={best_score:.4f})")
    return best_metrics


# ============================================================================
# Main entry point
# ============================================================================

def main(presence_path, background_path, env_dir, output_dir, model_path, output_path, mask_shp_path):
    presence_data = pd.read_csv(presence_path)
    background_data = pd.read_csv(background_path)

    dataset = HabitatDataset(presence_data, background_data, env_dir)

    input_dim = len(dataset.env_files)
    hidden_dim = 1024
    output_dim = 512
    learning_rate = 1e-4

    model = SimCLRModel(input_dim, hidden_dim, output_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    if USE_SPATIAL_CV:
        print("Training with spatial cross-validation...")
        best_val_metrics = train_with_spatial_cv(
            dataset, model, optimizer, scheduler, output_dir, model_path)
    else:
        print("Training without spatial cross-validation...")
        n_samples = len(dataset)
        class_counts = dataset.data['label'].value_counts()
        class_weights = n_samples / (2 * class_counts)
        sample_weights = [class_weights[label] for label in dataset.data['label']]

        indices = np.arange(len(dataset))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42,
            stratify=dataset.data['label'])
        train_weights = [sample_weights[i] for i in train_idx]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_idx), replacement=True)

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                  batch_size=2014, sampler=train_sampler)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                                batch_size=2014, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(100):
            train_loss = train_simclr(model, train_loader, optimizer)
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, _ in val_loader:
                    if len(data) < 2:
                        continue
                    data = data.cuda()
                    _, z1 = model(data)
                    _, z2 = model(data)
                    z1 = F.normalize(z1, dim=1)
                    z2 = F.normalize(z2, dim=1)
                    similarity = torch.matmul(z1, z2.T) / 0.5
                    targets = torch.arange(len(data)).cuda()
                    val_loss += F.cross_entropy(similarity, targets).item()
            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f"SimCLR epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= 15:
                    print(f"SimCLR early stopping at epoch {epoch + 1}")
                    break

        rf_classifier, calibrator = train_rf_classifier(
            model, train_loader, val_loader)
        best_val_metrics = evaluate_model(model, val_loader)

        torch.save({
            'simclr_state_dict': model.state_dict(),
            'rf_classifier': rf_classifier,
            'calibrator': calibrator,
        }, model_path)
        print("Model saved.")

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    ckpt = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt['simclr_state_dict'])
    model.rf_classifier = ckpt['rf_classifier']
    model.calibrator = ckpt['calibrator']
    model.eval()

    full_loader = DataLoader(dataset, batch_size=2014, shuffle=False)
    print("\nFinal model evaluation:")
    final_metrics = evaluate_model(model, full_loader)
    for k in ['auc', 'pr_auc', 'tss', 'specificity', 'ece', 'threshold']:
        print(f"  {k.upper()}: {final_metrics[k]:.4f}")

    excel_output_path = os.path.join(output_dir, f"all_points_evaluation_{try_times}.xlsx")
    save_evaluation_results(final_metrics, excel_output_path)

    save_rf_and_calibrator(model, output_dir, try_times)

    # HSI mapping (PPM-lite: Platt-calibrated logits + quantile normalization)
    print("\nGenerating HSI map...")
    predict_hsi_with_ppm_lite(
        model, env_dir, output_path, mask_shp_path
    )

    print("\nOutput summary:")
    print(f"  Model:         {model_path}")
    print(f"  RF classifier: {os.path.join(output_dir, f'rf_classifier_{try_times}.joblib')}")
    print(f"  Calibrator:    {os.path.join(output_dir, f'calibrator_{try_times}.joblib')}")
    if USE_SPATIAL_CV:
        print(f"  CV results:    {os.path.join(output_dir, f'spatial_cv_results_{try_times}.csv')}")
        print(f"  CV summary:    {os.path.join(output_dir, f'cv_summary_{try_times}.csv')}")
    print(f"  Evaluation:    {excel_output_path}")
    print(f"  HSI map:       {output_path}")
    print("\nDone.")


if __name__ == "__main__":
    set_random_seed(RANDOM_STATE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f"best_model_{try_times}.pth")
    output_path = os.path.join(OUTPUT_DIR, f"hsi_prediction_{try_times}.tif")

    main(
        presence_path=PRESENCE_POINTS_FILE,
        background_path=BACKGROUND_POINTS_FILE,
        env_dir=ENV_VAR_RASTER_FILES,
        output_dir=OUTPUT_DIR,
        model_path=model_path,
        output_path=output_path,
        mask_shp_path=STUDY_AREA_SHP_FILE
    )
