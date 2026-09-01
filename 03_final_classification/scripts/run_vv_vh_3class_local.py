"""
Local adaptation of SAR_Ship_Detection_FINAL_3Class_VV_VH_TimeSeries.ipynb

Same decision logic as the Kaggle notebook (VV/VH multi-date sampling, AIS-anchored
vessel-evidence model, calibrated threshold, static/rescue rules), but reads:
  - the already-computed v4_1 object table (FINAL_CLASSIFIED_SAR_CANDIDATES.csv)
  - cached VV/VH scene arrays (scene_crops/*.npz) with Sentinel-1 GCP tie points,
    instead of georeferenced GeoTIFFs + rasterio.

Geocoding: the npz files store the SAFE product's ground-control-point grid
(geo_line/geo_pixel/geo_lat/geo_lon, a regular 10x21 grid here). Forward
(line,pixel)->(lat,lon) uses bilinear interpolation over that grid (the standard
GCP interpolation model). Inverse (lat,lon)->(line,pixel) uses a cubic-griddata
initial guess refined by Newton iteration against the same forward model -
verified to converge to <0.01 px round-trip error.
"""
import os, re, json, math, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator, griddata
from pyproj import Transformer

from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score, confusion_matrix, balanced_accuracy_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 180)

# =============================
# USER CONFIGURATION - edit this, or set the SAR_PROJECT_ROOT environment
# variable, to point at your own working project folder (the one containing
# the v4_1 results export and the KAGGLE RESULTS/.../scene_crops cache -
# see the main README for exactly what's expected under each).
# =============================
ROOT = Path(os.environ.get("SAR_PROJECT_ROOT", ".")).resolve()
DET_CSV = ROOT / "SAR_Ship_Project_V4_1_RESULTS_WITH_FIGURES" / "FINAL_EXPORT" / "FINAL_CLASSIFIED_SAR_CANDIDATES.csv"
SCENE_CROP_DIR = ROOT / "KAGGLE RESULTS" / "SAR_Ship_Project_FULL" / "SAR_Ship_Project" / "results_v3" / "scene_crops"
OUTPUT_DIR = ROOT / "SAR_Ship_Project_V4_1_VV_VH_3CLASS_RESULTS"
FIG_DIR = OUTPUT_DIR / "FIGURES" / "final_3class_overlays"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- config (same values as the notebook) ----
CLUSTER_EPS_M = 30.0
TARGET_RADIUS_M = 25.0
BG_INNER_RADIUS_M = 60.0
BG_OUTER_RADIUS_M = 120.0
MIN_VALID_PIXELS = 6

MIN_VESSEL_CONTRAST_DB = 2.0
AIS_CONTRAST_QUANTILE = 0.10
STATIC_BRIGHT_PERSISTENCE = 0.82
STATIC_POSITION_STD_M = 20.0
STATIC_REL_MAD_MAX = 0.30

TARGET_AIS_RECALL = 0.95
MAX_ANCHOR_NEGATIVE_FPR = 0.20
FALLBACK_VESSEL_THRESHOLD = 0.45

RESCUE_MAX_BRIGHT_PERSISTENCE = 0.60
RESCUE_MIN_SPIKE_DB = 1.0
REVIEW_MARGIN = 0.08
RANDOM_STATE = 42

print("=" * 80)
print("STEP 1: Load v4_1 object-level detections")
print("=" * 80)

raw = pd.read_csv(DET_CSV)
print("Loaded rows:", len(raw))

det = pd.DataFrame()
det['object_id'] = np.arange(len(raw))
det['scene_id'] = raw['scene_id'].astype(str)
det['date'] = det['scene_id']  # scene_id already encodes the acquisition date, e.g. S01_20240602
det['lon'] = pd.to_numeric(raw['longitude'], errors='coerce')
det['lat'] = pd.to_numeric(raw['latitude'], errors='coerce')
det['ais_matched_raw'] = raw['ais_matched'].astype(bool)
det['ais_match_quality'] = raw['ais_match_quality'].astype(str)
# "AIS-supported" must mean a credible match (HIGH/MODERATE quality), matching v4_1's own
# definition of credible AIS. A LOW-quality (weak time-alignment) match is still useful
# evidence for the vessel-score model below, but must not alone promote an object to
# AIS_SUPPORTED_VESSEL - that would misrepresent uncertain AIS association as certainty.
det['ais_matched'] = det['ais_matched_raw'] & det['ais_match_quality'].isin(['HIGH', 'MODERATE'])
det['on_land'] = raw['center_on_land'].astype(bool)
det['confidence'] = pd.to_numeric(raw['confidence_max'], errors='coerce')
det['distance_to_land_m'] = pd.to_numeric(raw['distance_to_coast_m'], errors='coerce')
det['target_land_overlap_fraction'] = pd.to_numeric(raw['target_land_overlap_fraction'], errors='coerce')
det['water_fraction'] = (1.0 - det['target_land_overlap_fraction']).clip(0, 1)
det['major_m'] = pd.to_numeric(raw['sar_apparent_length_m'], errors='coerce')
det['minor_m'] = pd.to_numeric(raw['sar_apparent_width_m'], errors='coerce')
det['aspect_ratio'] = pd.to_numeric(raw['sar_aspect_ratio'], errors='coerce')
det['area_m2'] = pd.to_numeric(raw['sar_signature_area_m2'], errors='coerce')
det['old_class'] = raw['final_class'].astype(str)

det = det.dropna(subset=['lon', 'lat']).reset_index(drop=True)
det['object_id'] = np.arange(len(det))
print("Usable objects (with lon/lat):", len(det))
print("Scenes:", det['date'].nunique())
print("AIS matches:", int(det['ais_matched'].sum()))
print(det['old_class'].value_counts())

print("\n" + "=" * 80)
print("STEP 2: Build VV/VH scene manifest from cached npz scene crops")
print("=" * 80)

manifest_rows = []
for p in sorted(SCENE_CROP_DIR.glob("*_VVVH.npz")):
    scene_id = p.stem.replace("_VVVH", "")
    manifest_rows.append({'scene_id': scene_id, 'date': scene_id, 'npz_path': str(p)})
manifest = pd.DataFrame(manifest_rows)
print("Scene crops found:", len(manifest))

available_dates = set(manifest['date'])
missing_dates = sorted(set(det['date'].dropna()) - available_dates)
if missing_dates:
    print("WARNING: detection dates without cached VV/VH crop:", missing_dates)
det = det[det['date'].isin(available_dates)].copy().reset_index(drop=True)
print("Detections retained after date match:", len(det))


# ---- Sentinel-1 GCP-grid geocoding ----
class SceneRaster:
    """Wraps one cached VV/VH crop + its Sentinel-1 GCP tie-point grid."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.vv = d['vv_db']
        self.vh = d['vh_db']
        self.row0 = int(d['row0'])
        self.col0 = int(d['col0'])
        self.pixel_spacing_m = float(d['pixel_spacing_m'])
        gl = d['geo_line'].astype(float)
        gp = d['geo_pixel'].astype(float)
        glat = d['geo_lat'].astype(float)
        glon = d['geo_lon'].astype(float)
        self._glat, self._glon, self._gl, self._gp = glat, glon, gl, gp

        lines = np.unique(gl)
        pixels = np.unique(gp)
        lat_grid = np.full((len(lines), len(pixels)), np.nan)
        lon_grid = np.full((len(lines), len(pixels)), np.nan)
        li = {v: i for i, v in enumerate(lines)}
        pi = {v: i for i, v in enumerate(pixels)}
        for L, P, LA, LO in zip(gl, gp, glat, glon):
            lat_grid[li[L], pi[P]] = LA
            lon_grid[li[L], pi[P]] = LO
        self.lines, self.pixels = lines, pixels
        self.fwd_lat = RegularGridInterpolator((lines, pixels), lat_grid)
        self.fwd_lon = RegularGridInterpolator((lines, pixels), lon_grid)
        self.lmin, self.lmax = lines.min(), lines.max()
        self.pmin, self.pmax = pixels.min(), pixels.max()

    def lonlat_to_rowcol(self, lon, lat):
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        lat = np.atleast_1d(np.asarray(lat, dtype=float))

        guess_l = griddata((self._glat, self._glon), self._gl, (lat, lon), method='cubic')
        guess_p = griddata((self._glat, self._glon), self._gp, (lat, lon), method='cubic')
        bad = np.isnan(guess_l) | np.isnan(guess_p)
        if bad.any():
            guess_l[bad] = griddata((self._glat, self._glon), self._gl, (lat[bad], lon[bad]), method='nearest')
            guess_p[bad] = griddata((self._glat, self._glon), self._gp, (lat[bad], lon[bad]), method='nearest')

        l, p = guess_l.copy(), guess_p.copy()
        eps = 1.0
        for _ in range(12):
            l = np.clip(l, self.lmin, self.lmax)
            p = np.clip(p, self.pmin, self.pmax)
            pts = np.column_stack([l, p])
            lat_c = self.fwd_lat(pts); lon_c = self.fwd_lon(pts)
            lp = np.column_stack([np.clip(l + eps, self.lmin, self.lmax), p])
            lat_dl = (self.fwd_lat(lp) - lat_c) / eps
            lon_dl = (self.fwd_lon(lp) - lon_c) / eps
            pp = np.column_stack([l, np.clip(p + eps, self.pmin, self.pmax)])
            lat_dp = (self.fwd_lat(pp) - lat_c) / eps
            lon_dp = (self.fwd_lon(pp) - lon_c) / eps
            det_j = lat_dl * lon_dp - lat_dp * lon_dl
            det_j = np.where(np.abs(det_j) < 1e-12, 1e-12, det_j)
            Flat = lat_c - lat; Flon = lon_c - lon
            inv00 = lon_dp / det_j; inv01 = -lat_dp / det_j
            inv10 = -lon_dl / det_j; inv11 = lat_dl / det_j
            dl = -(inv00 * Flat + inv01 * Flon)
            dp = -(inv10 * Flat + inv11 * Flon)
            l = l + dl; p = p + dp

        row = l - self.row0
        col = p - self.col0
        return row, col


def robust_mad(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    med = np.median(a)
    return np.median(np.abs(a - med))


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def choose_utm_epsg(lon, lat):
    zone = int((float(lon) + 180) // 6) + 1
    return (32600 if float(lat) >= 0 else 32700) + zone


def sample_window(arr, row, col, mpp, target_radius_m, bg_inner_m, bg_outer_m):
    """arr already in dB. row/col are float pixel coords (arr indexing convention)."""
    h, w = arr.shape
    half = int(math.ceil(bg_outer_m / mpp)) + 2
    r0 = int(round(row)) - half
    c0 = int(round(col)) - half
    r1 = r0 + 2 * half + 1
    c1 = c0 + 2 * half + 1
    if r1 <= 0 or c1 <= 0 or r0 >= h or c0 >= w:
        return None
    rr0, rr1 = max(r0, 0), min(r1, h)
    cc0, cc1 = max(c0, 0), min(c1, w)
    sub = arr[rr0:rr1, cc0:cc1]
    if sub.size == 0:
        return None
    yy = (np.arange(rr0, rr1) - row) * mpp
    xx = (np.arange(cc0, cc1) - col) * mpp
    YY, XX = np.meshgrid(yy, xx, indexing='ij')
    dist = np.sqrt(XX ** 2 + YY ** 2)
    finite = np.isfinite(sub)
    target = sub[(dist <= target_radius_m) & finite]
    bg = sub[(dist >= bg_inner_m) & (dist <= bg_outer_m) & finite]
    if len(target) < MIN_VALID_PIXELS or len(bg) < MIN_VALID_PIXELS:
        return None
    bg_med = float(np.nanmedian(bg))
    target_med = float(np.nanmedian(target))
    target_p95 = float(np.nanpercentile(target, 95))
    return {
        'target_median_db': target_med,
        'target_p95_db': target_p95,
        'background_median_db': bg_med,
        'contrast_db': target_p95 - bg_med,
        'bright_fraction_3db': float(np.mean(target >= (bg_med + 3.0))),
        'n_target_px': int(len(target)),
        'n_bg_px': int(len(bg)),
    }


print("\n" + "=" * 80)
print("STEP 3: Spatial-temporal clustering of repeated detections (DBSCAN, eps=%gm)" % CLUSTER_EPS_M)
print("=" * 80)

med_lon = float(det['lon'].median())
med_lat = float(det['lat'].median())
utm_epsg = choose_utm_epsg(med_lon, med_lat)
ll_to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{utm_epsg}', always_xy=True)
det['x_m'], det['y_m'] = ll_to_utm.transform(det['lon'].to_numpy(), det['lat'].to_numpy())
coords = det[['x_m', 'y_m']].to_numpy(float)
labels = DBSCAN(eps=CLUSTER_EPS_M, min_samples=1, metric='euclidean').fit_predict(coords)
det['temporal_cluster_id'] = labels.astype(int)

n_scenes_total = manifest['date'].nunique()
cluster_base = det.groupby('temporal_cluster_id').agg(
    cluster_lon=('lon', 'median'), cluster_lat=('lat', 'median'),
    n_detection_rows=('object_id', 'size'), n_dates_detected=('date', 'nunique'),
    x_mean=('x_m', 'mean'), y_mean=('y_m', 'mean'),
    x_std=('x_m', 'std'), y_std=('y_m', 'std'),
).reset_index()
cluster_base['position_std_m'] = np.sqrt(cluster_base['x_std'].fillna(0) ** 2 + cluster_base['y_std'].fillna(0) ** 2)
cluster_base['detection_persistence'] = cluster_base['n_dates_detected'] / max(n_scenes_total, 1)
print("Spatial-temporal clusters:", len(cluster_base))
print(cluster_base[['n_dates_detected', 'detection_persistence', 'position_std_m']].describe())

print("\n" + "=" * 80)
print("STEP 4: Sample VV/VH backscatter time series at every cluster, every scene date")
print("=" * 80)

clusters_records = cluster_base[['temporal_cluster_id', 'cluster_lon', 'cluster_lat']].to_dict('records')
cluster_lons = np.array([c['cluster_lon'] for c in clusters_records])
cluster_lats = np.array([c['cluster_lat'] for c in clusters_records])
cluster_ids = np.array([c['temporal_cluster_id'] for c in clusters_records])

ts_rows = []
for i, sc in manifest.iterrows():
    print(f"  Sampling {i+1}/{len(manifest)}: {sc['date']}")
    scene = SceneRaster(sc['npz_path'])
    rows_f, cols_f = scene.lonlat_to_rowcol(cluster_lons, cluster_lats)
    mpp = scene.pixel_spacing_m
    for cid, r, c in zip(cluster_ids, rows_f, cols_f):
        vv = sample_window(scene.vv, r, c, mpp, TARGET_RADIUS_M, BG_INNER_RADIUS_M, BG_OUTER_RADIUS_M)
        vh = sample_window(scene.vh, r, c, mpp, TARGET_RADIUS_M, BG_INNER_RADIUS_M, BG_OUTER_RADIUS_M)
        if vv is None and vh is None:
            continue
        rec = {'temporal_cluster_id': int(cid), 'date': sc['date']}
        if vv is not None:
            rec.update({f'vv_{k}': v for k, v in vv.items()})
        if vh is not None:
            rec.update({f'vh_{k}': v for k, v in vh.items()})
        ts_rows.append(rec)
    del scene

ts = pd.DataFrame(ts_rows)
print("Time-series rows:", len(ts))
for c in ['vv_contrast_db', 'vh_contrast_db']:
    if c not in ts.columns:
        ts[c] = np.nan
ts['combined_contrast_db'] = ts[['vv_contrast_db', 'vh_contrast_db']].max(axis=1, skipna=True)
ts['vv_minus_vh_peak_db'] = ts.get('vv_target_p95_db', np.nan) - ts.get('vh_target_p95_db', np.nan)
ts.to_csv(OUTPUT_DIR / 'CLUSTER_VV_VH_TIME_SERIES.csv', index=False)

print("\n" + "=" * 80)
print("STEP 5: Attach current-scene VV/VH evidence + temporal summaries to each object")
print("=" * 80)

cur_cols = ['temporal_cluster_id', 'date', 'vv_target_p95_db', 'vh_target_p95_db',
            'vv_background_median_db', 'vh_background_median_db', 'vv_contrast_db', 'vh_contrast_db',
            'combined_contrast_db', 'vv_minus_vh_peak_db', 'vv_bright_fraction_3db', 'vh_bright_fraction_3db']
cur_cols = [c for c in cur_cols if c in ts.columns]
det = det.merge(ts[cur_cols], on=['temporal_cluster_id', 'date'], how='left')
det = det.merge(cluster_base[['temporal_cluster_id', 'n_dates_detected', 'detection_persistence', 'position_std_m']],
                 on='temporal_cluster_id', how='left')

med_all = ts.groupby('temporal_cluster_id')['combined_contrast_db'].median().rename('ts_median_contrast_db')
mad_all = ts.groupby('temporal_cluster_id')['combined_contrast_db'].apply(robust_mad).rename('ts_mad_contrast_db')
max_all = ts.groupby('temporal_cluster_id')['combined_contrast_db'].max().rename('ts_max_contrast_db')
valid_all = ts.groupby('temporal_cluster_id')['combined_contrast_db'].count().rename('n_valid_ts_dates')
cluster_ts_summary = pd.concat([med_all, mad_all, max_all, valid_all], axis=1).reset_index()
cluster_ts_summary['ts_relative_mad'] = cluster_ts_summary['ts_mad_contrast_db'] / (cluster_ts_summary['ts_median_contrast_db'].abs() + 1.0)
det = det.merge(cluster_ts_summary, on='temporal_cluster_id', how='left')
det['temporal_spike_db'] = det['combined_contrast_db'] - det['ts_median_contrast_db']

print("\n" + "=" * 80)
print("STEP 6: Learn per-scene bright-vessel contrast threshold from AIS positives")
print("=" * 80)

ais_ref = det[det['ais_matched'] & det['combined_contrast_db'].notna()].copy()
print("AIS-matched objects with VV/VH features:", len(ais_ref))
global_ais_thr = max(MIN_VESSEL_CONTRAST_DB, float(ais_ref['combined_contrast_db'].quantile(AIS_CONTRAST_QUANTILE)))
scene_thr = {}
for dt, g in ais_ref.groupby('date'):
    if len(g) >= 5:
        scene_thr[dt] = max(MIN_VESSEL_CONTRAST_DB, float(g['combined_contrast_db'].quantile(AIS_CONTRAST_QUANTILE)))
    else:
        scene_thr[dt] = global_ais_thr
manifest['vessel_contrast_thr_db'] = manifest['date'].map(scene_thr).fillna(global_ais_thr)
print("Global AIS-derived contrast floor:", round(global_ais_thr, 2), "dB")

thr_map = dict(zip(manifest['date'], manifest['vessel_contrast_thr_db']))
ts['scene_contrast_thr_db'] = ts['date'].map(thr_map).fillna(global_ais_thr)
ts['bright_like_vessel'] = ts['combined_contrast_db'] >= ts['scene_contrast_thr_db']
bright_summary = ts.groupby('temporal_cluster_id').agg(
    n_valid_ts_dates2=('combined_contrast_db', 'count'), n_bright_dates=('bright_like_vessel', 'sum'),
).reset_index()
bright_summary['bright_persistence'] = bright_summary['n_bright_dates'] / bright_summary['n_valid_ts_dates2'].clip(lower=1)
det = det.merge(bright_summary[['temporal_cluster_id', 'bright_persistence']], on='temporal_cluster_id', how='left')
det['scene_contrast_thr_db'] = det['date'].map(thr_map).fillna(global_ais_thr)
det['current_bright_like_vessel'] = det['combined_contrast_db'] >= det['scene_contrast_thr_db']

print("\n" + "=" * 80)
print("STEP 7: Geometry plausibility")
print("=" * 80)

geom_known = det['major_m'].notna() & det['minor_m'].notna()
det['geometry_plausible'] = True
det.loc[geom_known, 'geometry_plausible'] = (
    det.loc[geom_known, 'major_m'].between(4, 500) &
    det.loc[geom_known, 'minor_m'].between(1.5, 180) &
    det.loc[geom_known, 'aspect_ratio'].between(1.0, 25.0) &
    (det.loc[geom_known, 'area_m2'] <= 60000)
)
print("Geometry available:", int(geom_known.sum()), "/", len(det))
print("Broadly plausible geometry:", int(det['geometry_plausible'].sum()), "/", len(det))

print("\n" + "=" * 80)
print("STEP 8: Build conservative positive/negative anchors")
print("=" * 80)

pos_anchor = det['ais_matched']
strong_static_signature = (
    (det['bright_persistence'] >= STATIC_BRIGHT_PERSISTENCE) &
    (det['position_std_m'] <= STATIC_POSITION_STD_M) &
    (det['ts_relative_mad'] <= STATIC_REL_MAD_MAX) &
    (det['n_valid_ts_dates'] >= 4)
)
old_probable_false = det['old_class'].astype(str).str.contains('PROBABLE_FALSE|FALSE_ALARM', regex=True, na=False)
old_static = det['old_class'].astype(str).str.contains('STATIC', regex=True, na=False)

neg_anchor = (
    det['on_land'] |
    (old_probable_false & ~det['current_bright_like_vessel']) |
    (old_static & strong_static_signature & ~det['geometry_plausible']) |
    (strong_static_signature & ~det['geometry_plausible'])
)
neg_anchor = neg_anchor & ~pos_anchor
det['positive_anchor'] = pos_anchor
det['negative_anchor'] = neg_anchor
print("Positive AIS anchors:", int(pos_anchor.sum()))
print("High-confidence negative anchors:", int(neg_anchor.sum()))
print("Strong static signatures:", int(strong_static_signature.sum()))

print("\n" + "=" * 80)
print("STEP 9: Train interpretable vessel-evidence model")
print("=" * 80)

feature_cols = ['vv_contrast_db', 'vh_contrast_db', 'combined_contrast_db', 'vv_minus_vh_peak_db',
                'temporal_spike_db', 'bright_persistence', 'detection_persistence', 'position_std_m',
                'ts_relative_mad', 'major_m', 'minor_m', 'area_m2', 'aspect_ratio',
                'confidence', 'water_fraction', 'distance_to_land_m']
# Deliberately no AIS-relatedness feature here: since the positive anchors ARE credible-AIS
# objects, any "has a nearby AIS ping" feature would be near-perfectly correlated with the
# training label and would make the model lean on AIS proximity instead of learning the
# actual SAR/VV-VH vessel signature - defeating the point of scoring non-AIS objects.
feature_cols = [c for c in feature_cols if c in det.columns and det[c].notna().any()]

anchor_mask = det['positive_anchor'] | det['negative_anchor']
X_anchor = det.loc[anchor_mask, feature_cols].copy()
y_anchor = det.loc[anchor_mask, 'positive_anchor'].astype(int).to_numpy()

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=3000, C=1.0, random_state=RANDOM_STATE)),
])

use_model = (np.sum(y_anchor == 1) >= 20 and np.sum(y_anchor == 0) >= 20 and len(feature_cols) >= 4)
calibrated_threshold = FALLBACK_VESSEL_THRESHOLD
cv_info = {}

if use_model:
    min_class = int(min(np.sum(y_anchor == 1), np.sum(y_anchor == 0)))
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_anchor, y_anchor, cv=cv, method='predict_proba')[:, 1]

    candidates = np.linspace(0.05, 0.95, 181)
    rows = []
    for t in candidates:
        pred = (oof >= t).astype(int)
        rec = recall_score(y_anchor, pred, zero_division=0)
        cm = confusion_matrix(y_anchor, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(fp + tn, 1)
        bal = balanced_accuracy_score(y_anchor, pred)
        rows.append((t, rec, fpr, bal))
    thr_df = pd.DataFrame(rows, columns=['threshold', 'ais_recall', 'negative_fpr', 'balanced_accuracy'])

    feasible = thr_df[(thr_df['ais_recall'] >= TARGET_AIS_RECALL) & (thr_df['negative_fpr'] <= MAX_ANCHOR_NEGATIVE_FPR)]
    if not feasible.empty:
        calibrated_threshold = float(feasible.sort_values(['threshold', 'balanced_accuracy'], ascending=[False, False]).iloc[0]['threshold'])
    else:
        high_recall = thr_df[thr_df['ais_recall'] >= TARGET_AIS_RECALL]
        if not high_recall.empty:
            calibrated_threshold = float(high_recall['threshold'].max())
        else:
            calibrated_threshold = float(thr_df.sort_values('balanced_accuracy', ascending=False).iloc[0]['threshold'])

    model.fit(X_anchor, y_anchor)
    det['vessel_score'] = model.predict_proba(det[feature_cols])[:, 1]

    pred_oof = (oof >= calibrated_threshold).astype(int)
    cm = confusion_matrix(y_anchor, pred_oof, labels=[0, 1])
    cv_info = {
        'threshold': calibrated_threshold,
        'oof_ais_recall': float(recall_score(y_anchor, pred_oof, zero_division=0)),
        'oof_balanced_accuracy': float(balanced_accuracy_score(y_anchor, pred_oof)),
        'oof_confusion_matrix': cm.tolist(),
        'n_positive_anchors': int(np.sum(y_anchor == 1)),
        'n_negative_anchors': int(np.sum(y_anchor == 0)),
        'features': feature_cols,
    }
    print("Model-based vessel score enabled")
    print(json.dumps(cv_info, indent=2))
else:
    sar = sigmoid((det['combined_contrast_db'] - det['scene_contrast_thr_db']) / 1.5)
    transient = (1.0 - det['bright_persistence'].fillna(0.5)).clip(0, 1)
    spike = sigmoid(det['temporal_spike_db'].fillna(0.0) / 1.5)
    geom = det['geometry_plausible'].astype(float)
    static_pen = strong_static_signature.astype(float)
    land_pen = det['on_land'].astype(float)
    det['vessel_score'] = (0.48 * sar + 0.24 * transient + 0.18 * spike + 0.10 * geom - 0.30 * static_pen - 0.50 * land_pen).clip(0, 1)
    calibrated_threshold = FALLBACK_VESSEL_THRESHOLD
    cv_info = {'mode': 'fallback_rule_score', 'threshold': calibrated_threshold}
    print("WARNING: too few conservative anchors. Using transparent fallback vessel score.")

print("Final vessel-score threshold:", round(calibrated_threshold, 3))

print("\n" + "=" * 80)
print("STEP 10: FINAL THREE-CLASS DECISION")
print("=" * 80)

det['hard_static_signature'] = strong_static_signature
rescue_vessel = (
    (~det['ais_matched']) & (~det['on_land']) & det['geometry_plausible'] & det['current_bright_like_vessel'] &
    ((det['bright_persistence'] <= RESCUE_MAX_BRIGHT_PERSISTENCE) | (det['temporal_spike_db'] >= RESCUE_MIN_SPIKE_DB))
)
hard_static_reject = (
    det['hard_static_signature'] & (det['vessel_score'] < max(calibrated_threshold + 0.15, 0.65)) & (~det['ais_matched'])
)

final = np.full(len(det), 'STATIC_OR_PROBABLE_FALSE_ALARM', dtype=object)
final[det['ais_matched'].to_numpy()] = 'AIS_SUPPORTED_VESSEL'
non_ais_vessel = (
    (~det['ais_matched']) & (~det['on_land']) & (~hard_static_reject) &
    ((det['vessel_score'] >= calibrated_threshold) | rescue_vessel)
)
final[non_ais_vessel.to_numpy()] = 'NON_AIS_SUPPORTED_VESSEL'
det['final_class_3'] = final

reason = []
for i, r in det.iterrows():
    if r['ais_matched']:
        reason.append('AIS match')
    elif bool(r['on_land']):
        reason.append('on-land hard rejection')
    elif bool(hard_static_reject.loc[i]):
        reason.append('high fixed temporal persistence + low vessel evidence')
    elif bool(rescue_vessel.loc[i]) and r['vessel_score'] < calibrated_threshold:
        reason.append('VV/VH vessel rescue: strong current contrast + transient/spike evidence')
    elif r['vessel_score'] >= calibrated_threshold:
        reason.append(f'vessel score >= calibrated threshold ({calibrated_threshold:.2f})')
    else:
        reason.append(f'vessel score below calibrated threshold ({calibrated_threshold:.2f})')
det['classification_reason'] = reason

det['review_flag'] = (
    ((det['vessel_score'] - calibrated_threshold).abs() <= REVIEW_MARGIN) |
    (hard_static_reject & (det['vessel_score'] >= calibrated_threshold - 0.10)) |
    (rescue_vessel & (det['vessel_score'] < calibrated_threshold))
) & (~det['ais_matched'])

print("\nFINAL CLASS COUNTS")
print(det['final_class_3'].value_counts())
print("\nQA review flags:", int(det['review_flag'].sum()))
assert set(det['final_class_3'].unique()).issubset({'AIS_SUPPORTED_VESSEL', 'NON_AIS_SUPPORTED_VESSEL', 'STATIC_OR_PROBABLE_FALSE_ALARM'})

print("\n" + "=" * 80)
print("STEP 11: Redistribution of old 5-class labels into the new 3 classes")
print("=" * 80)
matrix = pd.crosstab(det['old_class'].replace('', 'NO_OLD_CLASS'), det['final_class_3'])
print(matrix)
matrix.to_csv(OUTPUT_DIR / 'OLD_TO_NEW_CLASS_MATRIX.csv')

old_uncertain = det['old_class'].astype(str).str.contains('UNCERTAIN', na=False)
print("\nOld UNCERTAIN objects:", int(old_uncertain.sum()))
if old_uncertain.any():
    print(det.loc[old_uncertain, 'final_class_3'].value_counts())

print("\n" + "=" * 80)
print("STEP 12: Diagnostics + figures")
print("=" * 80)

fig, ax = plt.subplots(figsize=(10, 5))
for label, mask in [
    ('AIS supported', det['ais_matched']),
    ('Old uncertain', old_uncertain & ~det['ais_matched']),
    ('Old probable false', old_probable_false & ~det['ais_matched']),
]:
    vals = det.loc[mask, 'vessel_score'].dropna()
    if len(vals):
        ax.hist(vals, bins=30, alpha=0.45, label=f'{label} (n={len(vals)})')
ax.axvline(calibrated_threshold, linestyle='--', linewidth=2, label=f'final threshold={calibrated_threshold:.2f}')
ax.set_xlabel('Vessel evidence score'); ax.set_ylabel('Objects')
ax.set_title('Vessel-score diagnostic (local VV/VH run)')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'vessel_score_diagnostic.png', dpi=200, bbox_inches='tight')
plt.close(fig)

class_counts = det['final_class_3'].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
colors = {'AIS_SUPPORTED_VESSEL': 'limegreen', 'NON_AIS_SUPPORTED_VESSEL': 'deepskyblue', 'STATIC_OR_PROBABLE_FALSE_ALARM': 'red'}
ax.bar(class_counts.index, class_counts.values, color=[colors.get(c, 'gray') for c in class_counts.index])
for i, v in enumerate(class_counts.values):
    ax.text(i, v, str(v), ha='center', va='bottom')
ax.set_ylabel('Count')
ax.set_title('Final 3-class distribution (all scenes)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'final_3class_distribution.png', dpi=200, bbox_inches='tight')
plt.close(fig)

scene_counts = pd.crosstab(det['date'], det['final_class_3']).reset_index()
scene_counts.to_csv(OUTPUT_DIR / 'final_3class_scene_counts.csv', index=False)

print("\n" + "=" * 80)
print("STEP 13: Per-scene classification overlays")
print("=" * 80)

CLASS_STYLE = {
    'AIS_SUPPORTED_VESSEL': ('limegreen', 'AIS-supported vessel'),
    'NON_AIS_SUPPORTED_VESSEL': ('deepskyblue', 'Non-AIS-supported vessel'),
    'STATIC_OR_PROBABLE_FALSE_ALARM': ('red', 'Static / probable false alarm'),
}

for _, row in manifest.iterrows():
    g = det[det['date'] == row['date']]
    if g.empty:
        continue
    scene = SceneRaster(row['npz_path'])
    bg = scene.vv
    max_size = 1800
    scale = max(bg.shape[1] / max_size, bg.shape[0] / max_size, 1)
    out_h, out_w = max(1, int(bg.shape[0] / scale)), max(1, int(bg.shape[1] / scale))
    small = bg[::max(1, int(scale)), ::max(1, int(scale))]
    vmin, vmax = np.nanpercentile(small[np.isfinite(small)], [2, 98]) if np.isfinite(small).any() else (-25, 5)

    rows_f, cols_f = scene.lonlat_to_rowcol(g['lon'].to_numpy(), g['lat'].to_numpy())
    # convert to display (downsampled) pixel coords
    disp_row = rows_f / max(1, int(scale))
    disp_col = cols_f / max(1, int(scale))

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(small, cmap='gray', vmin=vmin, vmax=vmax, origin='upper')
    tmp = g.copy()
    tmp['_dr'] = disp_row; tmp['_dc'] = disp_col
    for cls, (color, label) in CLASS_STYLE.items():
        m = tmp[tmp['final_class_3'] == cls]
        if m.empty:
            continue
        ax.scatter(m['_dc'], m['_dr'], s=34, marker='s', facecolors='none', edgecolors=color,
                   linewidths=1.1, label=f'{label} ({len(m)})')
    ax.set_title(f"{row['date']} - final 3-class object classification (local VV/VH run)")
    ax.set_axis_off()
    ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    out = FIG_DIR / f"{row['date']}_FINAL_3CLASS_OVERLAY.png"
    plt.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print("Saved", out)
    del scene

print("\n" + "=" * 80)
print("STEP 14: Save final tables")
print("=" * 80)

keep_cols = [c for c in [
    'object_id', 'scene_id', 'date', 'lon', 'lat', 'old_class',
    'ais_matched', 'ais_matched_raw', 'ais_match_quality', 'on_land', 'water_fraction', 'distance_to_land_m',
    'major_m', 'minor_m', 'area_m2', 'aspect_ratio', 'confidence',
    'vv_target_p95_db', 'vh_target_p95_db', 'vv_background_median_db', 'vh_background_median_db',
    'vv_contrast_db', 'vh_contrast_db', 'combined_contrast_db', 'vv_minus_vh_peak_db',
    'temporal_cluster_id', 'n_dates_detected', 'detection_persistence', 'bright_persistence',
    'position_std_m', 'ts_median_contrast_db', 'ts_mad_contrast_db', 'ts_relative_mad', 'temporal_spike_db',
    'geometry_plausible', 'vessel_score', 'final_class_3', 'classification_reason', 'review_flag'
] if c in det.columns]

final_table = det[keep_cols].copy()
final_table.to_csv(OUTPUT_DIR / 'FINAL_3CLASS_OBJECTS.csv', index=False)
det.to_csv(OUTPUT_DIR / 'FINAL_3CLASS_OBJECTS_FULL_AUDIT.csv', index=False)
manifest.to_csv(OUTPUT_DIR / 'SCENE_MANIFEST_USED.csv', index=False)

with open(OUTPUT_DIR / 'CLASSIFICATION_CONFIG_AND_CV.json', 'w') as f:
    json.dump({
        'cluster_eps_m': CLUSTER_EPS_M, 'target_radius_m': TARGET_RADIUS_M,
        'bg_inner_radius_m': BG_INNER_RADIUS_M, 'bg_outer_radius_m': BG_OUTER_RADIUS_M,
        'min_vessel_contrast_db': MIN_VESSEL_CONTRAST_DB, 'ais_contrast_quantile': AIS_CONTRAST_QUANTILE,
        'static_bright_persistence': STATIC_BRIGHT_PERSISTENCE, 'static_position_std_m': STATIC_POSITION_STD_M,
        'static_rel_mad_max': STATIC_REL_MAD_MAX, 'target_ais_recall': TARGET_AIS_RECALL,
        'final_vessel_threshold': calibrated_threshold, 'cv': cv_info,
    }, f, indent=2)

summary = {
    'n_objects': int(len(det)), 'n_scenes': int(det['date'].nunique()),
    'class_counts': det['final_class_3'].value_counts().to_dict(),
    'old_uncertain_count': int(old_uncertain.sum()),
    'old_uncertain_to_non_ais_vessel': int(((old_uncertain) & (det['final_class_3'] == 'NON_AIS_SUPPORTED_VESSEL')).sum()),
    'old_uncertain_to_static_false': int(((old_uncertain) & (det['final_class_3'] == 'STATIC_OR_PROBABLE_FALSE_ALARM')).sum()),
    'qa_review_flags': int(det['review_flag'].sum()),
    'vessel_score_threshold': float(calibrated_threshold),
}
with open(OUTPUT_DIR / 'FINAL_SUMMARY.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nDONE. Primary output:", OUTPUT_DIR / 'FINAL_3CLASS_OBJECTS.csv')
