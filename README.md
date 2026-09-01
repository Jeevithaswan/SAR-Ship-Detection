# SAR Ship Detection — 3-Class Classification (Sentinel-1, Singapore Strait)

Detects ships in Sentinel-1 SAR imagery (YOLO11 detector, trained on SSDD) and
classifies every detection into the three categories requested:

1. **AIS-supported vessel** — a credible (HIGH/MODERATE quality) AIS match.
2. **Non-AIS-supported vessel** — no credible AIS match, but the Sentinel-1
   VV/VH signature (contrast, temporal spike, geometry) looks like a vessel.
3. **Static / probable false alarm** — on land, or a fixed scatterer confirmed
   by its VV/VH backscatter *time series*, or weak/implausible SAR evidence.

This repo holds exactly the three stages that actually produced the checked-in
results — no dead ends, no unused alternates.

## Pipeline (run in this order)

```
01_model_training/
    Ship_Detection_3_0_Modified_Visual_Realtime.ipynb
    -> Google Colab. Prepares SSDD (COCO -> YOLO), trains the two frozen
       YOLO11s detectors used everywhere downstream:
         E0 = baseline, E1 = sensor-mix
       Output: E0_YOLO11s_SSDD_Baseline.pt, E1_YOLO11s_SSDD_SensorMix.pt
       (already in models/ in this repo).

02_detection_pipeline/
    SAR_Ship_Detection_FINAL_Kaggle_V4_1.ipynb
    -> Kaggle. Tiled E0/E1 inference over the controlled Sentinel-1 scene
       catalogue, GSHHG shoreline context, Global Fishing Watch AIS matching,
       SAR feature extraction. Produces the object-level candidate table with
       an initial 5-class label (AIS_MATCHED_VESSEL / NON_AIS_LIKELY_VESSEL /
       STATIC_MARITIME_OBJECT / PROBABLE_FALSE_ALARM / UNCERTAIN) - this is
       the input the next stage re-classifies into the final 3 classes.

03_final_classification/
    SAR_Ship_Detection_FINAL_3Class_VV_VH_TimeSeries.ipynb
    -> Kaggle. Takes the 5-class table from stage 2 plus co-registered VV/VH
       rasters, samples backscatter across *every* scene date at every
       detection location (not just the dates something fired), and
       re-derives the 3 final classes from that time series plus an
       AIS-anchored vessel-evidence model (95% target recall on known AIS
       vessels). This is what produced results_2026-08-31/.
    scripts/run_vv_vh_3class_local.py
    -> Local (non-Kaggle) port of the same logic: reads stage 2's object
       table + cached VV/VH scene arrays (Sentinel-1 GCP tie points,
       Newton-refined geocoding) with no SAFE data, GFW token, or internet
       access required. Used to reproduce results_2026-08-31/ on a laptop.
    scripts/make_property_figures.py
    -> Generates the geometry / SAR-signature / VV-VH-evidence figures in
       results_2026-08-31/figures/.
    scripts/retitle_classification_maps.py
    -> Cleans up the scene-classification-map titles for reporting (keeps
       scene ID + date, drops code-ish phrasing).
```

## Results summary (development scenes, 11,914 objects) — 2026-08-31 run

| Class | Count | % |
|---|---|---|
| Non-AIS-supported vessel | 5,923 | 49.7% |
| AIS-supported vessel | 3,481 | 29.2% |
| Static / probable false alarm | 2,510 | 21.1% |

Model quality (cross-validated against known AIS vessels): 95.0% recall,
94.0% balanced accuracy, calibrated vessel-score threshold 0.40. Full numbers
in `results_2026-08-31/FINAL_SUMMARY.json` and `CLASSIFICATION_CONFIG_AND_CV.json`.

## Not in this repo

- An earlier notebook that re-derives the 3 classes by re-running the entire
  stage-2 pipeline end to end with different decision logic, and an even
  earlier pre-v4.1 pipeline version — both superseded, both kept (not
  deleted) in the sibling `SAR-Ship-Detection-Full-Pipeline` folder for
  provenance, not duplicated here.
- Raw SSDD dataset, raw Sentinel-1 SAFE scenes, cached VV/VH `.npz` arrays,
  full per-object CSVs, and the full set of per-scene map overlays — all
  regenerable from the notebooks/scripts above, too large for a code repo.
  One sample classification map is kept in `results_2026-08-31/figures/`.

## Running it

- **01**: open in Colab, mount Drive, run top to bottom.
- **02**: Kaggle only — attach `models/*.pt` and the Sentinel-1 scene inputs
  as Kaggle datasets, run top to bottom.
- **03 (Kaggle notebook)**: attach stage 2's output table + VV/VH rasters,
  run top to bottom. **Or (local)**: `python scripts/run_vv_vh_3class_local.py`
  with pandas/sklearn/matplotlib/scipy/rasterio/pyproj installed, pointed at
  stage 2's object CSV and the cached VV/VH scene arrays.
