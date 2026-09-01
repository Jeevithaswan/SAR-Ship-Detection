"""
Geometry + SAR-signature figures for the final 3-class result.

Reads FINAL_3CLASS_OBJECTS.csv (produced by run_vv_vh_3class_local.py) and
writes report-ready PNGs, mirroring the style of the v4_1 target_property_plots
but for the new 3-class scheme, plus new figures that show *why* the VV/VH
time-series evidence separated static objects from vessels.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(r"c:\Users\Jeevitha.Balaraman\OneDrive - Swan Corp\Documents\Jeevi\SHIP DETECTION")
RESULTS_DIR = ROOT / "SAR_Ship_Project_V4_1_VV_VH_3CLASS_RESULTS"
OUT_DIR = RESULTS_DIR / "FIGURES" / "target_property_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- validated-leaning categorical triple (blue / green / red from the design-system
# categorical ramp), same hue identity as the classification-map overlays already
# delivered, with distinct marker shapes as a secondary (non-color) encoding ----
CLASS_ORDER = ['AIS_SUPPORTED_VESSEL', 'NON_AIS_SUPPORTED_VESSEL', 'STATIC_OR_PROBABLE_FALSE_ALARM']
CLASS_LABEL = {
    'AIS_SUPPORTED_VESSEL': 'AIS-supported\nvessel',
    'NON_AIS_SUPPORTED_VESSEL': 'Non-AIS-supported\nvessel',
    'STATIC_OR_PROBABLE_FALSE_ALARM': 'Static / probable\nfalse alarm',
}
CLASS_COLOR = {
    'AIS_SUPPORTED_VESSEL': '#008300',
    'NON_AIS_SUPPORTED_VESSEL': '#2a78d6',
    'STATIC_OR_PROBABLE_FALSE_ALARM': '#e34948',
}
CLASS_MARKER = {
    'AIS_SUPPORTED_VESSEL': 'o',
    'NON_AIS_SUPPORTED_VESSEL': 's',
    'STATIC_OR_PROBABLE_FALSE_ALARM': '^',
}

SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_SECONDARY, 'text.color': INK_PRIMARY,
    'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'grid.color': GRID,
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
})

det = pd.read_csv(RESULTS_DIR / 'FINAL_3CLASS_OBJECTS.csv', low_memory=False)
det = det[det['final_class_3'].isin(CLASS_ORDER)].copy()

with open(RESULTS_DIR / 'CLASSIFICATION_CONFIG_AND_CV.json') as f:
    cfg = json.load(f)
THRESHOLD = cfg['final_vessel_threshold']


def class_series(col, clip=None):
    out = []
    for c in CLASS_ORDER:
        v = pd.to_numeric(det.loc[det['final_class_3'] == c, col], errors='coerce').dropna()
        if clip:
            v = v[(v >= clip[0]) & (v <= clip[1])]
        out.append(v.to_numpy())
    return out


def legend_handles(marker=True):
    if marker:
        return [Line2D([0], [0], marker=CLASS_MARKER[c], color='none', markerfacecolor=CLASS_COLOR[c],
                        markeredgecolor='white', markersize=9, label=CLASS_LABEL[c].replace('\n', ' '))
                for c in CLASS_ORDER]
    return [Patch(facecolor=CLASS_COLOR[c], edgecolor='white', label=CLASS_LABEL[c].replace('\n', ' '))
            for c in CLASS_ORDER]


def styled_box(ax, data, title, ylabel, ylim=None):
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                     medianprops=dict(color=INK_PRIMARY, linewidth=1.6),
                     whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED))
    for patch, c in zip(bp['boxes'], CLASS_ORDER):
        patch.set_facecolor(CLASS_COLOR[c])
        patch.set_alpha(0.75)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.2)
    # jittered raw points, thinned for large classes, for an honest look at spread
    rng = np.random.default_rng(0)
    for i, (c, vals) in enumerate(zip(CLASS_ORDER, data), start=1):
        if len(vals) == 0:
            continue
        sample = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
        jitter = rng.uniform(-0.12, 0.12, size=len(sample))
        ax.scatter(np.full(len(sample), i) + jitter, sample, s=6, color=INK_PRIMARY, alpha=0.08, linewidths=0)
    # counts go into the tick labels themselves - an offset annotation collides with them
    ax.set_xticks(range(1, len(CLASS_ORDER) + 1))
    ax.set_xticklabels([f'{CLASS_LABEL[c]}\n(n={len(v):,})' for c, v in zip(CLASS_ORDER, data)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)


def small_multiple_density(fig_title, xlabel, ylabel, xy_by_class, xlim, ylim, bins=45, out_path=None, annotate=None):
    """3-panel small multiples (one per class), 2D histogram density on a shared
    LOG color scale - the honest way to compare ~thousands of points across
    classes without an illegible overlaid scatter. Log scale matters here: a
    single "one-off transient detection" cell is 100x denser than the rest of
    the plot and would otherwise wash out every other cell on a linear scale."""
    from matplotlib.colors import LogNorm

    xedges = np.linspace(*xlim, bins + 1)
    yedges = np.linspace(*ylim, bins + 1)
    hists = []
    for c in CLASS_ORDER:
        x, y = xy_by_class[c]
        h, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        hists.append(h)
    vmax = max(h.max() for h in hists)
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharex=True, sharey=True,
                              gridspec_kw={'wspace': 0.10})
    im = None
    for ax, c, h in zip(axes, CLASS_ORDER, hists):
        hm = np.ma.masked_where(h == 0, h)
        cmap = plt.get_cmap('Blues').copy()
        cmap.set_bad(SURFACE)
        im = ax.imshow(hm.T, origin='lower', extent=[*xlim, *ylim], aspect='auto', cmap=cmap, norm=norm)
        n = int(h.sum())
        ax.set_title(f'{CLASS_LABEL[c].replace(chr(10), " ")}', fontsize=11.5, color=CLASS_COLOR[c], pad=16)
        ax.text(0.5, 1.015, f'n={n:,}', transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, color=INK_MUTED)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)
    axes[0].set_ylabel(ylabel)
    fig.supxlabel(xlabel, fontsize=11, color=INK_SECONDARY, y=-0.02)
    if annotate:
        annotate(axes)
    fig.suptitle(fig_title, fontsize=13.5, fontweight='bold', y=1.14)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label('Objects per cell (log scale)', color=INK_SECONDARY)
    cbar.outline.set_edgecolor(GRID)
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


print("Building geometry + SAR-signature figures ->", OUT_DIR)

# 1) Apparent length by class
fig, ax = plt.subplots(figsize=(8, 6))
styled_box(ax, class_series('major_m', clip=(0, 800)), 'Apparent SAR length by final class', 'Apparent length (m)')
plt.tight_layout()
plt.savefig(OUT_DIR / '01_APPARENT_LENGTH_BY_CLASS.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 2) Apparent width by class
fig, ax = plt.subplots(figsize=(8, 6))
styled_box(ax, class_series('minor_m', clip=(0, 300)), 'Apparent SAR width by final class', 'Apparent width (m)')
plt.tight_layout()
plt.savefig(OUT_DIR / '02_APPARENT_WIDTH_BY_CLASS.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 3) Aspect ratio by class
fig, ax = plt.subplots(figsize=(8, 6))
styled_box(ax, class_series('aspect_ratio', clip=(0, 12)), 'Length-to-width aspect ratio by final class', 'Aspect ratio (length / width)')
plt.tight_layout()
plt.savefig(OUT_DIR / '03_ASPECT_RATIO_BY_CLASS.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 4) Length vs width density, small multiples by class (12k points -> a scatter is illegible; density is honest)
xlim, ylim = (0, 300), (0, 800)
xy = {}
for c in CLASS_ORDER:
    g = det[det['final_class_3'] == c]
    x = pd.to_numeric(g['minor_m'], errors='coerce'); y = pd.to_numeric(g['major_m'], errors='coerce')
    m = x.notna() & y.notna() & x.between(*xlim) & y.between(*ylim)
    xy[c] = (x[m].to_numpy(), y[m].to_numpy())


def _diag(axes):
    for ax in axes:
        ax.plot(xlim, [xlim[0] * (ylim[1] / xlim[1]), ylim[1]], color=INK_MUTED, linewidth=0.8, linestyle='--')


small_multiple_density('SAR-signature geometry: length vs. width, by final class',
                        'Apparent width (m)', 'Apparent length (m)', xy, xlim, ylim,
                        out_path=OUT_DIR / '04_LENGTH_VS_WIDTH_BY_CLASS.png', annotate=_diag)

# 5) Combined VV/VH contrast by class
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
styled_box(axes[0], class_series('vv_contrast_db', clip=(-10, 40)), 'VV target-vs-background contrast', 'Contrast (dB)')
styled_box(axes[1], class_series('vh_contrast_db', clip=(-10, 40)), 'VH target-vs-background contrast', 'Contrast (dB)')
for ax in axes:  # plt.subplots(sharey=True) auto-hides non-first-column tick labels; these panels sit side by side, not stacked, so both need their own axis read
    ax.tick_params(labelleft=True)
plt.tight_layout()
plt.savefig(OUT_DIR / '05_VV_VH_CONTRAST_BY_CLASS.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 6) VV vs VH contrast density, small multiples by class (polarimetric signature)
lims = (-10, 40)
xy = {}
for c in CLASS_ORDER:
    g = det[det['final_class_3'] == c]
    x = pd.to_numeric(g['vv_contrast_db'], errors='coerce'); y = pd.to_numeric(g['vh_contrast_db'], errors='coerce')
    m = x.notna() & y.notna() & x.between(*lims) & y.between(*lims)
    xy[c] = (x[m].to_numpy(), y[m].to_numpy())


def _diag2(axes):
    for ax in axes:
        ax.plot(lims, lims, color=INK_MUTED, linewidth=0.8, linestyle='--')


small_multiple_density('Polarimetric SAR signature: VV vs. VH contrast, by final class',
                        'VV contrast (dB)', 'VH contrast (dB)', xy, lims, lims,
                        out_path=OUT_DIR / '06_VV_VS_VH_CONTRAST_BY_CLASS.png', annotate=_diag2)

# 7) THE key figure: how the VV/VH time series identifies static objects.
# x = spatial position stability across dates, y = temporal brightness persistence.
# Density small multiples, not an overlaid scatter: with ~12k points and a
# discretized y-axis (k of n valid dates), an overlay is just colored noise.
xlim, ylim = (0, 50), (0, 1.0)
xy = {}
for c in CLASS_ORDER:
    g = det[det['final_class_3'] == c]
    x = pd.to_numeric(g['position_std_m'], errors='coerce')
    y = pd.to_numeric(g['bright_persistence'], errors='coerce')
    m = x.notna() & y.notna() & x.between(*xlim)
    xy[c] = (x[m].to_numpy(), y[m].to_numpy())


def _static_zone(axes):
    for ax in axes:
        ax.axvspan(0, 20, color=INK_PRIMARY, alpha=0.05, zorder=5)
        ax.axhspan(0.82, 1.0, color=INK_PRIMARY, alpha=0.05, zorder=5)
    axes[-1].annotate('fixed-scatterer zone:\nlow position drift +\nbright on nearly every date',
                       xy=(48, 0.97), fontsize=8.5, color=INK_SECONDARY, ha='right', va='top')


small_multiple_density('VV/VH time-series static-object evidence, by final class',
                        'Position dispersion across repeat detections (m)',
                        'Fraction of scene dates location was "bright like a vessel"',
                        xy, xlim, ylim, bins=40,
                        out_path=OUT_DIR / '07_STATIC_EVIDENCE_PERSISTENCE_VS_DISPERSION.png',
                        annotate=_static_zone)

# 8) Temporal spike (dB) by class - transient brightness vs this location's own history
fig, ax = plt.subplots(figsize=(8, 6))
styled_box(ax, class_series('temporal_spike_db', clip=(-20, 30)), 'Temporal brightness spike vs. this location\'s own VV/VH history', 'Spike above median (dB)')
ax.axhline(0, color=INK_MUTED, linewidth=1)
plt.tight_layout()
plt.savefig(OUT_DIR / '08_TEMPORAL_SPIKE_BY_CLASS.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 9) Vessel evidence score distribution by class, with the calibrated threshold marked
fig, ax = plt.subplots(figsize=(9, 6))
bins = np.linspace(0, 1, 41)
for c in CLASS_ORDER:
    v = pd.to_numeric(det.loc[det['final_class_3'] == c, 'vessel_score'], errors='coerce').dropna()
    ax.hist(v, bins=bins, color=CLASS_COLOR[c], alpha=0.55, label=f'{CLASS_LABEL[c].replace(chr(10), " ")} (n={len(v)})',
            edgecolor='white', linewidth=0.3)
ax.axvline(THRESHOLD, color=INK_PRIMARY, linewidth=1.6, linestyle='--')
ax.text(THRESHOLD - 0.03, ax.get_ylim()[1] if ax.get_ylim()[1] else 1, f'calibrated threshold = {THRESHOLD:.2f}',
        ha='right', va='top', fontsize=9.5, color=INK_SECONDARY)
ax.set_xlabel('Vessel evidence score'); ax.set_ylabel('Objects')
ax.set_title('Vessel evidence score by final class (AIS-anchored model, 95% target recall)')
ax.grid(True, axis='y', linewidth=0.8); ax.set_axisbelow(True)
# legend sits in the empty band right of the threshold line, short of the tall
# bar at score~1.0, so neither the dashed line nor the peak bar cuts through it
ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.88, 0.90))
plt.tight_layout()
plt.savefig(OUT_DIR / '09_VESSEL_SCORE_BY_CLASS.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# 10) Signature area by class
fig, ax = plt.subplots(figsize=(8, 6))
styled_box(ax, class_series('area_m2', clip=(0, 120000)), 'SAR signature area by final class', 'Signature area (m^2)')
plt.tight_layout()
plt.savefig(OUT_DIR / '10_SIGNATURE_AREA_BY_CLASS.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print("Saved 10 figures to", OUT_DIR)
for p in sorted(OUT_DIR.glob('*.png')):
    print(' ', p.name)
