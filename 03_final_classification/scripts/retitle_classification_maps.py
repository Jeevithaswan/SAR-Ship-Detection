"""
Replaces the plain "SCENEID_DATE - <code-ish description>" title baked into
each classification-map PNG with a clean, manager-presentable heading, while
leaving every pixel of the actual map/legend untouched.

Method: the title text only darkens a small fraction of each row's pixels;
the real map content (grayscale SAR image, or a full-width axes border/
gridline) darkens most of the row. So the boundary between "title band" and
"real content" is found per-image as the first row where >50% of pixels
differ from the background color - robust to any small variation in how
each figure was originally rendered, without touching the classification
data itself.
"""
import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Edit this, or set the SAR_PROJECT_ROOT environment variable, to point at
# your own working project folder (see the main README).
ROOT = Path(os.environ.get("SAR_PROJECT_ROOT", ".")).resolve()

TARGETS = [
    (ROOT / "SAR_Ship_Project_V4_1_VV_VH_3CLASS_RESULTS" / "FIGURES" / "final_3class_overlays", "Final Ship Classification"),
    (ROOT / "SAR_Ship_Project_V4_1_RESULTS_WITH_FIGURES" / "FINAL_EXPORT" / "FIGURES" / "classification_maps" / "sar_overlays", "Object-Level Classification"),
    (ROOT / "SAR_Ship_Project_V4_1_RESULTS_WITH_FIGURES" / "FINAL_EXPORT" / "FIGURES" / "classification_maps" / "geographic_maps", "Geographic Classification Map"),
    (ROOT / "SAR_Ship_Project_V4_1_RESULTS_WITH_FIGURES" / "FINAL_EXPORT" / "FIGURES" / "classification_maps" / "vessel_only", "Vessel-Only Classification"),
]

SCENE_DATE_RE = re.compile(r'(S\d+)_(\d{8})')

# A bold sans-serif truetype font. Tries common Windows/macOS/Linux locations
# in order; falls back to PIL's bundled default (much lower quality, but the
# script still runs) if none are found.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeuib.ttf",       # Windows
    r"C:\Windows\Fonts\arialbd.ttf",        # Windows fallback
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
]
FONT_PATH_BOLD = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)


def format_date(datestr):
    dt = datetime.strptime(datestr, "%Y%m%d")
    day = dt.day  # avoid %-d (not portable on Windows)
    return f"{day} {dt.strftime('%b %Y')}"


def find_content_top(arr, bg, tol=6, frac_thresh=0.5):
    diff = np.abs(arr.astype(int) - bg.astype(int)).max(axis=2)
    nonbg_frac = (diff > tol).mean(axis=1)
    rows = np.where(nonbg_frac > frac_thresh)[0]
    return int(rows[0]) if len(rows) else int(arr.shape[0] * 0.05)


def retitle(path: Path, heading: str):
    m = SCENE_DATE_RE.search(path.stem)
    if not m:
        print('  SKIP (no scene/date match):', path.name)
        return
    scene, datestr = m.group(1), m.group(2)
    title_text = f"{heading} \u2014 Scene {scene} \u00b7 {format_date(datestr)}"

    im = Image.open(path).convert('RGB')
    arr = np.array(im)
    bg = arr[2, 2].copy()
    content_top = find_content_top(arr, bg)

    draw = ImageDraw.Draw(im)
    # wipe the old title band completely
    draw.rectangle([0, 0, im.width, content_top], fill=tuple(int(x) for x in bg))

    # size the font to the image width, then center the title in the band
    font_size = max(14, min(30, im.width // 68))
    if FONT_PATH_BOLD:
        font = ImageFont.truetype(FONT_PATH_BOLD, font_size)
    else:
        print('  WARNING: no bold truetype font found on this system, using PIL default (lower quality).')
        font = ImageFont.load_default(size=font_size)
    bbox = draw.textbbox((0, 0), title_text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (im.width - text_w) // 2
    y = max(2, (content_top - text_h) // 2 - bbox[1])
    draw.text((x, y), title_text, font=font, fill=(11, 11, 11))

    im.save(path)
    print('  OK:', path.name, '->', title_text)


def main():
    total = 0
    for folder, heading in TARGETS:
        if not folder.exists():
            print('MISSING FOLDER:', folder)
            continue
        print('=' * 90)
        print(folder, '  [heading:', heading, ']')
        for p in sorted(folder.glob('*.png')):
            retitle(p, heading)
            total += 1
    print('\nDone. Retitled', total, 'images.')


if __name__ == '__main__':
    # Guarded deliberately: this script overwrites images in place. Importing
    # it (e.g. to reuse retitle()/find_content_top() elsewhere) must NOT
    # trigger a full batch run - only `python retitle_classification_maps.py` does.
    main()
