from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import shapefile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe

ROOT = Path(r'C:\Users\ASUS\Documents\ChatGPT\灾害信息处理')
DATA = ROOT / 'figure1_qgis'
OUT = ROOT / 'figure1_v2'
OUT.mkdir(parents=True, exist_ok=True)

# Publication baseline
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6.0,
    'axes.labelsize': 6.0,
    'xtick.labelsize': 5.2,
    'ytick.labelsize': 5.2,
    'axes.linewidth': 0.55,
    'xtick.major.width': 0.45,
    'ytick.major.width': 0.45,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.dpi': 300,
})

COL = {
    'ocean': '#F5F8FA',
    'land': '#F1EFE8',
    'land_hi': '#F7F4EA',
    'border': '#AAB2B6',
    'coast': '#89949A',
    'grid': '#DDE3E6',
    'source': '#2C6FB0',
    'target': '#C94747',
    'target_edge': '#7F1D1D',
    'diag': '#8A4F9E',
    'extent_fill': '#E59A3A',
    'extent_edge': '#C56A12',
    'text': '#20252A',
    'muted': '#5A646B',
}

REGIONS = [
    dict(id='1', short='Wenchuan', lon=103.32, lat=31.00, w=102.95, s=30.55, e=103.95, n=31.60),
    dict(id='2', short='Jiuzhai Valley', lon=103.86, lat=33.19, w=103.55, s=32.90, e=104.15, n=33.50),
    dict(id='3', short='Moxitaidi', lon=102.24, lat=29.68, w=102.00, s=29.45, e=102.45, n=29.90),
    dict(id='4', short='Longxi River', lon=103.58, lat=31.08, w=103.40, s=30.95, e=103.75, n=31.25),
    dict(id='5', short='Hokkaido Iburi-Tobu', lon=141.93, lat=42.69, w=141.50, s=42.45, e=142.25, n=43.05),
    dict(id='6', short='Lombok', lon=116.40, lat=-8.35, w=115.90, s=-8.90, e=116.80, n=-7.90),
    dict(id='7', short='Palu', lon=119.87, lat=-0.89, w=119.65, s=-1.15, e=120.10, n=-0.10),
]
SOURCES = [
    dict(id='S1', short='Moxi', lon=102.16, lat=29.65),
    dict(id='S2', short='Bijie', lon=105.29, lat=27.30),
]

# Load Natural Earth polygons once.
shape_readers = {
    'coarse': shapefile.Reader(str(DATA / 'basemap' / 'ne_110m_admin_0_countries.shp')),
    'fine': shapefile.Reader(str(DATA / 'basemap' / 'ne_10m_admin_0_countries.shp')),
}


def iter_polygons(reader: shapefile.Reader):
    for sr in reader.iterShapeRecords():
        name = sr.record.as_dict().get('ADMIN', '')
        pts = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]
        for i in range(len(parts) - 1):
            part = pts[parts[i]:parts[i + 1]]
            if len(part) >= 3:
                yield name, np.asarray(part, dtype=float)


def draw_land(ax, reader, extent, highlight=()):
    xmin, ymin, xmax, ymax = extent
    pad_x = (xmax - xmin) * 0.15
    pad_y = (ymax - ymin) * 0.15
    for name, part in iter_polygons(reader):
        if part[:, 0].max() < xmin - pad_x or part[:, 0].min() > xmax + pad_x:
            continue
        if part[:, 1].max() < ymin - pad_y or part[:, 1].min() > ymax + pad_y:
            continue
        fc = COL['land_hi'] if name in highlight else COL['land']
        ax.add_patch(Polygon(part, closed=True, facecolor=fc, edgecolor=COL['coast'],
                             linewidth=0.22, zorder=1, joinstyle='round'))


def setup_map(ax, extent, *, ticks=True, grid=True, aspect_ref=None):
    xmin, ymin, xmax, ymax = extent
    ax.set_facecolor(COL['ocean'])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if aspect_ref is not None:
        ax.set_aspect(1.0 / math.cos(math.radians(aspect_ref)), adjustable='box')
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#7B858B')
        s.set_linewidth(0.55)
    if grid:
        ax.grid(color=COL['grid'], linewidth=0.25, alpha=0.75, zorder=0)
    if ticks:
        ax.tick_params(direction='out', pad=1.5, colors='#3D454A')
    else:
        ax.set_xticks([]); ax.set_yticks([])
    return ax


def degfmt(v, _):
    v = int(round(v))
    if v < 0:
        return f'{abs(v)}°S'
    return f'{v}°E' if v != 0 else '0°'


def draw_extent(ax, r, lw=0.6, alpha=0.28, z=3):
    ax.add_patch(Rectangle((r['w'], r['s']), r['e'] - r['w'], r['n'] - r['s'],
                           facecolor=COL['extent_fill'], edgecolor=COL['extent_edge'],
                           linewidth=lw, alpha=alpha, zorder=z))


def target_marker(ax, lon, lat, number=None, label=None, *, size=27, z=7, source_adjacent=False):
    edge = COL['diag'] if source_adjacent else COL['target_edge']
    face = 'white' if source_adjacent else COL['target']
    ax.scatter([lon], [lat], s=size, marker='o', facecolor=face, edgecolor=edge,
               linewidth=0.85, zorder=z)
    if number:
        ax.text(lon, lat, str(number), ha='center', va='center', fontsize=4.1,
                color=edge if source_adjacent else 'white', weight='bold', zorder=z + 1)


def source_marker(ax, lon, lat, size=27, z=8):
    ax.scatter([lon], [lat], s=size, marker='s', facecolor=COL['source'],
               edgecolor='white', linewidth=0.75, zorder=z)


HALO = [pe.withStroke(linewidth=1.7, foreground='white')]


def ann(ax, text, xy, xytext, *, color=COL['text'], size=5.1, weight='normal', ha='left', va='center'):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords='data', ha=ha, va=va,
                fontsize=size, color=color, weight=weight, zorder=12,
                arrowprops=dict(arrowstyle='-', color='#6C7479', linewidth=0.45,
                                shrinkA=1.5, shrinkB=2.5),
                path_effects=HALO)


def panel_heading(ax, letter, text):
    # Panel names are defined in the manuscript caption; keeping only the
    # letter here prevents titles from overflowing narrow map panels.
    ax.text(0.018, 0.975, letter, transform=ax.transAxes,
            ha='left', va='top', fontsize=8.0, fontweight='bold', color=COL['text'],
            zorder=40, bbox=dict(facecolor='white', edgecolor='#7B858B',
                                 linewidth=0.35, boxstyle='round,pad=0.20'))

def north_arrow(ax, x=0.94, y=0.90):
    ax.add_patch(FancyArrowPatch((x, y - 0.08), (x, y), transform=ax.transAxes,
                                 arrowstyle='-|>', mutation_scale=7, linewidth=0.55,
                                 color='#2A3034', zorder=20, clip_on=False))
    ax.text(x, y + 0.005, 'N', transform=ax.transAxes, ha='center', va='bottom',
            fontsize=5.1, color='#2A3034', weight='bold', zorder=21)


def scale_bar(ax, km, lat, *, xfrac=0.72, yfrac=0.10):
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    span_x = xmax - xmin; span_y = ymax - ymin
    length = km / (111.32 * math.cos(math.radians(lat)))
    x0 = xmin + xfrac * span_x
    if x0 + length > xmax - 0.04 * span_x:
        x0 = xmax - 0.04 * span_x - length
    y0 = ymin + yfrac * span_y
    ax.add_patch(Rectangle((x0 - 0.025 * span_x, y0 - 0.055 * span_y),
                           length + 0.05 * span_x, 0.12 * span_y,
                           facecolor='white', edgecolor='none', alpha=0.88, zorder=18))
    ax.plot([x0, x0 + length], [y0, y0], color='#20252A', linewidth=1.25,
            solid_capstyle='butt', zorder=20)
    ax.plot([x0, x0], [y0 - 0.018 * span_y, y0 + 0.018 * span_y], color='#20252A', lw=0.55, zorder=20)
    ax.plot([x0 + length, x0 + length], [y0 - 0.018 * span_y, y0 + 0.018 * span_y], color='#20252A', lw=0.55, zorder=20)
    ax.text(x0 + length / 2, y0 + 0.027 * span_y, f'{km:g} km',
            ha='center', va='bottom', fontsize=4.6, color='#20252A', zorder=20)


# ---------------------------------------------------------------------------
# Figure: 160 x 127 mm, matching the manuscript insertion ratio (1.260).
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(6.3, 4.99139), dpi=300)
gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0], hspace=0.29, wspace=0.22,
                      left=0.052, right=0.988, bottom=0.065, top=0.945)
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])
axd = fig.add_subplot(gs[1, 2])

# Panel a: Asia-Pacific overview
ext_a = (75, -15, 155, 50)
setup_map(axa, ext_a, ticks=True, grid=True, aspect_ref=20)
draw_land(axa, shape_readers['coarse'], ext_a, highlight={'China', 'Japan', 'Indonesia'})
axa.set_xticks(np.arange(80, 156, 10))
axa.set_yticks(np.arange(-10, 51, 10))
axa.xaxis.set_major_formatter(FuncFormatter(degfmt))
axa.yaxis.set_major_formatter(FuncFormatter(degfmt))

for r in REGIONS:
    draw_extent(axa, r, lw=0.55, alpha=0.22, z=3)
    target_marker(axa, r['lon'], r['lat'], r['id'], size=34, z=9,
                  source_adjacent=(r['id'] == '3'))
# D5-13: at the overview scale the two source squares coincide, so a single
# 'Source domain' marker is drawn here and the S1/S2 detail is left to panel (b).
_src_lon = sum(s['lon'] for s in SOURCES) / len(SOURCES)
_src_lat = sum(s['lat'] for s in SOURCES) / len(SOURCES)
source_marker(axa, _src_lon, _src_lat, size=31, z=10)

# Context labels and direct leader labels in the overview.
for text, x, y in [
    ('China', 96, 37), ('Japan', 138.5, 37.4), ('Indonesia', 115.5, -4.5),
    ('Philippines', 122.0, 15.0), ('Vietnam', 106.5, 17.6), ('Thailand', 100.0, 15.0),
]:
    axa.text(x, y, text, fontsize=5.0, color='#697278', ha='center', va='center', zorder=5)

# Compact legend and region key.
leg_x, leg_y = 0.012, 0.016
leg_w, leg_h = 0.340, 0.340
axa.add_patch(Rectangle((leg_x, leg_y), leg_w, leg_h, transform=axa.transAxes,
                        facecolor='white', edgecolor='#7B858B', linewidth=0.45,
                        alpha=0.95, zorder=25))
axa.text(leg_x + 0.014, leg_y + leg_h - 0.028, 'Marker legend and region key',
         transform=axa.transAxes, fontsize=4.9, va='center', weight='bold', zorder=30)
axa.scatter([leg_x + 0.027], [leg_y + leg_h - 0.078], s=23, marker='s',
            transform=axa.transAxes, facecolor=COL['source'], edgecolor='white',
            linewidth=0.55, zorder=30, clip_on=False)
axa.text(leg_x + 0.047, leg_y + leg_h - 0.078, 'Source domain',
         transform=axa.transAxes, fontsize=4.6, va='center', zorder=30)
axa.scatter([leg_x + 0.027], [leg_y + leg_h - 0.126], s=23, marker='o',
            transform=axa.transAxes, facecolor=COL['target'], edgecolor=COL['target_edge'],
            linewidth=0.55, zorder=30, clip_on=False)
axa.text(leg_x + 0.047, leg_y + leg_h - 0.126, 'Held-out target region (1-7)',
         transform=axa.transAxes, fontsize=4.6, va='center', zorder=30)
axa.scatter([leg_x + 0.027], [leg_y + leg_h - 0.174], s=23, marker='o',
            transform=axa.transAxes, facecolor='white', edgecolor=COL['diag'],
            linewidth=0.7, zorder=30, clip_on=False)
axa.text(leg_x + 0.047, leg_y + leg_h - 0.174, 'Source-adjacent diagnostic (3)',
         transform=axa.transAxes, fontsize=4.6, va='center', zorder=30)
axa.add_patch(Rectangle((leg_x + 0.016, leg_y + 0.067), 0.022, 0.030,
                        transform=axa.transAxes, facecolor=COL['extent_fill'],
                        edgecolor=COL['extent_edge'], linewidth=0.45, alpha=0.28, zorder=30))
axa.text(leg_x + 0.047, leg_y + 0.082, 'Approximate locator extent',
         transform=axa.transAxes, fontsize=4.6, va='center', zorder=30)
axa.text(leg_x + 0.014, leg_y + 0.014,
         '1 Wenchuan; 2 Jiuzhaigou; 3 Moxitaidi; 4 Longxi River;\n5 Hokkaido Iburi-Tobu; 6 Lombok; 7 Palu; S1 Moxi; S2 Bijie',
         transform=axa.transAxes, fontsize=4.25, va='bottom', linespacing=1.15,
         color=COL['muted'], zorder=30)

panel_heading(axa, 'a', 'Regional\noverview')

# Panel b: Sichuan and source domain
ext_b = (100.65, 26.40, 106.60, 34.15)
setup_map(axb, ext_b, ticks=True, grid=True, aspect_ref=30)
draw_land(axb, shape_readers['fine'], ext_b, highlight={'China'})
axb.set_xticks([101, 103, 105])
axb.set_yticks([27, 29, 31, 33])
axb.xaxis.set_major_formatter(FuncFormatter(degfmt)); axb.yaxis.set_major_formatter(FuncFormatter(degfmt))
for r in REGIONS[:4]:
    draw_extent(axb, r, lw=0.5, alpha=0.16, z=3)
    target_marker(axb, r['lon'], r['lat'], r['id'], size=31, z=9, source_adjacent=(r['id'] == '3'))
for s in SOURCES:
    source_marker(axb, s['lon'], s['lat'], size=29, z=10)
ann(axb, '1 Wenchuan', (103.32, 31.00), (100.85, 31.75), color=COL['target_edge'], size=5.0, weight='bold', ha='left')
ann(axb, '2 Jiuzhaigou', (103.86, 33.19), (104.42, 32.90), color=COL['target_edge'], size=5.0, weight='bold', ha='right')
ann(axb, '3 Moxitaidi', (102.24, 29.68), (100.72, 29.05), color=COL['diag'], size=5.0, weight='bold', ha='left')
ann(axb, '4 Longxi River', (103.58, 31.08), (104.45, 30.30), color=COL['target_edge'], size=5.0, weight='bold', ha='right')
ann(axb, 'S1 Moxi', (102.16, 29.65), (100.80, 30.20), color=COL['source'], size=4.8, weight='bold', ha='left')
ann(axb, 'S2 Bijie', (105.29, 27.30), (105.25, 28.15), color=COL['source'], size=4.8, weight='bold', ha='left')
scale_bar(axb, 50, 30.5, xfrac=0.60, yfrac=0.08)
north_arrow(axb, x=0.94, y=0.80)
panel_heading(axb, 'b', 'Sichuan and\nsource domain')

# Panel c: Hokkaido Iburi-Tobu
ext_c = (140.35, 41.35, 143.00, 43.65)
setup_map(axc, ext_c, ticks=True, grid=True, aspect_ref=42.5)
draw_land(axc, shape_readers['fine'], ext_c, highlight={'Japan'})
axc.set_xticks([141, 142, 143]); axc.set_yticks([42, 43])
axc.xaxis.set_major_formatter(FuncFormatter(degfmt)); axc.yaxis.set_major_formatter(FuncFormatter(degfmt))
r = REGIONS[4]
draw_extent(axc, r, lw=0.7, alpha=0.34, z=3)
target_marker(axc, r['lon'], r['lat'], r['id'], size=33, z=9)
ann(axc, '5 Hokkaido Iburi-Tobu', (141.93, 42.69), (140.50, 43.30), color=COL['target_edge'], size=5.0, weight='bold', ha='left')
scale_bar(axc, 20, 42.7, xfrac=0.60, yfrac=0.08)
north_arrow(axc)
panel_heading(axc, 'c', 'Hokkaido\nIburi-Tobu')

# Panel d: Lombok and Palu
ext_d = (115.15, -10.00, 120.55, 1.00)
setup_map(axd, ext_d, ticks=True, grid=True, aspect_ref=-5.0)
draw_land(axd, shape_readers['fine'], ext_d, highlight={'Indonesia'})
axd.set_xticks([116, 118, 120]); axd.set_yticks([-8, -6, -4, -2, 0])
axd.xaxis.set_major_formatter(FuncFormatter(degfmt)); axd.yaxis.set_major_formatter(FuncFormatter(degfmt))
for idx in (5, 6):
    r = REGIONS[idx]
    draw_extent(axd, r, lw=0.7, alpha=0.34, z=3)
    target_marker(axd, r['lon'], r['lat'], r['id'], size=33, z=9)
ann(axd, '6 Lombok', (116.40, -8.35), (115.27, -9.35), color=COL['target_edge'], size=5.0, weight='bold', ha='left')
ann(axd, '7 Palu', (119.87, -0.89), (117.35, -0.35), color=COL['target_edge'], size=5.0, weight='bold', ha='left')
scale_bar(axd, 200, -5.0, xfrac=0.60, yfrac=0.08)
north_arrow(axd)
panel_heading(axd, 'd', 'Lombok and\nPalu')


# --------------------------- automated text QA ---------------------------
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
all_texts = []
for t in fig.findobj(matplotlib.text.Text):
    try:
        txt = t.get_text().strip()
        if t.get_visible() and txt:
            all_texts.append((t, txt, t.get_window_extent(renderer)))
    except Exception:
        pass
W, H = fig.bbox.width, fig.bbox.height
clipped = [(txt, (bb.x0, bb.y0, bb.x1, bb.y1)) for t, txt, bb in all_texts
           if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1]
overlaps = []
for i, (ta_obj, ta, a) in enumerate(all_texts):
    for tb_obj, tb, b in all_texts[i+1:]:
        ix = min(a.x1, b.x1) - max(a.x0, b.x0)
        iy = min(a.y1, b.y1) - max(a.y0, b.y0)
        if ix > 1 and iy > 1:
            inter = ix * iy
            smaller = min(a.width * a.height, b.width * b.height)
            ratio = inter / smaller if smaller else 0
            if ratio > 0.35 and not (ta.isdigit() and tb.isdigit()):
                overlaps.append((round(ratio, 2), ta, tb))
# Strictly check custom texts against their own panel box. Axis/tick/title items are excluded.
axis_text_ids = set()
for ax in fig.axes:
    for t in [ax.title, ax.xaxis.label, ax.yaxis.label, ax.xaxis.offsetText, ax.yaxis.offsetText]:
        if t is not None: axis_text_ids.add(id(t))
    for t in (*ax.get_xticklabels(), *ax.get_xticklabels(minor=True), *ax.get_yticklabels(), *ax.get_yticklabels(minor=True)):
        axis_text_ids.add(id(t))
panel_overflow = []
for t, txt, bb in all_texts:
    if id(t) in axis_text_ids or t.axes is None:
        continue
    ab = t.axes.bbox
    limit_x = max(8.0, ab.width * 0.012)
    limit_y = max(8.0, ab.height * 0.012)
    dx = max(ab.x0 - bb.x0, bb.x1 - ab.x1, 0.0)
    dy = max(ab.y0 - bb.y0, bb.y1 - ab.y1, 0.0)
    if dx > limit_x or dy > limit_y:
        panel_overflow.append((round(dx,1), round(dy,1), txt, t.axes.get_position().bounds,
                               (round(bb.x0,1),round(bb.y0,1),round(bb.x1,1),round(bb.y1,1)),
                               (round(ab.x0,1),round(ab.y0,1),round(ab.x1,1),round(ab.y1,1))))
print('TEXT_QA clipped=', len(clipped), 'overlaps=', len(overlaps), 'panel_overflow=', len(panel_overflow))
for row in clipped[:20]: print('CLIP', row)
for row in overlaps[:30]: print('OVERLAP', row)
for row in panel_overflow[:80]: print('PANEL_OVERFLOW', row)
fig.savefig(OUT / 'Figure1_study_area_v2.png', dpi=600, facecolor='white')
fig.savefig(OUT / 'Figure1_study_area_v2.pdf', facecolor='white')
fig.savefig(OUT / 'Figure1_study_area_v2.svg', facecolor='white')
plt.close(fig)

# 600-dpi TIFF for journal systems that require raster.
from PIL import Image
im = Image.open(OUT / 'Figure1_study_area_v2.png').convert('RGB')
im.save(OUT / 'Figure1_study_area_v2.tif', compression='tiff_lzw', dpi=(600, 600))

print('Generated:')
for p in sorted(OUT.glob('Figure1_study_area_v2.*')):
    print(p, p.stat().st_size)
