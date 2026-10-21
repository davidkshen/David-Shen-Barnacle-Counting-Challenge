"""Brief idea: 'Exploring the data' - sizes, spacing and density from the labels.

DALI: aperture centers/radii decoded from the supplied masks (rectified 1024 quadrat).
Roboflow: adult-barnacle boxes (classes 0,1,2,6,7) in longest-side-1024 coordinates.
Metrics:
  size           equivalent radius (px), and diameter relative to the quadrat
  spacing        nearest-neighbour (NN) center distance; NN distance / (r_i + r_j) ('touching' ~ 1)
  aggregation    Clark-Evans R = mean NN / (0.5 / sqrt(density)); R<1 clustered, ~1 random, >1 regular
                 (computed inside the convex hull-free bounding area; edge effects not corrected)
  local density  apertures per 128-px tile; coefficient of variation across tiles
  size-density   Spearman correlation between radius and local neighbour count (r < 25 px)
Outputs results/data_exploration.json and figures/size_spacing.png.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os
import common2 as C2  # noqa: E402
bnp = C2.bnp
import rf_bench as B  # noqa: E402

R = Path(__file__).resolve().parent / 'results'
R.mkdir(exist_ok=True)


def stats(xyr, area_wh):
    xy, r = xyr[:, :2], xyr[:, 2]
    t = cKDTree(xy)
    d, j = t.query(xy, k=2)
    nn = d[:, 1]
    touch = nn / (r + r[j[:, 1]])
    W, H = area_wh
    dens = len(xy) / (W * H)
    ce = nn.mean() / (0.5 / np.sqrt(dens))
    nb = np.array([len(t.query_ball_point(p, 25)) - 1 for p in xy])
    rho = spearmanr(r, nb).statistic if len(xy) > 5 else np.nan
    q = lambda v: [float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])]  # noqa: E731
    return dict(n=int(len(xy)), radius_p5_25_50_75_95=q(r), nn_distance_p5_25_50_75_95=q(nn),
                nn_over_sum_radii_median=float(np.median(touch)), share_touching_or_overlapping=float(np.mean(touch < 1.05)),
                clark_evans_R=float(ce), radius_vs_local_count_spearman=float(rho))


def main():
    out = {}
    C = bnp.cases()
    for k in (1, 2):
        g = C[k]['gt']
        s = stats(g, (1024, 1024))
        ij = np.floor(g[:, :2] / 128).astype(int)
        tc = np.bincount(ij[:, 1] * 8 + ij[:, 0], minlength=64)
        s['per_tile_count_mean'] = float(tc.mean())
        s['per_tile_count_cv'] = float(tc.std() / tc.mean())
        s['empty_tiles'] = int((tc == 0).sum())
        out[f'dali_photo{k}'] = s
        print(f'DALI photo {k}:', json.dumps({a: b for a, b in s.items() if 'p5' not in a}))
    rows = B.load()
    per = []
    for r in rows:
        t, _ = B.targets(r)
        if len(t) < 6:
            continue
        s = stats(t, (r['width'], r['height']))
        per.append(dict(id=r['id'], split=r['split'], **{k: v for k, v in s.items()}))
    agg = lambda key, i=None: [float(x) for x in np.nanpercentile([p[key][i] if i is not None else p[key] for p in per], [10, 50, 90])]  # noqa: E731
    out['roboflow_per_image'] = dict(images=len(per),
                                     targets_per_image_p10_50_90=agg('n'),
                                     median_radius_p10_50_90=agg('radius_p5_25_50_75_95', 2),
                                     nn_over_sum_radii_p10_50_90=agg('nn_over_sum_radii_median'),
                                     clark_evans_R_p10_50_90=agg('clark_evans_R'),
                                     radius_vs_local_count_spearman_p10_50_90=agg('radius_vs_local_count_spearman'))
    print('Roboflow per-image summary:', json.dumps(out['roboflow_per_image']))
    # within-image size spread vs between-image spread (how 'local' is size?)
    med = np.array([p['radius_p5_25_50_75_95'][2] for p in per])
    iqr = np.array([p['radius_p5_25_50_75_95'][3] - p['radius_p5_25_50_75_95'][1] for p in per])
    out['roboflow_size_locality'] = dict(between_image_median_radius_cv=float(med.std() / med.mean()),
                                         within_image_iqr_over_median_median=float(np.median(iqr / med)),
                                         median_radius_range=[float(med.min()), float(med.max())])
    print('size locality:', out['roboflow_size_locality'])
    (R / 'data_exploration.json').write_text(json.dumps(dict(summary=out, roboflow_images=per), indent=1))
    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURF, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
    BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
    plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'font.size': 10, 'axes.labelcolor': INK2,
                         'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True,
                         'grid.color': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.axisbelow': True})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, col in ((1, BLUE), (2, ORANGE)):
        g = C[k]['gt']
        ax[0].hist(g[:, 2], bins=np.arange(2, 18, 0.5), color=col, alpha=.55, label=f'Photo {k} (n={len(g)})')
        d, j = cKDTree(g[:, :2]).query(g[:, :2], k=2)
        ax[1].hist(d[:, 1] / (g[:, 2] + g[j[:, 1], 2]), bins=np.arange(0, 4, 0.08), color=col, alpha=.55)
    ax[0].set_xlabel('Aperture radius (px, rectified quadrat)')
    ax[0].set_ylabel('Apertures')
    ax[0].set_title('Size', loc='left')
    ax[1].axvline(1, color=MUTED, lw=1)
    ax[1].set_xlabel('Nearest-neighbour distance / (sum of radii)')
    ax[1].set_title('Spacing (1 = rims touching)', loc='left')
    ax[2].scatter(med, [p['clark_evans_R'] for p in per], s=10, color=AQUA, alpha=.6, label='Roboflow photo')
    for k, col in ((1, BLUE), (2, ORANGE)):
        ax[2].scatter([out[f'dali_photo{k}']['radius_p5_25_50_75_95'][2]], [out[f'dali_photo{k}']['clark_evans_R']],
                      s=70, color=col, edgecolor=SURF, linewidth=2, zorder=3, label=f'DALI photo {k}')
    ax[2].axhline(1, color=MUTED, lw=1)
    ax[2].set_xlabel('Median radius (px)')
    ax[2].set_ylabel('Clark-Evans R (<1 clustered)')
    ax[2].set_title('Clustering vs size', loc='left')
    h, l = ax[0].get_legend_handles_labels()
    h2, l2 = ax[2].get_legend_handles_labels()
    fig.legend(h + h2[:1], l + l2[:1], loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    (Path(__file__).resolve().parent / 'figures').mkdir(exist_ok=True)
    fig.savefig(str(Path(__file__).resolve().parent / 'figures') + '/size_spacing.png', dpi=160)


if __name__ == '__main__':
    main()
