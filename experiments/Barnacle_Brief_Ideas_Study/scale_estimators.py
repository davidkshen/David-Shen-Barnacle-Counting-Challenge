"""Label-free estimates of barnacle scale in a photo (no detector involved).

log_scale   characteristic scale of blobs: sigma maximising the mean of the top 2% of
            scale-normalised |LoG| responses; returned as an equivalent radius (sqrt(2)*sigma)
acf_spacing typical centre-to-centre spacing: first peak of the radially averaged
            autocorrelation of the band-passed image. Motivated by the data exploration:
            barnacles are evenly spaced (Clark-Evans R > 1 in DALI), so texture repeats at
            the spacing distance.
Validation (labels used only to CHECK the estimators, never inside them): Spearman correlation
with the true median radius / median nearest-neighbour distance on Roboflow dev photos, and
the photo-2/photo-1 ratio on DALI (true radius ratio 1.22, true spacing ratio from labels).
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402

SIGMAS = np.geomspace(1.2, 40, 30)


def _gray(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.GaussianBlur(g, (0, 0), 25)
    sd = np.sqrt(cv2.GaussianBlur((g - mu) ** 2, (0, 0), 25)) + 1e-3
    return (g - mu) / sd


def log_scale(rgb):
    g = _gray(rgb)
    e = []
    for s in SIGMAS:
        # pyramid: for large sigma, blur at reduced resolution (sigma_eff ~ 2 px), same normalisation
        k = max(1, int(s // 2))
        gs = cv2.resize(g, (g.shape[1] // k, g.shape[0] // k), interpolation=cv2.INTER_AREA) if k > 1 else g
        se = s / k
        r = np.abs((se ** 2) * ndi.gaussian_laplace(gs, se))
        e.append(np.mean(np.sort(r.ravel())[-max(10, r.size // 50):]))
    e = np.array(e)
    i = int(np.argmax(e))
    if 0 < i < len(e) - 1:   # parabolic refinement in log-sigma
        y0, y1, y2 = e[i - 1], e[i], e[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        ls = np.log(SIGMAS[i]) + d * (np.log(SIGMAS[1]) - np.log(SIGMAS[0]))
    else:
        ls = np.log(SIGMAS[i])
    return float(np.sqrt(2) * np.exp(ls))


def acf_spacing(rgb, rmin=4, rmax=150):
    g = _gray(rgb)
    g = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 30)   # band-pass
    g -= g.mean()
    H, W = g.shape
    F = np.fft.rfft2(g, s=(2 * H, 2 * W))
    ac = np.fft.irfft2(np.abs(F) ** 2, s=(2 * H, 2 * W))
    ac = np.fft.fftshift(ac)
    cy, cx = H, W
    yy, xx = np.indices(ac.shape)
    rr = np.hypot(yy - cy, xx - cx).astype(int)
    sel = rr <= rmax
    prof = np.bincount(rr[sel], ac[sel]) / np.maximum(np.bincount(rr[sel]), 1)
    prof = prof / prof[0]
    prof = ndi.gaussian_filter1d(prof, 1.0)
    for r in range(rmin, len(prof) - 1):   # first local maximum after the central peak
        if prof[r] > prof[r - 1] and prof[r] >= prof[r + 1]:
            return float(r)
    return float('nan')


def true_stats(xyr):
    if len(xyr) < 6:
        return np.nan, np.nan
    d, _ = cKDTree(xyr[:, :2]).query(xyr[:, :2], k=2)
    return float(np.median(xyr[:, 2])), float(np.median(d[:, 1]))


def main():
    out = {}
    D = C2.dali()
    for k in (1, 2):
        rad, nn = true_stats(D[k]['gt'])
        out[f'dali_photo{k}'] = dict(log_radius=log_scale(D[k]['im']), acf_spacing=acf_spacing(D[k]['im']),
                                     true_median_radius=rad, true_median_nn=nn)
        print(k, out[f'dali_photo{k}'], flush=True)
    a, b = out['dali_photo1'], out['dali_photo2']
    out['dali_ratio_2_over_1'] = dict(log=b['log_radius'] / a['log_radius'], acf=b['acf_spacing'] / a['acf_spacing'],
                                      true_radius=b['true_median_radius'] / a['true_median_radius'],
                                      true_nn=b['true_median_nn'] / a['true_median_nn'])
    print('DALI ratios', out['dali_ratio_2_over_1'])
    rows = C2.rf_rows('val')[::2] + C2.rf_rows('train')[::18]
    per = []
    for r in rows:
        t = C2.rf_targets(r)
        rad, nn = true_stats(t)
        if not np.isfinite(rad):
            continue
        im = C2.rf_img(r)
        per.append(dict(id=r['id'], split=r['split'], true_radius=rad, true_nn=nn, log_radius=log_scale(im),
                        acf_spacing=acf_spacing(im)))
    P = {k: np.array([p[k] for p in per], float) for k in ('true_radius', 'true_nn', 'log_radius', 'acf_spacing')}
    ok = np.isfinite(P['acf_spacing'])
    out['roboflow_validation'] = dict(
        photos=len(per),
        spearman_log_vs_true_radius=float(spearmanr(P['log_radius'], P['true_radius']).statistic),
        spearman_acf_vs_true_nn=float(spearmanr(P['acf_spacing'][ok], P['true_nn'][ok]).statistic),
        spearman_acf_vs_true_radius=float(spearmanr(P['acf_spacing'][ok], P['true_radius'][ok]).statistic),
        median_abs_log2_error_log=float(np.median(np.abs(np.log2(P['log_radius'] / P['true_radius']) -
                                                         np.median(np.log2(P['log_radius'] / P['true_radius']))))),
        median_abs_log2_error_acf=float(np.median(np.abs(np.log2(P['acf_spacing'][ok] / P['true_nn'][ok]) -
                                                         np.median(np.log2(P['acf_spacing'][ok] / P['true_nn'][ok]))))),
        acf_failures=int((~ok).sum()))
    print('Roboflow validation', out['roboflow_validation'])
    v = out['roboflow_validation']
    # Pre-declared rule: use the estimator whose photo-to-photo RATIOS are more accurate
    # (smaller median |log2 error| after removing a constant bias) on the labelled check photos.
    out['chosen_estimator'] = 'log' if v['median_abs_log2_error_log'] <= v['median_abs_log2_error_acf'] else 'acf'
    print('chosen estimator:', out['chosen_estimator'])
    C2.save(str(Path(__file__).resolve().parent / 'results') + '/scale_estimators.json', dict(summary=out, per_photo=per))


if __name__ == '__main__':
    main()
