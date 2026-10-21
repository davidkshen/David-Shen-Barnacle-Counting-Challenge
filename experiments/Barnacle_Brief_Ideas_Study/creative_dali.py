"""New label-free post-processing ideas, tested on DALI cross-photo transfer (one-photo models).

1. no_overlap   Physical prior from the data exploration: two openings cannot overlap (only 1.3% /
                0% of nearest neighbours are closer than the sum of radii). After the detector's
                own NMS, drop a lower-scored detection whose centre is closer than 1.0 x (r_i + r_j)
                using predicted radii. alpha = 1.0 is fixed by the physical argument, not tuned.
                The score threshold is re-chosen on SOURCE validation with the same filter.
2. otsu         Label-free threshold on the NEW photo: Otsu's method on the logit of candidate
                scores (candidates above 0.01). Needs no labels from either photo.
3. conf_scale   Label-free scale search: run the new photo at f in {0.7, 0.85, 1.0, 1.2, 1.4} and keep
                the factor whose 50 most confident detections have the highest mean confidence.
                The source threshold is kept. (An image-statistics scale estimator was also built,
                but it failed its validation; see scale_estimators.json.)
4. hough_vote   Geometric second opinion: Hough circles (settings and radius range from the SOURCE
                photo, classical_cv.py) are found on the new photo; a detector candidate with a
                circle centre within max(3, 0.75 r) gets score + w * (circle vote level / 40).
                w and the threshold are chosen on SOURCE validation (x >= 768) only.
Baselines: reported setting (source threshold) and the oracle best threshold (diagnostic only).
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import scale_estimators as SE  # noqa: E402
import classical_cv as CV  # noqa: E402
bnp = C2.bnp

OUT = Path(__file__).resolve().parent / 'results'
FULL = (0, 0, 1024, 1024)
SV = (768, 0, 1024, 1024)
THRS = np.unique(np.r_[np.linspace(0.005, 0.95, 190), 0.001])


def no_overlap(p, alpha=1.0):
    if not len(p):
        return p
    p = p[np.argsort(-p[:, 3])]
    t = cKDTree(p[:, :2])
    alive = np.ones(len(p), bool)
    rmax = p[:, 2].max()
    for i in range(len(p)):
        if not alive[i]:
            continue
        for j in t.query_ball_point(p[i, :2], alpha * (p[i, 2] + rmax)):
            if j > i and alive[j] and np.hypot(*(p[j, :2] - p[i, :2])) < alpha * (p[i, 2] + p[j, 2]):
                alive[j] = False
    return p[alive]


def best_thr(p, gt):
    best = (0.25, -1)
    for t in THRS:
        f = bnp.metrics(p[p[:, 3] >= t], gt)['f1']
        if f > best[1]:
            best = (float(t), f)
    return best


def otsu_thr(p):
    s = p[p[:, 3] > 0.01, 3]
    lg = np.log(s / (1 - s))
    return float(1 / (1 + np.exp(-threshold_otsu(lg))))


def scaled_predict(mp, key, rgb, f):
    if abs(f - 1) < 1e-3:
        return bnp.cached_predict(mp, key, rgb, FULL)[0]
    n = int(round(1024 * f))
    big = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_LINEAR if f > 1 else cv2.INTER_AREA)
    p = bnp.cached_predict(mp, f'{key}_f{f:.4f}', big, (0, 0, n, n))[0].copy()
    p[:, :3] /= f
    return p


def hough_circles(rgb, rr, param1):
    return CV.hough_candidates(rgb, rr[0], rr[1], param1)


def hough_rescore(p, h, w):
    if not len(p) or not len(h) or w == 0:
        return p
    t = cKDTree(h[:, :2])
    d, j = t.query(p[:, :2])
    ok = d <= np.maximum(3.0, 0.75 * p[:, 2])
    q = p.copy()
    q[ok, 3] = q[ok, 3] + w * h[j[ok], 3] / 40.0
    return q


def summarize(p, gt):
    m = bnp.metrics(p, gt)
    return {k: m[k] for k in ('precision', 'recall', 'f1', 'predicted_count', 'reference_count', 'signed_error')}


def main():
    D = C2.dali()
    out = {}
    for tag, src, tgt in [('pretrained_from2', 2, 1), ('pretrained_from1', 1, 2), ('scratch_from2', 2, 1), ('scratch_from1', 1, 2)]:
        mp = bnp.MODELS / f'{tag}.pt'
        thr0 = json.loads((bnp.SELECTIONS / f'{tag}_selection.json').read_text())['chosen']['threshold']
        p = bnp.cached_predict(mp, f'img{tgt}', D[tgt]['im'], FULL)[0]
        gt = D[tgt]['gt']
        ps = bnp.cached_predict(mp, f'img{src}', D[src]['im'], SV)[0]
        gs = bnp.gt_region(src, SV)
        r = {}
        r['reported'] = dict(threshold=thr0, **summarize(p[p[:, 3] >= thr0], gt))
        o = best_thr(p, gt)
        r['oracle_threshold_DIAGNOSTIC'] = dict(threshold=o[0], f1=o[1])
        # 1 no-overlap
        pn, psn = no_overlap(p), no_overlap(ps)
        tn = best_thr(psn, gs)[0]
        r['no_overlap'] = dict(threshold=tn, **summarize(pn[pn[:, 3] >= tn], gt))
        # 2 otsu
        to = otsu_thr(p)
        r['otsu_label_free_threshold'] = dict(threshold=to, **summarize(p[p[:, 3] >= to], gt))
        to2 = otsu_thr(pn)
        r['no_overlap+otsu'] = dict(threshold=to2, **summarize(pn[pn[:, 3] >= to2], gt))
        # 3 confidence-guided scale search (label-free)
        conf = {}
        for f in (0.7, 0.85, 1.0, 1.2, 1.4):
            pf_ = scaled_predict(mp, f'img{tgt}', D[tgt]['im'], f)
            conf[f] = (float(np.sort(pf_[:, 3])[-50:].mean()), pf_)
        f = max(conf, key=lambda k: conf[k][0])
        pf = conf[f][1]
        r['conf_scale'] = dict(factor=f, mean_top50_conf={str(k): v[0] for k, v in conf.items()},
                               threshold=thr0, **summarize(pf[pf[:, 3] >= thr0], gt))
        r['conf_scale_oracle_threshold_DIAGNOSTIC'] = dict(f1=best_thr(pf, gt)[1])
        pfn = no_overlap(pf)
        r['conf_scale+no_overlap'] = dict(factor=f, threshold=tn, **summarize(pfn[pfn[:, 3] >= tn], gt))
        # 4 Hough second opinion (settings from the source photo's classical-CV tuning)
        cvres = json.loads((OUT / 'classical_cv.json').read_text())[f'dali_{src}to{tgt}']
        rr, p1 = cvres['radius_range'], cvres['hough']['setting']['param1']
        hs = hough_circles(D[src]['im'], rr, p1)
        hs = hs[hs[:, 0] >= 768]
        ht = hough_circles(D[tgt]['im'], rr, p1)
        best = None
        for w in (0.0, 0.1, 0.2, 0.4, 0.8):
            t, f = best_thr(hough_rescore(ps, hs, w), gs)
            if best is None or f > best[2]:
                best = (w, t, f)
        w, th, _ = best
        q = hough_rescore(p, ht, w)
        r['hough_vote'] = dict(weight=w, threshold=th, **summarize(q[q[:, 3] >= th], gt))
        out[tag] = r
        print(tag, ' | '.join(f"{k}: {v['f1']:.3f}" for k, v in r.items()), f'(conf-scale f={f:.2f})', flush=True)
    C2.save(OUT / 'creative_dali.json', out)


if __name__ == '__main__':
    main()
