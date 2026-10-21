"""Can a label-free signal tell which photos the model is getting wrong? (reliability flag)

On the 97 external test photos, at each model's chosen fixed scale (detectors_external.py), compare
per-photo F1 (needs labels; used only to CHECK the signal) with label-free signals:
  top_conf    mean of the 20 most confident detections
  separation  Otsu between-class variance ratio of logit(confidence) for candidates > 0.01
              (how bimodal the score distribution is)
  agreement   share of one model's detections (at its DALI threshold) that the other one-photo
              model also finds within max(3, 0.5 r): two models trained on different photos agreeing
Reported: Spearman correlation with per-photo F1, and how much reviewing the worst-flagged 25% of
photos would capture of the photos that are actually worst (bottom-quartile F1).
Exploratory: three signals, one dataset, no threshold tuned.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.filters import threshold_otsu
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import detectors_external as DE  # noqa: E402
bnp = C2.bnp
OUT = Path(__file__).resolve().parent / 'results'


def separation(p):
    s = p[p[:, 3] > 0.01, 3]
    if len(s) < 10:
        return 0.0
    lg = np.log(s / (1 - s))
    t = threshold_otsu(lg)
    a, b = lg[lg < t], lg[lg >= t]
    if not len(a) or not len(b):
        return 0.0
    w0, w1 = len(a) / len(lg), len(b) / len(lg)
    return float(w0 * w1 * (a.mean() - b.mean()) ** 2 / lg.var())


def agreement(a, b):
    if not len(a):
        return 0.0
    if not len(b):
        return 0.0
    d, j = cKDTree(b[:, :2]).query(a[:, :2])
    return float(np.mean(d <= np.maximum(3.0, 0.5 * a[:, 2])))


def main():
    de = json.loads((OUT / 'detectors_external.json').read_text())
    test = C2.rf_rows('test')
    refs = [C2.rf_targets(r) for r in test]
    out = {}
    preds = {}
    for tag in ('pretrained_from2', 'pretrained_from1'):
        f = de[tag]['fixed_scale']['factor']
        thr = de[tag]['dali_threshold']
        preds[tag] = ([DE.predict_at(tag, r, f) for r in test], thr)
    for tag, other in (('pretrained_from2', 'pretrained_from1'), ('pretrained_from1', 'pretrained_from2')):
        P, thr = preds[tag]
        Q, thr_o = preds[other]
        f1 = np.array([bnp.metrics(p[p[:, 3] >= thr], g)['f1'] for p, g in zip(P, refs)])
        sig = dict(top_conf=np.array([np.sort(p[:, 3])[-20:].mean() if len(p) else 0 for p in P]),
                   separation=np.array([separation(p) for p in P]),
                   agreement=np.array([agreement(p[p[:, 3] >= thr], q[q[:, 3] >= thr_o]) for p, q in zip(P, Q)]))
        worst = f1 <= np.percentile(f1, 25)
        r = dict(per_photo_f1_median=float(np.median(f1)), per_photo_f1_p10_p90=[float(np.percentile(f1, 10)), float(np.percentile(f1, 90))])
        for k, v in sig.items():
            rho = spearmanr(v, f1).statistic
            flagged = v <= np.percentile(v, 25)          # lowest signal = least trustworthy
            r[k] = dict(spearman_with_f1=float(rho), worst_quartile_caught=float((flagged & worst).sum() / worst.sum()))
        out[tag] = r
        print(tag, json.dumps(r), flush=True)
    C2.save(OUT / 'reliability_flags.json', out)


if __name__ == '__main__':
    main()
