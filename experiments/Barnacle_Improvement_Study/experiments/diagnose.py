"""Task 1: why does cross-photo transfer fail?

Part A - threshold vs ranking. For each one-photo base model:
  * F1 at the source-validation threshold (what was reported),
  * best F1 at ANY threshold on the target (an oracle that peeks at target labels;
    it is a diagnostic upper bound for threshold-only fixes, not a result),
  * recall ceiling = recall of all candidates down to confidence 0.001.
If oracle F1 >> reported F1, the model ranks barnacles above background but its
confidence scale shifted between photos. If oracle F1 is also low, ranking failed.

Part B - error characteristics at the reported operating point, for each
reference (hit/miss) and each prediction (TP/FP): size, local density,
brightness, blur, background colour, interior darkness, and tile-seam distance.
"""
import sys
import numpy as np
import cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp

M = bnp.MODELS
SEL = bnp.SELECTIONS
FULL = (0, 0, 1024, 1024)
THRS = np.unique(np.r_[np.linspace(0.005, 0.95, 190), 0.001])


def f1_curve(p, s, gt):
    rows = []
    for t in THRS:
        m = bnp.metrics(p[s >= t], gt)
        rows.append(dict(threshold=float(t), **{k: m[k] for k in
                    ('precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'predicted_count')}))
    return rows


def local_maps(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    blur = cv2.blur(lap * lap, (31, 31))                       # local sharpness (high = sharp)
    bright = cv2.blur(g, (15, 15))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    redness = cv2.blur(lab[..., 1] - 128, (31, 31))           # a* channel: red rock vs grey shell
    return g, blur, bright, redness


def interior_ratio(g, x, y, r):
    """Mean grey of inner disk (0.5r) / mean grey of ring (1.0r-1.6r): dark-centred openings < 1."""
    r = max(r, 3.0)
    x0, x1 = int(max(0, x - 2 * r)), int(min(g.shape[1], x + 2 * r + 1))
    y0, y1 = int(max(0, y - 2 * r)), int(min(g.shape[0], y + 2 * r + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    inner = g[y0:y1, x0:x1][d <= 0.5 * r]
    ring = g[y0:y1, x0:x1][(d >= r) & (d <= 1.6 * r)]
    return float(inner.mean() / max(ring.mean(), 1.0)) if len(inner) and len(ring) else np.nan


SEAMS = np.array([224, 416, 608, 800])   # tile-core boundaries for the full-image window grid


def seam_dist(x, y):
    return float(min(np.abs(SEAMS - x).min(), np.abs(SEAMS - y).min()))


def describe(points, rgb, gt_all, maps):
    g, blur, bright, red = maps
    out = []
    for x, y, r in points[:, :3]:
        xi, yi = int(np.clip(round(x), 0, 1023)), int(np.clip(round(y), 0, 1023))
        dens = int((np.hypot(gt_all[:, 0] - x, gt_all[:, 1] - y) < 25).sum())
        out.append(dict(radius=float(r), density25=dens, brightness=float(bright[yi, xi]),
                        sharpness=float(np.log10(blur[yi, xi] + 1)), redness=float(red[yi, xi]),
                        interior_ratio=interior_ratio(g, x, y, r), seam_distance=seam_dist(x, y)))
    return out


def summarize(rows, key):
    v = np.array([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(n=int(len(v)), median=float(np.median(v)), p25=float(np.percentile(v, 25)),
                p75=float(np.percentile(v, 75)))


def main():
    out = {}
    for tag, src, tgt in [('scratch_from1', 1, 2), ('pretrained_from1', 1, 2),
                          ('scratch_from2', 2, 1), ('pretrained_from2', 2, 1)]:
        thr = bnp.json.loads((SEL / f'{tag}_selection.json').read_text())['chosen']['threshold']
        C = bnp.cases()
        p, s, b = bnp.cached_predict(M / f'{tag}.pt', f'img{tgt}', C[tgt]['im'], FULL)
        gt = C[tgt]['gt']
        sv = (768, 0, 1024, 1024)
        ps, ss, _ = bnp.cached_predict(M / f'{tag}.pt', f'img{src}', C[src]['im'], sv)
        gts = bnp.gt_region(src, sv)
        tcurve, scurve = f1_curve(p, s, gt), f1_curve(ps, ss, gts)
        best = max(tcurve, key=lambda r: r['f1'])
        rep = bnp.metrics(p[s >= thr], gt)
        ceiling = bnp.metrics(p, gt)['recall']
        # score distributions of matched vs unmatched candidates at the floor
        mt = bnp.match_objects(p, gt)
        is_tp = np.zeros(len(p), bool)
        is_tp[mt[:, 0]] = True
        # Part B at the reported operating point
        pr = p[s >= thr]
        mm = bnp.match_objects(pr, gt)
        hit = np.zeros(len(gt), bool)
        hit[mm[:, 1]] = True
        tpp = np.zeros(len(pr), bool)
        tpp[mm[:, 0]] = True
        maps = local_maps(C[tgt]['im'])
        refs = describe(gt, C[tgt]['im'], gt, maps)
        preds = describe(pr, C[tgt]['im'], gt, maps)
        # FP subtypes: near an unmatched/matched reference (localisation/duplicate) vs isolated
        fp_idx = np.flatnonzero(~tpp)
        near = []
        for i in fp_idx:
            d = np.hypot(gt[:, 0] - pr[i, 0], gt[:, 1] - pr[i, 1])
            near.append(bool((d < 2 * np.maximum(3, .75 * gt[:, 2])).any()))
        keys = ['radius', 'density25', 'brightness', 'sharpness', 'redness', 'interior_ratio', 'seam_distance']
        out[tag] = dict(
            source=src, target=tgt, source_threshold=thr,
            reported=dict(rep), oracle_best=best, recall_ceiling_at_0p001=ceiling,
            source_val_best=max(scurve, key=lambda r: r['f1']),
            score_median_true_candidates=float(np.median(s[is_tp])) if is_tp.any() else None,
            score_median_false_candidates=float(np.median(s[~is_tp])),
            n_candidates_at_floor=int(len(p)),
            hits={k: summarize([r for r, h in zip(refs, hit) if h], k) for k in keys},
            misses={k: summarize([r for r, h in zip(refs, hit) if not h], k) for k in keys},
            true_pos={k: summarize([r for r, t in zip(preds, tpp) if t], k) for k in keys},
            false_pos={k: summarize([r for r, t in zip(preds, tpp) if not t], k) for k in keys},
            false_pos_near_reference_fraction=float(np.mean(near)) if near else None,
            target_curve=tcurve, source_curve=scurve)
        # per-reference table for later plots
        np.savez_compressed(bnp.RESULTS / f'diag_{tag}.npz', gt=gt, hit=hit, pred=pr, tp=tpp,
                            ref_feats=np.array([[r[k] for k in keys] for r in refs], float),
                            pred_feats=np.array([[r[k] for k in keys] for r in preds], float).reshape(-1, len(keys)),
                            keys=np.array(keys))
        print(tag, 'reported F1 %.3f | oracle-threshold F1 %.3f at t=%.3f (source t=%.3f) | recall ceiling %.3f'
              % (rep['f1'], best['f1'], best['threshold'], thr, ceiling), flush=True)
    bnp.save_json(bnp.RESULTS / 'diagnosis.json', out)


if __name__ == '__main__':
    main()
