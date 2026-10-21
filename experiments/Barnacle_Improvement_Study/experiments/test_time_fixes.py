"""Task 2: label-free test-time fixes for automatic transfer to a new photo.

Pre-declared variants (fixed before seeing their target scores):
  base    : saved model, source-validation threshold (reported setting)
  colour  : Reinhard LAB mean/std transfer of the target quadrat to the SOURCE
            photo's statistics (no target labels)
  scale   : resize target by f = median predicted radius on source validation /
            median predicted radius on target (both at the source threshold,
            both model outputs, no labels), f clipped to [0.6, 1.6]
  flips   : 4-view flip test-time augmentation with score averaging; the threshold
            is re-selected on the SOURCE validation region with the same pipeline
  colour+scale, colour+scale+flips
Diagnostic only (uses target labels, never a deployable result):
  oracle_scale : f from the true reference radius ratio (source train / target)
Evaluation: full target photo AND reserved stripe x>=832, center F1 (0.75 r rule).
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
SV = (768, 0, 1024, 1024)


def lab_stats(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)


def colour_transfer(rgb, src_stats):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0)
    out = (lab - mu) / np.maximum(sd, 1e-3) * src_stats[1] + src_stats[0]
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def predict_scaled(mp, key, rgb, f, bounds=FULL):
    """Resize the whole rectified image by f, run the same tiling, map back."""
    if abs(f - 1) < 1e-6:
        return bnp.cached_predict(mp, key, rgb, bounds)
    n = int(round(1024 * f))
    big = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_LINEAR if f > 1 else cv2.INTER_AREA)
    bb = tuple(int(round(v * f)) for v in bounds)
    bb = (bb[0], bb[1], min(bb[2], n), min(bb[3], n))
    p, s, b = bnp.cached_predict(mp, f'{key}_f{f:.4f}', big, bb)
    p = p.copy()
    p[:, :3] /= f
    return p, s, b / f


def flip_views(mp, key, rgb, f, bounds=FULL):
    views = []
    for fx, fy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        im = rgb[:, ::-1] if fx else rgb
        im = im[::-1] if fy else im
        x0, y0, x1, y1 = bounds
        fb = (1024 - x1 if fx else x0, 1024 - y1 if fy else y0, 1024 - x0 if fx else x1, 1024 - y0 if fy else y1)
        p, s, _ = predict_scaled(mp, f'{key}_flip{fx}{fy}', np.ascontiguousarray(im), f, fb)
        p = p.copy()
        if fx:
            p[:, 0] = 1024 - p[:, 0]
        if fy:
            p[:, 1] = 1024 - p[:, 1]
        views.append(p)
    return fuse(views)


def fuse(views):
    """Greedy clustering of centers across views; score = mean over views (missing view = 0)."""
    allp = np.concatenate([np.column_stack([v, np.full(len(v), i)]) for i, v in enumerate(views)])
    order = np.argsort(-allp[:, 3])
    allp = allp[order]
    used = np.zeros(len(allp), bool)
    out = []
    from scipy.spatial import cKDTree
    tree = cKDTree(allp[:, :2])
    for i in range(len(allp)):
        if used[i]:
            continue
        rad = max(3.0, 0.5 * allp[i, 2])
        nb = [j for j in tree.query_ball_point(allp[i, :2], rad) if not used[j]]
        best = {}
        for j in nb:
            v = int(allp[j, 4])
            if v not in best or allp[j, 3] > allp[best[v], 3]:
                best[v] = j
        idx = list(best.values())
        used[nb] = True
        w = allp[idx, 3]
        xy = (allp[idx, :2] * w[:, None]).sum(0) / w.sum()
        r = float((allp[idx, 2] * w).sum() / w.sum())
        out.append([xy[0], xy[1], r, w.sum() / len(views)])
    p = np.asarray(out, np.float32)
    return p, p[:, 3]


def evaluate(p, s, thr, k):
    full = bnp.metrics(p[s >= thr], bnp.cases()[k]['gt'])
    stripe_mask = bnp.in_box(p, (832, 0, 1024, 1024))
    stripe = bnp.metrics(p[(s >= thr) & stripe_mask], bnp.gt_region(k, (832, 0, 1024, 1024)))
    keep = ('precision', 'recall', 'f1', 'predicted_count', 'reference_count', 'signed_error')
    return dict(full={q: full[q] for q in keep}, stripe={q: stripe[q] for q in keep})


def main():
    C = bnp.cases()
    out = {}
    for tag, src, tgt in [('pretrained_from1', 1, 2), ('pretrained_from2', 2, 1),
                          ('scratch_from1', 1, 2), ('scratch_from2', 2, 1)]:
        mp = M / f'{tag}.pt'
        thr = bnp.json.loads((SEL / f'{tag}_selection.json').read_text())['chosen']['threshold']
        src_im, tgt_im = C[src]['im'], C[tgt]['im']
        stats = lab_stats(src_im)
        tgt_col = colour_transfer(tgt_im, stats)
        # label-free scale factor from model outputs only
        ps, ss, _ = bnp.cached_predict(mp, f'img{src}', src_im, SV)
        pt, st, _ = bnp.cached_predict(mp, f'img{tgt}', tgt_im, FULL)
        ptc, stc, _ = bnp.cached_predict(mp, f'img{tgt}_col', tgt_col, FULL)
        r_src = np.median(ps[ss >= thr, 2])
        f_plain = float(np.clip(r_src / np.median(pt[st >= thr, 2]), .6, 1.6)) if (st >= thr).any() else 1.0
        f_col = float(np.clip(r_src / np.median(ptc[stc >= thr, 2]), .6, 1.6)) if (stc >= thr).any() else 1.0
        train_ref = bnp.gt_region(src, (0, 0, 640, 1024))
        f_oracle = float(np.median(train_ref[:, 2]) / np.median(C[tgt]['gt'][:, 2]))
        # flip TTA threshold re-selected on SOURCE validation only
        pv, sv_ = flip_views(mp, f'img{src}', src_im, 1.0, SV)
        tta_thr = bnp.choose_confidence(pv, sv_, bnp.gt_region(src, SV))[0]['threshold']
        res = {}
        p, s = pt, st
        res['base'] = evaluate(p, s, thr, tgt)
        res['colour'] = evaluate(ptc, stc, thr, tgt)
        p, s, _ = predict_scaled(mp, f'img{tgt}', tgt_im, f_plain)
        res['scale'] = evaluate(p, s, thr, tgt)
        p, s = flip_views(mp, f'img{tgt}', tgt_im, 1.0)
        res['flips'] = evaluate(p, s, tta_thr, tgt)
        p, s, _ = predict_scaled(mp, f'img{tgt}_col', tgt_col, f_col)
        res['colour+scale'] = evaluate(p, s, thr, tgt)
        p, s = flip_views(mp, f'img{tgt}_col', tgt_col, f_col)
        res['colour+scale+flips'] = evaluate(p, s, tta_thr, tgt)
        p, s, _ = predict_scaled(mp, f'img{tgt}', tgt_im, f_oracle)
        res['DIAGNOSTIC_oracle_scale'] = evaluate(p, s, thr, tgt)
        out[tag] = dict(source=src, target=tgt, source_threshold=thr, flips_threshold=tta_thr,
                        scale_factor_label_free=f_plain, scale_factor_after_colour=f_col,
                        scale_factor_oracle=f_oracle, variants=res)
        line = ' | '.join(f"{k}: {v['full']['f1']:.3f}/{v['stripe']['f1']:.3f}" for k, v in res.items())
        print(f'{tag} f={f_plain:.2f} fcol={f_col:.2f} f*={f_oracle:.2f} tta_t={tta_thr:.3f} :: {line}', flush=True)
    bnp.save_json(bnp.RESULTS / 'test_time_fixes.json', out)


if __name__ == '__main__':
    main()
