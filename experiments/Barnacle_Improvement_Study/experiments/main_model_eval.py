"""Freeze the MAIN submission model and evaluate it exactly as reported.

Needs the files that were NOT in the uploaded archives:
  models/mixed_0.5_yolo11n.pt  (repo root)
  models/default.json
Pass them with --model / --config. Runs on CPU without PyTorch.

What it does (nothing is re-selected; threshold comes from default.json):
  1. Reserved stripes x>=832 of photo 1 and photo 2: per-photo and pooled
     P/R/F1 (0.75 r rule), strict F1 (0.5 r), predicted vs reference counts,
     signed and absolute count error. Expected (reported): pooled F1 0.941,
     photo 1 303/15/18, photo 2 80/10/5.
  2. Unseen photo 1 (no labels): prediction count (reported 1,143) and runtime.
  3. Optional --new IMAGE:CORNERS_JSON:POINTS_CSV for independently annotated
     new quadrats (rectified x,y[,radius] of every target; see README). Scored
     with a declared fixed tolerance when no radius is given.
Writes results/main_model_eval.json. If the model is missing, it says so and stops.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp

STRIPE = (832, 0, 1024, 1024)


def score(pred, gt, tol_px=None):
    if tol_px is not None:
        gt = np.column_stack([gt[:, :2], np.full(len(gt), tol_px / 0.75)])
    m = bnp.metrics(pred, gt)
    m['strict_f1'] = bnp.metrics(pred, gt, tolerance=.5)['f1'] if tol_px is None else None
    m['abs_error'] = abs(m['signed_error'])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(bnp.ROOT.parents[1] / 'models/mixed_0.5_yolo11n.pt'))
    ap.add_argument('--config', default=str(bnp.ROOT.parents[1] / 'models/default.json'))
    ap.add_argument('--new', action='append', default=[],
                    help='IMAGE:CORNERS_JSON:POINTS_CSV (complete rectified point labels)')
    ap.add_argument('--tolerance-px', type=float, default=5.0,
                    help='match distance for new point labels without radius (declared, not tuned)')
    a = ap.parse_args()
    mp, cp = Path(a.model), Path(a.config)
    if not mp.exists() or not cp.exists():
        raise SystemExit(f'Main model or config not found:\n  {mp}\n  {cp}\nThese were not in the uploaded '
                         'archives. Pass --model and --config with their real locations.')
    cfg = json.loads(cp.read_text())
    thr = cfg.get('threshold', cfg.get('conf', cfg.get('confidence')))
    if thr is None:
        raise SystemExit(f'No threshold key found in {cp}: {list(cfg)}')
    thr = float(thr)
    model = bnp.model(mp)
    out = dict(model=str(mp), config=cfg, threshold=thr, checkpoint_meta=dict(version=model.meta['version'],
               epoch=model.meta['epoch']))
    C = bnp.cases()
    rows, tot = [], dict(tp=0, fp=0, fn=0, pred=0, ref=0)
    for k in (1, 2):
        t0 = time.time()
        p, s, _ = bnp.tile_predict_np(model, C[k]['im'], STRIPE)
        dt = time.time() - t0
        pred = p[s >= thr]
        gt = bnp.gt_region(k, STRIPE)
        m = score(pred, gt)
        rows.append(dict(photo=k, region='x>=832', seconds=round(dt, 1), **m))
        for key, src in (('tp', 'tp'), ('fp', 'fp'), ('fn', 'fn'), ('pred', 'predicted_count'), ('ref', 'reference_count')):
            tot[key] += m[src]
        print(f"photo {k}: ref {m['reference_count']} pred {m['predicted_count']} TP {m['tp']} FP {m['fp']} FN {m['fn']} "
              f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} strict {m['strict_f1']:.3f} count err {m['signed_error']:+d}")
    pooled = dict(tp=tot['tp'], fp=tot['fp'], fn=tot['fn'], reference_count=tot['ref'], predicted_count=tot['pred'],
                  precision=tot['tp'] / max(1, tot['pred']), recall=tot['tp'] / max(1, tot['ref']),
                  f1=2 * tot['tp'] / max(1, 2 * tot['tp'] + tot['fp'] + tot['fn']), signed_error=tot['pred'] - tot['ref'])
    print(f"pooled: F1 {pooled['f1']:.3f} (reported 0.941), count {tot['pred']} vs {tot['ref']}")
    out['reserved_stripes'] = rows
    out['pooled'] = pooled
    out['matches_reported'] = dict(
        pooled_f1_0p941=round(pooled['f1'], 3) == 0.941,
        photo1_303_15_18=[rows[0]['tp'], rows[0]['fp'], rows[0]['fn']] == [303, 15, 18],
        photo2_80_10_5=[rows[1]['tp'], rows[1]['fp'], rows[1]['fn']] == [80, 10, 5])
    try:
        im = bnp.rectified('unseen_img1.png')
        t0 = time.time()
        p, s, _ = bnp.tile_predict_np(model, im)
        dt = time.time() - t0
        n = int((s >= thr).sum())
        out['unseen_img1'] = dict(predictions=n, reported=1143, seconds_full_quadrat=round(dt, 1),
                                  status='unverified prediction count; no reference labels')
        print(f'unseen_img1: {n} detections (reported 1,143), {dt:.1f} s on this CPU')
    except FileNotFoundError:
        out['unseen_img1'] = 'unseen_img1.png not found'
    news = []
    for spec in a.new:
        img, corners, pts = spec.split(':')
        rgb, _ = bnp.rectify(bnp.read_rgb(img), json.loads(Path(corners).read_text()))
        lab = np.loadtxt(pts, delimiter=',', skiprows=1, ndmin=2)
        t0 = time.time()
        p, s, _ = bnp.tile_predict_np(model, rgb)
        dt = time.time() - t0
        pred = p[s >= thr]
        if lab.shape[1] >= 3:
            m = score(pred, lab[:, :3])
            rule = '0.75 x labelled radius (min 3 px)'
        else:
            m = score(pred, lab[:, :2], tol_px=a.tolerance_px)
            rule = f'fixed {a.tolerance_px} px (no radii supplied)'
        news.append(dict(image=img, matching_rule=rule, seconds=round(dt, 1), **m))
        print(f"NEW {img}: ref {m['reference_count']} pred {m['predicted_count']} F1 {m['f1']:.3f} count err {m['signed_error']:+d} [{rule}]")
    out['new_images'] = news
    bnp.save_json(bnp.RESULTS / 'main_model_eval.json', out)


if __name__ == '__main__':
    main()
