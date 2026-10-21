"""Summarise exported timed-review JSON files (from make_review_kit.py pages).

pilot export  -> per condition: seconds per tile, seconds per final object,
                 deletions/additions per tile, and (if the photo has supplied
                 reference labels) precision/recall/F1 of the REVIEWED points
                 against those labels, so speed and quality are read together.
audit export  -> corrected whole-quadrat count with approximate 95% t interval
                 (barnacles.audit.corrected_count, model coefficient fixed at 1),
                 the manual-sample-only estimate (coefficient 0), and the frozen
                 model's precision/recall/F1 inside the audited tiles, using the
                 human final points as reference with a declared fixed tolerance.
Measured times are what the browser recorded while the tab was visible; they
are one reviewer's times on one occasion unless several exports are passed.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp  # noqa: E402
from barnacles.audit import corrected_count  # noqa: E402

LABELLED = {'img1.png': 1, 'img2.png': 2}


def tol_ref(xy, tol):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.column_stack([xy, np.full(len(xy), tol / 0.75)])


def pilot(d, tol):
    k = LABELLED.get(d['image'])
    out = {}
    for c in sorted({t['condition'] for t in d['tiles']}):
        tt = [t for t in d['tiles'] if t['condition'] == c and t['done']]
        if not tt:
            continue
        objs = sum(len(t['final_points']) for t in tt)
        row = dict(tiles=len(tt), threshold=tt[0]['threshold'],
                   seconds_per_tile=float(np.mean([t['seconds'] for t in tt])),
                   seconds_per_object=float(sum(t['seconds'] for t in tt) / max(1, objs)),
                   deletes_per_tile=float(np.mean([t['deletes'] for t in tt])),
                   adds_per_tile=float(np.mean([t['adds'] for t in tt])),
                   model_markers_per_tile=float(np.mean([t['model_markers_shown'] for t in tt])),
                   uncertain_marks=int(sum(p['uncertain'] for t in tt for p in t['final_points'])))
        if k:
            tp = fp = fn = 0
            for t in tt:
                b = (t['tile_id'] % 8 * 128, t['tile_id'] // 8 * 128, t['tile_id'] % 8 * 128 + 128, t['tile_id'] // 8 * 128 + 128)
                pts = np.array([[p['x'], p['y'], 0, 1] for p in t['final_points']], float).reshape(-1, 4)
                m = bnp.metrics(pts, bnp.gt_region(k, b))
                tp, fp, fn = tp + m['tp'], fp + m['fp'], fn + m['fn']
            row['reviewed_vs_supplied_labels'] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp),
                                                      recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn),
                                                      rule='supplied radius 0.75 r (benchmark rule)')
        out[c] = row
    return dict(kind='pilot', image=d['image'], conditions=d['random_sampling'].get('conditions'), by_condition=out)


def audit(d, tol, model_override=None, thr_override=None):
    tiles = [t for t in d['tiles'] if t['done']]
    ids = [t['tile_id'] for t in tiles]
    if len(tiles) < len(d['tiles']):
        raise SystemExit(f'{len(d["tiles"]) - len(tiles)} audit tiles are not marked done; finish them first.')
    human = [len(t['final_points']) for t in tiles]
    blind = bool(d['random_sampling'].get('blind'))
    if model_override and not blind:
        raise SystemExit('A non-blind audit showed one model\'s markers; scoring a different model on it is biased.')
    thr = float(thr_override) if thr_override is not None else d['random_sampling']['audit_threshold']
    mfile = model_override or d['model']['file']
    mpath = Path(mfile) if Path(mfile).exists() else bnp.MODELS / mfile
    img = bnp.find_image(d['image']) if not Path(d['image']).exists() else Path(d['image'])
    rgb, _ = bnp.rectify(bnp.read_rgb(img), d['corners'])
    p, s, _ = bnp.tile_predict_np(bnp.model(mpath), rgb)
    pred = p[s >= thr]
    from barnacles.audit import tile_counts
    mc = tile_counts(np.clip(pred[:, :2], 0, 1023.999)).tolist()
    if not model_override and thr_override is None and mc != d['model_tile_counts']:
        raise SystemExit('Recomputed model tile counts differ from the frozen counts in the export.')
    est = corrected_count(mc, ids, human, beta=1.0)
    man = corrected_count(mc, ids, human, beta=0.0)
    tp = fp = fn = 0
    for t in tiles:
        b = (t['tile_id'] % 8 * 128, t['tile_id'] // 8 * 128, t['tile_id'] % 8 * 128 + 128, t['tile_id'] // 8 * 128 + 128)
        ref = tol_ref([[q['x'], q['y']] for q in t['final_points']], tol)
        m = bnp.metrics(pred[bnp.in_box(pred, b)], ref)
        tp, fp, fn = tp + m['tp'], fp + m['fp'], fn + m['fn']
    return dict(kind='audit', image=d['image'], blind=blind, model=str(mpath.name), threshold=thr,
                model_count=int(sum(mc)), corrected=est, manual_sample_only=man,
                seconds_total=float(sum(t['seconds'] for t in tiles)),
                frozen_model_in_audited_tiles=dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp),
                                                   recall=tp / max(1, tp + fn), f1=2 * tp / max(1, 2 * tp + fp + fn),
                                                   rule=f'fixed {tol} px to human points'),
                caution=('Not blind: reviewer saw model markers, which can anchor counts.' if not d['random_sampling'].get('blind') else
                         'Blind counts.'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exports', nargs='+')
    ap.add_argument('--tolerance-px', type=float, default=5.0)
    ap.add_argument('--model', help='blind audits only: score this frozen model instead of the kit model')
    ap.add_argument('--threshold', type=float, help='frozen threshold to use with --model')
    ap.add_argument('--out', default=str(bnp.RESULTS / 'review_measurements.json'))
    a = ap.parse_args()
    res = []
    for f in a.exports:
        d = json.loads(Path(f).read_text())
        r = pilot(d, a.tolerance_px) if d['mode'] == 'pilot' else audit(d, a.tolerance_px, a.model, a.threshold)
        r['file'] = f
        res.append(r)
        print(json.dumps(r, indent=2))
    bnp.save_json(a.out, res)


if __name__ == '__main__':
    main()
