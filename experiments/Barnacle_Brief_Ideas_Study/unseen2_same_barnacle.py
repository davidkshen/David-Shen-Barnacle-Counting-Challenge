"""Rescore unseen_img2 with the 'same barnacle' rule, ADDED after seeing the overlays (the 8-px rule set
beforehand marked many dots that sit on the right barnacle, off-centre from the drawn line, as wrong).
Rule: a dot matches a mark if it is within min(15 px, half the distance from that mark to its nearest
neighbouring mark); one-to-one matching as everywhere else. Reads unseen2_raw_preds.json (YOLO models incl.
the main model), probe_unseen2.json (Roboflow-trained probe) and recomputes Hough at native size."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import classical_cv as CV  # noqa: E402
W3 = str(Path(__file__).resolve().parent / 'results') + '/'


def main():
    M = json.load(open(W3 + 'red_marks.json'))
    P = np.array([[d['x'], d['y']] for d in M], np.float32)
    nn = cKDTree(P).query(P, 2)[0][:, 1]
    ref = np.column_stack([P, np.minimum(15, 0.5 * nn)]).astype(np.float32)

    def sc(p):
        m = C2.bnp.metrics(np.asarray(p, np.float32).reshape(-1, 4), ref, tolerance=1.0)
        return dict(f1=m['f1'], tp=m['tp'], fp=m['fp'], fn=m['fn'], count=m['predicted_count'])
    raw = json.load(open(W3 + 'unseen2_raw_preds.json'))
    res = json.load(open(W3 + 'unseen2_results.json'))
    out = dict(rule='per-mark tolerance min(15 px, half the distance to the nearest other mark); ADDED after seeing overlays', per_model={})
    for tag, m in res['models'].items():
        fs = sorted({'1.0', str(m['label_free_factor']), '2.048'}, key=float)
        out['per_model'][tag] = {f: sc([q for q in raw[tag][f] if q[3] >= m['threshold']]) for f in fs}
    pr = json.load(open(W3 + 'probe_unseen2.json'))
    r8 = np.column_stack([P, np.full(len(P), 8.0)]).astype(np.float32)
    out['per_model']['probe_roboflow'] = {}
    for f, p in pr['preds'].items():
        q = np.array([x for x in p if x[3] >= pr['threshold']], np.float32).reshape(-1, 4)
        out['per_model']['probe_roboflow'][f] = dict(sc(q), f1_8px=C2.bnp.metrics(q, r8, tolerance=1.0)['f1'])
    im = np.asarray(Image.open(__import__('os').environ['BARNACLE_ASSISTANT'] + '/data/unseen_img2.png').convert('RGB'))
    cfg = json.load(open(str(Path(__file__).resolve().parent / 'results') + '/classical_cv.json'))
    for src in ('dali_1to2', 'dali_2to1'):
        c = cfg[src]
        h = CV.hough_candidates(im, c['radius_range'][0], c['radius_range'][1], c['hough']['setting']['param1'])
        out['per_model']['hough_tuned_on_photo' + src[5]] = {'1.0': sc(h[h[:, 3] >= c['hough']['threshold']])}
    q = np.array([x for x in raw['main_model']['1.0'] if x[3] >= res['models']['main_model']['threshold']], np.float32)
    mt = C2.bnp.match_objects(q, ref, 1.0)
    e = q[np.setdiff1d(np.arange(len(q)), mt[:, 0])]
    out['main_model_crop_extras_within_12px_of_edge'] = int(((e[:, 0] < 12) | (e[:, 1] < 12) | (e[:, 0] > 488) | (e[:, 1] > 488)).sum())
    out['reference_marks'] = int(len(P))
    json.dump(out, open(W3 + 'unseen2_same_barnacle.json', 'w'), indent=1)
    for k, v in out['per_model'].items():
        print(k.ljust(24), {f: (round(x['f1'], 3), x['count']) for f, x in v.items()})


if __name__ == '__main__':
    main()
