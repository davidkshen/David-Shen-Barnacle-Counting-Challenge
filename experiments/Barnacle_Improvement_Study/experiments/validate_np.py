"""Check the NumPy detector against prediction CSVs saved by the original Ultralytics runs.

For each saved model, predict the same region, apply the recorded threshold, and
compare (a) prediction counts, (b) one-to-one coordinate agreement,
(c) confidence agreement, (d) recomputed metrics vs the recorded evaluation JSON.
"""
import sys
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp

R = bnp.SELECTIONS
M = bnp.MODELS
out = []
jobs = sys.argv[1:] or ['pretrained_from1:2:832', 'scratch_from1:2:832', 'pretrained_from2:1:832',
                        'scratch_from2:1:832', 'pretrained_adapt_to2:2:832', 'scratch_adapt_to2:2:832',
                        'pretrained_adapt_to1:1:832', 'scratch_adapt_to1:1:832',
                        'scratch_mixed_styles:1:832', 'scratch_mixed_styles:2:832',
                        'pretrained_from1:2:0', 'pretrained_from2:1:0', 'scratch_from1:2:0', 'scratch_from2:1:0']
for job in jobs:
    tag, k, x0 = job.split(':')
    k, x0 = int(k), int(x0)
    bounds = (x0, 0, 1024, 1024)
    thr = json.loads((R / f'{tag}_selection.json').read_text())['chosen']['threshold']
    saved = np.loadtxt(R / f'{tag}_on{k}_x{x0}.csv', delimiter=',', skiprows=1, ndmin=2)
    t = time.time()
    p, s, b = bnp.cached_predict(M / f'{tag}.pt', f'img{k}', bnp.cases()[k]['im'], bounds)
    dt = time.time() - t
    mine = p[s >= thr]
    d = np.linalg.norm(mine[:, None, :2] - saved[None, :, :2], axis=2) if len(mine) and len(saved) else np.zeros((0, 0))
    i, j = linear_sum_assignment(d) if d.size else ([], [])
    close = d[i, j] < 0.5 if d.size else np.array([])
    dconf = np.abs(mine[i, 3] - saved[j, 3]) if d.size else np.array([])
    gt = bnp.gt_region(k, bounds)
    met = bnp.metrics(mine, gt)
    ev = json.loads((R / f'{tag}_evaluation.json').read_text())
    rec = [e for e in ev if e['photo'] == k and e['bounds'][0] == x0][0]
    row = dict(tag=tag, photo=k, bounds=bounds, threshold=thr, seconds=round(dt, 1),
               saved_count=len(saved), numpy_count=len(mine), paired_within_0p5px=int(close.sum()),
               max_coord_diff_px=float(d[i, j].max()) if d.size else None,
               max_conf_diff=float(dconf.max()) if len(dconf) else None,
               median_conf_diff=float(np.median(dconf)) if len(dconf) else None,
               recorded_f1=rec['f1'], numpy_f1=met['f1'], recorded_tp=rec['tp'], numpy_tp=met['tp'])
    out.append(row)
    print(json.dumps(row), flush=True)
bnp.save_json(bnp.RESULTS / 'numpy_validation.json', out)
