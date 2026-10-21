"""Unlabelled photos: how much do counts from different saved models disagree?

unseen_img1 has no reference labels, so no accuracy is computed. The spread of
counts across models (each at its own pre-selected threshold) is a warning signal
about count uncertainty on a new photo, not an error estimate. unseen_img2 has
no frame; the pipeline abstains rather than inventing a quadrat.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp

MODELS = ['pretrained_from1', 'pretrained_from2', 'scratch_from1', 'scratch_from2', 'scratch_mixed_styles']


def main():
    frames = json.loads((bnp.DATA / 'frames.json').read_text())
    out = dict(unseen_img2='abstained: no green quadrat frame corners available; no count produced')
    im = bnp.rectified('unseen_img1.png')
    rows = []
    for tag in MODELS:
        thr = json.loads((bnp.SELECTIONS / f'{tag}_selection.json').read_text())['chosen']['threshold']
        t0 = time.time()
        p, s, _ = bnp.cached_predict(bnp.MODELS / f'{tag}.pt', 'unseen_img1', im)
        rows.append(dict(model=tag, threshold=thr, count=int((s >= thr).sum()), seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    c = np.array([r['count'] for r in rows if r['model'].startswith('pretrained') or r['model'] == 'scratch_mixed_styles'])
    out['unseen_img1'] = dict(corners=frames.get('unseen_img1.png'), per_model=rows,
                              reported_main_model_count=1143,
                              spread_pretrained_and_mixed=[int(c.min()), int(c.max())],
                              status='unverified; disagreement is not an accuracy estimate')
    bnp.save_json(bnp.RESULTS / 'unseen_counts.json', out)


if __name__ == '__main__':
    main()
