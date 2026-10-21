"""Re-create the Roboflow-trained probe (round 6, probe.py rf_experiment: same seed, same 100 photos,
same caps; HistGradientBoosting, spacing 3, threshold chosen then on validation = 0.7773) and apply it to
unseen_img2 at the pre-declared size factors. The probe was not saved in round 6, so it is retrained here.
Reproduction check: validation F1 at the recorded setting must match results/probe.json['rf']['val']['f1'].
"""
import json
import pickle
import sys
from pathlib import Path
import time
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import probe as PR  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

W3 = str(Path(__file__).resolve().parent / 'results') + '/'
GRID = [0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.048]


def main():
    rec = json.loads(open(str(Path(__file__).resolve().parent / 'results') + '/probe.json').read())['rf']
    sp, thr = rec['chosen']['spacing'], rec['chosen']['threshold']
    t0 = time.time()
    rng = np.random.default_rng(1)
    dev = C2.rf_rows('train')
    dev = [dev[i] for i in rng.choice(len(dev), 100, replace=False)]
    tr = ((C2.rf_img(r), (0, 0, r['width'], r['height']), C2.rf_targets(r), C2.rf_others(r)) for r in dev)
    X, y = PR.collect(tr, rng, neg_per_pos=1, max_neg_tile=100, max_pos_tile=100)
    print('cells', len(y), 'recorded', rec['train_cells'], f'{time.time() - t0:.0f}s', flush=True)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X -= mu
    X /= sd
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0).fit(X, y)
    del X
    pickle.dump(dict(clf=clf, mu=mu, sd=sd, spacing=sp, threshold=thr), open(W3 + 'probe_rf_hgb.pkl', 'wb'))
    print('fitted', f'{time.time() - t0:.0f}s', flush=True)
    val = C2.rf_rows('val')
    vb = [(0, 0, r['width'], r['height']) for r in val]
    vmaps = [PR.prob_maps(C2.rf_img(r), b, {'hgb': clf}, (mu, sd)) for r, b in zip(val, vb)]
    _, vm = PR.apply(vmaps, vb, [C2.rf_targets(r) for r in val], 'hgb', sp, thr)
    print('val F1', vm['f1'], 'recorded', rec['val']['f1'], flush=True)
    im = np.asarray(Image.open(__import__('os').environ['BARNACLE_ASSISTANT'] + '/data/unseen_img2.png').convert('RGB'))
    out = dict(reproduction=dict(train_cells=int(len(y)), recorded_train_cells=rec['train_cells'],
                                 val_f1=vm['f1'], recorded_val_f1=rec['val']['f1']),
               spacing=sp, threshold=thr, preds={})
    for f in GRID:
        w = h = round(500 * f)
        x = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC) if abs(f - 1) > 1e-3 else im
        mp = PR.prob_maps(x, (0, 0, w, h), {'hgb': clf}, (mu, sd))
        pk = PR.peaks(mp['hgb'], (0, 0, w, h), sp)
        pk[:, :3] /= f
        out['preds'][str(f)] = pk.tolist()
        print('factor', f, 'peaks >= thr', int((pk[:, 3] >= thr).sum()), flush=True)
    json.dump(out, open(W3 + 'probe_unseen2.json', 'w'))
    print('done', f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
