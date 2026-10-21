"""Main model (mixed_0.5_yolo11n.pt, threshold 0.37 from default.json) on unseen_img2, with exactly the
settings used for the other saved models in unseen2_eval.py (sizes, label-free size pick, Otsu,
8-px rule) plus the same-barnacle rule added later (unseen2_same_barnacle.json)."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unseen2_eval as U  # noqa
bnp = U.bnp
MP = __import__('os').environ.get('BARNACLE_MAIN_MODEL', str(Path(__file__).resolve().parents[2] / 'models/mixed_0.5_yolo11n.pt'))
THR = json.load(open(__import__('os').environ.get('BARNACLE_MAIN_CONFIG', str(Path(__file__).resolve().parents[2] / 'models/default.json'))))['threshold']
im = np.asarray(Image.open(__import__('os').environ['BARNACLE_ASSISTANT'] + '/data/unseen_img2.png').convert('RGB'))
M = json.load(open(str(Path(__file__).resolve().parent / 'results') + '/red_marks.json')); P = np.array([[d['x'], d['y']] for d in M], np.float32)
nn = cKDTree(P).query(P, 2)[0][:, 1]
same = np.column_stack([P, np.minimum(15, 0.5 * nn)]).astype(np.float32)
per, raw = {}, {}
for f in U.GRID:
    x, n = U.resized(im, f)
    p = bnp.cached_predict(MP, f'unseen_img2_f{f:.4f}', x, (0, 0, n, n))[0].copy()
    p[:, :3] /= (n / 500)
    raw[str(f)] = p.tolist()
    q = p[p[:, 3] >= THR]
    s = bnp.metrics(q, same, tolerance=1.0)
    t = U.otsu_thr(p)
    per[str(f)] = dict(top50_conf=float(np.sort(p[:, 3])[-50:].mean()) if len(p) >= 50 else 0.0,
                       own=U.score(q), own_strict=U.score(q, 5.0),
                       same_barnacle=dict(count=int(s['predicted_count']), tp=int(s['tp']), fp=int(s['fp']), fn=int(s['fn']), f1=s['f1']),
                       otsu=dict(threshold=t, **U.score(p[p[:, 3] >= t])))
    print(f, 'count', len(q), '| 8px F1', round(per[str(f)]['own']['f1'], 3), '| same-barnacle F1', round(s['f1'], 3), s['tp'], s['fp'], s['fn'], flush=True)
lf = max(U.GRID, key=lambda f: per[str(f)]['top50_conf'])
out = dict(model=MP, threshold=THR, label_free_factor=lf, per_factor=per)
json.dump(out, open(str(Path(__file__).resolve().parent / 'results') + '/main_unseen2.json', 'w'), indent=1)
r = json.load(open(str(Path(__file__).resolve().parent / 'results') + '/unseen2_raw_preds.json')); r['main_model'] = raw
json.dump(r, open(str(Path(__file__).resolve().parent / 'results') + '/unseen2_raw_preds.json', 'w'))
res = json.load(open(str(Path(__file__).resolve().parent / 'results') + '/unseen2_results.json'))
res['models']['main_model'] = dict(description='COCO-pretrained, trained on both photos (main model)', threshold=THR, label_free_factor=lf, per_factor=per)
json.dump(res, open(str(Path(__file__).resolve().parent / 'results') + '/unseen2_results.json', 'w'), indent=1)
print('label-free factor', lf)
