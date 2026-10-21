"""Apply the reproduced Roboflow probe (probe_rf_hgb.pkl) to unseen_img2 at the pre-declared sizes.
Images are padded (edge reflection) to a multiple of 16 px and at least 256 px so the network input is valid;
peaks in the padding are dropped."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
W3 = str(Path(__file__).resolve().parent / 'results') + '/'
import common2 as C2  # noqa
import probe as PR  # noqa
GRID = [0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.048]
d = pickle.load(open(W3 + 'probe_rf_hgb.pkl', 'rb'))
im = np.asarray(Image.open(__import__('os').environ['BARNACLE_ASSISTANT'] + '/data/unseen_img2.png').convert('RGB'))
out = dict(spacing=d['spacing'], threshold=d['threshold'], preds={})
for f in GRID:
    n = round(500 * f)
    x = im if abs(f - 1) < 1e-3 else cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    N = max(256, int(np.ceil(n / 16) * 16))
    xp = cv2.copyMakeBorder(x, 0, N - n, 0, N - n, cv2.BORDER_REFLECT_101)
    mp = PR.prob_maps(xp, (0, 0, N, N), {'hgb': d['clf']}, (d['mu'], d['sd']))
    pk = PR.peaks(mp['hgb'], (0, 0, N, N), d['spacing'])
    pk = pk[(pk[:, 0] < n) & (pk[:, 1] < n)]
    pk[:, :3] /= (n / 500)
    out['preds'][str(f)] = pk.tolist()
    print(f, int((pk[:, 3] >= d['threshold']).sum()), flush=True)
json.dump(out, open(W3 + 'probe_unseen2.json', 'w'))
