"""Side-by-side overlays: green = dot on a marked barnacle, red = extra dot, yellow ring = marked barnacle missed."""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
W3 = str(Path(__file__).resolve().parent / 'results') + '/'


def panel(im, p, title, S=2):
    from scipy.spatial import cKDTree
    P = np.array([[d['x'], d['y']] for d in json.load(open(W3 + 'red_marks.json'))], np.float32)
    nn = cKDTree(P).query(P, 2)[0][:, 1]
    ref = np.column_stack([P, np.minimum(15, 0.5 * nn)]).astype(np.float32)  # same-barnacle rule
    m = C2.bnp.match_objects(np.asarray(p, np.float32).reshape(-1, 4), ref, tolerance=1.0) if len(p) else np.empty((0, 2), int)
    mp, mr = set(m[:, 0].tolist()), set(m[:, 1].tolist())
    out = cv2.resize(im, (500 * S, 500 * S), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].copy()
    for j, r in enumerate(ref):
        if j not in mr:
            cv2.circle(out, (int(r[0] * S), int(r[1] * S)), 16, (0, 230, 255), 3, cv2.LINE_AA)
    for i, q in enumerate(p):
        c = (int(q[0] * S), int(q[1] * S))
        if i in mp:
            cv2.circle(out, c, 6, (60, 200, 60), -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(out, c, (40, 40, 240), cv2.MARKER_TILTED_CROSS, 14, 3, cv2.LINE_AA)
    bar = np.full((70, out.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(bar, f'{len(mp)} found   {len(p) - len(mp)} extra   {len(ref) - len(mr)} missed   (dots: {len(p)}, marks: {len(ref)})',
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def main(specs, name):
    im = np.asarray(Image.open(__import__('os').environ['BARNACLE_ASSISTANT'] + '/data/unseen_img2.png').convert('RGB'))
    raw = json.load(open(W3 + 'unseen2_raw_preds.json'))
    res = json.load(open(W3 + 'unseen2_results.json'))
    panels = []
    for tag, f, title in specs:
        p = np.array(raw[tag][str(f)], np.float32)
        p = p[p[:, 3] >= res['models'][tag]['threshold']]
        panels.append(panel(im, p, title))
    gap = np.full((panels[0].shape[0], 16, 3), 255, np.uint8)
    row = panels[0]
    for q in panels[1:]:
        row = np.hstack([row, gap, q])
    cv2.imwrite(W3 + name, row)


if __name__ == '__main__':
    main(json.loads(sys.argv[1]), sys.argv[2])
