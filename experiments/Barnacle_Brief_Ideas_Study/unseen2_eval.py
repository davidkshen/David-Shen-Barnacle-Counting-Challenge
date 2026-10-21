"""Score every saved model on unseen_img2 against the user's red marks (135 in the current annotation; an earlier one had 138).

Decisions fixed BEFORE looking at any model output on this photo:
  reference      centroids of the red marks (red_marks.json, from the user's annotated copy)
  match          one-to-one, a dot counts if within 8 px of a mark (500-px image; about the half-width of an
                 opening here and under half the 18-px smallest gap between marks). Strict check: 5 px.
  size           the photo has no quadrat frame, so size is a choice. Reported for every model:
                   native       the 500-px photo as is (factor 1.0)
                   as_quadrat   stretched to 1024 px, as the pipeline treats a rectified quadrat (factor 2.048)
                   label_free   the factor in GRID where the model's 50 most confident dots are most confident
                 The F1-vs-size curve over GRID uses the marks and is a DIAGNOSTIC, not a result.
  threshold      each model's own threshold from its DALI selection; the label-free Otsu threshold is a
                 secondary row for the COCO-pretrained models.
  Hough          the settings tuned on DALI photo 1 and on DALI photo 2 in round 6 (classical_cv.json).
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import classical_cv as CV  # noqa: E402
from creative_dali import otsu_thr  # noqa: E402

bnp = C2.bnp
W3 = str(Path(__file__).resolve().parent / 'results') + '/'
GRID = [0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.048]
TT = __import__('os').environ['BARNACLE_THREE_TRACK'] + '/results/'
MODELS = {  # tag: (plain description, threshold)
    'pretrained_from1': 'COCO-pretrained, trained on photo 1',
    'pretrained_from2': 'COCO-pretrained, trained on photo 2',
    'pretrained_adapt_to1': 'COCO-pretrained, photo 2 + a small piece of photo 1',
    'pretrained_adapt_to2': 'COCO-pretrained, photo 1 + a small piece of photo 2',
    'scratch_mixed_styles': 'From scratch, trained on both photos',
    'scratch_from1': 'From scratch, trained on photo 1',
    'scratch_from2': 'From scratch, trained on photo 2',
    'scratch_adapt_to1': 'From scratch, photo 2 + a small piece of photo 1',
    'scratch_adapt_to2': 'From scratch, photo 1 + a small piece of photo 2',
}


def thr_of(tag):
    if tag == 'scratch_mixed_styles':
        return json.load(open(TT + 'mixed_scratch_summary.json'))['threshold']
    return json.load(open(TT + f'{tag}_selection.json'))['chosen']['threshold']


def refs(tol):
    m = json.load(open(W3 + 'red_marks.json'))
    return np.array([[d['x'], d['y'], tol] for d in m], np.float32)


def score(p, tol=8.0):
    m = bnp.metrics(np.asarray(p, np.float32).reshape(-1, 4), refs(tol), tolerance=1.0)
    return dict(count=int(m['predicted_count']), tp=int(m['tp']), fp=int(m['fp']), fn=int(m['fn']),
                precision=m['precision'], recall=m['recall'], f1=m['f1'])


def resized(im, f):
    if abs(f - 1) < 1e-3:
        return im, 500
    n = round(500 * f)
    return cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC), n


def main():
    im = np.asarray(Image.open(__import__('os').environ['BARNACLE_ASSISTANT'] + '/data/unseen_img2.png').convert('RGB'))
    out = dict(reference_marks=len(json.load(open(W3 + 'red_marks.json'))), grid=GRID, models={}, hough={})
    raw = {}
    for tag, desc in MODELS.items():
        thr = thr_of(tag)
        per = {}
        raw[tag] = {}
        for f in GRID:
            x, n = resized(im, f)
            p = bnp.cached_predict(bnp.MODELS / f'{tag}.pt', f'unseen_img2_f{f:.4f}', x, (0, 0, n, n))[0].copy()
            p[:, :3] /= (n / 500)
            raw[tag][str(f)] = p.tolist()
            top = float(np.sort(p[:, 3])[-50:].mean()) if len(p) >= 50 else 0.0
            per[str(f)] = dict(top50_conf=top, own=score(p[p[:, 3] >= thr]), own_strict=score(p[p[:, 3] >= thr], 5.0))
            if tag.startswith('pretrained'):
                t = otsu_thr(p)
                per[str(f)]['otsu'] = dict(threshold=t, **score(p[p[:, 3] >= t]))
        lf = max(GRID, key=lambda f: per[str(f)]['top50_conf'])
        out['models'][tag] = dict(description=desc, threshold=thr, label_free_factor=lf, per_factor=per)
        print(tag, 'thr', round(thr, 3), '| native', per['1.0']['own'], '| quadrat', per['2.048']['own']['f1'],
              '| label-free x', lf, per[str(lf)]['own']['f1'], flush=True)
    cv_cfg = json.load(open(str(Path(__file__).resolve().parent / 'results') + '/classical_cv.json'))
    for src in ('dali_1to2', 'dali_2to1'):
        cfg = cv_cfg[src]
        rr, p1, t = cfg['radius_range'], cfg['hough']['setting']['param1'], cfg['hough']['threshold']
        per = {}
        for f in GRID:
            x, n = resized(im, f)
            c = CV.hough_candidates(x, rr[0], rr[1], p1)
            c[:, :3] /= (n / 500)
            per[str(f)] = dict(own=score(c[c[:, 3] >= t]))
        out['hough']['tuned_on_photo' + src[5]] = dict(radius_range=rr, param1=p1, threshold=t, per_factor=per)
        print('hough', src, {f: round(per[str(f)]['own']['f1'], 3) for f in (1.0, 2.048)}, flush=True)
    json.dump(out, open(W3 + 'unseen2_results.json', 'w'), indent=1)
    json.dump(raw, open(W3 + 'unseen2_raw_preds.json', 'w'))


if __name__ == '__main__':
    main()
