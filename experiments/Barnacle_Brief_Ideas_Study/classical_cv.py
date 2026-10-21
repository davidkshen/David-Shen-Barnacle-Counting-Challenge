"""Brief idea: traditional computer vision (kernels, circles, ellipses), measured like the detector.

Four label-free candidate generators, each producing (x, y, radius, score):
  log     scale-normalised Laplacian-of-Gaussian blobs (dark, bright or either polarity)
  ring    the original 'bright rim, dark centre' hypothesis as a matched kernel:
          mean(annulus 0.8r-1.3r) - mean(disk 0-0.5r), divided by local std (or the reverse polarity)
  hough   cv2.HoughCircles (edge-based circle voting); the accumulator threshold is the score knob
  ellipse adaptive threshold -> contours -> cv2.fitEllipse, kept if size/axis-ratio/solidity plausible
Only the SOURCE labels choose the method's settings and score threshold (max center F1); the
TARGET is scored once. Protocols:
  DALI    tune on photo k's left region x<640, score the full other photo (both directions)
  ROBOFLOW tune on 20 dev-train photos (fixed random sample), score the 97 grouped test photos
Radius search range comes from the SOURCE labels only (radius_range).
For speed, the Roboflow run only (PYRAMID, CAP, NTHR) computes large-radius LoG responses on a
downsampled image, keeps the 4,000 strongest LoG/ring candidates per photo and sweeps 60 thresholds.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402

OUT = Path(__file__).resolve().parent / 'results'


def gray(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mu = cv2.GaussianBlur(g, (0, 0), 25)
    sd = np.sqrt(cv2.GaussianBlur((g - mu) ** 2, (0, 0), 25)) + 1e-3
    return (g - mu) / sd


def nms(c, k):
    """Greedy: keep highest score, drop others whose center is within k * max(r_i, r_j)."""
    if not len(c):
        return c
    c = c[np.argsort(-c[:, 3])]
    t = cKDTree(c[:, :2])
    alive = np.ones(len(c), bool)
    rmax = c[:, 2].max()
    for i in range(len(c)):
        if not alive[i]:
            continue
        for j in t.query_ball_point(c[i, :2], k * max(c[i, 2], rmax)):
            if j > i and alive[j] and np.hypot(*(c[j, :2] - c[i, :2])) < k * max(c[i, 2], c[j, 2]):
                alive[j] = False
    return c[alive]


_STACKS = {}


def _log_stack(g, radii):
    key = ('log', id(g), tuple(np.round(radii, 3)))
    if key not in _STACKS:
        out = []
        for s in radii / np.sqrt(2):
            k = max(1, int(s // 3)) if PYRAMID else 1   # pyramid for large sigma (Roboflow run only, for speed)
            if k > 1:
                gs = cv2.resize(g, (g.shape[1] // k, g.shape[0] // k), interpolation=cv2.INTER_AREA)
                r = -((s / k) ** 2) * ndi.gaussian_laplace(gs, s / k)
                r = cv2.resize(r, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                r = -(s ** 2) * ndi.gaussian_laplace(g, s)
            out.append(r)
        _STACKS.clear()
        _STACKS[key] = np.stack(out)
    return _STACKS[key]


def log_candidates(g, radii, polarity):
    stack = _log_stack(g, radii).copy()   # bright blob -> positive
    if polarity == 'dark':
        stack = -stack
    elif polarity == 'abs':
        stack = np.abs(stack)
    mx = ndi.maximum_filter(stack, size=(3, 5, 5))
    pk = (stack == mx) & (stack > 0.05)
    s, y, x = np.nonzero(pk)
    return np.column_stack([x, y, radii[s], stack[s, y, x]]).astype(np.float32)


def ring_candidates(g, radii, polarity):
    key = ('ring', id(g), tuple(np.round(radii, 3)))
    if key in _STACKS:
        stack = _STACKS[key].copy()
        if polarity == 'bright_centre':
            stack = -stack
        mx = ndi.maximum_filter(stack, size=(3, 5, 5))
        pk = (stack == mx) & (stack > 0.05)
        s, y, x = np.nonzero(pk)
        return np.column_stack([x, y, radii[s], stack[s, y, x]]).astype(np.float32)
    resp = []
    for r in radii:
        n = int(np.ceil(1.4 * r))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        d = np.hypot(xx, yy)
        ann = ((d >= 0.8 * r) & (d <= 1.3 * r)).astype(np.float32)
        disk = (d <= max(0.5 * r, 1.0)).astype(np.float32)
        k = ann / ann.sum() - disk / disk.sum()
        resp.append(cv2.filter2D(g, -1, k))
    stack = np.stack(resp)
    _STACKS.clear()
    _STACKS[key] = stack.copy()
    if polarity == 'bright_centre':
        stack = -stack
    mx = ndi.maximum_filter(stack, size=(3, 5, 5))
    pk = (stack == mx) & (stack > 0.05)
    s, y, x = np.nonzero(pk)
    return np.column_stack([x, y, radii[s], stack[s, y, x]]).astype(np.float32)


def hough_candidates(rgb, rmin, rmax, param1):
    g8 = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 3)
    out = []
    for p2 in (8, 10, 13, 16, 20, 25, 32, 40):
        c = cv2.HoughCircles(g8, cv2.HOUGH_GRADIENT, dp=1, minDist=max(2, rmin), param1=param1, param2=p2,
                             minRadius=int(rmin), maxRadius=int(rmax))
        if c is not None:
            c = c[0]
            out.append(np.column_stack([c[:, :3], np.full(len(c), p2)]))
    if not out:
        return np.zeros((0, 4), np.float32)
    allc = np.concatenate(out)
    # a circle found at a stricter accumulator threshold also appears at looser ones; keep its max
    allc = allc[np.argsort(-allc[:, 3])]
    return nms(allc.astype(np.float32), 0.5)


def ellipse_candidates(g, rmin, rmax, block, polarity):
    g8 = np.clip(g * 40 + 128, 0, 255).astype(np.uint8)
    mode = cv2.THRESH_BINARY_INV if polarity == 'dark' else cv2.THRESH_BINARY
    bw = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, mode, block | 1, 5)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 6:
            continue
        a = cv2.contourArea(c)
        req = np.sqrt(a / np.pi)
        if not (rmin <= req <= rmax):
            continue
        (x, y), (MA, ma), _ = cv2.fitEllipse(c)
        if min(MA, ma) <= 0 or max(MA, ma) / min(MA, ma) > 2.5:
            continue
        hull = cv2.convexHull(c)
        sol = a / max(cv2.contourArea(hull), 1e-6)
        out.append([x, y, req, sol])
    return np.asarray(out, np.float32).reshape(-1, 4)


def candidates(method, rgb, g, setting, rr):
    rmin, rmax = rr
    radii = np.geomspace(rmin, rmax, 10).astype(np.float32)
    if method == 'log':
        c = log_candidates(g, radii, setting['polarity'])
    elif method == 'ring':
        c = ring_candidates(g, radii, setting['polarity'])
    elif method == 'hough':
        c = hough_candidates(rgb, rmin, rmax, setting['param1'])
    else:
        c = ellipse_candidates(g, rmin, rmax, setting['block'], setting['polarity'])
    if method in ('log', 'ring'):
        c = nms(c[np.argsort(-c[:, 3])][:CAP], setting['nms'])
    return c


NTHR = 120
PYRAMID = False   # the DALI results were computed at full resolution; the Roboflow run sets True
CAP = 20000   # strongest candidates kept per image (DALI run); the Roboflow run uses 4000

SETTINGS = {
    'log': [dict(polarity=p, nms=k) for p in ('dark', 'bright', 'abs') for k in (0.8, 1.3)],
    'ring': [dict(polarity=p, nms=k) for p in ('dark_centre', 'bright_centre') for k in (0.8, 1.3)],
    'hough': [dict(param1=p) for p in (60, 120)],
    'ellipse': [dict(block=b, polarity=p) for b in (21, 51) for p in ('dark', 'bright')],
}


def best_threshold(cands, refs):
    """cands/refs: lists over images. Pooled F1 over a threshold sweep."""
    allscore = np.concatenate([c[:, 3] for c in cands]) if cands else np.zeros(0)
    if not len(allscore):
        return 0.0, dict(f1=0.0)
    ths = np.unique(np.quantile(allscore, np.linspace(0, 0.995, NTHR)))
    best = (None, dict(f1=-1))
    for t in ths:
        m = C2.pooled([c[c[:, 3] >= t] for c in cands], refs)
        if m['f1'] > best[1]['f1']:
            best = (float(t), m)
    return best


def run_protocol(name, src_imgs, src_refs, tgt_imgs, tgt_refs, rr):
    res = {}
    for method, settings in SETTINGS.items():
        t0 = time.time()
        best = None
        g_src = [gray(im) for im in src_imgs]
        per_setting = [[] for _ in settings]
        for im, g in zip(src_imgs, g_src):            # image-outer loop: each filter stack is built once
            for k, st in enumerate(settings):
                per_setting[k].append(candidates(method, im, g, st, rr))
        for st, cs in zip(settings, per_setting):
            thr, m = best_threshold(cs, src_refs)
            if best is None or m['f1'] > best['source']['f1']:
                best = dict(setting=st, threshold=thr, source=m)
        g_tgt = [gray(im) for im in tgt_imgs]
        ct = [candidates(method, im, g, best['setting'], rr) for im, g in zip(tgt_imgs, g_tgt)]
        ct = [c[c[:, 3] >= best['threshold']] for c in ct]
        best['target'] = C2.pooled(ct, tgt_refs)
        best['target_count'] = C2.count_errors(ct, tgt_refs)
        best['seconds'] = round(time.time() - t0, 1)
        res[method] = best
        print(f"{name} {method:8s} setting {best['setting']} | source F1 {best['source']['f1']:.3f} -> target F1 "
              f"{best['target']['f1']:.3f} (P {best['target']['precision']:.2f} R {best['target']['recall']:.2f}) "
              f"count err {best['target_count']['weighted_abs_pct']:.0f}%  [{best['seconds']}s]", flush=True)
    return res


def radius_range(refs):
    """Pre-declared rule: search radii from 0.6 x 5th to 1.3 x 95th percentile of SOURCE label radii."""
    r = np.concatenate([g[:, 2] for g in refs])
    return float(max(2.0, 0.6 * np.percentile(r, 5))), float(1.3 * np.percentile(r, 95))


def main():
    import sys as _s
    out = json.loads((OUT / 'classical_cv.json').read_text()) if (OUT / 'classical_cv.json').exists() else {}
    if 'roboflow' not in _s.argv[1:]:
        D = C2.dali()
        for src, tgt in ((1, 2), (2, 1)):
            sim = D[src]['im'][:, :640].copy()
            sref = D[src]['gt'][D[src]['gt'][:, 0] < 640]
            rr = radius_range([sref])
            print(f'DALI {src}->{tgt}: radius range from source labels {rr[0]:.1f}-{rr[1]:.1f}px', flush=True)
            res = run_protocol(f'DALI {src}->{tgt}', [sim], [sref], [D[tgt]['im']], [D[tgt]['gt']], rr)
            res['radius_range'] = rr
            out[f'dali_{src}to{tgt}'] = res
            C2.save(OUT / 'classical_cv.json', out)
    if 'dali' not in _s.argv[1:]:
        global CAP, NTHR, PYRAMID
        CAP, NTHR, PYRAMID = 4000, 60, True
        dev = C2.rf_rows('train')
        rng = np.random.default_rng(7)
        dev = [dev[i] for i in rng.choice(len(dev), 20, replace=False)]
        test = C2.rf_rows('test')
        dref = [C2.rf_targets(r) for r in dev]
        rr = radius_range(dref)
        print(f'ROBOFLOW radius range from dev labels {rr[0]:.1f}-{rr[1]:.1f}px', flush=True)
        res = run_protocol('ROBOFLOW', [C2.rf_img(r) for r in dev], dref,
                           [C2.rf_img(r) for r in test], [C2.rf_targets(r) for r in test], rr)
        res['radius_range'] = rr
        out['roboflow'] = res
        C2.save(OUT / 'classical_cv.json', out)


if __name__ == '__main__':
    main()
