"""'Pretrained CNN features + scikit-learn classifier' detector (a linear/GBM probe).

Why: combines two ideas from the brief (CNN features, scikit-learn classification) and needs no
PyTorch training. The COCO-pretrained YOLO11n (outside weights; never trained on barnacles) is
frozen and used only as a feature extractor. Each 4-px cell of the image gets a 448-d descriptor
(neck outputs P3 64ch + P4 128ch + P5 256ch, upsampled to stride 8 at the 2x tile scale).
A classifier learns 'barnacle centre here?'; local maxima of the probability map are detections.

Cell labels: positive if within max(2 px, 0.35 r) of a target centre; negative if farther than
0.9 r from every target; in between ignored. Non-target labelled objects (snails, spat, larvae)
are explicitly negatives (hard negatives).
Classifiers: logistic regression (C in 0.01/0.1/1) and HistGradientBoosting; peak spacing 2/3/5
cells. Classifier, C, spacing and threshold are chosen on a validation set only.

Experiments (training data -> test):
  dali_k   train photo k left region x<640, validate photo k x>=768, test the full other photo
  rf       train 100 Roboflow dev-train photos, validate on the 26 val photos, test on the 97 test photos;
           the same trained probe is also applied to both full DALI photos (shells vs openings!)
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
bnp = C2.bnp

OUT = Path(__file__).resolve().parent / 'results'
COCO = bnp.PROJECT / 'yolo11n.pt'
CELL = 4.0          # px in the 1024-scale image per stride-8 cell at the 2x tile scale
_M = None


def model():
    global _M
    if _M is None:
        _M = bnp.Model(COCO)
    return _M


def tile_features(rgb, bounds):
    """Yield (core_mask_cells, cell_xy (n,2), feats (n,448)) per tile, cells restricted to the tile core."""
    ws = bnp.windows(bounds)
    m = model()
    for s in range(0, len(ws), 4):
        chunk = ws[s:s + 4]
        x = []
        for w in chunk:
            x0, y0, x1, y1 = w['crop']
            t = rgb[y0:y1, x0:x1]
            t = cv2.resize(t, ((x1 - x0) * 2, (y1 - y0) * 2), interpolation=cv2.INTER_LINEAR)
            x.append(t)
        x = np.stack(x).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        _, feats = m.raw(np.ascontiguousarray(x), return_features=True)
        p3, p4, p5 = feats
        p4 = p4.repeat(2, 2).repeat(2, 3)
        p5 = p5.repeat(4, 2).repeat(4, 3)
        f = np.concatenate([p3, p4, p5], 1)                       # N,448,h,w
        for j, w in enumerate(chunk):
            x0, y0, _, _ = w['crop']
            h, wd = f.shape[2], f.shape[3]
            gy, gx = np.mgrid[0:h, 0:wd]
            cx = x0 + (gx + 0.5) * CELL
            cy = y0 + (gy + 0.5) * CELL
            c = w['core']
            keep = (cx >= c[0]) & (cx < c[2]) & (cy >= c[1]) & (cy < c[3])
            yield np.column_stack([cx[keep], cy[keep]]).astype(np.float32), f[j][:, keep].T.astype(np.float32)


def cell_labels(xy, targets, others):
    lab = np.full(len(xy), -1, np.int8)
    if len(targets):
        d, i = cKDTree(targets[:, :2]).query(xy)
        r = targets[i, 2]
        lab[d > 0.9 * r] = 0
        lab[d < np.maximum(2.0, 0.35 * r)] = 1
    else:
        lab[:] = 0
    if len(others):
        d2, i2 = cKDTree(others[:, :2]).query(xy)
        lab[(d2 < 0.6 * others[i2, 2]) & (lab != 1)] = 0
    return lab


def collect(images, rng, neg_per_pos=4, max_neg_tile=400, max_pos_tile=None):
    X, y = [], []
    for rgb, bounds, targets, others in images:
        for xy, f in tile_features(rgb, bounds):
            lab = cell_labels(xy, targets, others)
            pos = np.flatnonzero(lab == 1)
            if max_pos_tile and len(pos) > max_pos_tile:
                pos = rng.choice(pos, max_pos_tile, replace=False)
            neg = np.flatnonzero(lab == 0)
            k = min(len(neg), max(max_neg_tile // 4, neg_per_pos * len(pos)), max_neg_tile)
            neg = rng.choice(neg, k, replace=False) if k else neg[:0]
            sel = np.r_[pos, neg]
            X.append(f[sel].astype(np.float16))
            y.append(lab[sel])
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


def fit_variants(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X -= mu
    X /= sd
    Z = X
    out = {}
    for C in (0.01, 0.1, 1.0):
        out[f'logreg_C{C}'] = LogisticRegression(C=C, max_iter=400).fit(Z, y)
    out['hgb'] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                random_state=0).fit(Z, y)
    return out, (mu, sd)


def prob_maps(rgb, bounds, variants, norm):
    """Probability per cell for every variant, rasterised at CELL resolution over the bounds."""
    x0, y0, x1, y1 = bounds
    H, W = int(np.ceil((y1 - y0) / CELL)), int(np.ceil((x1 - x0) / CELL))
    maps = {k: np.zeros((H, W), np.float32) for k in variants}
    mu, sd = norm
    for xy, f in tile_features(rgb, bounds):
        Z = (f - mu) / sd
        gi = np.clip(((xy[:, 1] - y0) / CELL).astype(int), 0, H - 1)
        gj = np.clip(((xy[:, 0] - x0) / CELL).astype(int), 0, W - 1)
        for k, clf in variants.items():
            maps[k][gi, gj] = clf.predict_proba(Z)[:, 1]
    return maps


def peaks(pmap, bounds, spacing, floor=0.02):
    sm = ndi.gaussian_filter(pmap, 0.7)
    mx = ndi.maximum_filter(sm, size=2 * spacing + 1)
    pk = (sm == mx) & (sm > floor)
    i, j = np.nonzero(pk)
    x = bounds[0] + (j + 0.5) * CELL
    y = bounds[1] + (i + 0.5) * CELL
    return np.column_stack([x, y, np.full(len(x), spacing * CELL), sm[i, j]]).astype(np.float32)


def choose(maps_list, bounds_list, refs):
    """maps_list: per image {variant: map}. Returns best (variant, spacing, threshold, metrics)."""
    best = None
    for v in maps_list[0]:
        for sp in (2, 3, 5):
            cands = [peaks(m[v], b, sp) for m, b in zip(maps_list, bounds_list)]
            allsc = np.concatenate([c[:, 3] for c in cands])
            for t in np.unique(np.quantile(allsc, np.linspace(0, 0.995, 100))):
                met = C2.pooled([c[c[:, 3] >= t] for c in cands], refs)
                if best is None or met['f1'] > best[3]['f1']:
                    best = (v, sp, float(t), met)
    return best


def apply(maps_list, bounds_list, refs, v, sp, t):
    c = [peaks(m[v], b, sp) for m, b in zip(maps_list, bounds_list)]
    c = [x[x[:, 3] >= t] for x in c]
    return c, dict(**C2.pooled(c, refs), **C2.count_errors(c, refs))


def dali_experiments(out):
    D = C2.dali()
    rng = np.random.default_rng(0)
    for src, tgt in ((1, 2), (2, 1)):
        t0 = time.time()
        g = D[src]['gt']
        tr = [(D[src]['im'], (0, 0, 640, 1024), g[g[:, 0] < 640 + 10], np.zeros((0, 3), np.float32))]
        X, y = collect(tr, rng, neg_per_pos=6, max_neg_tile=1500)
        variants, norm = fit_variants(X, y)
        vb = (768, 0, 1024, 1024)
        vmaps = [prob_maps(D[src]['im'], vb, variants, norm)]
        v, sp, t, vm = choose(vmaps, [vb], [g[g[:, 0] >= 768]])
        tb = (0, 0, 1024, 1024)
        tmaps = [prob_maps(D[tgt]['im'], tb, variants, norm)]
        _, tm = apply(tmaps, [tb], [D[tgt]['gt']], v, sp, t)
        oracle = choose(tmaps, [tb], [D[tgt]['gt']])
        out[f'dali_{src}to{tgt}'] = dict(train_samples=int(len(y)), positives=int(y.sum()), chosen=dict(variant=v, spacing=sp, threshold=t),
                                         source_val=vm, target=tm, target_oracle_DIAGNOSTIC=oracle[3]['f1'],
                                         seconds=round(time.time() - t0))
        print(f"probe DALI {src}->{tgt}: {v} sp{sp} | source-val F1 {vm['f1']:.3f} -> target F1 {tm['f1']:.3f} "
              f"(P {tm['precision']:.2f} R {tm['recall']:.2f}, count {tm['weighted_abs_pct']:.0f}%) oracle {oracle[3]['f1']:.3f}", flush=True)
        C2.save(OUT / 'probe.json', out)


def rf_experiment(out, n_train=100):
    """Memory-bounded: photos are loaded one at a time; at most 100 positive and 100 negative
    cells per tile (a first run with 150 photos and no caps ran out of memory)."""
    rng = np.random.default_rng(1)
    t0 = time.time()
    dev = C2.rf_rows('train')
    dev = [dev[i] for i in rng.choice(len(dev), n_train, replace=False)]
    tr = ((C2.rf_img(r), (0, 0, r['width'], r['height']), C2.rf_targets(r), C2.rf_others(r)) for r in dev)
    X, y = collect(tr, rng, neg_per_pos=1, max_neg_tile=100, max_pos_tile=100)
    variants, norm = fit_variants(X, y)
    print(f'probe RF trained on {len(dev)} photos, {len(y)} cells ({int(y.sum())} positive) in {time.time() - t0:.0f}s', flush=True)
    val = C2.rf_rows('val')
    vb = [(0, 0, r['width'], r['height']) for r in val]
    vmaps = [prob_maps(C2.rf_img(r), b, variants, norm) for r, b in zip(val, vb)]
    v, sp, t, vm = choose(vmaps, vb, [C2.rf_targets(r) for r in val])
    del vmaps
    test = C2.rf_rows('test')
    tb = [(0, 0, r['width'], r['height']) for r in test]
    tmaps = [prob_maps(C2.rf_img(r), b, {v: variants[v]}, norm) for r, b in zip(test, tb)]
    refs = [C2.rf_targets(r) for r in test]
    cands, tm = apply(tmaps, tb, refs, v, sp, t)
    tm['f1_group_ci95'] = C2.group_bootstrap(cands, refs, [r['group'] for r in test])
    D = C2.dali()
    dres = {}
    for k in (1, 2):
        b = (0, 0, 1024, 1024)
        mp = [prob_maps(D[k]['im'], b, {v: variants[v]}, norm)]
        _, dm = apply(mp, [b], [D[k]['gt']], v, sp, t)
        orc = choose(mp, [b], [D[k]['gt']])
        dres[f'photo{k}'] = dict(**dm, oracle_DIAGNOSTIC=orc[3]['f1'])
    out['rf'] = dict(train_photos=len(dev), train_cells=int(len(y)), chosen=dict(variant=v, spacing=sp, threshold=t),
                     val=vm, test=tm, dali_full_photos=dres, seconds=round(time.time() - t0))
    print(f"probe RF: {v} sp{sp} | val F1 {vm['f1']:.3f} -> test F1 {tm['f1']:.3f} {tm['f1_group_ci95']} count {tm['weighted_abs_pct']:.0f}% | "
          f"DALI photo1 F1 {dres['photo1']['f1']:.3f} photo2 {dres['photo2']['f1']:.3f}", flush=True)
    C2.save(OUT / 'probe.json', out)


if __name__ == '__main__':
    out = {}
    if 'rf' not in sys.argv[1:]:
        dali_experiments(out)
    if 'dali' not in sys.argv[1:]:
        rf_experiment(out)
