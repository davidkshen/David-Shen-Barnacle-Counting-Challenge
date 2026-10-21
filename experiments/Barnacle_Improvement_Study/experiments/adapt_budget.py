"""Task 3: how small can the per-image annotation budget be?

PRE-DECLARED DESIGN (fixed before looking at any stripe score)
Bases      : the four saved ONE-PHOTO models (never trained on the target photo).
Target pool: 48 cells of 128x128 px in x<768 of the rectified target photo.
Reserved   : stripe x>=832 (never labelled, never used for any choice).
Draws      : 20 random cell orders per target photo (seed 1000+draw); budget K
             uses the first K cells (nested). Same draws for every base/method.
Budgets    : K = 1, 2, 4, 8 cells  (1.6%, 3.1%, 6.25%, 12.5% of the quadrat).
Annotation : every reference center inside a chosen cell (simulated from the
             supplied masks = a complete point annotation of that cell).
Methods    :
  none      - base model, source-validation threshold.
  threshold - one number re-chosen on the labelled cells (max F1, ties ->
              precision); needs >=3 labelled positives, else keep source threshold.
  refit     - last-layer refit: new logit = base logit + w.x + b, x = standardised
              penultimate class features + log radius; L2 penalty chosen by
              leave-one-cell-out CV log-loss (K>=2; K=1 uses 0.03); keep a
              candidate if new probability >= 0.5 (no calibration cells needed).
              Needs >=3 positives and >=3 negatives, else falls back to threshold.
Candidates : base detections down to confidence 0.001 (all methods share them).
Reported   : stripe precision/recall/F1, predicted vs reference count, signed
             error; mean, SD and spread across the 20 draws (= patch-choice
             sensitivity, NOT an independent-image confidence interval).
Effort     : labelled objects (point clicks) and, alternatively, the corrections
             (deletions + additions) needed to turn base predictions inside the
             cells into complete labels.
"""
import sys
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp

M = bnp.MODELS
SEL = bnp.SELECTIONS
STRIPE = (832, 0, 1024, 1024)
CELLS = [(x, y, x + 128, y + 128) for y in range(0, 1024, 128) for x in range(0, 768, 128)]
BUDGETS = [1, 2, 4, 8]
NDRAW = 20
ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3]


def fit_offset_logistic(X, y, off, alpha):
    n, d = X.shape

    def f(theta):
        w, b = theta[:d], theta[d]
        z = off + X @ w + b
        # log(1+exp(-z)) for y=1, log(1+exp(z)) for y=0
        loss = np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + alpha * w @ w
        p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        g = p - y
        return loss, np.r_[X.T @ g / n + 2 * alpha * w, g.mean()]
    r = minimize(f, np.zeros(d + 1), jac=True, method='L-BFGS-B', options=dict(maxiter=500))
    return r.x[:d], r.x[d]


def logloss(z, y):
    return float(np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)))


def label_candidates(p, gt):
    y = np.zeros(len(p))
    if len(p) and len(gt):
        m = bnp.match_objects(p, gt)
        y[m[:, 0]] = 1
    return y


def stripe_stats(keep, p, gt_s, in_s):
    m = bnp.metrics(p[keep & in_s], gt_s)
    return {k: m[k] for k in ('precision', 'recall', 'f1', 'tp', 'fp', 'fn',
                              'predicted_count', 'reference_count', 'signed_error')}


def run_base(tag, tgt):
    C = bnp.cases()
    thr0 = bnp.json.loads((SEL / f'{tag}_selection.json').read_text())['chosen']['threshold']
    p, s, b, F, lg = bnp.cached_feats(M / f'{tag}.pt', f'img{tgt}', C[tgt]['im'])
    gt = C[tgt]['gt']
    in_s = bnp.in_box(p, STRIPE)
    gt_s = gt[bnp.in_box(gt, STRIPE)]
    X_all = np.column_stack([F, np.log(np.maximum(p[:, 2], 1.0))])
    cell_of = np.full(len(p), -1)
    gcell_of = np.full(len(gt), -1)
    for ci, c in enumerate(CELLS):
        cell_of[bnp.in_box(p, c)] = ci
        gcell_of[bnp.in_box(gt, c)] = ci
    # per-cell candidate labels (matching within each cell)
    ylab = np.zeros(len(p))
    for ci, c in enumerate(CELLS):
        idx = np.flatnonzero(cell_of == ci)
        ylab[idx] = label_candidates(p[idx], gt[gcell_of == ci])
    none = stripe_stats(s >= thr0, p, gt_s, in_s)
    rows = []
    for d in range(NDRAW):
        order = np.random.default_rng(1000 + d).permutation(len(CELLS))
        for K in BUDGETS:
            cells = order[:K]
            sel = np.isin(cell_of, cells)
            gsel = np.isin(gcell_of, cells)
            npos_ref = int(gsel.sum())
            # corrections needed to produce complete labels from base predictions
            base_in = sel & (s >= thr0)
            mm = bnp.metrics(p[base_in], gt[gsel])
            corrections = int(mm['fp'] + mm['fn'])
            # --- threshold-only
            if npos_ref >= 3:
                thr = bnp.choose_confidence(p[sel], s[sel], gt[gsel])[0]['threshold']
            else:
                thr = thr0
            th = stripe_stats(s >= thr, p, gt_s, in_s)
            # --- last-layer refit
            yl = ylab[sel]
            if yl.sum() >= 3 and (1 - yl).sum() >= 3:
                Xs = X_all[sel]
                mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
                Z = (Xs - mu) / sd
                off = lg[sel]
                if K >= 2:
                    cv = []
                    csel = cell_of[sel]
                    for a in ALPHAS:
                        zs = np.zeros(len(yl))
                        for ci in cells:
                            tr, te = csel != ci, csel == ci
                            if not te.any():
                                continue
                            if yl[tr].sum() < 1:
                                zs[te] = off[te]
                                continue
                            w, b0 = fit_offset_logistic(Z[tr], yl[tr], off[tr], a)
                            zs[te] = off[te] + Z[te] @ w + b0
                        cv.append(logloss(zs, yl))
                    alpha = ALPHAS[int(np.argmin(cv))]
                else:
                    alpha = 0.03
                w, b0 = fit_offset_logistic(Z, yl, off, alpha)
                znew = lg + ((X_all - mu) / sd) @ w + b0
                keep = znew >= 0           # probability >= 0.5
                rf = stripe_stats(keep, p, gt_s, in_s)
                rf_mode = 'refit'
            else:
                alpha = None
                rf = th
                rf_mode = 'fallback_threshold'
            rows.append(dict(draw=d, K=K, cells=[CELLS[c] for c in cells], labelled_objects=npos_ref,
                             corrections_from_base=corrections, area_fraction=K * 128 * 128 / 1024 ** 2,
                             threshold_chosen=float(thr), alpha=alpha, refit_mode=rf_mode,
                             none=none, threshold=th, refit=rf))
    return dict(tag=tag, target=tgt, source_threshold=thr0, stripe_reference=len(gt_s), rows=rows)


def summarise(res):
    out = {}
    for K in BUDGETS:
        rr = [r for r in res['rows'] if r['K'] == K]
        o = dict(labelled_objects=dict(mean=float(np.mean([r['labelled_objects'] for r in rr])),
                                       min=int(min(r['labelled_objects'] for r in rr)),
                                       max=int(max(r['labelled_objects'] for r in rr))),
                 corrections_from_base=dict(mean=float(np.mean([r['corrections_from_base'] for r in rr]))),
                 refit_fallbacks=int(sum(r['refit_mode'] != 'refit' for r in rr)))
        for meth in ('none', 'threshold', 'refit'):
            f1 = np.array([r[meth]['f1'] for r in rr])
            ce = np.array([abs(r[meth]['signed_error']) / r[meth]['reference_count'] for r in rr])
            se = np.array([r[meth]['signed_error'] for r in rr])
            o[meth] = dict(f1_mean=float(f1.mean()), f1_sd=float(f1.std(ddof=1)), f1_min=float(f1.min()),
                           f1_p10=float(np.percentile(f1, 10)), f1_median=float(np.median(f1)),
                           f1_p90=float(np.percentile(f1, 90)), f1_max=float(f1.max()),
                           abs_count_error_pct_mean=float(100 * ce.mean()),
                           abs_count_error_pct_p90=float(100 * np.percentile(ce, 90)),
                           signed_error_mean=float(se.mean()),
                           precision_mean=float(np.mean([r[meth]['precision'] for r in rr])),
                           recall_mean=float(np.mean([r[meth]['recall'] for r in rr])))
        out[K] = o
    return out


def main():
    allres = {}
    for tag, tgt in [('pretrained_from2', 1), ('pretrained_from1', 2), ('scratch_from2', 1), ('scratch_from1', 2)]:
        res = run_base(tag, tgt)
        res['summary'] = summarise(res)
        allres[tag] = res
        for K, o in res['summary'].items():
            print(f"{tag} K={K} labels~{o['labelled_objects']['mean']:.0f} fixes~{o['corrections_from_base']['mean']:.0f} | "
                  f"none {o['none']['f1_mean']:.3f} | thr {o['threshold']['f1_mean']:.3f}±{o['threshold']['f1_sd']:.3f} "
                  f"[{o['threshold']['f1_min']:.3f},{o['threshold']['f1_max']:.3f}] | refit {o['refit']['f1_mean']:.3f}±{o['refit']['f1_sd']:.3f} "
                  f"[{o['refit']['f1_min']:.3f},{o['refit']['f1_max']:.3f}] cnt% thr {o['threshold']['abs_count_error_pct_mean']:.1f} refit {o['refit']['abs_count_error_pct_mean']:.1f} fb={o['refit_fallbacks']}",
                  flush=True)
        bnp.save_json(bnp.RESULTS / 'adapt_budget.json', allres)


if __name__ == '__main__':
    main()
