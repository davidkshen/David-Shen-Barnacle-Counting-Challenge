"""Post-hoc diagnostics for Task 3 (labelled as post hoc in the write-up).

(1) Was the refit failure a ranking failure or a decision-rule failure?
    -> stripe F1 of refit scores at the best possible stripe threshold (oracle,
       diagnostic only) vs the same oracle for the base scores.
(2) Can the labelled cells themselves warn you that cheap adaptation will not
    be enough?  -> in-cell F1 after threshold choice (a quantity the user
    actually sees) vs the stripe F1 they will get.
Reuses exactly the same draws/cells as adapt_budget.py.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as bnp
import adapt_budget as A

THRS = np.unique(np.r_[np.linspace(0.001, 0.99, 400)])


def oracle_f1(score, p, gt_s, in_s):
    best = 0.0
    for t in np.unique(np.quantile(score[in_s], np.linspace(0, 1, 300))):
        best = max(best, bnp.metrics(p[in_s & (score >= t)], gt_s)['f1'])
    return best


def main():
    C = bnp.cases()
    out = {}
    res = bnp.json.loads((bnp.RESULTS / 'adapt_budget.json').read_text())
    for tag, tgt in [('pretrained_from2', 1), ('pretrained_from1', 2), ('scratch_from2', 1), ('scratch_from1', 2)]:
        thr0 = res[tag]['source_threshold']
        p, s, b, F, lg = bnp.cached_feats(A.M / f'{tag}.pt', f'img{tgt}', C[tgt]['im'])
        gt = C[tgt]['gt']
        in_s = bnp.in_box(p, A.STRIPE)
        gt_s = gt[bnp.in_box(gt, A.STRIPE)]
        X_all = np.column_stack([F, np.log(np.maximum(p[:, 2], 1.0))])
        cell_of = np.full(len(p), -1)
        gcell_of = np.full(len(gt), -1)
        for ci, c in enumerate(A.CELLS):
            cell_of[bnp.in_box(p, c)] = ci
            gcell_of[bnp.in_box(gt, c)] = ci
        ylab = np.zeros(len(p))
        for ci in range(len(A.CELLS)):
            idx = np.flatnonzero(cell_of == ci)
            ylab[idx] = A.label_candidates(p[idx], gt[gcell_of == ci])
        base_oracle = oracle_f1(s, p, gt_s, in_s)
        rows = []
        for r in res[tag]['rows']:
            if r['K'] not in (2, 8):
                continue
            order = np.random.default_rng(1000 + r['draw']).permutation(len(A.CELLS))
            cells = order[:r['K']]
            sel = np.isin(cell_of, cells)
            gsel = np.isin(gcell_of, cells)
            incell_base = bnp.metrics(p[sel & (s >= thr0)], gt[gsel])['f1']
            incell_thr = bnp.metrics(p[sel & (s >= r['threshold_chosen'])], gt[gsel])['f1'] if gsel.sum() else np.nan
            ro = None
            if r['refit_mode'] == 'refit':
                Xs = X_all[sel]
                mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
                w, b0 = A.fit_offset_logistic((Xs - mu) / sd, ylab[sel], lg[sel], r['alpha'])
                znew = lg + ((X_all - mu) / sd) @ w + b0
                ro = oracle_f1(znew, p, gt_s, in_s)
            rows.append(dict(draw=r['draw'], K=r['K'], incell_f1_base_threshold=incell_base,
                             incell_f1_after_threshold=incell_thr, stripe_f1_threshold=r['threshold']['f1'],
                             stripe_f1_refit=r['refit']['f1'], refit_oracle_stripe_f1=ro))
        out[tag] = dict(base_oracle_stripe_f1=base_oracle, rows=rows)
        for K in (2, 8):
            rr = [x for x in rows if x['K'] == K]
            ro = [x['refit_oracle_stripe_f1'] for x in rr if x['refit_oracle_stripe_f1'] is not None]
            print(f"{tag} K={K}: base oracle {base_oracle:.3f} | refit oracle mean {np.mean(ro):.3f} | "
                  f"in-cell F1 after thr mean {np.nanmean([x['incell_f1_after_threshold'] for x in rr]):.3f} "
                  f"(range {np.nanmin([x['incell_f1_after_threshold'] for x in rr]):.3f}-{np.nanmax([x['incell_f1_after_threshold'] for x in rr]):.3f}) "
                  f"-> stripe thr F1 {np.mean([x['stripe_f1_threshold'] for x in rr]):.3f}", flush=True)
    bnp.save_json(bnp.RESULTS / 'adapt_diagnostics.json', out)


if __name__ == '__main__':
    main()
