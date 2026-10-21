"""Task 5b: combined human-in-the-loop workflow (ORACLE SIMULATION, post hoc design).

Workflow on one new photo with a frozen one-photo base model:
  1. Fully point-label K=2 random 128-px tiles (the SAME 20 draws as adapt_budget.py).
  2. Re-choose the confidence threshold on those 2 tiles (threshold-only adaptation).
     The detector is then frozen.
  3. Audit n further tiles drawn uniformly at random from the remaining 62.
  4. Count = exact count of the 2 labelled tiles
           + difference estimator over the 62-tile remainder (finite-population t interval).
Compared with: (a) the same audit size using the source threshold (no step 2),
               (b) manual sampling with the same total human-counted tiles.
Mask labels stand in for complete human counts; no human audit was done.
"""
import sys
import numpy as np
from scipy.stats import t as tdist
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp
from barnacles.audit import tile_counts

M = bnp.MODELS
NS = [8, 16]
AUDITS_PER_DRAW = 500
CELL_TO_TILE = {}
for ci, (x0, y0, x1, y1) in enumerate([(x, y, x + 128, y + 128) for y in range(0, 1024, 128) for x in range(0, 768, 128)]):
    CELL_TO_TILE[ci] = (y0 // 128) * 8 + x0 // 128


def estimate(mc, rc, known, ids, beta):
    rest = np.setdiff1d(np.arange(64), known)
    Mr = len(rest)
    n = len(ids)
    d = rc[ids] - beta * mc[ids]
    est = rc[known].sum() + beta * mc[rest].sum() + Mr * d.mean()
    se = Mr * np.sqrt((1 - n / Mr) * d.var(ddof=1) / n)
    h = tdist.ppf(.975, n - 1) * se
    return est, h


def main():
    C = bnp.cases()
    R = bnp.json.loads((bnp.RESULTS / 'adapt_budget.json').read_text())
    out = {}
    for tag, tgt in [('pretrained_from2', 1), ('pretrained_from1', 2), ('scratch_from2', 1), ('scratch_from1', 2)]:
        p, s, _ = bnp.cached_predict(M / f'{tag}.pt', f'img{tgt}', C[tgt]['im'])
        rc = tile_counts(C[tgt]['gt'][:, :2])
        truth = rc.sum()
        thr0 = R[tag]['source_threshold']
        mc0 = tile_counts(np.clip(p[s >= thr0, :2], 0, 1023.999))
        res = {}
        for n in NS:
            acc = {k: dict(err=[], cov=[], half=[], effort=[]) for k in ('adapted+audit', 'source+audit', 'manual')}
            for r in [r for r in R[tag]['rows'] if r['K'] == 2]:
                order = np.random.default_rng(1000 + r['draw']).permutation(48)
                known = np.array([CELL_TO_TILE[c] for c in order[:2]])
                mca = tile_counts(np.clip(p[s >= r['threshold_chosen'], :2], 0, 1023.999))
                rest = np.setdiff1d(np.arange(64), known)
                rng = np.random.default_rng(50000 + r['draw'] * 100 + n)
                for _ in range(AUDITS_PER_DRAW):
                    ids = rng.choice(rest, n, replace=False)
                    for k, mc, beta in (('adapted+audit', mca, 1.0), ('source+audit', mc0, 1.0), ('manual', mc0, 0.0)):
                        e, h = estimate(mc, rc, known, ids, beta)
                        acc[k]['err'].append(e - truth)
                        acc[k]['cov'].append(abs(e - truth) <= h)
                        acc[k]['half'].append(h)
                        acc[k]['effort'].append(rc[known].sum() + rc[ids].sum())
            res[n] = {k: dict(halfwidth_pct=float(100 * np.mean(v['half']) / truth),
                              coverage95=float(np.mean(v['cov'])),
                              rmse_pct=float(100 * np.sqrt(np.mean(np.square(v['err']))) / truth),
                              within5pct=float(np.mean(np.abs(v['err']) <= .05 * truth)),
                              objects_counted_mean=float(np.mean(v['effort'])),
                              tiles_counted=n + 2) for k, v in acc.items()}
            line = ' | '.join(f"{k}: ±{v['halfwidth_pct']:.1f}% cov {v['coverage95']:.3f} w5 {v['within5pct']:.2f}" for k, v in res[n].items())
            print(f"{tag} n={n}+2 tiles (~{res[n]['manual']['objects_counted_mean']:.0f} objects) :: {line}", flush=True)
        out[tag] = dict(target=tgt, reference_total=int(truth), by_audit_size=res)
    bnp.save_json(bnp.RESULTS / 'combined_workflow_sim.json', out)


if __name__ == '__main__':
    main()
