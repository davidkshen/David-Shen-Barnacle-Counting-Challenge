"""Task 5: how wide is the count margin of error with a random tile audit?

ORACLE SIMULATION: supplied mask labels stand in for a human's complete tile
counts; no new human audit was performed. The detector is frozen first
(one-photo base model on the opposite photo = a genuinely unseen photo for it).
For each audit size n of the 64 tiles (128 px), 5000 random audits:
  model + audit : N_model + 64 * mean(human - model)   (difference estimator)
  manual sample : 64 * mean(human)                     (no model)
Both with the project's approximate t interval and finite-population correction
(barnacles.audit.corrected_count, beta=1 or beta=0).
Reported: bias, RMSE, % of audits within 5% of the reference total, 95%
interval coverage and mean half-width (% of total), and objects counted.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp
from barnacles.audit import tile_counts, corrected_count

M = bnp.MODELS
SEL = bnp.SELECTIONS
NS = [4, 8, 16, 32]
REPEATS = 5000


def simulate(model_counts, ref_counts, n, beta, seed):
    rng = np.random.default_rng(seed)
    truth = ref_counts.sum()
    est, cov, half, effort = [], [], [], []
    for _ in range(REPEATS):
        ids = rng.choice(64, n, replace=False)
        a = corrected_count(model_counts, ids, ref_counts[ids], beta=beta)
        lo, hi = a['interval95']
        est.append(a['estimate'])
        cov.append(lo <= truth <= hi)
        half.append((hi - lo) / 2)
        effort.append(ref_counts[ids].sum())
    e = np.asarray(est) - truth
    return dict(n=n, bias_pct=float(100 * e.mean() / truth), rmse_pct=float(100 * np.sqrt((e ** 2).mean()) / truth),
                within5pct=float(np.mean(np.abs(e) <= .05 * truth)), within10pct=float(np.mean(np.abs(e) <= .10 * truth)),
                coverage95=float(np.mean(cov)), mean_halfwidth_pct=float(100 * np.mean(half) / truth),
                objects_counted_mean=float(np.mean(effort)))


def main():
    C = bnp.cases()
    out = {}
    for tag, tgt in [('pretrained_from2', 1), ('pretrained_from1', 2), ('scratch_from2', 1), ('scratch_from1', 2)]:
        thr = bnp.json.loads((SEL / f'{tag}_selection.json').read_text())['chosen']['threshold']
        p, s, _ = bnp.cached_predict(M / f'{tag}.pt', f'img{tgt}', C[tgt]['im'])
        pred = p[s >= thr]
        mc = tile_counts(np.clip(pred[:, :2], 0, 1023.999))
        rc = tile_counts(C[tgt]['gt'][:, :2])
        res = dict(target=tgt, threshold=thr, model_total=int(mc.sum()), reference_total=int(rc.sum()),
                   raw_model_error_pct=float(100 * (mc.sum() - rc.sum()) / rc.sum()),
                   tile_residual_sd=float(np.std(rc - mc, ddof=1)), tile_count_sd=float(np.std(rc, ddof=1)),
                   model_plus_audit=[simulate(mc, rc, n, 1.0, 7 + n) for n in NS],
                   manual_sample=[simulate(mc, rc, n, 0.0, 7 + n) for n in NS])
        out[tag] = res
        print(f"\n{tag} -> photo {tgt}: model {res['model_total']} vs ref {res['reference_total']} "
              f"({res['raw_model_error_pct']:+.1f}%); tile SD residual {res['tile_residual_sd']:.1f} vs count {res['tile_count_sd']:.1f}")
        for a, b in zip(res['model_plus_audit'], res['manual_sample']):
            print(f"  n={a['n']:2d} (~{a['objects_counted_mean']:.0f} objects): model+audit ±{a['mean_halfwidth_pct']:.1f}% cover {a['coverage95']:.3f} "
                  f"within5% {a['within5pct']:.2f} | manual ±{b['mean_halfwidth_pct']:.1f}% cover {b['coverage95']:.3f} within5% {b['within5pct']:.2f}")
    bnp.save_json(bnp.RESULTS / 'audit_sim.json', out)


if __name__ == '__main__':
    main()
