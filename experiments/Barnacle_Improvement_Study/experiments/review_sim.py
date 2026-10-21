"""Task 4: does a higher-recall operating point reduce review effort?  (SIMULATION)

No human was timed. For every threshold we count what a reviewer would have to
fix on a full new photo: deletions (false positives) and additions (misses).
Effort is expressed in 'deletion units' under explicit, varied assumptions:
  cost = v * (correct markers to glance at) + 1 * deletions + c * additions
  c = cost of finding+adding a missed barnacle relative to deleting a marker
      (1, 2, 3, 5 tested; plausibly >1 because misses must be searched for)
  v = cost of glancing at a correct marker (0 or 0.25)
Manual counting of the whole quadrat is taken as c * N (every barnacle found and
clicked like an addition). These are assumptions to be replaced by measured
times from the timed-review protocol.
"""
import json
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp

M = bnp.MODELS
SEL = bnp.SELECTIONS
THRS = np.unique(np.r_[np.linspace(0.005, 0.95, 190), 0.001])


def main():
    C = bnp.cases()
    out = {}
    for tag, tgt in [('pretrained_from2', 1), ('pretrained_from1', 2), ('scratch_from2', 1), ('scratch_from1', 2)]:
        thr0 = bnp.json.loads((SEL / f'{tag}_selection.json').read_text())['chosen']['threshold']
        p, s, _ = bnp.cached_predict(M / f'{tag}.pt', f'img{tgt}', C[tgt]['im'])
        gt = C[tgt]['gt']
        N = len(gt)
        curve = []
        for t in THRS:
            m = bnp.metrics(p[s >= t], gt)
            curve.append(dict(threshold=float(t), tp=m['tp'], fp=m['fp'], fn=m['fn'], f1=m['f1'],
                              precision=m['precision'], recall=m['recall']))
        f1best = max(curve, key=lambda r: r['f1'])
        src = min(curve, key=lambda r: abs(r['threshold'] - thr0))
        src = dict(src, **{k: v for k, v in bnp.metrics(p[s >= thr0], gt).items() if k in ('tp', 'fp', 'fn', 'f1', 'precision', 'recall')})
        src['threshold'] = thr0
        scen = []
        for v in (0.0, 0.25):
            for c in (1, 2, 3, 5):
                def cost(r):
                    return v * r['tp'] + r['fp'] + c * r['fn']
                best = min(curve, key=cost)
                manual = c * N
                scen.append(dict(add_cost=c, glance_cost=v, manual_cost=manual,
                                 best_threshold=best['threshold'], best_recall=best['recall'],
                                 best_precision=best['precision'], best_fp=best['fp'], best_fn=best['fn'],
                                 best_cost_pct_of_manual=100 * cost(best) / manual,
                                 f1opt_cost_pct_of_manual=100 * cost(f1best) / manual,
                                 source_cost_pct_of_manual=100 * cost(src) / manual))
        # Deployable operating point: the threshold re-chosen on 2 labelled cells
        # (the same 20 draws as adapt_budget.py) - uses no labels outside those cells.
        two = None
        ab = bnp.RESULTS / 'adapt_budget.json'
        if ab.exists():
            ths = [r['threshold_chosen'] for r in json.loads(ab.read_text())[tag]['rows'] if r['K'] == 2]
            two = dict(thresholds_median=float(np.median(ths)), thresholds_range=[float(min(ths)), float(max(ths))], by_cost={})
            for v in (0.0, 0.25):
                for c in (1, 3):
                    e = []
                    for t in ths:
                        m = bnp.metrics(p[s >= t], gt)
                        e.append(100 * (v * m['tp'] + m['fp'] + c * m['fn']) / (c * N))
                    two['by_cost'][f'v{v}_c{c}'] = dict(mean_pct_of_manual=float(np.mean(e)), min=float(min(e)), max=float(max(e)))
        out[tag] = dict(target=tgt, reference_count=N, source_operating_point=src, f1_optimal=f1best,
                        scenarios=scen, two_cell_threshold=two, curve=curve,
                        note='best_* thresholds are chosen with the target labels (oracle); two_cell_threshold is deployable')
        print(f"\n{tag} -> photo {tgt} (N={N}); source t={thr0:.3f}: FP {src['fp']} FN {src['fn']} | F1-opt t={f1best['threshold']:.3f}: FP {f1best['fp']} FN {f1best['fn']}")
        for sc in scen:
            print(f"  v={sc['glance_cost']:.2f} c={sc['add_cost']}: best t={sc['best_threshold']:.3f} (R {sc['best_recall']:.2f}, P {sc['best_precision']:.2f}, FP {sc['best_fp']}, FN {sc['best_fn']}) "
                  f"effort {sc['best_cost_pct_of_manual']:.0f}% of manual | at F1-opt {sc['f1opt_cost_pct_of_manual']:.0f}% | at source t {sc['source_cost_pct_of_manual']:.0f}%")
    bnp.save_json(bnp.RESULTS / 'review_sim.json', out)


if __name__ == '__main__':
    main()
