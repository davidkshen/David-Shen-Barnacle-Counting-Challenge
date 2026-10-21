"""POST HOC extension (decided after seeing that validation F1 peaked at the edge of the declared
grid, f = 0.5): add f = 0.7 for the photo-2 pretrained model.
  fixed_0.7        val and test at 0.7 (DALI threshold and val-calibrated threshold)
  confidence_wide  confidence-guided scale over {0.7, 0.5, 0.35}
Reported separately from the pre-declared results and labelled post hoc.
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import detectors_external as DE  # noqa: E402
OUT = Path(__file__).resolve().parent / 'results'


def main():
    tag = 'pretrained_from2'
    val, test = C2.rf_rows('val'), C2.rf_rows('test')
    val_r = [C2.rf_targets(r) for r in val]
    test_r = [C2.rf_targets(r) for r in test]
    groups = [r['group'] for r in test]
    dthr = json.loads((C2.bnp.SELECTIONS / f'{tag}_selection.json').read_text())['chosen']['threshold']
    vp = [DE.predict_at(tag, r, 0.7) for r in val]
    tp = [DE.predict_at(tag, r, 0.7) for r in test]
    out = dict(note='POST HOC: added after the declared grid peaked at its edge',
               fixed_0p7=dict(val_f1_dali_threshold=C2.pooled([p[p[:, 3] >= dthr] for p in vp], val_r)['f1'],
                              dali_threshold=DE.score(tp, test_r, groups, dthr),
                              calibrated=DE.score(tp, test_r, groups, DE.best_thr(vp, val_r)[0])))
    def pick(r):
        best = None
        for f in (0.7, 0.5, 0.35):
            p = DE.predict_at(tag, r, f)
            sc = float(np.sort(p[:, 3])[-20:].mean()) if len(p) else 0.0
            if best is None or sc > best[0]:
                best = (sc, f, p)
        return best[1], best[2]
    vc = [pick(r) for r in val]
    tc = [pick(r) for r in test]
    out['confidence_wide'] = dict(factor_counts={str(f): int(sum(x[0] == f for x in tc)) for f in (0.7, 0.5, 0.35)},
                                  dali_threshold=DE.score([x[1] for x in tc], test_r, groups, dthr),
                                  calibrated=DE.score([x[1] for x in tc], test_r, groups, DE.best_thr([x[1] for x in vc], val_r)[0]))
    print(json.dumps({k: (v if isinstance(v, str) else {a: (b['f1'] if isinstance(b, dict) and 'f1' in b else b) for a, b in v.items()}) for k, v in out.items()}), flush=True)
    C2.save(OUT / 'external_extension_posthoc.json', out)


if __name__ == '__main__':
    main()
