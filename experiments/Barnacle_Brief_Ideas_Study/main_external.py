"""Main model (mixed_0.5_yolo11n.pt, threshold 0.37 from default.json) on the round-6 external Roboflow
benchmark (corrected split: 26 validation photos / 5 groups, 97 test photos / 9 groups).

Declared before running (same protocol as detectors_external.py, one change noted):
  fixed_scale       one factor for all photos, chosen on the 26 validation photos by pooled F1 at 0.37,
                    from {1.0, 0.7, 0.5, 0.35, 0.25}. (Round 6 used {0.5, 0.35, 0.25} and found the best
                    factor at the edge, so 1.0 and 0.7 are included from the start here.)
  thresholds        own (0.37); calibrated (re-chosen on the 26 validation photos at the chosen factor);
                    label-free Otsu per photo (falls back to 0.37 if fewer than 10 scores > 0.01).
  confidence_scale  per photo, label-free: factor in the same set whose 20 most confident detections
                    have the highest mean confidence; own threshold.
Not comparable with the other session's 0.733 / 0.756 (different split, different scale handling).
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import detectors_external as DE  # noqa: E402
from creative_dali import otsu_thr  # noqa: E402
bnp = C2.bnp
MP = __import__('os').environ.get('BARNACLE_MAIN_MODEL', str(Path(__file__).resolve().parents[2] / 'models/mixed_0.5_yolo11n.pt'))
THR = json.load(open(__import__('os').environ.get('BARNACLE_MAIN_CONFIG', str(Path(__file__).resolve().parents[2] / 'models/default.json'))))['threshold']
FACTORS = (1.0, 0.7, 0.5, 0.35, 0.25)


def predict(row, f):
    im = C2.rf_img(row)
    if abs(f - 1) > 1e-3:
        w, h = max(32, round(row['width'] * f)), max(32, round(row['height'] * f))
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    else:
        w, h = row['width'], row['height']
    p = bnp.cached_predict(MP, f"rf_{row['id']}" + ('' if abs(f - 1) < 1e-3 else f'_f{f:.4f}'), im, (0, 0, w, h))[0].copy()
    p[:, 0] *= row['width'] / w
    p[:, 1] *= row['height'] / h
    p[:, 2] *= row['width'] / w
    return p


def main():
    val, test = C2.rf_rows('val'), C2.rf_rows('test')
    val_r = [C2.rf_targets(r) for r in val]
    test_r = [C2.rf_targets(r) for r in test]
    groups = [r['group'] for r in test]
    vp = {f: [predict(r, f) for r in val] for f in FACTORS}
    val_f1 = {str(f): C2.pooled([p[p[:, 3] >= THR] for p in vp[f]], val_r)['f1'] for f in FACTORS}
    print('validation F1 by factor', {k: round(v, 3) for k, v in val_f1.items()}, flush=True)
    fx = max(FACTORS, key=lambda f: val_f1[str(f)])
    tp = {f: [predict(r, f) for r in test] for f in FACTORS}
    cal_t = DE.best_thr(vp[fx], val_r)[0]
    ot = []
    for p in tp[fx]:
        t = otsu_thr(p) if (p[:, 3] > 0.01).sum() >= 10 else THR
        ot.append(p[p[:, 3] >= t])
    otsu = dict(**C2.pooled(ot, test_r), **C2.count_errors(ot, test_r), f1_group_ci95=C2.group_bootstrap(ot, test_r, groups))

    def pick(preds_by_f, i):
        sc = {f: (float(np.sort(preds_by_f[f][i][:, 3])[-20:].mean()) if len(preds_by_f[f][i]) else 0.0) for f in FACTORS}
        return max(FACTORS, key=lambda f: sc[f])
    cf = [pick(tp, i) for i in range(len(test))]
    cpred = [tp[f][i] for i, f in enumerate(cf)]
    out = dict(model=MP, threshold=THR, factors=FACTORS, validation_f1_own_threshold=val_f1, chosen_factor=fx,
               fixed_scale=dict(own_threshold=DE.score(tp[fx], test_r, groups, THR),
                                calibrated=DE.score(tp[fx], test_r, groups, cal_t),
                                label_free_otsu=otsu),
               confidence_scale=dict(factor_counts={str(f): int(sum(c == f for c in cf)) for f in FACTORS},
                                     own_threshold=DE.score(cpred, test_r, groups, THR)),
               test_f1_by_factor_DIAGNOSTIC={str(f): C2.pooled([p[p[:, 3] >= THR] for p in tp[f]], test_r)['f1'] for f in FACTORS},
               n_test=len(test), n_test_groups=len(set(groups)), n_val=len(val))
    for k in ('own_threshold', 'calibrated', 'label_free_otsu'):
        m = out['fixed_scale'][k]
        print(k, round(m['f1'], 3), m['f1_group_ci95'], 'P', round(m['precision'], 2), 'R', round(m['recall'], 2),
              'count err', round(m['weighted_abs_pct']), flush=True)
    print('confidence scale', round(out['confidence_scale']['own_threshold']['f1'], 3), out['confidence_scale']['factor_counts'])
    print('diagnostic test F1 by factor', {k: round(v, 3) for k, v in out['test_f1_by_factor_DIAGNOSTIC'].items()})
    C2.save(str(Path(__file__).resolve().parent / 'results') + '/main_external.json', out)


if __name__ == '__main__':
    main()
