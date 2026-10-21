"""Saved DALI-trained detectors on the external Roboflow benchmark, with scale handling.

Why scale handling: Roboflow photos are close-ups (~5 x 5 cm), so at longest-side 1024 a shell
has a median radius of ~17 px (up to ~46 px) while DALI openings are ~7-8 px. Run at native scale
the photo-1 model fires on gravel texture (F1 0.048 in a first run; that run was dropped from this repo).

Inference: the photo is resized by a factor f (longest side 1024*f), tiled exactly like a DALI
quadrat (256-px tiles, 192 stride, upsampled to the 512-px network input), merged with the
project's core ownership + NMS, and mapped back to longest-side-1024 coordinates for scoring.

Variants (declared before results):
  fixed_scale      one factor for all photos, chosen on the 26 validation photos from {0.5, 0.35, 0.25}
                   by pooled F1 at the model's own DALI threshold
  confidence_scale per photo, label-free: run at every factor in {0.5, 0.35, 0.25, 0.18} and keep the
                   factor whose 20 most confident detections have the highest mean confidence
                   (a detector should be most confident at the object size it was trained on).
  auto_scale       per photo, label-free: f = clip(s_DALI / s_photo, 0.12, 1.0), with s the scale
                   estimator chosen in scale_estimators.py, s_DALI measured on the model's own
                   DALI training photo. Run only if that estimator passed its validation
                   (Spearman >= 0.5 with true size on labelled check photos).
Thresholds: 'dali_threshold' (the model's own; purely external) and 'calibrated' (re-chosen on the
the 26 validation photos at the same scale setting; uses labels from other survey groups).
Fusion: the two one-photo pretrained models combined (score = mean of the two, 0 if missing).
Scored on the 97 grouped test photos; F1 CI resamples the 9 survey groups.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
import scale_estimators as SE  # noqa: E402
bnp = C2.bnp

OUT = Path(__file__).resolve().parent / 'results'
MODELS = ['pretrained_from2', 'pretrained_from1', 'scratch_mixed_styles']
SOURCE_PHOTO = {'pretrained_from1': (1,), 'pretrained_from2': (2,), 'scratch_mixed_styles': (1, 2)}
FIXED = (0.5, 0.35, 0.25)
GRID = (0.5, 0.35, 0.25, 0.18)
THRS = np.unique(np.r_[np.linspace(0.005, 0.95, 190), 0.001])


def predict_at(tag, row, f):
    im = C2.rf_img(row)
    f = float(f)
    if abs(f - 1) > 1e-3:
        w, h = max(32, round(row['width'] * f)), max(32, round(row['height'] * f))
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    else:
        w, h = row['width'], row['height']
    p = bnp.cached_predict(bnp.MODELS / f'{tag}.pt', f"rf_{row['id']}" + ('' if abs(f - 1) < 1e-3 else f'_f{f:.4f}'),
                           im, (0, 0, w, h))[0].copy()
    p[:, 0] *= row['width'] / w
    p[:, 1] *= row['height'] / h
    p[:, 2] *= row['width'] / w
    return p


def fuse(pa, pb):
    from scipy.spatial import cKDTree
    out = []
    for a, b in zip(pa, pb):
        if not len(a) or not len(b):
            q = a if len(a) else b
            out.append(np.column_stack([q[:, :3], q[:, 3] / 2]) if len(q) else q)
            continue
        tb = cKDTree(b[:, :2])
        used = np.zeros(len(b), bool)
        rows = []
        for i in np.argsort(-a[:, 3]):
            d, j = tb.query(a[i, :2])
            if d <= max(3.0, 0.5 * max(a[i, 2], b[j, 2])) and not used[j]:
                used[j] = True
                w = np.array([a[i, 3], b[j, 3]])
                xy = (a[i, :2] * w[0] + b[j, :2] * w[1]) / w.sum()
                rows.append([xy[0], xy[1], (a[i, 2] + b[j, 2]) / 2, w.mean()])
            else:
                rows.append([*a[i, :3], a[i, 3] / 2])
        for j in np.flatnonzero(~used):
            rows.append([*b[j, :3], b[j, 3] / 2])
        out.append(np.asarray(rows, np.float32))
    return out


def best_thr(preds, refs):
    best = (0.25, dict(f1=-1))
    for t in THRS:
        m = C2.pooled([p[p[:, 3] >= t] for p in preds], refs)
        if m['f1'] > best[1]['f1']:
            best = (float(t), m)
    return best


def score(preds, refs, groups, thr):
    tp = [p[p[:, 3] >= thr] for p in preds]
    return dict(threshold=thr, **C2.pooled(tp, refs), **C2.count_errors(tp, refs),
                f1_group_ci95=C2.group_bootstrap(tp, refs, groups))


def main():
    val, test = C2.rf_rows('val'), C2.rf_rows('test')
    val_r = [C2.rf_targets(r) for r in val]
    test_r = [C2.rf_targets(r) for r in test]
    groups = [r['group'] for r in test]
    se = json.loads((OUT / 'scale_estimators.json').read_text())
    est = se['summary']['chosen_estimator']
    fn = SE.log_scale if est == 'log' else SE.acf_spacing
    key = 'log_radius' if est == 'log' else 'acf_spacing'
    cache_s = OUT / 'rf_photo_scales.json'
    S = json.loads(cache_s.read_text()) if cache_s.exists() else {}
    for r in val + test:
        if r['id'] not in S:
            S[r['id']] = fn(C2.rf_img(r))
    C2.save(cache_s, S)
    D = C2.dali()
    out = dict(test_images=len(test), test_groups=len(set(groups)), test_targets=int(sum(len(g) for g in test_r)),
               calibration_images=len(val), scale_estimator=est)
    keep = {}
    for tag in MODELS:
        t0 = time.time()
        dthr = json.loads((bnp.SELECTIONS / f'{tag}_selection.json').read_text())['chosen']['threshold']
        res = dict(dali_threshold=dthr)
        # fixed global scale chosen on val at the DALI threshold
        vf = {}
        for f in FIXED:
            vp = [predict_at(tag, r, f) for r in val]
            vf[f] = (C2.pooled([p[p[:, 3] >= dthr] for p in vp], val_r)['f1'], vp)
        fstar = max(vf, key=lambda f: vf[f][0])
        res['fixed_scale_val_f1'] = {str(f): vf[f][0] for f in FIXED}
        tp = [predict_at(tag, r, fstar) for r in test]
        cal = best_thr(vf[fstar][1], val_r)[0]
        res['fixed_scale'] = dict(factor=fstar, dali_threshold=score(tp, test_r, groups, dthr),
                                  calibrated=score(tp, test_r, groups, cal),
                                  oracle_threshold_DIAGNOSTIC=best_thr(tp, test_r)[1]['f1'])
        # per-photo confidence-guided scale (label-free)
        def conf_pick(r):
            best = None
            for f in GRID:
                p = predict_at(tag, r, f)
                top = np.sort(p[:, 3])[-20:] if len(p) else np.zeros(1)
                sc = float(top.mean())
                if best is None or sc > best[0]:
                    best = (sc, f, p)
            return best[1], best[2]
        vc = [conf_pick(r) for r in val]
        tc = [conf_pick(r) for r in test]
        vpc, tpc = [x[1] for x in vc], [x[1] for x in tc]
        cal_c = best_thr(vpc, val_r)[0]
        res['confidence_scale'] = dict(factor_counts={str(f): int(sum(x[0] == f for x in tc)) for f in GRID},
                                       dali_threshold=score(tpc, test_r, groups, dthr),
                                       calibrated=score(tpc, test_r, groups, cal_c),
                                       oracle_threshold_DIAGNOSTIC=best_thr(tpc, test_r)[1]['f1'])
        keep_c = (vpc, tpc)
        # per-photo estimator-based scale (only if the estimator validated)
        s_dali = float(np.mean([se['summary'][f'dali_photo{k}'][key] for k in SOURCE_PHOTO[tag]]))
        fac = lambda r: float(np.clip(s_dali / S[r['id']], 0.12, 1.0))  # noqa: E731
        ok_est = se['summary']['roboflow_validation'][
            'spearman_log_vs_true_radius' if est == 'log' else 'spearman_acf_vs_true_nn'] >= 0.5
        if ok_est:
            vpa = [predict_at(tag, r, round(fac(r), 3)) for r in val]
            tpa = [predict_at(tag, r, round(fac(r), 3)) for r in test]
            cal_a = best_thr(vpa, val_r)[0]
            res['auto_scale'] = dict(s_dali=s_dali, factor_median=float(np.median([fac(r) for r in test])),
                                     dali_threshold=score(tpa, test_r, groups, dthr),
                                     calibrated=score(tpa, test_r, groups, cal_a),
                                     oracle_threshold_DIAGNOSTIC=best_thr(tpa, test_r)[1]['f1'])
        else:
            vpa, tpa = vpc, tpc
            res['auto_scale'] = 'skipped: scale estimator failed its validation (see scale_estimators.json)' 
        res['seconds'] = round(time.time() - t0)
        out[tag] = res
        keep[tag] = (vf[fstar][1], tp, keep_c[0], keep_c[1])
        print(f"{tag}: fixed f={fstar} (val F1 {res['fixed_scale_val_f1']}) test F1 dali-thr {res['fixed_scale']['dali_threshold']['f1']:.3f} "
              f"cal {res['fixed_scale']['calibrated']['f1']:.3f} oracle {res['fixed_scale']['oracle_threshold_DIAGNOSTIC']:.3f} | "
              f"confidence-scale {res['confidence_scale']['factor_counts']} dali-thr {res['confidence_scale']['dali_threshold']['f1']:.3f} "
              f"cal {res['confidence_scale']['calibrated']['f1']:.3f} | auto: "
              f"{res['auto_scale'] if isinstance(res['auto_scale'], str) else round(res['auto_scale']['dali_threshold']['f1'], 3)} "
              f"[{res['seconds']}s]", flush=True)
        C2.save(OUT / 'detectors_external.json', out)
    for mode, iv, it in (('fixed_scale', 0, 1), ('confidence_scale', 2, 3)):
        va = fuse(keep['pretrained_from1'][iv], keep['pretrained_from2'][iv])
        ta = fuse(keep['pretrained_from1'][it], keep['pretrained_from2'][it])
        cal = best_thr(va, val_r)[0]
        out[f'fusion_{mode}'] = dict(calibrated=score(ta, test_r, groups, cal),
                                     no_tuning_threshold_0p29=score(ta, test_r, groups, 0.29))
        print(f"fusion {mode}: cal {out[f'fusion_{mode}']['calibrated']['f1']:.3f} no-tuning {out[f'fusion_{mode}']['no_tuning_threshold_0p29']['f1']:.3f}", flush=True)
    C2.save(OUT / 'detectors_external.json', out)


if __name__ == '__main__':
    main()
