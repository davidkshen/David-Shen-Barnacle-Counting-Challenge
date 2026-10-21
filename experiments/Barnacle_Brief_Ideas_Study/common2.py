"""Shared helpers for the brief-ideas study: DALI cases, Roboflow benchmark, pooled metrics.

Expected layout (override with environment variables):
  <parent>/Barnacle_Brief_Ideas_Study/      (this folder)
  <parent>/Barnacle_Improvement_Study/      BARNACLE_IMPROVEMENT  (NumPy YOLO, common.py)
  <parent>/Barnacle_Three_Track_Study/      BARNACLE_THREE_TRACK  (saved models, DALI data)
  <parent>/Barnacle_Assistant/              BARNACLE_ASSISTANT    (unseen photos)
  Roboflow export folders (Archive*, Archive*_label*) in BARNACLE_RF_EXPORT
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('BARNACLE_THREE_TRACK', str(ROOT.parent / 'Barnacle_Three_Track_Study'))
os.environ.setdefault('BARNACLE_ASSISTANT', str(ROOT.parent / 'Barnacle_Assistant'))
sys.path.insert(0, os.environ.get('BARNACLE_IMPROVEMENT', str(ROOT.parent / 'Barnacle_Improvement_Study')))
sys.path.insert(0, str(ROOT))
import common as bnp  # noqa: E402,F401  (paths, NumPy YOLO, metrics)
import rf_bench as B  # noqa: E402

metrics = bnp.metrics


def dali():
    return bnp.cases()


def rf_rows(split=None):
    return B.load(split)


def rf_img(row):
    return B.read(row)


def rf_targets(row):
    return B.targets(row)[0]


def rf_others(row):
    return B.targets(row)[1]


def pooled(preds, refs):
    tp = fp = fn = 0
    for p, g in zip(preds, refs):
        p = np.asarray(p, np.float32)
        m = metrics(p.reshape(-1, 4) if p.ndim == 1 else p, g)
        tp, fp, fn = tp + m['tp'], fp + m['fp'], fn + m['fn']
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                f1=2 * tp / max(1, 2 * tp + fp + fn))


def count_errors(preds, refs):
    e = np.array([len(p) - len(g) for p, g in zip(preds, refs)], float)
    n = np.array([len(g) for g in refs], float)
    return dict(mean_abs_per_image=float(np.abs(e).mean()), weighted_abs_pct=float(100 * np.abs(e).sum() / max(1, n.sum())),
                signed_total_pct=float(100 * e.sum() / max(1, n.sum())))


def group_bootstrap(preds, refs, groups, reps=2000, seed=0):
    """Percentile CI of pooled F1 resampling capture groups (photos within a group move together)."""
    rng = np.random.default_rng(seed)
    per = []
    for p, g in zip(preds, refs):
        m = metrics(np.asarray(p, np.float32), g)
        per.append((m['tp'], m['fp'], m['fn']))
    per = np.array(per, float)
    ug = sorted(set(groups))
    idx = {u: [i for i, x in enumerate(groups) if x == u] for u in ug}
    f = []
    for _ in range(reps):
        pick = rng.choice(len(ug), len(ug), replace=True)
        rows = np.concatenate([idx[ug[i]] for i in pick])
        tp, fp, fn = per[rows].sum(0)
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]


def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=bnp._np) + '\n')
