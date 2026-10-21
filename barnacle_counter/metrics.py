import numpy as np
from scipy.optimize import linear_sum_assignment


def match_objects(pred, ref, tolerance=.75):
    # pair dots with barnacles one-to-one. a pair only counts if the dot is within tolerance * radius
    # of the barnacle (min 3 px). first maximise the number of pairs, then prefer the closer ones.
    # a second dot on the same barnacle doesn't get a pair -> it's an extra (false positive).
    p, g = len(pred), len(ref)
    if not p or not g:
        return np.empty((0, 2), int)
    dist = np.linalg.norm(pred[:, None, :2] - ref[None, :, :2], axis=2)
    limit = np.maximum(3., tolerance * ref[:, 2])
    norm = dist / limit[None, :]
    valid = norm <= 1.
    reward = min(p, g) + 1  # big bonus per valid pair so 'more pairs' always wins over 'shorter distance'
    cost = np.where(valid, norm - reward, 0.)
    i, j = linear_sum_assignment(cost)
    keep = valid[i, j]
    return np.column_stack((i[keep], j[keep]))


def metrics(pred, ref, tolerance=.75):
    # precision = share of dots that are on a barnacle, recall = share of barnacles that got a dot,
    # F1 = both in one number. NOTE: F1 is not count accuracy, extras and misses can cancel in the total.
    m = match_objects(np.asarray(pred, np.float32).reshape(-1, 4) if len(pred) else np.zeros((0, 4), np.float32),
                      np.asarray(ref, np.float32), tolerance)
    tp = len(m)
    fp, fn = len(pred) - tp, len(ref) - tp
    return dict(reference_count=len(ref), predicted_count=len(pred), tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn) if tp + fn else 0.,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                signed_error=len(pred) - len(ref))


def count_error(pred_counts, ref_counts):
    # total |model count - true count| over all photos, as % of all true barnacles
    e = np.asarray(pred_counts, float) - np.asarray(ref_counts, float)
    return float(100 * np.abs(e).sum() / max(1, np.sum(ref_counts)))
