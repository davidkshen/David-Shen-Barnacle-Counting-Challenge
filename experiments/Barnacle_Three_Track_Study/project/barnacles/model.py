"""Train on one photograph, set thresholds on a separated source region."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from .core import (features, normalized_gray, select_detections, metrics,
                   gradient_patches, rotate_gradient_descriptor)


def fit_classifier(X, y, seed=42):
    keep = y >= 0
    yy = y[keep]
    if len(np.unique(yy)) != 2:
        raise ValueError('Both positive and negative source examples are required.')
    # Balanced weights prevent thousands of easy negatives dominating training.
    weights = np.where(yy == 1, len(yy)/(2*(yy == 1).sum()),
                       len(yy)/(2*(yy == 0).sum()))
    clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=9,
            learning_rate=.07, min_samples_leaf=25, l2_regularization=2.,
            early_stopping=False, random_state=seed)
    clf.fit(X[keep], yy, sample_weight=weights)
    return clf


def fit_patch_classifier(X, H, proposals, y, seed=42):
    keep = y >= 0
    # Augment ONLY source training examples. Siblings never cross the split.
    augmented = np.concatenate([fused_features(X[keep], H[keep], proposals[keep], q)
                                for q in range(4)])
    target = np.tile(y[keep], 4)
    if len(np.unique(target)) != 2:
        raise ValueError('Both target classes are required.')
    clf = ExtraTreesClassifier(n_estimators=160, max_depth=16,
            min_samples_leaf=3, max_features=.2, class_weight='balanced',
            n_jobs=2, random_state=seed)
    clf.fit(augmented, target)
    return clf


def fused_features(X, H, proposals, q=0):
    return np.column_stack((X, rotate_gradient_descriptor(H, q), np.log(proposals[:, 2])))


def predict_patch(clf, X, H, proposals):
    # Average four rotations at prediction time as well as training time.
    return np.mean([clf.predict_proba(fused_features(X, H, proposals, q))[:, 1]
                    for q in range(4)], axis=0)


def patch_stability_scores(rgb, proposals, clf, base_X=None, base_H=None):
    if base_X is None:
        gray = normalized_gray(rgb)
        base_X, _ = features(gray, proposals)
        base_H = gradient_patches(gray, proposals)
    values = [predict_patch(clf, base_X, base_H, proposals)]
    for gamma in (.75, 1.35):
        altered = np.rint(255*(rgb.astype(np.float32)/255)**gamma).astype(np.uint8)
        g = normalized_gray(altered); xx, _ = features(g, proposals)
        values.append(predict_patch(clf, xx, gradient_patches(g, proposals), proposals))
    rotated = np.ascontiguousarray(np.rot90(rgb))
    rp = proposals.copy(); rp[:, 0] = proposals[:, 1]; rp[:, 1] = rgb.shape[1]-1-proposals[:, 0]
    g = normalized_gray(rotated); xx, _ = features(g, rp)
    values.append(predict_patch(clf, xx, gradient_patches(g, rp), rp))
    v = np.stack(values, 1)
    return np.clip(v.mean(1)-v.std(1), 0, 1), v


def stability_scores(rgb, proposals, clf, columns=None, base_X=None):
    """Score the SAME proposed centers under controlled nuisance transforms.

    Conditional robustness: proposals are not regenerated here. Independent
    whole-pipeline perturbation tests are run separately in the experiment.
    The conservative mean-minus-SD score is a heuristic, NOT a probability.
    """
    def score(g, p):
        xx, _ = features(g, p)
        return clf.predict_proba(xx if columns is None else xx[:, columns])[:, 1]
    values = []
    if base_X is None:
        values.append(score(normalized_gray(rgb), proposals))
    else:
        values.append(clf.predict_proba(base_X if columns is None else base_X[:, columns])[:, 1])
    for gamma in (.75, 1.35):
        altered = np.rint(255*(rgb.astype(np.float32)/255)**gamma).astype(np.uint8)
        values.append(score(normalized_gray(altered), proposals))
    rotated = np.ascontiguousarray(np.rot90(rgb))
    rp = proposals.copy()
    rp[:, 0] = proposals[:, 1]
    rp[:, 1] = rgb.shape[1]-1-proposals[:, 0]
    values.append(score(normalized_gray(rotated), rp))
    v = np.stack(values, 1)
    return np.clip(v.mean(1)-v.std(1), 0, 1), v


def choose_threshold(proposals, scores, reference, mode='balanced'):
    """Use only source validation labels. Include proposal misses in recall.

    Balanced: maximize object F1.
    Conservative: maximize recall subject to empirical precision >=90% and
    >=20 detections. If infeasible, maximize F0.5 and report the fallback.
    Neither policy guarantees its source precision on a new photograph.
    """
    thresholds = np.unique(np.r_[.02, np.quantile(scores, np.linspace(.1, .999, 35)), .99])
    rows = []
    for threshold in thresholds:
        ids = select_detections(proposals, scores, threshold)
        m = metrics(proposals[ids], reference)
        m['threshold'] = float(threshold)
        p, r = m['precision'], m['recall']
        m['f05'] = 1.25*p*r/(.25*p+r) if p+r else 0.
        rows.append(m)
    fallback = False
    if mode == 'balanced':
        selected = max(rows, key=lambda m:(m['f1'],m['precision']))
    else:
        valid = [m for m in rows if m['precision'] >= .90 and m['predicted_count'] >= 20]
        fallback = not bool(valid)
        selected = max(valid, key=lambda m:(m['recall'],m['precision'])) if valid else max(rows, key=lambda m:m['f05'])
    return selected['threshold'], dict(selected,
            precision_target_feasible=(not fallback if mode != 'balanced' else None)), rows


def conditional_occlusion(gray, proposal, clf, patch_radius=48, cell=9, stride=8):
    """Local score-change map on normalized luminance, holding center/radius fixed.

    Replacing cells with 0 is a diagnostic intervention, possibly out of
    distribution. This does NOT explain proposal generation or establish human
    alignment, and it is not the ring/slit evidence map itself.
    """
    from .core import sample
    offsets = np.arange(-patch_radius, patch_radius+1)
    yy, xx = np.meshgrid(offsets, offsets, indexing='ij')
    crop = sample(gray, xx+proposal[0], yy+proposal[1]).astype(np.float32)
    p = proposal.copy(); p[:2] = patch_radius
    X, _ = features(crop, p[None, :])
    baseline = predict_patch(clf, X, gradient_patches(crop, p[None, :]), p[None, :])[0]
    impact = np.zeros_like(crop); weight = np.zeros_like(crop)
    for y in range(0, crop.shape[0]-cell+1, stride):
        for x in range(0, crop.shape[1]-cell+1, stride):
            altered = crop.copy(); altered[y:y+cell, x:x+cell] = 0.
            f, _ = features(altered, p[None, :])
            value = predict_patch(clf, f, gradient_patches(altered, p[None, :]), p[None, :])[0]
            delta = baseline-value
            impact[y:y+cell, x:x+cell] += delta; weight[y:y+cell, x:x+cell] += 1
    return crop, impact/np.maximum(weight, 1), float(baseline)
