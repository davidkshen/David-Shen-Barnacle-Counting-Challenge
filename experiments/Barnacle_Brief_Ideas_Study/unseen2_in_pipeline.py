"""unseen_img2 is an enlarged crop of unseen_img1 (found by SIFT matching after the first unseen_img2 tests).
So the faithful test of the deployed pipeline is: straighten unseen_img1 from its four frame corners
(frames.json), run each model once on the whole quadrat, and score the dots that fall inside the crop
region against the hand marks (red_marks.json) mapped into the same coordinates.

Mapping: similarity transform unseen_img2 -> unseen_img1 fitted on SIFT matches (RANSAC, fixed seed),
then the frame homography unseen_img1 -> 1024 x 1024 (identical to barnacles.core.rectify).
Matching: the same two rules as on unseen_img2, scaled into rectified pixels:
  same barnacle  min(15, half the distance to the nearest other mark) unseen_img2 px  (added after looking)
  strict         8 unseen_img2 px                                                   (declared beforehand)
Dots count if their centre lies inside the crop region; nothing else is tuned.
"""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
bnp = C2.bnp
W3 = str(Path(__file__).resolve().parent / 'results') + '/'
MP = __import__('os').environ.get('BARNACLE_MAIN_MODEL', str(Path(__file__).resolve().parents[2] / 'models/mixed_0.5_yolo11n.pt'))
THR_MAIN = json.load(open(__import__('os').environ.get('BARNACLE_MAIN_CONFIG', str(Path(__file__).resolve().parents[2] / 'models/default.json'))))['threshold']
TT = __import__('os').environ['BARNACLE_THREE_TRACK'] + '/results/'
MODELS = ['pretrained_from1', 'pretrained_from2', 'pretrained_adapt_to1', 'pretrained_adapt_to2', 'scratch_mixed_styles',
          'scratch_from1', 'scratch_from2', 'scratch_adapt_to1', 'scratch_adapt_to2']


def crop_transform():
    cv2.setRNGSeed(0)
    a = cv2.imread(str(bnp.EXTRA_DATA / 'unseen_img2.png'), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(bnp.EXTRA_DATA / 'unseen_img1.png'), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    src = np.float32([ka[g.queryIdx].pt for g in good])
    dst = np.float32([kb[g.trainIdx].pt for g in good])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    res = np.linalg.norm(src[inl.ravel() == 1] @ A[:, :2].T + A[:, 2] - dst[inl.ravel() == 1], axis=1)
    return A, int(inl.sum()), len(good), float(np.median(res))


def main():
    A, n_in, n_good, med_res = crop_transform()
    scale = float(np.hypot(A[0, 0], A[1, 0]))
    frames = json.load(open(bnp.DATA / 'frames.json'))
    Hr = cv2.getPerspectiveTransform(np.float32(frames['unseen_img1.png']), np.float32([[0, 0], [1023, 0], [1023, 1023], [0, 1023]]))
    to_rect = lambda xy: cv2.perspectiveTransform((np.asarray(xy, np.float32) @ A[:, :2].T + A[:, 2]).astype(np.float32).reshape(-1, 1, 2), Hr).reshape(-1, 2).astype(np.float32)  # noqa: E731
    M = json.load(open(W3 + 'red_marks.json'))
    P2 = np.array([[d['x'], d['y']] for d in M], np.float32)
    nn = cKDTree(P2).query(P2, 2)[0][:, 1]
    P = to_rect(P2)
    region = to_rect([[0, 0], [499, 0], [499, 499], [0, 499]])
    s = float(np.sqrt(cv2.contourArea(region) / (499.0 ** 2)))  # rectified px per unseen_img2 px
    tol_same = np.minimum(15, 0.5 * nn) * s
    im = bnp.rectified('unseen_img1.png')
    out = dict(reference_marks=int(len(P)), crop_in_unseen_img1=dict(similarity=A.tolist(), scale_unseen2_to_unseen1=scale, sift_matches=n_good, inliers=n_in,
                                        median_residual_px=med_res),
               rectified_px_per_unseen2_px=s, region_rectified=region.tolist(),
               region_share_of_quadrat=float(cv2.contourArea(region) / 1024 ** 2), models={})
    for tag in ['main_model'] + MODELS:
        if tag == 'main_model':
            p, sc, _ = bnp.cached_predict(MP, 'unseen_img1', im)
            thr = THR_MAIN
        else:
            p, sc, _ = bnp.cached_predict(bnp.MODELS / f'{tag}.pt', 'unseen_img1', im)
            thr = json.load(open(TT + f'{tag}_selection.json'))['chosen']['threshold'] if tag != 'scratch_mixed_styles' \
                else json.load(open(TT + 'mixed_scratch_summary.json'))['threshold']
        q = p[sc >= thr]
        inside = np.array([cv2.pointPolygonTest(region.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in q[:, :2]], bool)
        qi = q[inside]
        a = bnp.metrics(qi, np.column_stack([P, tol_same]).astype(np.float32), tolerance=1.0)
        b = bnp.metrics(qi, np.column_stack([P, np.full(len(P), 8 * s)]).astype(np.float32), tolerance=1.0)
        out['models'][tag] = dict(threshold=thr, whole_quadrat_count=int(len(q)), region_count=int(len(qi)),
                                  same_barnacle=dict(tp=a['tp'], fp=a['fp'], fn=a['fn'], f1=a['f1']),
                                  strict=dict(tp=b['tp'], fp=b['fp'], fn=b['fn'], f1=b['f1']))
        print(f"{tag:22s} quadrat {len(q):5d} | region {len(qi):3d} dots vs {len(P)} marks | same-barnacle {a['tp']}/{a['fp']}/{a['fn']} F1 {a['f1']:.3f} | strict F1 {b['f1']:.3f}", flush=True)
        if tag == 'main_model':
            np.save(W3 + 'pipeline_main_region_dots.npy', qi)
            mt = bnp.match_objects(qi, np.column_stack([P, tol_same]).astype(np.float32), tolerance=1.0)
            missed = np.setdiff1d(np.arange(len(P)), mt[:, 1])
            edge_d = [abs(cv2.pointPolygonTest(region.reshape(-1, 1, 2), (float(P[i, 0]), float(P[i, 1])), True)) for i in missed]
            out['models'][tag]['missed_within_5px_of_region_edge'] = int(sum(d <= 5 for d in edge_d))
    import pickle
    import probe as PR
    import classical_cv as CV

    def score_extra(tag, q):
        inside = np.array([cv2.pointPolygonTest(region.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in q[:, :2]], bool)
        qi = q[inside]
        a = bnp.metrics(qi, np.column_stack([P, tol_same]).astype(np.float32), tolerance=1.0)
        b = bnp.metrics(qi, np.column_stack([P, np.full(len(P), 8 * s)]).astype(np.float32), tolerance=1.0)
        out['models'][tag] = dict(whole_quadrat_count=int(len(q)), region_count=int(len(qi)),
                                  same_barnacle=dict(tp=a['tp'], fp=a['fp'], fn=a['fn'], f1=a['f1']),
                                  strict=dict(tp=b['tp'], fp=b['fp'], fn=b['fn'], f1=b['f1']))
        print(f"{tag:22s} quadrat {len(q):5d} | region {len(qi):3d} | same-barnacle {a['tp']}/{a['fp']}/{a['fn']} F1 {a['f1']:.3f} | strict F1 {b['f1']:.3f}", flush=True)
    d = pickle.load(open(W3 + 'probe_rf_hgb.pkl', 'rb'))
    mp = PR.prob_maps(im, (0, 0, 1024, 1024), {'hgb': d['clf']}, (d['mu'], d['sd']))
    pk = PR.peaks(mp['hgb'], (0, 0, 1024, 1024), d['spacing'])
    score_extra('probe_roboflow', pk[pk[:, 3] >= d['threshold']])
    cfg = json.load(open(str(Path(__file__).resolve().parent / 'results') + '/classical_cv.json'))
    for src in ('dali_1to2', 'dali_2to1'):
        c = cfg[src]
        h = CV.hough_candidates(im, c['radius_range'][0], c['radius_range'][1], c['hough']['setting']['param1'])
        score_extra('hough_tuned_on_photo' + src[5], h[h[:, 3] >= c['hough']['threshold']])
    np.save(W3 + 'pipeline_marks_rectified.npy', np.column_stack([P, tol_same]))
    print('crop: scale', round(scale, 4), 'inliers', n_in, '/', n_good, 'median residual', round(med_res, 2), 'px; rectified px per unseen2 px', round(s, 4))
    json.dump(out, open(W3 + 'unseen2_in_pipeline.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
