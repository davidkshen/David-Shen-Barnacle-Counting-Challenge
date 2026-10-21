"""Overlay for the normal-pipeline test: the straightened unseen_img1 around the marked region, with the main
model's dots (green = on a marked barnacle, red X = extra) and missed marks (yellow ring), same-barnacle rule."""
import json
import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common2 as C2  # noqa: E402
bnp = C2.bnp
W3 = str(Path(__file__).resolve().parent / 'results') + '/'


def main():
    d = json.load(open(W3 + 'unseen2_in_pipeline.json'))
    reg = np.array(d['region_rectified'], np.float32)
    q = np.load(W3 + 'pipeline_main_region_dots.npy')
    ref = np.load(W3 + 'pipeline_marks_rectified.npy').astype(np.float32)
    m = bnp.match_objects(q, ref, 1.0)
    mp, mr = set(m[:, 0].tolist()), set(m[:, 1].tolist())
    im = bnp.rectified('unseen_img1.png')
    x0, y0 = np.floor(reg.min(0)).astype(int) - 6
    x1, y1 = np.ceil(reg.max(0)).astype(int) + 6
    S = 3
    crop = cv2.resize(im[y0:y1, x0:x1], None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)[:, :, ::-1].copy()
    cv2.polylines(crop, [((reg - [x0, y0]) * S).astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 2, cv2.LINE_AA)
    for j, r in enumerate(ref):
        if j not in mr:
            cv2.circle(crop, (int((r[0] - x0) * S), int((r[1] - y0) * S)), 16, (0, 230, 255), 3, cv2.LINE_AA)
    for i, p in enumerate(q):
        c = (int((p[0] - x0) * S), int((p[1] - y0) * S))
        if i in mp:
            cv2.circle(crop, c, 6, (60, 200, 60), -1, cv2.LINE_AA)
        else:
            cv2.drawMarker(crop, c, (40, 40, 240), cv2.MARKER_TILTED_CROSS, 14, 3, cv2.LINE_AA)
    bar = np.full((70, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, 'Main model inside unseen_img1 (normal pipeline), marked area', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(bar, f'{len(mp)} found   {len(q) - len(mp)} extra   {len(ref) - len(mr)} missed   (dots: {len(q)}, marks: {len(ref)})',
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(W3 + 'unseen_img1_region_main_model.jpg', np.vstack([bar, crop]), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == '__main__':
    main()
