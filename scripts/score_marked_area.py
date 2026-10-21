# Scores a model on the part of unseen_img1 I counted by hand (= the unseen_img2 crop, 135 barnacles).
# Runs the normal pipeline on the WHOLE unseen_img1 (straighten frame -> tiles -> model), then only
# looks at dots inside the crop area. Expected for the main model: 130 dots, F1 0.966 (0.928 strict).
# usage: python scripts/score_marked_area.py [--model models/photo2_only_pretrained.pt --threshold 0.33]
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import barnacle_counter as bc  # noqa: E402
from barnacle_counter import marks as mk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(ROOT / 'models/mixed_0.5_yolo11n.pt'))
    ap.add_argument('--threshold', type=float, default=json.loads((ROOT / 'models/default.json').read_text())['threshold'])
    ap.add_argument('--marks', default=str(ROOT / 'data/unseen_img2_marked_135.png'))
    ap.add_argument('--backend', default='numpy')
    a = ap.parse_args()
    frames = json.loads((ROOT / 'data/frames.json').read_text())
    # 1. my hand marks, in crop pixels
    marks = mk.read_marks(a.marks, ROOT / 'data/unseen_img2.png')
    tol = mk.same_barnacle_tolerance(marks)
    # 2. where the crop sits inside unseen_img1, then inside the straightened frame
    A, fit = mk.find_crop(ROOT / 'data/unseen_img2.png', ROOT / 'data/unseen_img1.png')
    rgb, M = bc.rectify(bc.read_rgb(ROOT / 'data/unseen_img1.png'), frames['unseen_img1.png'])
    marks_f = mk.crop_to_frame(marks, A, M)
    area = mk.crop_to_frame([[0, 0], [499, 0], [499, 499], [0, 499]], A, M)
    s = float(np.sqrt(cv2.contourArea(area) / 499.0 ** 2))  # frame px per crop px (about 0.65)
    # 3. normal pipeline on the whole photo, keep dots whose centre is inside the hand-counted area
    det = bc.Detector(a.model, backend=a.backend)
    d = det.dots(rgb)
    d = d[d[:, 3] >= a.threshold]
    di = d[mk.inside(area, d[:, :2])]
    # 4. score: 'same barnacle' rule, and the stricter 8-crop-px rule that was set before looking
    same = bc.metrics(di, np.column_stack([marks_f, tol * s]), tolerance=1.0)
    strict = bc.metrics(di, np.column_stack([marks_f, np.full(len(marks_f), 8 * s)]), tolerance=1.0)
    print(f"crop found in unseen_img1: scale {fit['scale']:.4f}, {fit['inliers']} good matches, "
          f"median error {fit['median_residual_px']:.2f} px")
    print(f"whole frame: {len(d)} dots | marked area: {len(di)} dots vs {len(marks)} hand marks")
    print(f"same-barnacle rule: {same['tp']} found / {same['fp']} extra / {same['fn']} missed, F1 {same['f1']:.3f}")
    print(f"strict 8px rule:    F1 {strict['f1']:.3f}")
    out = dict(model=Path(a.model).name, threshold=a.threshold, marks=len(marks), crop_fit=fit,
               whole_frame_dots=int(len(d)), area_dots=int(len(di)), same_barnacle=same, strict=strict)
    (ROOT / 'results' / f'marked_area_{Path(a.model).stem}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
