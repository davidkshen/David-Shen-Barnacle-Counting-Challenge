# Re-checks the main model's headline numbers from scratch:
#   - F1 on the held-out right strips (x >= 832) of img1 and img2 -> should be 0.941
#   - number of dots on unseen_img1 -> should be 1,143
# usage: python scripts/check_main_model.py   (add --backend ultralytics if you have it)
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import barnacle_counter as bc  # noqa: E402

STRIP = (832, 0, 1024, 1024)  # the slice of each photo the model never trained on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='numpy')
    a = ap.parse_args()
    cfg = json.loads((ROOT / 'models/default.json').read_text())
    thr = cfg['threshold']  # 0.37, picked on the validation strip during training
    det = bc.Detector(ROOT / 'models/mixed_0.5_yolo11n.pt', backend=a.backend)
    frames = json.loads((ROOT / 'data/frames.json').read_text())
    tot = dict(tp=0, fp=0, fn=0)
    rows = []
    for k in (1, 2):
        rgb, M = bc.rectify(bc.read_rgb(ROOT / f'data/img{k}.png'), frames[f'img{k}.png'])
        ref = bc.load_reference(ROOT / f'data/mask{k}.png', M)
        ref = ref[bc.in_box(ref, STRIP)]
        # same as the original report: run the tiles on the strip only, then threshold
        d = det.dots(rgb, STRIP)
        m = bc.metrics(d[d[:, 3] >= thr], ref)
        rows.append(dict(photo=k, **m))
        for key in tot:
            tot[key] += m[key]
        print(f"photo {k}: {m['reference_count']} barnacles, {m['predicted_count']} dots, "
              f"{m['tp']} found / {m['fp']} extra / {m['fn']} missed, F1 {m['f1']:.3f}")
    f1 = 2 * tot['tp'] / (2 * tot['tp'] + tot['fp'] + tot['fn'])
    print(f'both strips: F1 {f1:.3f}  (reported: 0.941)')
    t0 = time.time()
    _, dots = bc.count_photo(det, ROOT / 'data/unseen_img1.png', frames['unseen_img1.png'], thr)
    print(f'unseen_img1: {len(dots)} dots in {time.time() - t0:.1f} s  (reported: 1,143)')
    out = dict(threshold=thr, strips=rows, pooled_f1=f1, unseen_img1_dots=int(len(dots)))
    (ROOT / 'results/check_main_model.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
