# Count barnacles in one photo.
# usage:
#   python scripts/predict.py --image data/unseen_img1.png               (corners looked up in data/frames.json)
#   python scripts/predict.py --image my.png --corners "[[x,y],[x,y],[x,y],[x,y]]"   (TL, TR, BR, BL inner corners)
#   add --cropped if the photo is already cropped to the inside of the frame (no corners needed)
# writes to --out: dots.csv (x, y, radius, confidence in the straightened 1024px frame), overlay.png, count.json
# NOTE: this gives a count, not an accuracy. you need hand labels to get an F1.
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import barnacle_counter as bc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--corners', help='JSON list of the 4 inner frame corners, TL TR BR BL')
    ap.add_argument('--cropped', action='store_true', help='photo is already just the inside of the frame')
    ap.add_argument('--model', default=str(ROOT / 'models/mixed_0.5_yolo11n.pt'))
    ap.add_argument('--threshold', type=float, default=json.loads((ROOT / 'models/default.json').read_text())['threshold'])
    ap.add_argument('--backend', default='numpy', help="'numpy' (no PyTorch) or 'ultralytics'")
    ap.add_argument('--out', default='output')
    a = ap.parse_args()

    img = bc.read_rgb(a.image)
    if a.cropped:
        # already the inside of the frame: just stretch it to 1024x1024 like the pipeline does.
        # careful: only right if the crop is the WHOLE frame. a zoomed-in piece will have barnacles too big.
        h, w = img.shape[:2]
        corners = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    elif a.corners:
        corners = json.loads(a.corners)
    else:
        corners = json.loads((ROOT / 'data/frames.json').read_text()).get(Path(a.image).name)
        if corners is None:
            sys.exit('no corners for this photo: pass --corners (or --cropped)')

    det = bc.Detector(a.model, backend=a.backend)
    rgb, dots = bc.count_photo(det, a.image, corners, a.threshold)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'dots.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'radius', 'confidence'])
        w.writerows(np.round(dots.astype(float), 3).tolist())
    # overlay: green circle per dot on the straightened frame
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    for x, y, r, _ in dots:
        dr.ellipse([x - r, y - r, x + r, y + r], outline=(0, 255, 0), width=1)
    im.save(out / 'overlay.png')
    (out / 'count.json').write_text(json.dumps(dict(
        image=str(a.image), count=int(len(dots)), threshold=a.threshold, model=Path(a.model).name,
        note='model count, not checked by a person. empty shells are not supposed to count but the model can still hit some.'),
        indent=1))
    print(f'{len(dots)} barnacles -> {out}/')


if __name__ == '__main__':
    main()
