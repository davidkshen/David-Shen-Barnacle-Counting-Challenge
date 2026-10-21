"""Build an offline, timed review page for one rectified quadrat.

Two modes:
  pilot : N random tiles split equally among two or more conditions (A, B, C...),
          each showing model markers at one confidence threshold, or none
          ('manual': counting from scratch). Assignment is random and unlabelled.
          Measures whether a recall-leaning operating point reduces correction
          time. Use on a photo with reference labels (img1/img2) so accuracy
          after review can also be scored.
  audit : n uniformly random tiles (drawn AFTER the model and threshold are
          frozen) at one threshold. Gives (i) a corrected whole-quadrat count
          with an approximate 95% interval, and (ii) the frozen model's
          precision/recall inside the audited tiles of a genuinely new photo.

Inference uses the PyTorch-free NumPy implementation (npyolo), which reproduces
Ultralytics predictions for the saved checkpoints (results/numpy_validation.json).

Examples
  python review/make_review_kit.py --mode pilot --image img1.png --model pretrained_from2.pt \
      --conditions 0.33 0.07 manual --tiles 18 --out review/kits/pilot_photo1.html
  python review/make_review_kit.py --mode audit --image unseen_img1.png --model PATH/mixed_0.5_yolo11n.pt \
      --conditions 0.37 --tiles 16 --blind --out review/kits/audit_unseen1.html
"""
import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as bnp  # noqa: E402
from barnacles.audit import tile_counts  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pilot', 'audit'], required=True)
    ap.add_argument('--image', required=True, help='file name inside the data folder, or a path')
    ap.add_argument('--corners', help='JSON [[TLx,TLy],[TRx,TRy],[BRx,BRy],[BLx,BLy]]; default: frames.json')
    ap.add_argument('--model', required=True)
    ap.add_argument('--conditions', nargs='+', required=True,
                    help="pilot: two or more of <threshold>|manual; audit: the frozen threshold")
    ap.add_argument('--blind', action='store_true', help='audit: hide model markers (independent counts)')
    ap.add_argument('--tiles', type=int, default=16)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    img_path = Path(a.image) if Path(a.image).exists() else bnp.find_image(a.image)
    if a.corners:
        corners = json.loads(a.corners)
    else:
        corners = json.loads((bnp.DATA / 'frames.json').read_text()).get(img_path.name)
        if corners is None:
            raise SystemExit('No frame corners known for this image: pass --corners (inside corners, clockwise from top left).')
    rgb, _ = bnp.rectify(bnp.read_rgb(img_path), corners)
    model_path = Path(a.model) if Path(a.model).exists() else bnp.MODELS / a.model
    p, s, _ = bnp.tile_predict_np(bnp.model(model_path), rgb)
    rng = np.random.default_rng(a.seed)
    if a.mode == 'pilot':
        conds = a.conditions
        if len(conds) < 2:
            raise SystemExit('pilot mode needs >=2 conditions, e.g. --conditions 0.33 0.10 manual')
        if a.tiles % len(conds):
            raise SystemExit(f'--tiles must be a multiple of the number of conditions ({len(conds)})')
        ids = rng.choice(64, a.tiles, replace=False)
        labels = [chr(65 + i) for i in range(len(conds))]
        assign = np.repeat(np.arange(len(conds)), a.tiles // len(conds))[rng.permutation(a.tiles)]
        tiles = [dict(id=int(i), condition=labels[c], threshold=None if conds[c] == 'manual' else float(conds[c]))
                 for i, c in zip(ids, assign)]
        intro = ('<p>This is a <b>timed pilot</b>. Tiles show model suggestions at different settings, '
                 'or none at all (count from scratch). You are not told which setting a tile uses. '
                 'Correct or count each tile completely.</p>')
        sampling = dict(design=f'pilot: {a.tiles} random tiles, {len(conds)} conditions randomly assigned equally',
                        seed=a.seed, tile_ids=[int(i) for i in ids],
                        conditions={l: c for l, c in zip(labels, conds)})
        mtc = None
    else:
        thr = float(a.conditions[0])
        ids = rng.choice(64, a.tiles, replace=False)
        tiles = [dict(id=int(i), condition='audit', threshold=None if a.blind else thr) for i in ids]
        intro = ('<p>This is a <b>random audit</b>. The model and threshold were frozen before these tiles were '
                 'drawn. Count every target in each tile' + (' from scratch (no model markers are shown).' if a.blind
                 else ', including ones the model missed.') + '</p>')
        sampling = dict(design='uniform random tiles without replacement, drawn after model freeze', seed=a.seed,
                        tile_ids=[int(i) for i in ids], total_tiles=64, blind=bool(a.blind), audit_threshold=thr)
        keep = s >= thr
        mtc = tile_counts(np.clip(p[keep, :2], 0, 1023.999)).tolist()
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=92)
    mhash = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    numeric = [float(c) for c in a.conditions if c != 'manual']
    floor = min(numeric) if numeric else 1.0
    data = dict(name=f'{a.mode.title()} review · {img_path.name}', mode=a.mode, image_name=img_path.name,
                corners=corners, model=dict(file=model_path.name, sha256_12=mhash, conditions=a.conditions),
                image='data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(),
                predictions=np.round(p[s >= floor], 3).tolist(), tiles=tiles,
                model_tile_counts=mtc, random_sampling=sampling, show_condition=False, intro=intro,
                scope=('Counts aperture targets as in the supplied masks; no live/dead judgement. '
                       'Frame corners were supplied manually.'),
                key=hashlib.sha256((img_path.name + mhash + a.mode + str(a.seed) + str(a.conditions) + str(a.blind)).encode()).hexdigest()[:12])
    html = (HERE / 'timed_review_template.html').read_text().replace('__DATA__', json.dumps(data))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print(json.dumps(dict(out=a.out, tiles=[t['id'] for t in tiles], conditions=[t['condition'] for t in tiles])))


if __name__ == '__main__':
    main()
