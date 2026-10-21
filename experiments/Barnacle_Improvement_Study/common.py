"""Shared paths and helpers for the improvement study.

Layout expected (override with environment variables):
  <parent>/Barnacle_Improvement_Study/   (this package)
  <parent>/Barnacle_Three_Track_Study/   BARNACLE_THREE_TRACK  (models, selections, project/)
  <parent>/Barnacle_Assistant/           BARNACLE_ASSISTANT    (extra data: unseen photos)

tile_predict_np mirrors barnacles.modern.tile_predict line for line, replacing
only the Ultralytics call with npyolo.yolo_np.Model.predict.
"""
import json
import os
import sys
import hashlib
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
THREE_TRACK = Path(os.environ.get('BARNACLE_THREE_TRACK', ROOT.parent / 'Barnacle_Three_Track_Study'))
ASSISTANT = Path(os.environ.get('BARNACLE_ASSISTANT', ROOT.parent / 'Barnacle_Assistant'))
PROJECT = THREE_TRACK / 'project'
DATA = PROJECT / 'data'
EXTRA_DATA = ASSISTANT / 'data'
MODELS = THREE_TRACK / 'models'
SELECTIONS = THREE_TRACK / 'results'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
CACHE = ROOT / 'cache'
for _p in (RESULTS, FIGURES, CACHE):
    _p.mkdir(exist_ok=True)
if not PROJECT.exists():
    raise SystemExit(f'Cannot find {PROJECT}. Place this folder next to Barnacle_Three_Track_Study '
                     'or set BARNACLE_THREE_TRACK.')
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / 'npyolo'))
from barnacles.modern import prepare_cases, windows, choose_confidence  # noqa: E402
from barnacles.core import metrics, match_objects, rectify, read_rgb  # noqa: E402
from yolo_np import Model  # noqa: E402

_CASES = None


def find_image(name):
    for d in (DATA, EXTRA_DATA):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def cases():
    global _CASES
    if _CASES is None:
        _CASES = prepare_cases(DATA)
    return _CASES


def rectified(name):
    frames = json.loads((DATA / 'frames.json').read_text())
    im, _ = rectify(read_rgb(find_image(name)), frames[name])
    return im


def tile_predict_np(model, rgb, bounds=(0, 0, 1024, 1024), imgsz=512, conf=.001):
    ws = windows(bounds)
    imgs = [rgb[w['crop'][1]:w['crop'][3], w['crop'][0]:w['crop'][2], ::-1].copy() for w in ws]
    results = model.predict(imgs, imgsz=imgsz, conf=conf, iou=.5, max_det=2000, batch=4)
    bb, scores = [], []
    for w, (b, s) in zip(ws, results):
        if not len(s):
            continue
        b = b + np.array([w['crop'][0], w['crop'][1], w['crop'][0], w['crop'][1]], np.float32)
        xy = (b[:, :2] + b[:, 2:]) / 2
        c = w['core']
        keep = (xy[:, 0] >= c[0]) & (xy[:, 0] < c[2]) & (xy[:, 1] >= c[1]) & (xy[:, 1] < c[3])
        bb.extend(b[keep])
        scores.extend(s[keep])
    if not scores:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty((0, 4), np.float32)
    b = np.asarray(bb, np.float32)
    s = np.asarray(scores, np.float32)
    xywh = np.column_stack((b[:, :2], b[:, 2:] - b[:, :2]))
    ids = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), conf, .5)
    ids = np.asarray(ids, dtype=int).ravel()
    b = b[ids]
    s = s[ids]
    xy = (b[:, :2] + b[:, 2:]) / 2
    radius = np.sqrt(np.prod(b[:, 2:] - b[:, :2], axis=1) / np.pi)
    p = np.column_stack((xy, radius, s)).astype(np.float32)
    return p, s, b


_MODELS = {}


def model(path):
    path = str(path)
    if path not in _MODELS:
        _MODELS[path] = Model(path)
    return _MODELS[path]


def _key(model_path, image_key, bounds):
    h = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:16]
    return CACHE / f'{Path(model_path).stem}_{h}_{image_key}_{"_".join(map(str, bounds))}.npz'


def cached_predict(model_path, image_key, rgb, bounds=(0, 0, 1024, 1024)):
    """Low-threshold (0.001) predictions, cached by model hash + image + bounds."""
    f = _key(model_path, image_key, bounds)
    if f.exists():
        z = np.load(f)
        return z['p'], z['s'], z['b']
    p, s, b = tile_predict_np(model(model_path), rgb, bounds)
    np.savez_compressed(f, p=p, s=s, b=b)
    return p, s, b


def in_box(xy, b):
    return (xy[:, 0] >= b[0]) & (xy[:, 0] < b[2]) & (xy[:, 1] >= b[1]) & (xy[:, 1] < b[3])


def gt_region(k, b):
    g = cases()[k]['gt']
    return g[in_box(g, b)]


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=_np) + '\n')


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def tile_predict_feats(model_obj, rgb, bounds=(0, 0, 1024, 1024), imgsz=512, conf=.001):
    """Same as tile_predict_np, but also returns the 80-d penultimate class features
    and raw class logit of the anchor behind every surviving detection."""
    ws = windows(bounds)
    imgs = [rgb[w['crop'][1]:w['crop'][3], w['crop'][0]:w['crop'][2], ::-1].copy() for w in ws]
    results = model_obj.predict(imgs, imgsz=imgsz, conf=conf, iou=.5, max_det=2000, batch=4, feats=True)
    bb, scores, ff, ll = [], [], [], []
    for w, (b, s, f, lg) in zip(ws, results):
        if not len(s):
            continue
        b = b + np.array([w['crop'][0], w['crop'][1], w['crop'][0], w['crop'][1]], np.float32)
        xy = (b[:, :2] + b[:, 2:]) / 2
        c = w['core']
        keep = (xy[:, 0] >= c[0]) & (xy[:, 0] < c[2]) & (xy[:, 1] >= c[1]) & (xy[:, 1] < c[3])
        bb.extend(b[keep]); scores.extend(s[keep]); ff.extend(f[keep]); ll.extend(lg[keep])
    b = np.asarray(bb, np.float32); s = np.asarray(scores, np.float32)
    f = np.asarray(ff, np.float32).reshape(len(s), -1); lg = np.asarray(ll, np.float32)
    xywh = np.column_stack((b[:, :2], b[:, 2:] - b[:, :2]))
    ids = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), conf, .5), dtype=int).ravel()
    b, s, f, lg = b[ids], s[ids], f[ids], lg[ids]
    xy = (b[:, :2] + b[:, 2:]) / 2
    radius = np.sqrt(np.prod(b[:, 2:] - b[:, :2], axis=1) / np.pi)
    p = np.column_stack((xy, radius, s)).astype(np.float32)
    return p, s, b, f, lg


def cached_feats(model_path, image_key, rgb, bounds=(0, 0, 1024, 1024)):
    f = _key(model_path, image_key + '_feats', bounds)
    if f.exists():
        z = np.load(f)
        return z['p'], z['s'], z['b'], z['f'].astype(np.float32), z['lg']
    p, s, b, ft, lg = tile_predict_feats(model(model_path), rgb, bounds)
    np.savez_compressed(f, p=p, s=s, b=b, f=ft.astype(np.float16), lg=lg)
    # Return the same float16-rounded features a cached call returns, so a fresh
    # run and a cached run give identical refit results.
    return p, s, b, ft.astype(np.float16).astype(np.float32), lg
