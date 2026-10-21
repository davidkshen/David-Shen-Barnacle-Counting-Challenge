import cv2
import numpy as np


def starts(low, high, tile=256, stride=192):
    # left/top edges of the tiles. last tile is pushed back so it ends exactly at the edge.
    if high - low <= tile:
        return [low]
    v = list(range(low, high - tile + 1, stride))
    if v[-1] != high - tile:
        v.append(high - tile)
    return v


def windows(bounds=(0, 0, 1024, 1024), tile=256, stride=192):
    # 256px tiles that slide by 192 -> neighbours overlap by 64px, so a barnacle cut by one tile
    # is whole in the next one. each tile also gets a 'core' (half way into each overlap).
    # a dot only counts in the tile whose core it lands in, so overlaps don't double count.
    x0, y0, x1, y1 = bounds
    xs, ys = starts(x0, x1, tile, stride), starts(y0, y1, tile, stride)
    xx = [min(x + tile, x1) for x in xs]
    yy = [min(y + tile, y1) for y in ys]
    cx = np.array([(a + b) / 2 for a, b in zip(xs, xx)])
    cy = np.array([(a + b) / 2 for a, b in zip(ys, yy)])
    xc = [x0] + ((cx[:-1] + cx[1:]) / 2).tolist() + [x1]
    yc = [y0] + ((cy[:-1] + cy[1:]) / 2).tolist() + [y1]
    out = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            out.append(dict(crop=(x, y, xx[i], yy[j]), core=(xc[i], yc[j], xc[i + 1], yc[j + 1])))
    return out


def tile_predict(predict_tiles, rgb, bounds=(0, 0, 1024, 1024), conf=.001):
    # predict_tiles(list of BGR tiles) -> list of (boxes xyxy, scores) per tile.
    # the model sees each 256px tile blown up to 512px (imgsz=512), which makes small barnacles big enough to find.
    # returns dots as rows of x, y, radius, confidence (in rectified pixels). conf is tiny on purpose:
    # keep everything here and apply the real threshold later.
    ws = windows(bounds)
    tiles = [rgb[w['crop'][1]:w['crop'][3], w['crop'][0]:w['crop'][2], ::-1].copy() for w in ws]
    results = predict_tiles(tiles)
    boxes, scores = [], []
    for w, (b, s) in zip(ws, results):
        if not len(s):
            continue
        b = b + np.array([w['crop'][0], w['crop'][1], w['crop'][0], w['crop'][1]], np.float32)  # tile -> frame coords
        xy = (b[:, :2] + b[:, 2:]) / 2
        c = w['core']
        keep = (xy[:, 0] >= c[0]) & (xy[:, 0] < c[2]) & (xy[:, 1] >= c[1]) & (xy[:, 1] < c[3])
        boxes.extend(b[keep])
        scores.extend(s[keep])
    if not scores:
        return np.empty((0, 4), np.float32)
    b = np.asarray(boxes, np.float32)
    s = np.asarray(scores, np.float32)
    # last clean-up: if two boxes still overlap a lot, keep the more confident one
    xywh = np.column_stack((b[:, :2], b[:, 2:] - b[:, :2]))
    ids = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), conf, .5), dtype=int).ravel()
    b, s = b[ids], s[ids]
    xy = (b[:, :2] + b[:, 2:]) / 2
    radius = np.sqrt(np.prod(b[:, 2:] - b[:, :2], axis=1) / np.pi)  # circle with the same area as the box
    return np.column_stack((xy, radius, s)).astype(np.float32)
