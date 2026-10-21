"""External benchmark built from the Roboflow 'Barnacle species identification2' export.

Classes (Roboflow orders names alphabetically; mapping inferred, consistent with box sizes):
  0 A.mod  1 C.mon  2 C.ste  3 Cyprid  4 L.sax  5 M.ner  6 P.per  7 S.bal  8 Spat
TARGET (counted)  : adult barnacle classes 0,1,2,6,7 (same policy as the other session's report)
NOT counted       : 3 Cyprid (larva), 8 Spat (juvenile), 4 L.sax and 5 M.ner (periwinkle snails)

Coordinates: each photo resized so its LONGEST side is 1024 px. A box becomes a center plus an
equivalent radius sqrt(w*h/pi); polygons use their bounding box. Matching uses the project rule
max(3 px, 0.75 r) one-to-one.

Split: photos grouped by a coarse survey-session proxy from the filename (group(): same date, or
same site + year); a group goes to TEST if sha1(group) mod 100 < 20, to DEV-VAL if 20 <= ... < 35,
else DEV-TRAIN. Roboflow's own random split is ignored because survey sessions span its splits.
(A first version of group() left most photos as single-photo groups, so photos from one survey
landed in both tuning and test; every external result was rerun after this fix.)
"""
import glob
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
from PIL import Image

RAW = Path(os.environ.get('BARNACLE_RF_EXPORT', Path(__file__).resolve().parent.parent / 'roboflow_export'))
OUT = Path(__file__).resolve().parent / 'data_external'
IMG = OUT / 'img1024'
NAMES = ['A.mod', 'C.mon', 'C.ste', 'Cyprid', 'L.sax', 'M.ner', 'P.per', 'S.bal', 'Spat']
TARGET = (0, 1, 2, 6, 7)


def stem(path):
    s = os.path.basename(str(path))
    s = re.sub(r'\.(jpg|txt)$', '', s)
    return re.sub(r'\.rf\.[0-9a-f]+$', '', s)


GENERIC = {'barnacle', 'barnacles', 'low', 'mid', 'upp', 'upper', 'lower', 'middle', 'jpg', 'jpeg', 'cropped',
           'usbarnacle', 'lsbarnacle', 'barnaclems', 'img', 'dsc', 'png'}


def group(s):
    """Survey-session proxy, deliberately coarse (coarser groups are safer against leakage):
    a YYYYMMDD date anywhere in the name -> all photos from that date;
    a dd_mm prefix (e.g. 04_10_...) -> that prefix;
    otherwise the first site-like word plus a 4-digit year (e.g. Jetty-2025, Skokholm-2025)."""
    k = re.sub(r'_(jpg|JPG|jpeg|JPEG|png)$', '', s)
    m = re.search(r'(20\d{6})', k)
    if m:
        return 'date_' + m.group(1)
    m = re.match(r'^(\d{2})_(\d{2})_', k)
    if m:
        return 'daymonth_' + m.group(1) + m.group(2)
    year = re.search(r'(20\d{2})', k)
    words = [w for w in re.split(r'[^A-Za-z]+', k) if len(w) >= 3 and w.lower() not in GENERIC]
    return 'site_' + (words[0] if words else 'unknown') + '_' + (year.group(1) if year else 'noyear')


def split_of(g):
    h = int(hashlib.sha1(g.encode()).hexdigest(), 16) % 100
    return 'test' if h < 20 else ('val' if h < 35 else 'train')


def build():
    IMG.mkdir(parents=True, exist_ok=True)
    imgs = {Path(f).stem: f for f in glob.glob(str(RAW / 'Archive*/*.jpg'))}
    labs = {Path(f).stem: f for f in glob.glob(str(RAW / 'Archive*label*/*.txt'))}
    rf_split = {}
    for d, s in (('Archive', 'train'), ('Archive_2', 'train'), ('Archive_3', 'train'),
                 ('Archive_valid', 'valid'), ('Archive_test', 'test')):
        for f in glob.glob(str(RAW / d / '*.jpg')):
            rf_split[Path(f).stem] = s
    rows = []
    for k in sorted(imgs):
        im = Image.open(imgs[k]).convert('RGB')
        W, H = im.size
        sc = 1024 / max(W, H)
        w2, h2 = round(W * sc), round(H * sc)
        out = IMG / (stem(k) + '.jpg')
        if not out.exists():
            im.resize((w2, h2), Image.LANCZOS).save(out, quality=95)
        objs = []
        for r in open(labs[k]).read().splitlines():
            r = r.split()
            if not r:
                continue
            c = int(r[0])
            v = np.array(r[1:], float)
            if len(v) == 4:
                cx, cy, w, h = v
            else:
                xs, ys = v[0::2], v[1::2]
                cx, cy, w, h = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, xs.max() - xs.min(), ys.max() - ys.min()
            bw, bh = w * w2, h * h2
            objs.append([cx * w2, cy * h2, float(np.sqrt(bw * bh / np.pi)), c, bw, bh])
        g = group(stem(k))
        rows.append(dict(id=stem(k), file=str(out.relative_to(OUT)), width=w2, height=h2, original_size=[W, H], group=g,
                         split=split_of(g), roboflow_split=rf_split[k], objects=objs))
    (OUT / 'bench.json').write_text(json.dumps(rows))
    return rows


def load(split=None):
    rows = json.loads((OUT / 'bench.json').read_text())
    return [r for r in rows if split is None or r['split'] == split]


def targets(row, classes=TARGET):
    o = np.array(row['objects'], float).reshape(-1, 6)
    keep = np.isin(o[:, 3], classes)
    return o[keep][:, :3].astype(np.float32), o[~keep][:, :4].astype(np.float32)


def read(row):
    f = Path(row['file'])
    if not f.is_absolute():
        f = OUT / f
        if not f.exists():
            raise FileNotFoundError(f'{f} missing: run rf_bench.py once to rebuild the 1024-px images from the Roboflow export')
    return np.asarray(Image.open(f).convert('RGB'))


if __name__ == '__main__':
    rows = build()
    from collections import Counter
    c = Counter(r['split'] for r in rows)
    g = Counter((r['split'], r['group']) for r in rows)
    print('images per split', dict(c), '| groups per split', dict(Counter(s for s, _ in g.keys())))
    for s in ('train', 'val', 'test'):
        rr = [r for r in rows if r['split'] == s]
        t = sum(len(targets(r)[0]) for r in rr)
        o = sum(len(targets(r)[1]) for r in rr)
        print(f'{s}: {len(rr)} images, {t} adult-barnacle targets, {o} other objects, '
              f'roboflow splits {dict(Counter(r["roboflow_split"] for r in rr))}')
