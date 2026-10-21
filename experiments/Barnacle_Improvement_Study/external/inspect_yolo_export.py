"""Inspect a Roboflow (YOLO-format) export BEFORE any scoring. No model is run.

Usage:  python external/inspect_yolo_export.py PATH/TO/EXPORT [--out results/external_inspection.json]
Accepts a folder with data.yaml and train/valid/test subfolders (images/, labels/),
or any partial batch with that layout.

Reports, per split and per class:
  * class names exactly as in data.yaml
  * annotation geometry: boxes vs polygons (segmentation rows), box size in pixels
    and relative to image size, aspect ratio -> are these apertures (~5-25 px in a
    1024 px quadrat) or whole shells / clumps?
  * images with NO label file vs an EMPTY label file (unannotated != negative)
  * original resolutions (no resizing is done)
  * near-duplicate / related captures across splits (8x8 average hash, Hamming <= 5)
  * possible Roboflow augmentation copies (".rf." names sharing a stem)
It does not decide compatibility; it gives the facts needed to decide whether a
center-F1 comparison with the DALI benchmark would be meaningful.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

IMG = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def names_from_yaml(p):
    if not p.exists():
        return None
    txt = p.read_text()
    m = re.search(r'names:\s*\[(.*?)\]', txt, re.S)
    if m:
        return [x.strip().strip('\'"') for x in m.group(1).split(',') if x.strip()]
    out = {}
    block = False
    for line in txt.splitlines():
        if line.strip().startswith('names:'):
            block = True
            continue
        if block:
            mm = re.match(r'\s+(\d+)\s*:\s*(.+)', line)
            if mm:
                out[int(mm.group(1))] = mm.group(2).strip().strip('\'"')
            elif line.strip().startswith('-'):
                out[len(out)] = line.strip()[1:].strip().strip('\'"')
            elif line.strip() and not line.startswith(' '):
                break
    return [out[k] for k in sorted(out)] if out else None


def ahash(path):
    im = Image.open(path).convert('L').resize((8, 8), Image.BILINEAR)
    a = np.asarray(im, float)
    return (a > a.mean()).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'results/external_inspection.json'))
    a = ap.parse_args()
    root = Path(a.root)
    names = names_from_yaml(root / 'data.yaml')
    report = dict(root=str(root), class_names=names, splits={})
    hashes = []
    stems = defaultdict(list)
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(q for q in (split_dir / 'images').glob('*') if q.suffix.lower() in IMG) if (split_dir / 'images').exists() else []
        if not imgs:
            continue
        lab_dir = split_dir / 'labels'
        res, geom = Counter(), Counter()
        no_label, empty_label = [], []
        per_class = defaultdict(lambda: dict(n=0, w_px=[], h_px=[], rel_w=[], aspect=[]))
        per_image_counts = []
        for ip in imgs:
            with Image.open(ip) as im:
                W, H = im.size
            res[f'{W}x{H}'] += 1
            lp = lab_dir / (ip.stem + '.txt')
            stems[re.sub(r'\.rf\.[0-9a-f]+$', '', re.sub(r'_(jpg|png|jpeg)$', '', ip.stem))].append(f'{split_dir.name}/{ip.name}')
            hashes.append((f'{split_dir.name}/{ip.name}', ahash(ip)))
            if not lp.exists():
                no_label.append(ip.name)
                continue
            rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
            if not rows:
                empty_label.append(ip.name)
            per_image_counts.append(len(rows))
            for r in rows:
                c = int(float(r[0]))
                v = np.array(r[1:], float)
                if len(v) == 4:
                    geom['box'] += 1
                    w, h = v[2], v[3]
                else:
                    geom['polygon'] += 1
                    xs, ys = v[0::2], v[1::2]
                    w, h = xs.max() - xs.min(), ys.max() - ys.min()
                d = per_class[c]
                d['n'] += 1
                d['w_px'].append(w * W)
                d['h_px'].append(h * H)
                d['rel_w'].append(w)
                d['aspect'].append((w * W) / max(h * H, 1e-6))
        cls = {}
        for c, d in sorted(per_class.items()):
            q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 50, 95])]  # noqa: E731
            cls[names[c] if names and c < len(names) else str(c)] = dict(
                objects=d['n'], width_px_p5_p50_p95=q(d['w_px']), height_px_p5_p50_p95=q(d['h_px']),
                width_fraction_of_image_p5_p50_p95=q(d['rel_w']), aspect_p5_p50_p95=q(d['aspect']))
        report['splits'][split_dir.name] = dict(
            images=len(imgs), resolutions=dict(res.most_common(10)), geometry=dict(geom),
            images_without_label_file=len(no_label), images_with_empty_label_file=len(empty_label),
            examples_without_label_file=no_label[:10], examples_empty_label_file=empty_label[:10],
            objects_per_labelled_image_p5_p50_p95=[float(x) for x in np.percentile(per_image_counts, [5, 50, 95])] if per_image_counts else None,
            per_class=cls)
    near = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i][0].split('/')[0] == hashes[j][0].split('/')[0]:
                continue
            d = int((hashes[i][1] != hashes[j][1]).sum())
            if d <= 5:
                near.append([hashes[i][0], hashes[j][0], d])
    report['cross_split_near_duplicates_ahash_le5'] = dict(count=len(near), examples=near[:30])
    groups = {k: v for k, v in stems.items() if len(v) > 1}
    report['shared_source_stems'] = dict(groups=len(groups), examples=dict(list(groups.items())[:15]),
                                         cross_split_groups=sum(len({x.split('/')[0] for x in v}) > 1 for v in groups.values()))
    report['notes'] = ['Images without label files are UNANNOTATED, not confirmed negatives.',
                       'Compare box sizes with DALI apertures (median radius ~7-8 px in a 1024 px quadrat) before any center-F1 comparison.',
                       'Roboflow published metrics belong to their models, not to this detector.']
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2)[:4000])


if __name__ == '__main__':
    main()
