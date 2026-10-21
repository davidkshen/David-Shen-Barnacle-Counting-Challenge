"""Training experiments that need PyTorch + Ultralytics (run on your own machine).

They could not run in the cloud workspace used for this study (no PyTorch
install allowed there). Each one answers a question the inference-only study
could not, and each is pre-declared here BEFORE any result exists.

Run from this folder, with Barnacle_Three_Track_Study next to the package:
  pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
  pip install -r ../../Barnacle_Three_Track_Study/requirements.txt
  python run_local.py seeds             # E1  ~6 runs
  python run_local.py matched_mixed     # E2  ~3 scratch runs (pairs with E1)
  python run_local.py budget_finetune   # E3  ~30 short runs
  python run_local.py summarize

E1 seeds - training-seed variability of the MAIN recipe.
   Same data, split, tiles (fraction 0.5, tile seed 42), optimizer and
   augmentation as modern_analysis/train_detector.py --mode mixed --fraction 0.5,
   with 100 epochs as recorded in the brief (the script's own default is 80:
   models/args.yaml confirms 100 epochs);
   only the training seed changes (42, 43, 44). COCO-pretrained YOLO11n.
   Report: per-seed stripe F1 per photo and pooled, mean and SD. Seed 42 should
   approximately reproduce 0.941 (CPU nondeterminism can move it slightly).
E2 matched_mixed - the clean pretraining comparison the existing study lacks.
   IDENTICAL to E1 except initialization: yolo11n.yaml, pretrained=False.
   Compare E1 vs E2 mean pooled F1 with the seed SD from both. This, not
   0.941 vs 0.889, is the defensible estimate of what COCO weights add here.
E3 budget_finetune - does WEIGHT fine-tuning beat threshold-only at small budgets?
   Base: pretrained_from{1,2}.pt (one-photo models, target never seen).
   Cells: the same 128-px cells and the same nested random draws (seed 1000+d)
   as experiments/adapt_budget.py, so results pair with its threshold-only rows.
   Budgets K = 2, 4, 8 cells; draws d = 0..4.  K-1 (K=2,4) or K-2 (K=8) cells
   train, the remaining cell(s) calibrate checkpoint + threshold (budget includes
   calibration). Source replay, 30 epochs, lr0 0.0003, no mosaic: the
   three-track adaptation settings. Weights are fine-tuned from the base, never
   reset. Each 128-px cell is trained at native scale inside a grey 256-px
   canvas. Scored on the untouched stripe x>=832 from whole-quadrat inference.
   Decision rule (declared now): weight fine-tuning is "worth it" at budget K only
   if its mean stripe F1 beats threshold-only by >= 0.03 AND its worst draw is
   not below threshold-only's worst draw.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
THREE = Path(os.environ.get('BARNACLE_THREE_TRACK', PKG.parent / 'Barnacle_Three_Track_Study'))
PROJECT = THREE / 'project'
sys.path.insert(0, str(PROJECT))
from barnacles.modern import prepare_cases, tile_predict, choose_confidence  # noqa: E402
from barnacles.core import metrics  # noqa: E402
from modern_analysis.train_detector import build_dataset  # noqa: E402

EPOCHS = 100   # main recipe as recorded in the brief; train_detector.py defaults to 80
OUT = HERE / 'outputs'
OUT.mkdir(exist_ok=True)
STRIPE = (832, 0, 1024, 1024)
CELLS = [(x, y, x + 128, y + 128) for y in range(0, 1024, 128) for x in range(0, 768, 128)]


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def region_gt(c, b):
    g = c['gt']
    return g[(g[:, 0] >= b[0]) & (g[:, 0] < b[2]) & (g[:, 1] >= b[1]) & (g[:, 1] < b[3])]


def train_main_recipe(data, tag, init, seed):
    """modern_analysis/train_detector.train() settings, plus init/seed switches."""
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(int(os.environ.get('BARNACLE_CPU_THREADS', '6')))
    scratch = init == 'scratch'
    yolo = YOLO('yolo11n.yaml' if scratch else str(PROJECT / 'yolo11n.pt'))
    yolo.train(data=str(data), epochs=EPOCHS, imgsz=512, batch=8, device='cpu', workers=0,
               optimizer='AdamW', lr0=.001, lrf=.1, cos_lr=True, warmup_epochs=3, weight_decay=.0005,
               patience=100, seed=seed, deterministic=True, pretrained=not scratch,
               degrees=180, translate=.1, scale=.3, fliplr=.5, flipud=.5,
               hsv_h=.015, hsv_s=.35, hsv_v=.35, mosaic=.3, close_mosaic=15, mixup=0.,
               max_det=2000, save=True, save_period=20, plots=False, verbose=False,
               cache='ram', project=str(OUT / 'runs'), name=tag, exist_ok=False)
    return OUT / 'runs' / tag


def train_adapt(data, tag, base):
    """The three-track adaptation settings (run_detectors.train with adapt=True)."""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    import torch
    torch.set_num_threads(int(os.environ.get('BARNACLE_CPU_THREADS', '6')))
    init_seeds(42, deterministic=True)
    model = YOLO(str(base))
    model.train(data=str(data), epochs=30, imgsz=512, batch=8, nbs=8, device='cpu', workers=0,
                optimizer='AdamW', lr0=.0003, lrf=.1, cos_lr=True, warmup_epochs=3, weight_decay=.0005,
                patience=31, seed=42, deterministic=True, pretrained=True,
                degrees=180, translate=.1, scale=.3, fliplr=.5, flipud=.5, hsv_h=.015, hsv_s=.35, hsv_v=.35,
                mosaic=0., close_mosaic=0, mixup=0., max_det=2000, save=True, save_period=10, plots=False,
                verbose=False, cache='ram', project=str(OUT / 'runs'), name=tag, exist_ok=False)
    return OUT / 'runs' / tag


def select(run, cases, regions):
    """Checkpoint + threshold by pooled center F1 on the given labelled regions only."""
    from ultralytics import YOLO
    best = None
    for ck in sorted((run / 'weights').glob('*.pt')):
        m = YOLO(ck)
        pp, ss, gg = [], [], []
        for j, (k, b) in enumerate(regions):
            p, s, _ = tile_predict(m, cases[k]['im'], b)
            g = region_gt(cases[k], b).copy()
            p[:, 0] += j * 2048
            g[:, 0] += j * 2048
            pp.append(p); ss.append(s); gg.append(g)
        ch, _ = choose_confidence(np.concatenate(pp), np.concatenate(ss), np.concatenate(gg))
        row = dict(checkpoint=str(ck), **ch)
        if best is None or (row['f1'], row['precision']) > (best['f1'], best['precision']):
            best = row
    return best


def stripe_eval(ck, thr, cases, photos):
    from ultralytics import YOLO
    m = YOLO(ck)
    rows = []
    for k in photos:
        t0 = time.time()
        p, s, _ = tile_predict(m, cases[k]['im'], STRIPE)
        g = region_gt(cases[k], STRIPE)
        mm = metrics(p[s >= thr], g)
        rows.append(dict(photo=k, seconds=round(time.time() - t0, 1), strict_f1=metrics(p[s >= thr], g, .5)['f1'], **mm))
    return rows


def build_cell_dataset(cases, folder, replay, target, train_cells, cal_cells):
    """Source replay tiles via build_dataset, plus each labelled 128-px target cell
    placed in a 256-px canvas at native scale with everything OUTSIDE the cell set
    to grey 114 (YOLO's letterbox grey). Native scale: inference upsamples 256-px
    tiles 2x to 512, so a bare 128-px crop would be upsampled 4x and train at the
    wrong apparent size. Grey-out: unlabelled barnacles outside the cell are never
    shown, so they cannot be learned as background."""
    from PIL import Image
    data = build_dataset(cases, folder, replay, [(target, cal_cells[0])])
    for split in ('train', 'val'):
        for f in (folder / f'images/{split}').glob(f'img{target}_*.png'):
            f.unlink()
            (folder / f'labels/{split}' / (f.stem + '.txt')).unlink()
    im = cases[target]['im']
    boxes = cases[target]['boxes']
    xy = (boxes[:, :2] + boxes[:, 2:]) / 2
    for split, cells in (('train', train_cells), ('val', cal_cells)):
        for j, (x0, y0, x1, y1) in enumerate(cells):
            cx0 = int(min(max(0, x0 - 64), 1024 - 256))
            cy0 = int(min(max(0, y0 - 64), 1024 - 256))
            canvas = np.full((256, 256, 3), 114, np.uint8)
            canvas[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = im[y0:y1, x0:x1]
            keep = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            b = boxes[keep].copy()
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], x0, x1) - cx0
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], y0, y1) - cy0
            name = f'cell{target}_{split}_{j:02d}'
            Image.fromarray(canvas).save(folder / f'images/{split}/{name}.png')
            lines = []
            for q in b:
                w, h = q[2] - q[0], q[3] - q[1]
                if w > 1 and h > 1:
                    lines.append(f'0 {(q[0] + q[2]) / 512:.8f} {(q[1] + q[3]) / 512:.8f} {w / 256:.8f} {h / 256:.8f}')
            (folder / f'labels/{split}/{name}.txt').write_text('\n'.join(lines) + '\n')
    return data


def select_cells(run, cases, target, cells):
    """Checkpoint + threshold on calibration cells, predicting the WHOLE quadrat
    (as deployed) and keeping detections inside the cells, so the cells are seen
    at the same 2x scale as everywhere else."""
    from ultralytics import YOLO
    gt = np.concatenate([region_gt(cases[target], c) for c in cells])
    best = None
    for ck in sorted((run / 'weights').glob('*.pt')):
        p, s, _ = tile_predict(YOLO(ck), cases[target]['im'])
        inside = np.zeros(len(p), bool)
        for c in cells:
            inside |= (p[:, 0] >= c[0]) & (p[:, 0] < c[2]) & (p[:, 1] >= c[1]) & (p[:, 1] < c[3])
        ch, _ = choose_confidence(p[inside], s[inside], gt)
        row = dict(checkpoint=str(ck), **ch)
        if best is None or (row['f1'], row['precision']) > (best['f1'], best['precision']):
            best = row
    return best


def stripe_eval_full(ck, thr, cases, k):
    """Whole-quadrat inference, then keep the stripe: the same protocol as
    experiments/adapt_budget.py, so E3 pairs with its threshold-only rows."""
    from ultralytics import YOLO
    t0 = time.time()
    p, s, _ = tile_predict(YOLO(ck), cases[k]['im'])
    p = p[s >= thr]
    p = p[(p[:, 0] >= STRIPE[0]) & (p[:, 0] < STRIPE[2])]
    g = region_gt(cases[k], STRIPE)
    return dict(photo=k, seconds=round(time.time() - t0, 1), strict_f1=metrics(p, g, .5)['f1'], **metrics(p, g))


def seeds(init='pretrained'):
    cases = prepare_cases(PROJECT / 'data')
    tr = [(k, (0, 0, 384, 1024)) for k in (1, 2)]
    va = [(k, (512, 0, 704, 1024)) for k in (1, 2)]
    res = []
    for seed in (42, 43, 44):
        tag = f'mixed05_{init}_seed{seed}'
        if (OUT / f'{tag}.json').exists():
            res.append(json.loads((OUT / f'{tag}.json').read_text()))
            continue
        data = build_dataset(cases, OUT / 'datasets' / tag, tr, va, fraction=0.5, seed=42)
        t0 = time.time()
        run = train_main_recipe(data, tag, init, seed)
        train_s = time.time() - t0
        best = select(run, cases, va)
        rows = stripe_eval(best['checkpoint'], best['threshold'], cases, (1, 2))
        tp, fp, fn = (sum(r[q] for r in rows) for q in ('tp', 'fp', 'fn'))
        r = dict(tag=tag, init=init, seed=seed, train_seconds=round(train_s), selection=best, stripes=rows,
                 pooled_f1=2 * tp / (2 * tp + fp + fn))
        save(OUT / f'{tag}.json', r)
        res.append(r)
        print(tag, 'pooled F1 %.3f' % r['pooled_f1'], flush=True)
    return res


def budget_finetune():
    cases = prepare_cases(PROJECT / 'data')
    for src, tgt in ((2, 1), (1, 2)):
        base = THREE / 'models' / f'pretrained_from{src}.pt'
        for d in range(5):
            order = np.random.default_rng(1000 + d).permutation(len(CELLS))
            for K in (2, 4, 8):
                tag = f'ft_from{src}_K{K}_d{d}'
                if (OUT / f'{tag}.json').exists():
                    continue
                cells = [CELLS[i] for i in order[:K]]
                ncal = 2 if K == 8 else 1
                train_cells, cal_cells = cells[:-ncal], cells[-ncal:]
                if not any(len(region_gt(cases[tgt], c)) for c in cal_cells):
                    save(OUT / f'{tag}.json', dict(tag=tag, skipped='calibration cells contain no targets'))
                    continue
                data = build_cell_dataset(cases, OUT / 'datasets' / tag, [(src, (0, 0, 640, 1024))],
                                          tgt, train_cells, cal_cells)
                t0 = time.time()
                run = train_adapt(data, tag, base)
                best = select_cells(run, cases, tgt, cal_cells)
                rows = [stripe_eval_full(best['checkpoint'], best['threshold'], cases, tgt)]
                save(OUT / f'{tag}.json', dict(tag=tag, source=src, target=tgt, draw=d, K=K,
                                               train_cells=train_cells, calibration_cells=cal_cells,
                                               labelled_objects=int(sum(len(region_gt(cases[tgt], c)) for c in cells)),
                                               seconds=round(time.time() - t0), selection=best, stripe=rows[0]))
                print(tag, 'stripe F1 %.3f' % rows[0]['f1'], flush=True)


def summarize():
    out = {}
    for init in ('pretrained', 'scratch'):
        rr = [json.loads(p.read_text()) for p in sorted(OUT.glob(f'mixed05_{init}_seed*.json'))]
        if rr:
            f = np.array([r['pooled_f1'] for r in rr])
            out[f'E1_E2_{init}'] = dict(n=len(f), pooled_f1=f.tolist(), mean=float(f.mean()),
                                        sd=float(f.std(ddof=1)) if len(f) > 1 else None)
    ft = [json.loads(p.read_text()) for p in sorted(OUT.glob('ft_from*_K*_d*.json'))]
    thr_path = PKG / 'results' / 'adapt_budget.json'
    thr = json.loads(thr_path.read_text()) if thr_path.exists() else {}
    for src, tgt in ((2, 1), (1, 2)):
        for K in (2, 4, 8):
            rows = [r for r in ft if r.get('source') == src and r.get('K') == K]
            if not rows:
                continue
            f = np.array([r['stripe']['f1'] for r in rows])
            draws = [r['draw'] for r in rows]
            tro = [x for x in thr.get(f'pretrained_from{src}', {}).get('rows', []) if x['K'] == K and x['draw'] in draws]
            tf = np.array([x['threshold']['f1'] for x in tro]) if tro else None
            out[f'E3_from{src}_K{K}'] = dict(draws=draws, finetune_f1=f.tolist(), finetune_mean=float(f.mean()),
                                             threshold_only_f1=tf.tolist() if tf is not None else None,
                                             threshold_only_mean=float(tf.mean()) if tf is not None else None,
                                             worth_it=(bool(f.mean() - tf.mean() >= .03 and f.min() >= tf.min())
                                                       if tf is not None else None))
    save(OUT / 'summary.json', out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['seeds', 'matched_mixed', 'budget_finetune', 'summarize'])
    a = ap.parse_args()
    if a.what == 'seeds':
        seeds('pretrained')
    elif a.what == 'matched_mixed':
        seeds('scratch')
    elif a.what == 'budget_finetune':
        budget_finetune()
    else:
        summarize()
