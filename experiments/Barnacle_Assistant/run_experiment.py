"""Reproduce the exploratory two-photograph experiment. No network or GPU."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import joblib
import numpy as np
import scipy
import sklearn
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from barnacles.core import (Config, read_rgb, rectify, load_reference, propose,
        features, gradient_patches, reference_targets, select_detections,
        metrics, match_objects, normalized_gray)
from barnacles.model import (fit_classifier, fit_patch_classifier, predict_patch,
        patch_stability_scores, choose_threshold, conditional_occlusion)
from barnacles.audit import tile_counts, simulate_audits, random_tiles
from barnacles.reviewer import create_review


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prepare(data_dir, frames, cfg):
    cases = {}
    for k in (1, 2):
        name = f'img{k}.png'; start = time.perf_counter()
        rgb = read_rgb(data_dir/name); im, matrix = rectify(rgb, frames[name], cfg.size)
        gt = load_reference(data_dir/f'mask{k}.png', matrix, cfg.size)
        p, g = propose(im, cfg); X, names = features(g, p); H = gradient_patches(g, p)
        cases[k] = dict(image=im, matrix=matrix, gt=gt, proposals=p, gray=g,
                        X=X, H=H, names=names, preprocessing_seconds=time.perf_counter()-start)
    return cases


def detection_overlay(image, predictions, reference, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    match = match_objects(predictions, reference)
    pi, gi = set(match[:, 0]), set(match[:, 1])
    fig, ax = plt.subplots(figsize=(10, 10)); ax.imshow(image)
    if len(match):
        pp = predictions[match[:, 0]]
        ax.scatter(pp[:, 0], pp[:, 1], s=15, facecolors='none', edgecolors='#54ffd6', linewidths=.8, label='Matched prediction')
    fp = [i for i in range(len(predictions)) if i not in pi]
    fn = [i for i in range(len(reference)) if i not in gi]
    if fp:
        pp = predictions[fp]; ax.scatter(pp[:, 0], pp[:, 1], s=12, marker='x', color='#ff7b80', linewidths=.7, label='Unmatched prediction')
    if fn:
        gg = reference[fn]; ax.scatter(gg[:, 0], gg[:, 1], s=10, facecolors='none', edgecolors='#ffd666', linewidths=.65, label='Unmatched reference')
    ax.set_title(title, fontsize=13); ax.legend(loc='lower left', fontsize=9, framealpha=.9)
    ax.axis('off'); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--frames', type=Path, default=Path('data/frames.json'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frames = json.loads(args.frames.read_text()); cfg = Config(); start_all = time.perf_counter()
    cv2.setNumThreads(2)
    cases = prepare(args.data_dir, frames, cfg)
    results = []; folds = []; audit_results = []; diagnostics = {}
    for k, s in cases.items():
        sensitivity = {}
        for alpha in (1, 16, 32, 64, 100, 128):
            sensitivity[str(alpha)] = len(load_reference(args.data_dir/f'mask{k}.png', s['matrix'], alpha=alpha))
        diagnostics[f'img{k}'] = dict(reference_count=len(s['gt']),
            aperture_radius_percentiles=np.percentile(s['gt'][:, 2], [10, 50, 90]).tolist(),
            label_count_by_alpha=sensitivity,
            proposal_count=len(s['proposals']), proposal_metrics=metrics(s['proposals'], s['gt']),
            preprocessing_seconds=s['preprocessing_seconds'])
        Image.fromarray(s['image']).save(args.output/f'img{k}_rectified.jpg', quality=94)
        np.savetxt(args.output/f'img{k}_reference_centers.csv', s['gt'], delimiter=',',
                   header='x,y,effective_aperture_radius', comments='')
    for source in (1, 2):
        target = 3-source; s = cases[source]; test = cases[target]
        p = s['proposals']; train = p[:, 0] < 368; val = p[:, 0] >= 656
        source_train_gt = s['gt'][s['gt'][:, 0] < 368]
        val_gt = s['gt'][s['gt'][:, 0] >= 656]
        y_train = reference_targets(p[train], source_train_gt)
        print(f'Train img{source} left region -> validate separated right region -> transfer img{target}', flush=True)
        t0 = time.perf_counter()
        structural = fit_classifier(s['X'][train], y_train)
        no_slit = np.array(['slit' not in x for x in s['names']])
        rim_only = fit_classifier(s['X'][train][:, no_slit], y_train)
        patch = fit_patch_classifier(s['X'][train], s['H'][train], p[train], y_train)
        fit_seconds = time.perf_counter()-t0
        patch_source = predict_patch(patch, s['X'], s['H'], p)
        patch_test = predict_patch(patch, test['X'], test['H'], test['proposals'])
        stable_source, source_views = patch_stability_scores(s['image'], p, patch, s['X'], s['H'])
        t0 = time.perf_counter()
        stable_test, test_views = patch_stability_scores(test['image'], test['proposals'], patch, test['X'], test['H'])
        stability_seconds = time.perf_counter()-t0
        scores = {
            'ring_filter': (p[:, 3], test['proposals'][:, 3]),
            'structural_without_slit': (rim_only.predict_proba(s['X'][:, no_slit])[:, 1], rim_only.predict_proba(test['X'][:, no_slit])[:, 1]),
            'structural': (structural.predict_proba(s['X'])[:, 1], structural.predict_proba(test['X'])[:, 1]),
            'structural_gradient': (patch_source, patch_test),
            'structural_gradient_stability': (stable_source, stable_test),
        }
        selections = {}; main_ids = None
        for method, (ss, ts) in scores.items():
            threshold, validation, curve = choose_threshold(p[val], ss[val], val_gt)
            ids = select_detections(test['proposals'], ts, threshold)
            result = dict(source_image=source, test_image=target, method=method,
                          threshold=threshold, source_validation=validation,
                          **metrics(test['proposals'][ids], test['gt']))
            results.append(result); selections[method] = dict(threshold=threshold, validation=validation)
            save_json(args.output/f'fold{source}_{method}_validation_curve.json', curve)
            print(method, 'test P/R/F1/count', round(result['precision'], 3), round(result['recall'], 3), round(result['f1'], 3), len(ids), flush=True)
            if method == 'structural_gradient_stability':
                main_ids = ids; main_threshold = threshold
        # Empirical high-precision policy, with an explicit infeasibility flag.
        high_threshold, high_val, _ = choose_threshold(p[val], stable_source[val], val_gt, mode='conservative')
        high_ids = select_detections(test['proposals'], stable_test, high_threshold)
        results.append(dict(source_image=source, test_image=target, method='conservative_policy',
            threshold=high_threshold, source_validation=high_val, **metrics(test['proposals'][high_ids], test['gt'])))
        predictions = test['proposals'][main_ids]
        create_review(test['image'], np.column_stack((predictions, stable_test[main_ids], test_views[main_ids].std(1))),
                      args.output/f'review_img{target}.html', f'Image {target}: transfer suggestions / human review required')
        model_counts = tile_counts(predictions); reference_counts = tile_counts(test['gt'])
        raw = metrics(predictions, test['gt'])
        for budget in (8, 16, 32):
            for baseline_name, counts in [('model_plus_audit', model_counts), ('manual_tiles_only', np.zeros(64))]:
                audit_results.append(dict(source_image=source, test_image=target, method=baseline_name,
                    **simulate_audits(counts, reference_counts, n=budget, repeats=1000)))
        perturbations = []
        for name in ('gamma_075', 'gamma_135', 'blur_08', 'rotate_90'):
            if name.startswith('gamma'):
                gamma = .75 if name == 'gamma_075' else 1.35
                altered = np.rint(255*(test['image'].astype(float)/255)**gamma).astype(np.uint8)
                reference = test['gt']
            elif name == 'blur_08':
                altered = cv2.GaussianBlur(test['image'], (0, 0), .8); reference = test['gt']
            else:
                altered = np.ascontiguousarray(np.rot90(test['image'])); reference = test['gt'].copy()
                reference[:, 0] = test['gt'][:, 1]; reference[:, 1] = 1023-test['gt'][:, 0]
            pp, gg = propose(altered, cfg); xx, _ = features(gg, pp); hh = gradient_patches(gg, pp)
            # Whole-pipeline test of SINGLE-VIEW model: includes proposal changes.
            sc = predict_patch(patch, xx, hh, pp)
            ii = select_detections(pp, sc, selections['structural_gradient']['threshold'])
            perturbations.append(dict(transform=name, **metrics(pp[ii], reference)))
        np.savetxt(args.output/f'img{target}_predictions.csv',
            np.column_stack((predictions, stable_test[main_ids], test_views[main_ids].std(1))),
            delimiter=',', header='x,y,proposal_radius,ring_response,stability_score,view_score_sd', comments='')
        save_json(args.output/f'img{target}_audit_plan.json', dict(
            grid=8, size=1024, seed=42, random_sample=random_tiles().tolist(),
            frozen_model_counts=model_counts.tolist(),
            instruction='Count every target center in every assigned tile, including missed detections. Do not replace difficult tiles.'))
        joblib.dump(dict(classifier=patch, threshold=main_threshold,
            single_view_threshold=selections['structural_gradient']['threshold'],
            config=asdict(cfg), source_image=source,
            note='Exploratory source-region fit. Scores are not calibrated probabilities.'),
            args.output/f'model_from_img{source}.joblib', compress=3)
        detection_overlay(test['image'], predictions, test['gt'],
            f'Transfer img{source} -> img{target} | P={raw["precision"]:.2f}, R={raw["recall"]:.2f}',
            args.output/f'img{target}_detection_overlay.png')
        # Choose at most two matched and two unmatched predictions for diagnostic maps.
        matched = match_objects(predictions, test['gt']); matched_ids = set(matched[:, 0])
        order = np.argsort(-stable_test[main_ids])
        selected = [i for i in order if i in matched_ids][:1] + [i for i in order if i not in matched_ids][:1]
        occlusions = []
        for number, i in enumerate(selected):
            crop, impact, baseline = conditional_occlusion(test['gray'], predictions[i], patch)
            np.savez_compressed(args.output/f'img{target}_occlusion{number}.npz', crop=crop, impact=impact,
                                baseline=baseline, proposal=predictions[i], matched=i in matched_ids)
            occlusions.append(dict(file=f'img{target}_occlusion{number}.npz', matched=i in matched_ids, score=baseline))
        folds.append(dict(source=source, target=target, fit_seconds=fit_seconds,
            test_stability_scoring_seconds=stability_seconds,
            source_training_candidates=int((y_train >= 0).sum()),
            source_training_positive_candidates=int((y_train == 1).sum()),
            source_training_reference_objects=len(source_train_gt),
            source_validation_reference_objects=len(val_gt),
            split=dict(train_x_less_than=368, validation_x_at_least=656, gap=288),
            thresholds=selections, conservative_validation=high_val,
            tolerance_sensitivity={str(t): metrics(predictions, test['gt'], t) for t in (.5, .75, 1.)},
            whole_pipeline_single_view_perturbations=perturbations,
            occlusions=occlusions,
            tile_model_reference_correlation=float(np.corrcoef(model_counts, reference_counts)[0, 1]),
            fixed_random_audit={'sample_ids':random_tiles().tolist(),
                              'reference_counts':[int(reference_counts[i]) for i in random_tiles()]},
            ))
    # Unlabeled quadrat uses the predefined img1-source model, not test selection.
    name = 'unseen_img1.png'
    unseen, _ = rectify(read_rgb(args.data_dir/name), frames[name])
    model = joblib.load(args.output/'model_from_img1.joblib')
    pp, gg = propose(unseen, cfg); xx, _ = features(gg, pp); hh = gradient_patches(gg, pp)
    us, uv = patch_stability_scores(unseen, pp, model['classifier'], xx, hh)
    ii = select_detections(pp, us, model['threshold'])
    create_review(unseen, np.column_stack((pp[ii], us[ii], uv[ii].std(1))), args.output/'review_unseen_img1.html',
                  'Unseen image 1: unverified suggestions / no reference labels')
    Image.fromarray(unseen).save(args.output/'unseen_img1_rectified.jpg', quality=94)
    np.savetxt(args.output/'unseen_img1_predictions.csv', np.column_stack((pp[ii], us[ii], uv[ii].std(1))),
               delimiter=',', header='x,y,proposal_radius,ring_response,stability_score,view_score_sd', comments='')
    unseen_summary = dict(predicted_count=len(ii), labeled=False,
                         note='Unverified suggestions only. No accuracy or count-error metric is available.')
    # Extra close-up has no frame or labels and a radically different pixel scale.
    unseen2_summary = dict(status='abstain', reason='No green quadrat or frame calibration; different scale; no labels.')
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data_dir.glob('*.png'))}
    summary = dict(config=asdict(cfg), diagnostics=diagnostics, results=results, folds=folds,
        audits=audit_results, unseen_img1=unseen_summary, unseen_img2=unseen2_summary,
        elapsed_seconds=time.perf_counter()-start_all, data_sha256=hashes,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__, opencv=cv2.__version__),
        evaluation_status='Exploratory cross-image transfer, n=2 photographs. Not an untouched test set or field validation.')
    save_json(args.output/'metrics.json', summary)
    with open(args.output/'metrics.csv', 'w') as f:
        keys=['source_image','test_image','method','reference_count','predicted_count','precision','recall','f1','signed_error','relative_error','threshold']
        w=csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(results)
    print('Finished. See', args.output/'metrics.json', flush=True)


if __name__ == '__main__':
    main()
