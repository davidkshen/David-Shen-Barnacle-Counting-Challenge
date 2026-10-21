"""Compare a conditional model occlusion map with REAL human diagnostic maps.

Example: python compare_importance.py results/img1_occlusion0.npz human1.png human2.png
Each human should independently paint the pixels used to recognize the target
on the SAME 97x97 crop (white=important, black=unmarked), without seeing the
machine map. Barnacle outline masks are NOT diagnostic importance maps.
"""
import argparse
import itertools
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('machine'); p.add_argument('human', nargs='+')
    p.add_argument('--output', default='importance_comparison.json')
    a = p.parse_args(); z = np.load(a.machine)
    machine = np.maximum(z['impact'], 0)
    people = [np.asarray(Image.open(x).convert('L'), dtype=float)/255 for x in a.human]
    if any(h.shape != machine.shape for h in people):
        raise ValueError(f'Every human map must have shape {machine.shape}; no automatic resizing is permitted.')
    # Match smoothing across human and model maps before a rank correlation.
    machine = gaussian_filter(machine, 2)
    people = [gaussian_filter(h, 2) for h in people]
    aggregate = np.mean(people, axis=0)
    inter = [correlation(x, y) for x, y in itertools.combinations(people, 2)]
    result = dict(spearman_model_human=correlation(machine, aggregate),
        individual_correlations=[correlation(machine, h) for h in people],
        inter_rater_correlations=inter, number_of_people=len(people),
        smoothing_sigma_pixels=2,
        caution='Exploratory conditional score-attribution comparison. No neural-alignment claim, no assumed independent pixels, no pixel-level p-value.')
    with open(a.output, 'w') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
