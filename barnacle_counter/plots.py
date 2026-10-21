import numpy as np
import matplotlib.pyplot as plt
from .metrics import match_objects


def show_match(ax, rgb, pred, ref, title='', tolerance=.75):
    # green dot = on a barnacle, red x = extra dot, yellow ring = barnacle the model missed
    m = match_objects(pred, ref, tolerance) if len(pred) else np.empty((0, 2), int)
    hit_p, hit_r = set(m[:, 0].tolist()), set(m[:, 1].tolist())
    ax.imshow(rgb)
    miss = [j for j in range(len(ref)) if j not in hit_r]
    extra = [i for i in range(len(pred)) if i not in hit_p]
    ax.scatter(pred[list(hit_p), 0], pred[list(hit_p), 1], s=8, c='lime')
    if extra:
        ax.scatter(pred[extra, 0], pred[extra, 1], s=40, c='red', marker='x')
    if miss:
        ax.scatter(ref[miss, 0], ref[miss, 1], s=120, facecolors='none', edgecolors='yellow', linewidths=1.5)
    ax.set_title(f'{title}\n{len(hit_p)} found, {len(extra)} extra, {len(miss)} missed', fontsize=10)
    ax.axis('off')


def show_dots(ax, rgb, pred, title=''):
    # just the model's dots, no answer key (for unlabelled photos)
    ax.imshow(rgb)
    ax.scatter(pred[:, 0], pred[:, 1], s=4, c='lime')
    ax.set_title(f'{title}\n{len(pred)} dots', fontsize=10)
    ax.axis('off')


def bars(ax, labels, values, title='', xmax=1.0):
    # quick horizontal bar chart for comparing scores
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color='#3a7d44')
    for yi, v in zip(y, values):
        ax.text(v + 0.01, yi, f'{v:.3f}', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, xmax)
    ax.set_title(title, fontsize=10)
