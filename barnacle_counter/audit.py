import numpy as np
from scipy.stats import t


def tile_counts(xy, size=1024, grid=8):
    # count dots per tile on an 8x8 grid (64 tiles of 128px)
    xy = np.asarray(xy)
    if not len(xy):
        return np.zeros(grid * grid, int)
    ij = np.floor(np.clip(xy[:, :2], 0, size - 1e-3) * grid / size).astype(int)
    return np.bincount(ij[:, 1] * grid + ij[:, 0], minlength=grid * grid)


def corrected_count(model_counts, sampled_ids, human_counts, beta=1.):
    # the audit: a person fully counts some RANDOM tiles. on those tiles we see how far off the model is,
    # and use that to correct the model's total. estimate = model total + 64 * mean(human - model).
    # beta=0 means 'ignore the model' (plain random sampling by hand), useful for comparison.
    # the 95% interval is approximate (t interval). it only works if the tiles were picked at random,
    # not 'tiles that looked wrong'.
    mc = np.asarray(model_counts, float) * beta
    ids = np.asarray(sampled_ids, int)
    h = np.asarray(human_counts, float)
    m, n = len(mc), len(ids)
    resid = h - mc[ids]
    est = mc.sum() + m * resid.mean()
    se = m * np.sqrt((1 - n / m) * resid.var(ddof=1) / n) if n > 1 else float('nan')
    half = float(t.ppf(.975, n - 1) * se) if n > 1 else float('nan')
    return dict(estimate=float(est), interval95=[float(est - half), float(est + half)], half_width=half)


def simulate_audits(model_counts, ref_counts, n=16, repeats=2000, beta=1., seed=7):
    # pretend the mask labels are a perfect human, run the audit many times with different random tiles,
    # and see how wide the interval is and how often it actually contains the true count
    rng = np.random.default_rng(seed)
    truth = float(np.sum(ref_counts))
    half, cover, err = [], [], []
    for _ in range(repeats):
        ids = rng.choice(len(model_counts), n, replace=False)
        a = corrected_count(model_counts, ids, np.asarray(ref_counts)[ids], beta)
        half.append(a['half_width'])
        cover.append(a['interval95'][0] <= truth <= a['interval95'][1])
        err.append(a['estimate'] - truth)
    return dict(n=n, mean_half_width_pct=100 * np.mean(half) / truth, coverage95=float(np.mean(cover)),
                within5pct=float(np.mean(np.abs(err) <= .05 * truth)))
