"""Design-based error correction using a uniform random tile audit.

Disagreement-based review is useful for repair but is NOT a random sample.
It must not be fed into this estimator without known inclusion probabilities.
"""
import numpy as np
from scipy.stats import t


def tile_ids(xy, size=1024, grid=8):
    xy = np.asarray(xy)
    if len(xy) == 0:
        return np.array([], int)
    if np.any(xy[:, :2] < 0) or np.any(xy[:, :2] >= size):
        raise ValueError('All centers must be inside the frame.')
    ij = np.floor(xy[:, :2]*grid/size).astype(int)
    return ij[:, 1]*grid+ij[:, 0]


def tile_counts(xy, size=1024, grid=8):
    return np.bincount(tile_ids(xy, size, grid), minlength=grid*grid)


def random_tiles(grid=8, n=16, seed=42):
    if not 1 <= n <= grid*grid:
        raise ValueError('Audit size must be between 1 and the number of tiles.')
    return np.random.default_rng(seed).choice(grid*grid, n, replace=False)


def corrected_count(model_counts, sampled_ids, human_counts, beta=1.):
    """Difference estimator beta*N_model + M*mean(human_i-beta*model_i).

    SRS without replacement. Approximate t interval with finite-population
    correction, not an exact coverage guarantee (especially for small n).
    beta must be fixed independently of this audit; beta=0 is manual sampling.
    """
    if not np.isfinite(beta):
        raise ValueError('Use a finite coefficient fixed before this audit sample.')
    model_counts = np.asarray(model_counts, float)*beta
    sampled_ids = np.asarray(sampled_ids, int)
    human_counts = np.asarray(human_counts, float)
    m = len(model_counts); n = len(sampled_ids)
    if n < 1 or n > m or len(np.unique(sampled_ids)) != n:
        raise ValueError('Need a nonempty random sample of distinct tiles.')
    if human_counts.shape != (n,) or np.any(human_counts < 0):
        raise ValueError('One nonnegative human count is required per sampled tile.')
    if np.any(sampled_ids < 0) or np.any(sampled_ids >= m):
        raise ValueError('Invalid tile ID.')
    residual = human_counts-model_counts[sampled_ids]
    estimate = model_counts.sum()+m*residual.mean()
    se = m*np.sqrt((1-n/m)*residual.var(ddof=1)/n) if n > 1 else float('nan')
    half = float(t.ppf(.975, n-1)*se) if n > 1 else float('nan')
    return dict(estimate=float(estimate), standard_error=float(se),
                interval95=[float(estimate-half), float(estimate+half)],
                sampled_tiles=n, total_tiles=m, fixed_model_coefficient=float(beta),
                note='Approximate design-based interval; assumes accurate, complete human tile counts.')


def simulate_audits(model_counts, reference_counts, n=16, repeats=1000, seed=42):
    """Oracle simulation using supplied labels instead of actual human review."""
    rng = np.random.default_rng(seed); m = len(model_counts)
    estimates = []; covered = []; widths = []
    truth = np.sum(reference_counts)
    for _ in range(repeats):
        ids = rng.choice(m, n, replace=False)
        a = corrected_count(model_counts, ids, reference_counts[ids])
        estimates.append(a['estimate']); lo, hi = a['interval95']
        covered.append(lo <= truth <= hi); widths.append(hi-lo)
    e = np.asarray(estimates)-truth
    return dict(n=n, repeats=repeats, bias=float(e.mean()), mae=float(np.abs(e).mean()),
                rmse=float(np.sqrt((e**2).mean())),
                within5pct=float((np.abs(e)<=.05*truth).mean()),
                coverage95=float(np.mean(covered)), mean_interval_width=float(np.mean(widths)))
