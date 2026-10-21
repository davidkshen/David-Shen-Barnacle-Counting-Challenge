"""Figures for the improvement report, drawn only from results/*.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'
F = ROOT / 'figures'
F.mkdir(exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
                     'axes.labelcolor': INK2, 'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 1, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
                     'axes.titlesize': 11, 'axes.titlecolor': INK, 'lines.linewidth': 2,
                     'lines.solid_capstyle': 'round'})
NAMES = {'pretrained_from2': 'Pretrained, photo 2 → 1', 'pretrained_from1': 'Pretrained, photo 1 → 2',
         'scratch_from2': 'Scratch, photo 2 → 1', 'scratch_from1': 'Scratch, photo 1 → 2'}


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_threshold_vs_ranking():
    d = load('diagnosis.json')
    if not d:
        return
    tags = ['pretrained_from2', 'pretrained_from1', 'scratch_from2', 'scratch_from1']
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(tags))[::-1]
    for yi, t in zip(y, tags):
        a, b = d[t]['reported']['f1'], d[t]['oracle_best']['f1']
        ax.plot([a, b], [yi, yi], color=GRID, lw=6, solid_capstyle='round', zorder=1)
        ax.scatter([a], [yi], s=70, color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
        ax.scatter([b], [yi], s=70, color=ORANGE, edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(max(a, b) + .02, yi, f'{a:.2f} → {b:.2f}', va='center', color=INK2, fontsize=9)
    ax.set_yticks(y, [NAMES[t] for t in tags])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Full opposite-photo center F1')
    ax.grid(axis='y', visible=False)
    ax.scatter([], [], s=50, color=BLUE, label='Source-photo threshold (reported)')
    ax.scatter([], [], s=50, color=ORANGE, label='Best target threshold (oracle, diagnostic only)')
    ax.set_ylim(-0.6, len(tags) - 0.4)
    ax.set_title('Wrong confidence cut-off vs wrong ranking', loc='left')
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(F / 'threshold_vs_ranking.png', dpi=170)
    plt.close(fig)


def fig_budget():
    d = load('adapt_budget.json')
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, t in zip(axes, ['pretrained_from2', 'pretrained_from1']):
        rows = d[t]['rows']
        Ks = [1, 2, 4, 8]
        none = rows[0]['none']['f1']
        ax.axhline(none, color=MUTED, lw=1, label='no adaptation (base model)')
        for meth, col, lab, dx in (('threshold', BLUE, 'threshold only', -0.06), ('refit', ORANGE, 'last-layer refit', 0.06)):
            means = []
            for K in Ks:
                v = np.array([r[meth]['f1'] for r in rows if r['K'] == K])
                xs = np.full(len(v), K) * np.exp(dx)
                ax.scatter(xs, v, s=10, color=col, alpha=.28, linewidth=0)
                means.append(v.mean())
            ax.plot(np.array(Ks) * np.exp(dx), means, color=col, marker='o', ms=6, mec=SURF, mew=2,
                    label=f'{lab} (mean; dots = draws)')
        ax.set_xscale('log', base=2)
        ax.set_xticks(Ks, ['1\n(1.6%)', '2\n(3.1%)', '4\n(6.3%)', '8\n(12.5%)'])
        ax.minorticks_off()
        ax.set_xlim(0.8, 10.5)
        ax.set_title(NAMES[t], loc='left')
        ax.set_xlabel('Fully labelled 128-px cells (share of quadrat)')
    axes[0].set_ylabel('Reserved-stripe F1 (20 random draws)')
    axes[0].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F / 'adaptation_budget.png', dpi=170)
    plt.close(fig)


def fig_audit():
    d = load('audit_sim.json')
    c = load('combined_workflow_sim.json')
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=False)
    for ax, t in zip(axes, ['pretrained_from2', 'pretrained_from1']):
        r = d[t]
        ns = [x['n'] for x in r['model_plus_audit']]
        a = [x['mean_halfwidth_pct'] for x in r['model_plus_audit']]
        m = [x['mean_halfwidth_pct'] for x in r['manual_sample']]
        ax.plot(ns, m, color=ORANGE, marker='o', ms=6, mec=SURF, mew=2, label='Manual random sample (no model)')
        ax.plot(ns, a, color=BLUE, marker='o', ms=6, mec=SURF, mew=2, label='Frozen model + random audit')
        if c and t in c:
            for n, v in c[t]['by_audit_size'].items():
                q = v['adapted+audit']
                ax.scatter([int(n) + 2], [q['halfwidth_pct']], s=60, marker='D', color=AQUA, edgecolor=SURF, linewidth=2, zorder=4,
                           label='2 labelled cells → re-pick threshold → audit (tiles = audit + 2)' if n == '8' else None)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ns, [f'{n}\n({n / 64:.0%})' for n in ns])
        ax.minorticks_off()
        ax.set_xlim(3.3, 40)
        ax.set_ylim(0, None)
        ax.set_title(NAMES[t], loc='left')
        ax.set_xlabel('Randomly audited tiles of 64 (share of quadrat)')
    axes[0].set_ylabel('Mean 95% interval half-width\n(% of true count)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(F / 'count_margin_of_error.png', dpi=170)
    plt.close(fig)


def fig_review():
    d = load('review_sim.json')
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, t in zip(axes, ['pretrained_from2', 'pretrained_from1']):
        cur = d[t]['curve']
        N = d[t]['reference_count']
        th = np.array([r['threshold'] for r in cur])
        for c, col in ((1, BLUE), (3, ORANGE)):
            eff = np.array([100 * (r['fp'] + c * r['fn']) / (c * N) for r in cur])
            ax.plot(th, eff, color=col, label=f'Adding a miss costs {c}× a deletion (assumed)')
            i = int(np.argmin(eff))
            ax.scatter([th[i]], [eff[i]], s=50, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        src = d[t]['source_operating_point']['threshold']
        ax.axvline(src, color=MUTED, lw=1)
        ax.text(src, 103, ' source threshold', color=INK2, fontsize=8.5, va='bottom')
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_ylim(0, 110)
        ax.set_title(NAMES[t], loc='left')
        ax.set_xlabel('Confidence threshold shown to the reviewer')
    axes[0].set_ylabel('Simulated correction effort\n(% of counting from scratch)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], marker='o', ls='', color=MUTED)], l + ['lowest-effort threshold'],
               loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(F / 'review_effort_simulation.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    fig_threshold_vs_ranking()
    fig_budget()
    fig_audit()
    fig_review()
    print(sorted(p.name for p in F.glob('*.png')))
