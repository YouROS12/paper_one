"""Generate all figures for paper_2 (plant disease instance segmentation).
All data sourced from real experiment logs in experiments/; no synthetic curves.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import csv
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

NAVY  = '#1a3a7a'
BLUE  = '#1a6fd4'
CYAN  = '#0077cc'
TEAL  = '#0a9278'
AMBER = '#b86e00'
ROSE  = '#c0325a'
LIGHT = '#f0f4ff'
GREY  = '#d6e0f5'

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')


def load_metric_from_csv(exp_name, metric_col_substr):
    """Read a per-epoch metric from an experiment's results.csv.
    Returns (epochs, values) lists, or ([], []) if file is missing/empty."""
    csv_path = os.path.join(EXPERIMENTS_DIR, exp_name, 'results.csv')
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return [], []
    epochs, values = [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            col = next((k for k in row if metric_col_substr in k), None)
            if col is None:
                break
            ep_key = next((k for k in row if 'epoch' in k.lower()), None)
            if ep_key is None:
                break
            try:
                epochs.append(int(row[ep_key].strip()))
                values.append(float(row[col].strip()))
            except (ValueError, KeyError):
                continue
    return epochs, values


# ── Figure 1: System Architecture ────────────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.5)
    ax.axis('off')
    fig.patch.set_facecolor(LIGHT)

    def box(cx, cy, w, h, color, label, sublabel=None, fs=9, lw=1.5):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle="round,pad=0.08",
                              facecolor=color, edgecolor='white',
                              linewidth=lw, zorder=3)
        ax.add_patch(rect)
        y_text = cy + (0.18 if sublabel else 0)
        ax.text(cx, y_text, label, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white', zorder=4)
        if sublabel:
            ax.text(cx, cy - 0.22, sublabel, ha='center', va='center',
                    fontsize=7, color='white', alpha=0.85, zorder=4)

    def arrow(x1, x2, y, color=BLUE):
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=1.8, mutation_scale=14), zorder=2)

    ax.text(7, 5.15, 'PlantSeg Instance Segmentation Pipeline',
            ha='center', va='center', fontsize=13, fontweight='bold', color=NAVY)

    for x, lbl in [(1.2, 'INPUT'), (3.8, 'PREPROCESSING'), (7.0, 'MODEL'),
                   (10.2, 'OUTPUT'), (12.8, 'EVALUATION')]:
        ax.text(x, 4.55, lbl, ha='center', va='center', fontsize=7.5,
                color=NAVY, fontweight='bold', alpha=0.6, fontfamily='monospace')

    box(1.2, 2.75, 1.7, 3.2, NAVY, 'PlantSeg', '11,458 images\n115 classes · 34 crops', fs=8.5)
    arrow(2.05, 2.55, 2.75)

    strat_colors = [BLUE, BLUE, '#2d6bbf', '#2d6bbf', TEAL, TEAL]
    strats = ['baseline', 'random_crop', 'center_crop',
              'bbox_guided ★', 'mask_guided ★', 'mask_guided_aware']
    block_x, block_w = 3.8, 2.0
    total_h = 3.0
    row_h = total_h / 6
    y_top = 2.75 + total_h / 2
    rect_bg = FancyBboxPatch((block_x - block_w/2 - 0.05, y_top - total_h - 0.05),
                             block_w + 0.1, total_h + 0.1,
                             boxstyle="round,pad=0.05",
                             facecolor=GREY, edgecolor=BLUE, linewidth=1.2, zorder=2)
    ax.add_patch(rect_bg)
    for i, (s, c) in enumerate(zip(strats, strat_colors)):
        cy = y_top - row_h * (i + 0.5)
        rect = FancyBboxPatch((block_x - block_w/2, cy - row_h*0.38),
                              block_w, row_h * 0.76,
                              boxstyle="round,pad=0.04",
                              facecolor=c, edgecolor='white', linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        ax.text(block_x, cy, s, ha='center', va='center',
                fontsize=7.2, color='white', fontweight='bold', zorder=4)

    arrow(4.82, 5.3, 2.75)
    box(7.0, 2.75, 2.8, 3.2, CYAN, 'YOLOv11m-seg',
        '~20M params · ~40 GFLOPs\n200 epochs · Single A100 · ~11.5h', fs=9)
    arrow(8.4, 8.9, 2.75)
    box(10.2, 2.75, 2.2, 3.2, AMBER, 'Instance Masks',
        'Per-lesion polygons\n+ bounding boxes', fs=8.5)
    arrow(11.3, 11.8, 2.75)

    eval_x = 12.8
    box(eval_x, 3.55, 1.8, 1.5, TEAL, 'mIoU', 'Pixel-level', fs=8.5)
    box(eval_x, 1.95, 1.8, 1.5, ROSE, 'mAP50', 'Instance-level', fs=8.5)
    ax.annotate('', xy=(eval_x, 3.55 - 0.75), xytext=(eval_x, 1.95 + 0.75),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2,
                                mutation_scale=10), zorder=2)
    ax.text(eval_x + 0.55, 2.75, 'Dual\nmetric', ha='left', va='center',
            fontsize=6.5, color='#555', style='italic')

    for cx, cy, col, lbl in [(3.2, 0.55, BLUE, 'Annotation-free'),
                              (5.1, 0.55, TEAL, 'Annotation-guided'),
                              (7.2, 0.55, '#e8b800', '★ Best results')]:
        rect = FancyBboxPatch((cx - 0.12, cy - 0.16), 0.24, 0.32,
                              boxstyle="round,pad=0.03",
                              facecolor=col, edgecolor='none', zorder=3)
        ax.add_patch(rect)
        ax.text(cx + 0.22, cy, lbl, ha='left', va='center', fontsize=7.5, color=NAVY)

    plt.tight_layout(pad=0.3)
    plt.savefig('figures/fig1_architecture.png', dpi=180,
                bbox_inches='tight', facecolor=LIGHT)
    plt.close()
    print('✓ fig1_architecture.png')


# ── Figure 2: Strategy Comparison Bar Chart ───────────────────────────────────
def fig_strategy_comparison():
    # Corrected: all values from test-set evaluation_results.json
    # mAP50 for random_crop corrected to test-set value (28.96%, not val 27.26%)
    strategies = ['baseline', 'random\ncrop', 'center\ncrop',
                  'mask\nguided\naware', 'mask\nguided', 'bbox\nguided']
    miou  = [27.93, 30.22, 31.59, 31.65, 33.72, 34.12]
    map50 = [29.83, 28.96, 30.48, 31.28, 32.69, 32.25]

    x = np.arange(len(strategies))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')

    bars1 = ax.bar(x - w/2, miou,  w, label='mIoU',  color=BLUE,  alpha=0.88, zorder=3)
    bars2 = ax.bar(x + w/2, map50, w, label='mAP50', color=CYAN, alpha=0.88, zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7.5, color=BLUE, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7.5, color=CYAN, fontweight='bold')

    bars1[5].set_edgecolor(AMBER); bars1[5].set_linewidth(2.2)  # bbox_guided best mIoU
    bars2[4].set_edgecolor(AMBER); bars2[4].set_linewidth(2.2)  # mask_guided best mAP50

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=8.5)
    ax.set_ylabel('Score (%)', fontsize=10)
    ax.set_ylim(24, 36.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_facecolor('#fafcff')
    ax.set_title('Preprocessing Strategy Comparison — PlantSeg Test Set\n'
                 'YOLOv11m-seg · 200 epochs · Single A100',
                 fontsize=10.5, color=NAVY, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)

    ax.annotate('Best mIoU\n(34.12%)', xy=(5 - w/2, 34.12), xytext=(4.0, 35.8),
                fontsize=7.5, color=BLUE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    ax.annotate('Best mAP50\n(32.69%)', xy=(4 + w/2, 32.69), xytext=(2.8, 35.2),
                fontsize=7.5, color=CYAN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CYAN, lw=1.2))

    plt.tight_layout()
    plt.savefig('figures/fig2_strategy_comparison.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ fig2_strategy_comparison.png')


# ── Figure 3: Metric Divergence ───────────────────────────────────────────────
def fig_metric_divergence():
    strategies = ['baseline', 'random_crop', 'center_crop',
                  'mask_guided\n_aware', 'mask_guided', 'bbox_guided']
    miou  = [27.93, 30.22, 31.59, 31.65, 33.72, 34.12]
    map50 = [29.83, 28.96, 30.48, 31.28, 32.69, 32.25]
    colors = [NAVY, ROSE, '#6678a0', AMBER, CYAN, BLUE]
    markers = ['o', 's', 'D', '^', 'P', '*']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.set_facecolor('#fafcff')
    for i, (s, m1, m2, c, mk) in enumerate(zip(strategies, miou, map50, colors, markers)):
        ax.scatter(m1, m2, c=c, marker=mk, s=130, zorder=4,
                   linewidths=0.8, edgecolors='white')
        offsets = [(0.15, 0.1), (-1.1, -0.45), (0.15, -0.45),
                   (-1.4, 0.1), (0.15, 0.1), (0.15, -0.45)]
        ax.text(m1 + offsets[i][0], m2 + offsets[i][1],
                s.replace('\n_', '_'), fontsize=7.5, color=c, fontweight='bold')

    lims = [26, 35.5]
    ax.plot(lims, lims, '--', color='#aaa', lw=1, label='mIoU = mAP50', zorder=1)
    ax.set_xlim(26, 35.5); ax.set_ylim(26, 35.5)
    ax.set_xlabel('mIoU (%)', fontsize=10); ax.set_ylabel('mAP50 (%)', fontsize=10)
    ax.set_title('mIoU vs mAP50 per Strategy\n(diagonal = perfect agreement)',
                 fontsize=10, color=NAVY)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, zorder=0)
    ax.annotate('random_crop\nanomaly', xy=(30.22, 28.96), xytext=(27.2, 28.2),
                fontsize=7.5, color=ROSE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ROSE, lw=1.2))

    ax2 = axes[1]
    ax2.set_facecolor('#fafcff')
    miou_rank  = [sorted(miou,  reverse=True).index(v) + 1 for v in miou]
    map50_rank = [sorted(map50, reverse=True).index(v) + 1 for v in map50]

    for i, (s, r1, r2, c) in enumerate(zip(strategies, miou_rank, map50_rank, colors)):
        ax2.plot([0, 1], [r1, r2], '-', color=c, lw=2, alpha=0.8, zorder=3)
        ax2.scatter([0, 1], [r1, r2], c=c, s=80, zorder=4)
        ax2.text(-0.08, r1, s.replace('\n_', '_'), ha='right', va='center',
                 fontsize=7.5, color=c, fontweight='bold')
        ax2.text(1.08, r2, f'{r2}→', ha='left', va='center', fontsize=7.5, color=c)

    i_rc = strategies.index('random_crop')
    ax2.plot([0, 1], [miou_rank[i_rc], map50_rank[i_rc]],
             '-', color=ROSE, lw=3, alpha=1.0, zorder=5)

    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(6.5, 0.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['mIoU rank', 'mAP50 rank'], fontsize=10)
    ax2.set_yticks(range(1, 7))
    ax2.set_ylabel('Rank (1 = best)', fontsize=10)
    ax2.set_title('Ranking Divergence Between Metrics\n(crossing lines = rank swap)',
                  fontsize=10, color=NAVY)
    ax2.grid(axis='y', alpha=0.25)

    fig.suptitle('Metric Divergence: Why Both mIoU and mAP50 Are Necessary',
                 fontsize=11.5, fontweight='bold', color=NAVY, y=1.01)
    plt.tight_layout()
    plt.savefig('figures/fig3_metric_divergence.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ fig3_metric_divergence.png')


# ── Figure 4: SOTA comparison ─────────────────────────────────────────────────
def fig_sota_comparison():
    methods = [
        'DeepLabv3\n(ResNet-50)',
        'DeepLabv3\n(ResNet-101)',
        'DeepLabv3+\n(ResNet-50)',
        'DeepLabv3+\n(ResNet-101)',
        'SAN\n(ViT-B/16)',
        'SAN\n(ViT-L/14)',
        'SegNeXt\n(MSCAN-L)',
        'Ours — baseline\n(YOLOv11m-seg)',
        'Ours — bbox_guided\n(YOLOv11m-seg) ★',
        'Ours — mask_guided\n(YOLOv11m-seg) ★',
    ]
    miou   = [17.24, 20.72, 25.08, 27.18, 34.79, 36.91, 44.52, 27.93, 34.12, 33.72]
    params = [40, 59, 43, 62, 86, 307, 49, 20, 20, 20]
    colors = ([GREY]*6 + ['#aec6cf'] + [NAVY, BLUE, CYAN])

    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.set_facecolor('#fafcff')
    bars = ax.barh(y, miou, color=colors, edgecolor='white', linewidth=0.8,
                   height=0.65, zorder=3)
    for i, (bar, v) in enumerate(zip(bars, miou)):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f'{v:.2f}%', va='center', fontsize=8,
                color=colors[i] if colors[i] not in (GREY, '#aec6cf') else '#555',
                fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=8.5)
    ax.set_xlabel('mIoU (%)', fontsize=10)
    ax.set_xlim(0, 50)
    ax.set_title('mIoU on PlantSeg (prior: semantic seg. | ours: instance seg.)',
                 fontsize=10.5, color=NAVY, pad=8)
    ax.axvline(34.12, color=BLUE, lw=1.2, ls='--', alpha=0.5, zorder=2)
    ax.grid(axis='x', alpha=0.25, zorder=0)
    ax.axhline(6.5, color='#aaa', lw=1, ls='-', zorder=2)
    ax.text(1, 6.65, 'Prior work (semantic segmentation)',
            fontsize=7.5, color='#888', style='italic')
    ax.text(1, 6.35, 'This work (instance segmentation)',
            fontsize=7.5, color=BLUE, style='italic', fontweight='bold')

    ax2 = axes[1]
    ax2.set_facecolor('#fafcff')
    for i, (p, c) in enumerate(zip(params, colors)):
        size = p / max(params) * 900 + 80
        ax2.scatter([0], [y[i]], s=size, c=c, alpha=0.8,
                    edgecolors='white', linewidth=0.8, zorder=3)
        ax2.text(0.18, y[i], f'{p}M', va='center', fontsize=8,
                 color=c if c not in (GREY, '#aec6cf') else '#555')
    ax2.set_xlim(-0.5, 0.8)
    ax2.set_ylim(-0.5, len(methods) - 0.5)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xticks([])
    ax2.set_title('Parameters\n(bubble ∝ size)', fontsize=9, color=NAVY)
    ax2.axhline(6.5, color='#aaa', lw=1, ls='-', zorder=2)
    ax2.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig('figures/fig4_sota_comparison.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ fig4_sota_comparison.png')


# ── Figure 5: Real convergence curves from training logs ─────────────────────
def fig_convergence():
    """Plot real val mAP50(B) curves from results.csv for all available experiments."""
    experiments = [
        ('bbox_guided_seed42',  'bbox_guided',  BLUE,  '-',  2.2),
        ('mask_guided_seed42',  'mask_guided',  CYAN,  '-',  2.2),
        ('center_crop_seed42',  'center_crop',  TEAL,  '--', 1.5),
        ('random_crop_seed42',  'random_crop',  ROSE,  '--', 1.5),
        ('baseline_seed42',     'baseline',     GREY,  ':',  1.5),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafcff')

    for exp_name, label, color, ls, lw in experiments:
        epochs, values = load_metric_from_csv(exp_name, 'mAP50(B)')
        if not epochs:
            print(f'  [warn] No data for {exp_name}')
            continue
        values_pct = [v * 100 for v in values]
        ax.plot(epochs, values_pct, color=color, lw=lw, ls=ls,
                label=label, alpha=0.9)
        ax.scatter([epochs[-1]], [values_pct[-1]], c=color, s=55, zorder=5)
        ax.text(epochs[-1] + 2, values_pct[-1],
                f'{values_pct[-1]:.1f}%', fontsize=7.5, color=color,
                fontweight='bold', va='center')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation mAP50 — Box (%)', fontsize=11)
    ax.set_title('Training Convergence: Validation mAP50 over 200 Epochs\n'
                 'Annotation-guided strategies (solid) consistently outperform annotation-free (dashed)',
                 fontsize=10.5, color=NAVY, pad=8)
    ax.legend(fontsize=9, framealpha=0.9, loc='upper left')
    ax.set_xlim(1, 210)
    ax.grid(alpha=0.25)

    ax.text(0.98, 0.05,
            'mask_guided_aware: convergence data unavailable\n'
            'Test results: mIoU=31.65%, mAP50=31.28%',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
            color=AMBER, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8e8', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figures/fig5_convergence.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('✓ fig5_convergence.png')


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    fig_architecture()
    fig_strategy_comparison()
    fig_metric_divergence()
    fig_sota_comparison()
    fig_convergence()
    print('\nAll figures saved to figures/')
