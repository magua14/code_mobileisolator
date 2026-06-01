import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import string

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
CSV_PATH   = os.path.join(DATA_DIR, 'lambda_validity_results_500.csv')
SAVE_DIR   = SCRIPT_DIR
LAMBDA_TARGETS = [1, 10, 20]

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

CONFIGS = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
COL_TITLES = [
    'Heterogeneous-Random',
    'Heterogeneous-Localized',
    'Homogeneous-Random',
    'Homogeneous-Localized'
]
METHODS = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']
METHOD_LABELS =['Adaptive TD', 'Adaptive T-Katz', 'Adaptive TIA', 'MobileIsolator']
METHOD_COLORS = {
    'Adaptive TD': '#0072B2',
    'Adaptive Katz': '#D55E00',
    'Adaptive TIA': '#009E73',
    'GNN-RL': '#CC79A7',
}
METHOD_MARKERS = {
    'Adaptive TD': 'o',
    'Adaptive Katz': 's',
    'Adaptive TIA': '^',
    'GNN-RL': 'D',
}

BOX_COLORS = ['#E69F00', '#56B4E9', '#009E73']
PANEL_LABELS = list(string.ascii_lowercase)[:24]

R_MIN, R_MAX = 0.008, 0.052
X_TICKS = [0.01, 0.02, 0.03, 0.04, 0.05]

FIG_W = 7.087
PAN_H = 1.2
LEG_H = 0.7
TOP_H = 0.3
FIG_H = 6 * PAN_H + LEG_H + TOP_H


def plot_all(csv_path: str = CSV_PATH, save_dir: str = SAVE_DIR):
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    gs_top = 1.0 - (TOP_H / FIG_H)
    gs_bot = LEG_H / FIG_H

    gs_blocks = fig.add_gridspec(3, 1, hspace=0.25, left=0.08, right=0.94, top=gs_top, bottom=gs_bot)

    axes = []
    for i in range(3):
        gs_row = gs_blocks[i].subgridspec(2, 4, hspace=0.25, wspace=0.38)
        row_axes = []
        for r in range(2):
            col_axes = []
            for c in range(4):
                ax = fig.add_subplot(gs_row[r, c])
                col_axes.append(ax)
            row_axes.append(col_axes)
        axes.append(row_axes)

    panel_idx = 0
    for block_idx, lam in enumerate(LAMBDA_TARGETS):
        df_lam = df[df['lambda_eff'] == lam].copy()
        if df_lam.empty:
            print(f"  [skip] λ={lam} not found in CSV.")
            continue

        for row_i_rel, (metric, ylabel) in enumerate([('pc', r'$P_C$'), ('anc', 'ANC')]):
            for col_i, config in enumerate(CONFIGS):
                ax = axes[block_idx][row_i_rel][col_i]
                sub = df_lam[df_lam['config'] == config].sort_values('attack_ratio')

                for method in METHODS:
                    m_sub = sub[sub['method'] == method]
                    if m_sub.empty:
                        continue

                    xs = m_sub['attack_ratio'].values
                    ys = m_sub[f'{metric}_mean'].values
                    errs = m_sub[f'{metric}_sem'].values
                    c = METHOD_COLORS[method]
                    mk = METHOD_MARKERS[method]

                    ax.plot(
                        xs, ys, color=c, marker=mk, markersize=2.5,
                        linewidth=0.8, markeredgecolor='white', markeredgewidth=0.3, zorder=3,
                    )
                    ax.fill_between(
                        xs, ys - errs, ys + errs,
                        color=c, alpha=0.12, linewidth=0, zorder=2,
                    )

                ax.set_xlim(R_MIN, R_MAX)
                ax.set_xticks(X_TICKS)

                ax.set_xticklabels([str(v) for v in X_TICKS])
                ax.set_xlabel('$r$', fontsize=8, labelpad=2)

                ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=7, pad=2)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5)

                ax.set_ylabel(ylabel, fontsize=8, labelpad=3)

                if block_idx == 0 and row_i_rel == 0:
                    ax.set_title(COL_TITLES[col_i], fontsize=8, pad=12)

                ax.text(-0.25, 0.99, f'({PANEL_LABELS[panel_idx]})', transform=ax.transAxes,
                        fontsize=10, fontweight='bold', ha='left', va='bottom')
                panel_idx += 1

    fig.canvas.draw()

    for block_idx, lam in enumerate(LAMBDA_TARGETS):
        bbox_tl = axes[block_idx][0][0].get_position()
        bbox_br = axes[block_idx][1][-1].get_position()

        px_L = 0.06
        px_R = 0.02
        py_T = 0.016
        py_B = 0.030

        if block_idx == 2:
            py_B = 0.030

        x0 = bbox_tl.x0 - px_L
        y0 = bbox_br.y0 - py_B
        width = bbox_br.x1 - bbox_tl.x0 + px_L + px_R
        height = bbox_tl.y1 - bbox_br.y0 + py_T + py_B

        face_color_rgba = mcolors.to_rgba(BOX_COLORS[block_idx], alpha=0.08)

        rect = FancyBboxPatch(
            (x0, y0), width, height,
            boxstyle="round,pad=0.0,rounding_size=0.015",
            transform=fig.transFigure,
            fill=True,
            facecolor=face_color_rgba,
            edgecolor=BOX_COLORS[block_idx],
            linewidth=1.5,
            linestyle='--',
            zorder=-5
        )
        fig.add_artist(rect)

        fig.text(x0 + width + 0.005, y0 + height / 2, f'$\\lambda = {lam}$',
                 transform=fig.transFigure, va='center', ha='left',
                 fontsize=10, fontweight='bold', color=BOX_COLORS[block_idx], rotation=-90)

    handles = [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                   markersize=3.5, linewidth=0.8, markeredgecolor='white',
                   markeredgewidth=0.3, label=METHOD_LABELS[i])
        for i, m in enumerate(METHODS)
    ]

    fig.legend(handles=handles, loc='lower center', ncol=len(METHODS),
               bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=8,
               handlelength=2.0, handletextpad=0.45, columnspacing=1.0)

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, 'lambda_validity_combined')
    fig.savefig(stem + '.pdf', dpi=300)
    fig.savefig(stem + '.png', dpi=300)
    print(f"Saved: {stem}.pdf / .png")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    print("Plotting combined lambda plot...")
    plot_all()