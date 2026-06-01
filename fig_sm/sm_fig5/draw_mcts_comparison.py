import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
CSV_PATH   = os.path.join(DATA_DIR, 'results_mcts_n50_sim20000.csv')
SAVE_DIR   = SCRIPT_DIR

mpl.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':     42,
    'ps.fonttype':      42,
    'font.size':        7,
    'axes.labelsize':   8,
    'axes.titlesize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
    'axes.linewidth':   0.5,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})

CONFIGS      = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
COL_TITLES = [
    'Heterogeneous-Random',
    'Heterogeneous-Localized',
    'Homogeneous-Random',
    'Homogeneous-Localized'
]
LAMBDA_VALS  = [1, 10, 20]
METHODS      = ['TIA', 'MCTS', 'GNN-RL']

METHOD_COLORS = {
    'TIA':    '#009E73',
    'MCTS':   '#34495e',
    'GNN-RL': '#CC79A7',
}
METHOD_MARKERS = {
    'TIA':    '^',
    'MCTS':   's',
    'GNN-RL': 'D',
}

LAMBDA_BG = {
    1:  '#E8F8F0',
    10: 'white',
    20: '#FDEDEC',
}

METHOD_DISPLAY = {
    'TIA':    'Adaptive TIA',
    'MCTS':  'MCTS',
    'GNN-RL':         'MobileIsolator',
}

PANEL_LABELS = list('abcdefgh')

N_METHODS = len(METHODS)
OFFSETS   = np.linspace(-0.22, 0.22, N_METHODS)

N_ROWS  = 2
N_COLS  = 4
FIG_W   = 7.087
PAN_H   = 1.50
LEG_H   = 0.38
TOP_H   = 0.05
FIG_H   = N_ROWS * PAN_H + LEG_H + TOP_H


def draw_panel(ax: plt.Axes, df: pd.DataFrame,
               config: str, metric: str, ylabel: str):

    x_pos = np.arange(len(LAMBDA_VALS))

    for m_idx, method in enumerate(METHODS):
        ys, es = [], []
        for lam in LAMBDA_VALS:
            row = df[(df['config']     == config) &
                     (df['method']     == method) &
                     (df['lambda_eff'] == lam)]
            if row.empty:
                ys.append(np.nan)
                es.append(0.0)
            else:
                ys.append(float(row.iloc[0][f'{metric}_mean']))
                es.append(float(row.iloc[0][f'{metric}_sem']))

        ax.errorbar(
            x_pos + OFFSETS[m_idx],
            ys, yerr=es,
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markersize=3.5,
            markeredgecolor='white', markeredgewidth=0.3,
            ecolor=METHOD_COLORS[method],
            elinewidth=0.7, capsize=1.5, capthick=0.5,
            linewidth=0, zorder=4,
            label=method,
        )

    for j, lam in enumerate(LAMBDA_VALS):
        ax.axvspan(j - 0.5, j + 0.5,
                   color=LAMBDA_BG[lam], alpha=1.0,
                   linewidth=0, zorder=0)

    lam_labels = [f'λ={v}' for v in LAMBDA_VALS]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lam_labels, fontsize=7)
    ax.set_xlim(-0.5, len(LAMBDA_VALS) - 0.5)

    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelsize=7, pad=1.5)
    ax.tick_params(axis='y', which='both', direction='in', right=True,
                   labelsize=7, pad=1.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.spines['top'].set_visible(True)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5, linestyle='--', zorder=1)
    ax.set_axisbelow(False)


def build_figure(csv_path: str = CSV_PATH, save_dir: str = SAVE_DIR):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={
            'hspace':  0.48,
            'wspace':  0.38,
            'left':    0.085,
            'right':   0.995,
            'top':     1.0 - TOP_H / FIG_H,
            'bottom':  LEG_H / FIG_H,
        },
    )

    panel_idx = 0
    for row_i, (metric, ylabel) in enumerate([('pc', r'$P_C$'), ('anc', 'ANC')]):
        for col_i, (config, title) in enumerate(zip(CONFIGS, COL_TITLES)):
            ax = axes[row_i, col_i]

            draw_panel(ax, df, config, metric, ylabel)

            if row_i == 0:
                ax.set_title(title, fontsize=10, pad=4)

            ax.text(
                -0.26, 1.01,
                f'({PANEL_LABELS[panel_idx]})',
                transform=ax.transAxes,
                fontsize=10, fontweight='bold',
                ha='left', va='bottom',
            )
            panel_idx += 1

    handles = [
        mlines.Line2D(
            [], [],
            marker=METHOD_MARKERS[m],
            color='w',
            markerfacecolor=METHOD_COLORS[m],
            markeredgecolor='white', markeredgewidth=0.3,
            markersize=4.5,
            label=METHOD_DISPLAY[m],
        )
        for m in METHODS
    ]
    legend_y = (LEG_H * 0.02) / FIG_H
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(METHODS),
        bbox_to_anchor=(0.54, legend_y),
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, 'mcts_comparison')
    fig.savefig(stem + '.pdf', dpi=300)
    fig.savefig(stem + '.png', dpi=300)
    print(f"Saved: {stem}.pdf / .png")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    build_figure()