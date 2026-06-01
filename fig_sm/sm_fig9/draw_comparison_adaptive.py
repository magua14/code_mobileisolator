import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
CSV_PATH   = os.path.join(DATA_DIR, 'comparison_adaptive_results.csv')
SAVE_DIR   = SCRIPT_DIR


mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size':         7,
    'axes.labelsize':    8,
    'axes.titlesize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size':  2.5, 'ytick.major.size':  2.5,
    'xtick.direction':   'in', 'ytick.direction':   'in',
    'figure.dpi':        300,  'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'figure.facecolor':  'white', 'axes.facecolor': 'white',
})


CONFIGS = [
    ('BA', 'random'),
    ('BA', 'localized'),
    ('WS', 'random'),
    ('WS', 'localized'),
]
# COL_TITLES  = ['BA-Random', 'BA-Localized', 'WS-Random', 'WS-Localized']
COL_TITLES = [
    'Heterogeneous-Random',
    'Heterogeneous-Localized',
    'Homogeneous-Random',
    'Homogeneous-Localized'
]
BASE_METHODS = ['TD', 'Katz', 'TIA', 'GNN-RL']
STRATEGIES   = ['Fully', 'Batch']

METHOD_COLORS = {
    'TD':     '#0072B2',
    'Katz':   '#D55E00',
    'TIA':    '#009E73',
    'GNN-RL': '#CC79A7',
}
METHOD_MARKERS = {
    'TD':     'o',
    'Katz':   's',
    'TIA':    '^',
    'GNN-RL': 'D',
}
STRATEGY_LINESTYLE = {'Fully': '-', 'Batch': '--'}

METHOD_DISPLAY = {
    'TD': 'Adaptive TD',
    'Katz': 'Adaptive T-Katz',
    'TIA': 'Adaptive TIA',
    'GNN-RL': 'MobileIsolator',
}

PANEL_LABELS = list('abcdefgh')

X_TICKS      = [1, 5, 10, 15, 20]


FIG_W  = 7.087
PAN_H  = 1.65
LEG_H  = 0.8
TOP_H  = 0.05

FIG_H  = 2 * PAN_H + LEG_H + TOP_H

GS_LEFT   = 0.085
GS_RIGHT  = 0.995
GS_TOP    = 1.0 - TOP_H / FIG_H
GS_BOTTOM = LEG_H / FIG_H
GS_HSPACE = 0.38
GS_WSPACE = 0.40


def plot(csv_path: str, save_dir: str):
    df = pd.read_csv(csv_path)

    lambda_vals = sorted(df['lambda_eff'].unique())
    x_max = max(lambda_vals) + 1

    fig, axes = plt.subplots(
        2, 4,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={
            'hspace':  GS_HSPACE,
            'wspace':  GS_WSPACE,
            'left':    GS_LEFT,
            'right':   GS_RIGHT,
            'top':     GS_TOP,
            'bottom':  GS_BOTTOM,
        },
    )

    panel_idx = 0
    for row_i, (metric, ylabel) in enumerate([('pc', r'$P_C$'), ('anc', 'ANC')]):
        for col_i, (graph_type, target_dist) in enumerate(CONFIGS):
            config_key = f"{graph_type}-{target_dist}"
            ax = axes[row_i][col_i]

            for bm in BASE_METHODS:
                c  = METHOD_COLORS.get(bm, '#333')
                mk = METHOD_MARKERS.get(bm, 'o')
                sub = df[
                    (df['config']      == config_key) &
                    (df['base_method'] == bm)
                ].sort_values('lambda_eff')

                for strategy in STRATEGIES:
                    s  = sub[sub['strategy'] == strategy]
                    if s.empty:
                        continue
                    xs   = s['lambda_eff'].values
                    ys   = s[f'{metric}_mean'].values
                    errs = s[f'{metric}_sem'].values
                    ls   = STRATEGY_LINESTYLE[strategy]

                    ax.plot(
                        xs, ys,
                        color=c, marker=mk, markersize=2.5,
                        linewidth=0.8, linestyle=ls,
                        markeredgecolor='white', markeredgewidth=0.3,
                        zorder=3,
                    )
                    ax.fill_between(
                        xs, ys - errs, ys + errs,
                        color=c, alpha=0.12, linewidth=0, zorder=2,
                    )

            ax.set_xticks(X_TICKS)
            ax.set_xticklabels([str(v) for v in X_TICKS])
            ax.set_xlim(0, x_max)
            ax.tick_params(
                which='both', direction='in',
                top=True, right=True,
                labelsize=7, pad=2,
            )
            ax.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=4, min_n_ticks=3))
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

            ax.set_xlabel(r'$\lambda$', fontsize=8, labelpad=2)
            ax.set_ylabel(ylabel, fontsize=8, labelpad=3)

            if row_i == 0:
                ax.set_title(COL_TITLES[col_i], fontsize=8, pad=4)

            ax.text(
                -0.25, 1.01,
                f'({PANEL_LABELS[panel_idx]})',
                transform=ax.transAxes,
                fontsize=8, fontweight='bold',
                ha='left', va='bottom',
            )
            panel_idx += 1

    handles = []
    for bm in BASE_METHODS:
        c  = METHOD_COLORS.get(bm, '#333')
        mk = METHOD_MARKERS.get(bm, 'o')
        display_name = METHOD_DISPLAY.get(bm, bm)
        for st in STRATEGIES:
            ls = STRATEGY_LINESTYLE[st]
            handles.append(
                plt.Line2D(
                    [], [],
                    color=c, marker=mk, markersize=3.5,
                    linewidth=0.8, linestyle=ls,
                    markeredgecolor='white', markeredgewidth=0.3,
                    label=f'{display_name} ({st})',
                )
            )

    legend_y = (LEG_H * 0.25) / FIG_H
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.54, legend_y),
        frameon=False,
        fontsize=7,
        handlelength=3.5,
        handletextpad=0.45,
        columnspacing=0.9,
        labelspacing=0.45,
    )

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, 'comparison_adaptive')
    fig.savefig(stem + '.pdf', dpi=300)
    fig.savefig(stem + '.png', dpi=300)
    print(f"Saved: {stem}.pdf / .png")
    plt.show()



if __name__ == '__main__':
    plot(CSV_PATH, SAVE_DIR)