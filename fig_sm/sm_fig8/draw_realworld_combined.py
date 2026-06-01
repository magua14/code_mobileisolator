import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import string

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_ROOT  = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_result')

SAVE_DIR = SCRIPT_DIR

SCENARIO_FILES = {
    'covid19':   'covid19/covid19_results.csv',
    'invasive':  'invasive_species/invasive_results.csv',
    'flood':     'flood/flood_results.csv',
    'smuggling': 'smuggling/smuggling_results.csv',
    'socialbot': 'socialbot/socialbot_results.csv',
    'fugitive':  'fugitive_chase/fugitive_results.csv',
}

COL_SCENARIO = 'scenario'
COL_METHOD = 'method'
COL_LAMBDA = 'lambda_eff'
COL_PC_MEAN = 'pc_mean'
COL_PC_SEM = 'pc_sem'
COL_ANC_MEAN = 'anc_mean'
COL_ANC_SEM = 'anc_sem'

SCENARIOS = [
    ('covid19', 'COVID-19'),
    ('invasive', 'Invasive Species'),
    ('flood', 'Urban Flood'),
    ('smuggling', 'Smuggling'),
    ('socialbot', 'Socialbot'),
    ('fugitive', 'Fugitive Chase'),
]

DOMAINS = [
    ('Public health', ['covid19', 'invasive'], '#E69F00'),
    ('Infrastructure protection', ['flood', 'smuggling'], '#56B4E9'),
    ('Security', ['socialbot', 'fugitive'], '#009E73'),
]

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

METHODS = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']
METHOD_LABELS = ['Adaptive TD', 'Adaptive T-Katz', 'Adaptive TIA', 'MobileIsolator']

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

PANEL_LABELS = list(string.ascii_lowercase)[:12]

FIG_W = 7.087
PAN_H = 1.20
LEG_H = 0.60
TOP_H = 0.05
ROW_SPC = 0.80
FIG_H = 3 * PAN_H + 2 * ROW_SPC + LEG_H + TOP_H


def plot_panel(ax, sub_df, metric_mean, metric_sem, ylabel):
    for method in METHODS:
        m_sub = sub_df[sub_df[COL_METHOD] == method].sort_values(COL_LAMBDA)
        if m_sub.empty:
            continue
        xs = m_sub[COL_LAMBDA].values
        ys = m_sub[metric_mean].values
        es = m_sub[metric_sem].values
        c = METHOD_COLORS[method]
        mk = METHOD_MARKERS[method]
        ax.plot(xs, ys, color=c, marker=mk, markersize=2.5,
                linewidth=0.8,
                markeredgecolor='white', markeredgewidth=0.3, zorder=3)
        ax.fill_between(xs, ys - es, ys + es, color=c,
                        alpha=0.12, linewidth=0, zorder=2)

    ax.set_xlabel(r'$\lambda$', fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=7, pad=2)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def build_figure():
    df_list = []
    for scen_key, rel_pattern in SCENARIO_FILES.items():
        matches = sorted(glob.glob(os.path.join(DATA_ROOT, rel_pattern)))
        if not matches:
            print(f"Warning: No file matched {rel_pattern} under {DATA_ROOT}")
            continue
        filepath = matches[-1]
        temp_df = pd.read_csv(filepath)
        temp_df[COL_SCENARIO] = scen_key
        df_list.append(temp_df)

    if not df_list:
        raise FileNotFoundError("No CSV files found. Please check DATA_ROOT and SCENARIO_FILES.")

    df = pd.concat(df_list, ignore_index=True)

    df[COL_METHOD] = df[COL_METHOD].replace({
        'TD': 'Adaptive TD',
        'Katz': 'Adaptive Katz',
        'TIA': 'Adaptive TIA'
    })

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs_top = 1.0 - TOP_H / FIG_H
    gs_bot = LEG_H / FIG_H

    gs_blocks = fig.add_gridspec(3, 1, hspace=ROW_SPC / PAN_H,
                                 left=0.08, right=0.91,
                                 top=gs_top, bottom=gs_bot)

    axes = []
    for r in range(3):
        gs_row = gs_blocks[r].subgridspec(1, 4, wspace=0.50)
        row_axes = [fig.add_subplot(gs_row[0, c]) for c in range(4)]
        axes.append(row_axes)

    panel_idx = 0
    scen_dict = dict(SCENARIOS)

    for r, (domain_label, domain_scens, box_color) in enumerate(DOMAINS):
        for s_idx, scen_key in enumerate(domain_scens):
            scen_df = df[df[COL_SCENARIO] == scen_key]

            col_pc = s_idx * 2
            ax_pc = axes[r][col_pc]
            plot_panel(ax_pc, scen_df, COL_PC_MEAN, COL_PC_SEM, r'$P_C$')

            col_anc = s_idx * 2 + 1
            ax_anc = axes[r][col_anc]
            plot_panel(ax_anc, scen_df, COL_ANC_MEAN, COL_ANC_SEM, 'ANC')

            for ax in (ax_pc, ax_anc):
                ax.text(-0.28, 1.05, f'({PANEL_LABELS[panel_idx]})',
                        transform=ax.transAxes,
                        fontsize=10, fontweight='bold',
                        ha='left', va='bottom')
                panel_idx += 1

    fig.canvas.draw()

    for r, (domain_label, domain_scens, box_color) in enumerate(DOMAINS):
        for s_idx, scen_key in enumerate(domain_scens):
            bbox_tl = axes[r][s_idx * 2].get_position()
            bbox_br = axes[r][s_idx * 2 + 1].get_position()

            px_L = 0.062
            px_R = 0.005
            py_T = 0.045
            py_B = 0.045

            x0 = bbox_tl.x0 - px_L
            y0 = bbox_br.y0 - py_B
            width = bbox_br.x1 - bbox_tl.x0 + px_L + px_R
            height = bbox_tl.y1 - bbox_br.y0 + py_T + py_B

            face_rgba = mcolors.to_rgba(box_color, alpha=0.07)
            rect = FancyBboxPatch(
                (x0, y0), width, height,
                boxstyle="round,pad=0.0,rounding_size=0.015",
                transform=fig.transFigure,
                fill=True, facecolor=face_rgba,
                edgecolor=box_color, linewidth=1.5,
                linestyle='--', zorder=-5,
            )
            fig.add_artist(rect)

            scen_display = scen_dict[scen_key]
            fig.text(x0 + width / 2, y0 + height + 0.008,
                     scen_display,
                     transform=fig.transFigure,
                     va='bottom', ha='center',
                     fontsize=9, fontweight='bold',
                     color=box_color)

    handles = [
        mlines.Line2D([], [],
                      color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                      markersize=3.5, linewidth=0.8,
                      markeredgecolor='white', markeredgewidth=0.3,
                      label=METHOD_LABELS[i])
        for i, m in enumerate(METHODS)
    ]

    fig.legend(handles=handles, loc='lower center',
               ncol=len(METHODS),
               bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=8,
               handlelength=2.0, handletextpad=0.45, columnspacing=1.0)

    os.makedirs(SAVE_DIR, exist_ok=True)
    stem = os.path.join(SAVE_DIR, 'draw_realworld_combined')
    fig.savefig(stem + '.pdf', dpi=300)
    fig.savefig(stem + '.png', dpi=300)
    print(f"Saved: {stem}.pdf / .png")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    build_figure()