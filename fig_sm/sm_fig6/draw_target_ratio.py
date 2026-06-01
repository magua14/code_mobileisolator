import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
CSV_PATH   = os.path.join(DATA_DIR, 'results_target_ratio_n1024_lambda20.0_generalization.csv')
SAVE_DIR   = SCRIPT_DIR

CONFIGS = [
    ('BA', 'random'),
    ('BA', 'localized'),
    ('WS', 'random'),
    ('WS', 'localized'),
]


mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
    'font.size':          7,
    'axes.labelsize':     8,
    'axes.titlesize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'axes.linewidth':     0.5,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.minor.width':  0.35,
    'ytick.minor.width':  0.35,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'xtick.minor.size':   1.5,
    'ytick.minor.size':   1.5,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
})

COLORS = {
    'Adaptive TD':   '#0072B2',
    'Adaptive Katz': '#D55E00',
    'Adaptive TIA':  '#009E73',
    'GNN-RL':        '#CC79A7',
}
MARKERS = {
    'Adaptive TD':   'o',
    'Adaptive Katz': 's',
    'Adaptive TIA':  '^',
    'GNN-RL':        'D',
}
METHOD_ORDER = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']

METHOD_DISPLAY = {
    'Adaptive TD':    'Adaptive TD',
    'Adaptive Katz':  'Adaptive T-Katz',
    'Adaptive TIA':   'Adaptive TIA',
    'GNN-RL':         'MobileIsolator',
}


df = pd.read_csv(CSV_PATH)


def draw_panel(ax, df_sub, x_col, y_mean_col, y_sem_col, xlabel, ylabel):
    for method in METHOD_ORDER:
        mdf = df_sub[df_sub['method'] == method].sort_values(x_col)
        if mdf.empty:
            continue
        x    = mdf[x_col].values
        mean = mdf[y_mean_col].values
        sem  = mdf[y_sem_col].values
        ax.plot(x, mean,
                color=COLORS[method], marker=MARKERS[method],
                markersize=2.5, linewidth=0.8,
                markerfacecolor=COLORS[method],
                markeredgecolor='white', markeredgewidth=0.3)
        ax.fill_between(x, mean - sem, mean + sem,
                        color=COLORS[method], alpha=0.15, linewidth=0)

    ax.set_xlabel(xlabel, fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
    ax.tick_params(which='both', direction='in',
                   top=True, right=True, labelsize=7, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


FIG_W = 7.087
ROW_H = 1.55
LEG_H = 0.28
FIG_H = 2 * ROW_H + LEG_H + 0.15

fig = plt.figure(figsize=(FIG_W, FIG_H))

gs = fig.add_gridspec(
    3, 4,
    height_ratios=[ROW_H, ROW_H, LEG_H],
    hspace=0.42,
    wspace=0.36,
    left=0.08, right=0.99,
    top=0.94,  bottom=0.04,
)

pc_axes  = [fig.add_subplot(gs[0, c]) for c in range(4)]
anc_axes = [fig.add_subplot(gs[1, c]) for c in range(4)]
ax_leg   = fig.add_subplot(gs[2, :])
ax_leg.set_axis_off()


for col, (gt, td) in enumerate(CONFIGS):
    config_key = f"{gt}-{td}"
    sub = df[df['config'] == config_key]

    draw_panel(pc_axes[col],  sub, 'target_ratio', 'pc_mean',  'pc_sem',
               xlabel='Target ratio', ylabel=r'$P_C$')
    draw_panel(anc_axes[col], sub, 'target_ratio', 'anc_mean', 'anc_sem',
               xlabel='Target ratio', ylabel='ANC')

    # pc_axes[col].set_title(f"{gt}-{td.capitalize()}", fontsize=7, pad=3)
    net_name = 'Heterogeneous' if gt == 'BA' else 'Homogeneous'
    pc_axes[col].set_title(f"{net_name}-{td.capitalize()}", fontsize=8, pad=6)

labels = list('abcdefgh')
idx = 0
for row_axes in [pc_axes, anc_axes]:
    for col, ax in enumerate(row_axes):
        xoff = -0.22 if col == 0 else -0.20
        ax.text(xoff, 1.10, labels[idx],
                transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')
        idx += 1


ax_leg.legend(
    handles=[
        mlines.Line2D([], [],
                      color=COLORS[m], marker=MARKERS[m],
                      markersize=4, linewidth=0.8,
                      markerfacecolor=COLORS[m],
                      markeredgecolor='white', markeredgewidth=0.3,
                      label=METHOD_DISPLAY[m])
        for m in METHOD_ORDER
    ],
    loc='center', ncol=len(METHOD_ORDER),
    frameon=False, fontsize=7,
    handlelength=1.8, handletextpad=0.4,
    columnspacing=1.2, borderpad=0,
)


if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)
    stem = os.path.join(SAVE_DIR, 'draw_target_ratio')
    fig.savefig(stem + '.pdf', dpi=300)
    fig.savefig(stem + '.png', dpi=300)
    plt.show()
    print(f"Saved: {stem}.pdf / .png")

plt.show()
print("Done.")