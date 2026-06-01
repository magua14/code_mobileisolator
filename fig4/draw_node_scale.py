import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from scipy import stats as scipy_stats

SCRIPT_DIR           = os.path.dirname(os.path.abspath(__file__))
DATA_DIR             = os.path.join(SCRIPT_DIR, 'data')
CSV_PATH             = os.path.join(DATA_DIR, 'results_node_scale_tr5_lambda20.0_final.csv')
RANDOM_WALK_CSV      = os.path.join(DATA_DIR, 'results_n1024.csv')
ROBUST_DEGREE_CSV    = os.path.join(DATA_DIR, 'robustness_n1024.csv')
ROBUST_SP_CSV        = os.path.join(DATA_DIR, 'robustness_startpoint_n1024.csv')
SAVE_DIR             = SCRIPT_DIR

BAR_GRAPH_TYPE = 'WS'
BAR_DIST       = 'random'

BAR_LAMBDA = 10
BAR_TAU    = 1

CONFIGS = [
    ('BA', 'random'),
    ('BA', 'localized'),
    ('WS', 'random'),
    ('WS', 'localized'),
]

TRAIN_LO = 50
TRAIN_HI = 100

SECOND_BEST = 'Adaptive TIA'
N_INSET = 5
INSET_BBOX = [0.6, 0.6, 0.35, 0.35]

PC_YLIM = {
    ('BA', 'random'): (0.1, 0.7),
    ('BA', 'localized'): (0.1, 0.7),
    ('WS', 'random'): (0.2, 1.1),
    ('WS', 'localized'): (0.2, 1.1),
}
ANC_YLIM = {
    ('BA', 'random'): (0.6, 0.92),
    ('BA', 'localized'): (0.6, 0.92),
    ('WS', 'random'): (0.58, 0.9),
    ('WS', 'localized'): (0.58, 0.9),
}

CONN_X_TIA = 128
CONN_X_GNNRL = 362


mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35,
    'ytick.minor.width': 0.35,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

COLORS = {
    'Adaptive TD': '#0072B2',
    'Adaptive Katz': '#D55E00',
    'Adaptive TIA': '#009E73',
    'GNN-RL': '#CC79A7',
}
MARKERS = {
    'Adaptive TD': 'o',
    'Adaptive Katz': 's',
    'Adaptive TIA': '^',
    'GNN-RL': 'D',
}
METHOD_ORDER = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']

METHOD_DISPLAY = {
    'Adaptive TD': 'Adaptive TD',
    'Adaptive Katz': 'Adaptive T-Katz',
    'Adaptive TIA': 'Adaptive TIA',
    'GNN-RL': 'MobileIsolator',
}

def _log2_fmt(v, _):
    exp = int(round(np.log2(v)))
    return f'$2^{{{exp}}}$'


LOG2_FMT = mpl.ticker.FuncFormatter(_log2_fmt)



def draw_panel(ax, df_sub, x_col, y_mean_col, y_sem_col, xlabel, ylabel):
    for method in METHOD_ORDER:
        mdf = df_sub[df_sub['method'] == method].sort_values(x_col)
        if mdf.empty:
            continue
        x = mdf[x_col].values
        mean = mdf[y_mean_col].values
        sem = mdf[y_sem_col].values
        ax.plot(x, mean,
                color=COLORS[method], marker=MARKERS[method],
                markersize=2.5, linewidth=0.8,
                markerfacecolor=COLORS[method],
                markeredgecolor='white', markeredgewidth=0.3,
                zorder=3)
        ax.fill_between(x, mean - sem, mean + sem,
                        color=COLORS[method], alpha=0.15, linewidth=0, zorder=2)

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(LOG2_FMT)
    ax.set_xlabel(xlabel, fontsize=8, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=3)
    ax.tick_params(which='both', direction='in',
                   top=True, right=True, labelsize=7, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def add_training_band(ax):
    ax.axvspan(TRAIN_LO, TRAIN_HI,
               color='#AAAAAA', alpha=0.18, linewidth=0, zorder=0)
    for x in (TRAIN_LO, TRAIN_HI):
        ax.axvline(x, color='#888888', linewidth=0.5,
                   linestyle=(0, (3, 2)), zorder=1)


def add_delta_inset(ax, df_sub, x_col):
    gnn_df = df_sub[df_sub['method'] == 'GNN-RL'].sort_values(x_col)
    second_df = df_sub[df_sub['method'] == SECOND_BEST].sort_values(x_col)
    if gnn_df.empty or second_df.empty:
        return

    merged = pd.merge(
        gnn_df[[x_col, 'pc_mean']].rename(columns={'pc_mean': 'gnn'}),
        second_df[[x_col, 'pc_mean']].rename(columns={'pc_mean': 'second'}),
        on=x_col
    )
    if merged.empty:
        return

    n_vals = merged[x_col].values
    indices = np.round(np.linspace(0, len(n_vals) - 1, N_INSET)).astype(int)
    indices = np.unique(indices)
    sel = merged.iloc[indices]

    delta = sel['second'].values - sel['gnn'].values
    x_ns = sel[x_col].values

    x_labels = [f'$2^{{{int(round(np.log2(v)))}}}$' for v in x_ns]
    x_pos = np.arange(len(x_ns))

    ax_ins = ax.inset_axes(INSET_BBOX)
    bar_colors = [('#E74C3C' if d > 0 else '#3498DB') for d in delta]
    ax_ins.bar(x_pos, delta, color=bar_colors,
               width=0.6, edgecolor='white', linewidth=0.3, zorder=3)
    ax_ins.axhline(0, color='#333333', linewidth=0.5, zorder=4)
    ax_ins.set_xticks(x_pos)
    ax_ins.set_xticklabels(x_labels, fontsize=4.5)
    ax_ins.tick_params(which='both', direction='in', labelsize=4.5,
                       pad=1, length=1.5, width=0.4)
    ax_ins.set_ylabel(r'$\Delta P_C$', fontsize=6, labelpad=2)
    ax_ins.set_xlabel('$N$', fontsize=6, labelpad=1)
    for sp in ax_ins.spines.values():
        sp.set_linewidth(0.4)

    return ax_ins


def _draw_inset_connectors(ax, ax_ins, df_sub, x_col):
    from matplotlib.patches import ConnectionPatch

    def nearest_y(method, x_target):
        mdf = df_sub[df_sub['method'] == method].sort_values(x_col)
        if mdf.empty:
            return None
        idx = (mdf[x_col] - x_target).abs().idxmin()
        return float(mdf.loc[idx, 'pc_mean'])

    y_tia = nearest_y(SECOND_BEST, CONN_X_TIA)
    y_gnnrl = nearest_y('GNN-RL', CONN_X_GNNRL)
    if y_tia is None or y_gnnrl is None:
        return

    conn_style = dict(
        arrowstyle='-', color='#5DADE2', linewidth=0.55, linestyle=(0, (3, 2)),
    )
    cp1 = ConnectionPatch(
        xyA=(0, 1), coordsA=ax_ins.transAxes,
        xyB=(CONN_X_TIA, y_tia), coordsB=ax.transData,
        zorder=0, **conn_style,
    )
    ax.add_artist(cp1)
    cp2 = ConnectionPatch(
        xyA=(0, 0), coordsA=ax_ins.transAxes,
        xyB=(CONN_X_GNNRL, y_gnnrl), coordsB=ax.transData,
        zorder=0, **conn_style,
    )
    ax.add_artist(cp2)


def _welch_pvalue(mean1, std1, n1, mean2, std2, n2):
    se = np.sqrt(std1**2 / n1 + std2**2 / n2)
    if se == 0:
        return 1.0
    t  = (mean1 - mean2) / se
    df = (std1**2/n1 + std2**2/n2)**2 /          ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
    return float(2 * scipy_stats.t.sf(abs(t), df))


def _pval_label(p):
    if   p < 0.001: return '***'
    elif p < 0.01:  return '**'
    elif p < 0.05:  return '*'
    else:           return 'ns'


def _annotate_pval(ax, x_gnnrl, x_heur, y_top, p, metric):
    label = _pval_label(p)
    if label == 'ns':
        return

    gap  = 0.012
    h    = 0.018
    y0   = y_top + gap
    y1   = y0 + h

    x_l  = min(x_gnnrl, x_heur)
    x_r  = max(x_gnnrl, x_heur)
    x_mid = (x_l + x_r) / 2.0

    ax.plot([x_l, x_l, x_r, x_r],
            [y0, y1, y1, y0],
            lw=0.6, color='#444444', zorder=10, clip_on=False)
    ax.text(x_mid, y1 + 0.004, label,
            ha='center', va='bottom',
            fontsize=6, color='#444444', zorder=10, clip_on=False)


def draw_condition_bar(ax, df_rw, df_deg, df_sp,
                       config_key, lambda_val, tau_val,
                       metric, ylabel, title=None):
    group_labels = ['Unbiased', 'Degree-biased', 'Outward-biased']
    n_groups = len(group_labels)
    x_pos    = np.arange(n_groups)
    # offsets  = np.linspace(-0.27, 0.27, len(METHOD_ORDER))
    offsets = np.linspace(-0.28, 0.28, len(METHOD_ORDER))

    def _get(df, method, key_col, key_val):
        mask = (df['method'] == method) & (df['config'] == config_key)
        if key_col == 'lambda_eff':
            mask &= (df[key_col] == key_val)
        else:
            mask &= np.isclose(df[key_col], key_val, atol=1e-5)
        row = df[mask]
        if row.empty:
            return np.nan, 0.0
        return float(row.iloc[0][f'{metric}_mean']), float(row.iloc[0][f'{metric}_sem'])

    bar_data = {}

    for m_idx, method in enumerate(METHOD_ORDER):
        y_rw,  e_rw  = _get(df_rw,  method, 'lambda_eff', lambda_val)
        y_deg, e_deg = _get(df_deg, method, 'tau',        tau_val)
        y_sp,  e_sp  = _get(df_sp,  method, 'tau',        tau_val)

        ys = [y_rw, y_deg, y_sp]
        es = [e_rw, e_deg, e_sp]

        ax.bar(
            x_pos + offsets[m_idx], ys,
            width=0.18,
            color=COLORS[method], alpha=0.85,
            edgecolor='white', linewidth=0.3,
            zorder=4,
        )
        ax.errorbar(
            x_pos + offsets[m_idx], ys, yerr=es,
            fmt='None',
            ecolor=COLORS[method],
            elinewidth=1.0, capsize=2.5, capthick=0.8,
            zorder=5,
        )

        for g_idx, (df_src, key_col, key_val) in enumerate([
                (df_rw,  'lambda_eff', lambda_val),
                (df_deg, 'tau',        tau_val),
                (df_sp,  'tau',        tau_val),
        ]):
            mask = (df_src['method'] == method) & (df_src['config'] == config_key)
            if key_col == 'lambda_eff':
                mask &= (df_src[key_col] == key_val)
            else:
                mask &= np.isclose(df_src[key_col], key_val, atol=1e-5)
            row = df_src[mask]
            if not row.empty:
                bar_data.setdefault(g_idx, {})[method] = {
                    'mean': float(row.iloc[0][f'{metric}_mean']),
                    'std':  float(row.iloc[0][f'{metric}_std']),
                    'n':    int(row.iloc[0]['n_runs']),
                    'top':  float(row.iloc[0][f'{metric}_mean']) +
                            float(row.iloc[0][f'{metric}_sem']),
                    'x':    float(x_pos[g_idx] + offsets[m_idx]),
                }

    # HEURISTICS = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA']
    HEURISTICS = ['Adaptive TIA', 'Adaptive Katz', 'Adaptive TD']
    for g_idx in range(n_groups):
        gd = bar_data.get(g_idx, {})
        if 'GNN-RL' not in gd:
            continue
        gnn = gd['GNN-RL']

        # for heur in HEURISTICS:
        #     if heur not in gd:
        #         continue
        #     h = gd[heur]
        #     p = _welch_pvalue(gnn['mean'], gnn['std'], gnn['n'],
        #                       h['mean'],   h['std'],   h['n'])
        #     y_top = max(gnn['top'], h['top'])
        #     _annotate_pval(ax, gnn['x'], h['x'], y_top, p, metric)
        bracket_y = max(gd[m]['top'] for m in gd)
        BRACKET_STEP = 0.04

        for heur in HEURISTICS:
            if heur not in gd:
                continue
            h = gd[heur]
            p = _welch_pvalue(gnn['mean'], gnn['std'], gnn['n'],
                              h['mean'], h['std'], h['n'])
            if metric == 'pc':
                y_top = max(gnn['top'], h['top'])
                _annotate_pval(ax, gnn['x'], h['x'], y_top, p, metric)
            else:
                _annotate_pval(ax, gnn['x'], h['x'], bracket_y, p, metric)
                bracket_y += BRACKET_STEP

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, fontsize=7, ha='center', va='top')
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)
    ax.tick_params(axis='x', which='both', bottom=False, top=False,
                   labelsize=7, pad=2)
    ax.tick_params(axis='y', which='both', direction='in', right=True,
                   labelsize=7, pad=1.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.spines['top'].set_visible(True)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5, linestyle='--', zorder=1)
    ax.set_axisbelow(False)


df     = pd.read_csv(CSV_PATH)
df_rw  = pd.read_csv(RANDOM_WALK_CSV)
df_deg = pd.read_csv(ROBUST_DEGREE_CSV)
df_sp  = pd.read_csv(ROBUST_SP_CSV)

BAR_CONFIG_KEY = f'{BAR_GRAPH_TYPE}-{BAR_DIST}'
_net  = 'Heterogeneous' if BAR_GRAPH_TYPE == 'BA' else 'Homogeneous'
_dist = BAR_DIST.capitalize()
BAR_SUBTITLE = f'{_net}-{_dist}'

FIG_W = 7.087
ROW_H = 1.55
MID_LEG_H = 0.50
BOT_LEG_H = 0.4
TOP_H = 0.25
FIG_H = 3 * ROW_H + MID_LEG_H + BOT_LEG_H + TOP_H

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs_top = 1.0 - (TOP_H / FIG_H)

gs = fig.add_gridspec(
    4, 1,
    height_ratios=[2 * ROW_H, MID_LEG_H, ROW_H, BOT_LEG_H],
    hspace=0.0,
    left=0.08, right=0.96,
    top=gs_top, bottom=0.02
)

gs_net = gs[0].subgridspec(2, 4, hspace=0.35, wspace=0.38)
pc_axes = [fig.add_subplot(gs_net[0, c]) for c in range(4)]
anc_axes = [fig.add_subplot(gs_net[1, c]) for c in range(4)]

ax_mid_leg = fig.add_subplot(gs[1])
ax_mid_leg.set_axis_off()

gs_bar = gs[2].subgridspec(1, 2, wspace=0.30)
ax_bar1 = fig.add_subplot(gs_bar[0, 0])
ax_bar2 = fig.add_subplot(gs_bar[0, 1])

ax_bot_leg = fig.add_subplot(gs[3])
ax_bot_leg.set_axis_off()

for col, (gt, td) in enumerate(CONFIGS):
    config_key = f"{gt}-{td}"
    sub = df[df['config'] == config_key]

    draw_panel(pc_axes[col], sub, 'n_nodes', 'pc_mean', 'pc_sem',
               xlabel='$N$', ylabel=r'$P_C$')
    draw_panel(anc_axes[col], sub, 'n_nodes', 'anc_mean', 'anc_sem',
               xlabel='$N$', ylabel='ANC')

    key = (gt, td)
    if PC_YLIM.get(key) is not None:
        pc_axes[col].set_ylim(*PC_YLIM[key])
    if ANC_YLIM.get(key) is not None:
        anc_axes[col].set_ylim(*ANC_YLIM[key])

    add_training_band(pc_axes[col])
    add_training_band(anc_axes[col])

    ax_ins = add_delta_inset(pc_axes[col], sub, 'n_nodes')
    if ax_ins is not None:
        _draw_inset_connectors(pc_axes[col], ax_ins, sub, 'n_nodes')

    net_name = 'Heterogeneous' if gt == 'BA' else 'Homogeneous'
    pc_axes[col].set_title(f"{net_name}-{td.capitalize()}", fontsize=8, pad=8)

for col in range(4):
    if col in (0, 1):
        pc_axes[col].set_yticks([0.1, 0.3, 0.5, 0.7])

for col in range(4):
    anc_axes[col].set_yticks([0.6, 0.7, 0.8, 0.9])

curve_handles = [
    mlines.Line2D([], [],
                  color=COLORS[m], marker=MARKERS[m],
                  markersize=4, linewidth=0.8,
                  markerfacecolor=COLORS[m],
                  markeredgecolor='white', markeredgewidth=0.3,
                  label=METHOD_DISPLAY.get(m, m))
    for m in METHOD_ORDER
]
train_patch = mpatches.Patch(
    facecolor='#AAAAAA', alpha=0.5, edgecolor='none',
    label='Training range'
)
ax_mid_leg.legend(
    handles=curve_handles + [train_patch],
    loc='center', ncol=len(METHOD_ORDER) + 1,
    bbox_to_anchor=(0.5, 0.4),
    frameon=False, fontsize=7,
    handlelength=1.8, handletextpad=0.4,
    columnspacing=1.0, borderpad=0,
)

draw_condition_bar(ax_bar1, df_rw, df_deg, df_sp,
                   BAR_CONFIG_KEY, BAR_LAMBDA, BAR_TAU,
                   metric='pc', ylabel=r'$P_C$',
                   title=BAR_SUBTITLE)
ax_bar1.set_ylim(bottom=0.2)
draw_condition_bar(ax_bar2, df_rw, df_deg, df_sp,
                   BAR_CONFIG_KEY, BAR_LAMBDA, BAR_TAU,
                   metric='anc', ylabel='ANC',
                   title=BAR_SUBTITLE)
ax_bar2.set_ylim(bottom=0.55)

rect_handles =[
    mpatches.Patch(
        facecolor=COLORS[m],
        edgecolor='white',
        linewidth=0.3,
        alpha=0.85,
        label=METHOD_DISPLAY.get(m, m)
    )
    for m in METHOD_ORDER
]

ax_bot_leg.legend(
    handles=rect_handles,
    loc='center', ncol=len(METHOD_ORDER),
    bbox_to_anchor=(0.5, 0.2),
    frameon=False, fontsize=7,
    handlelength=1.2,
    handleheight=0.8,
    handletextpad=0.4,
    columnspacing=1.0, borderpad=0,
)

labels = list('abcdefghij')
idx = 0
for row_axes in [pc_axes, anc_axes]:
    for col, ax in enumerate(row_axes):
        xoff = -0.22 if col == 0 else -0.22
        ax.text(xoff, 1.08, labels[idx],
                transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')
        idx += 1

ax_bar1.text(-0.11, 1.05, labels[idx], transform=ax_bar1.transAxes,
             fontsize=10, fontweight='bold', va='top', ha='left')
idx += 1
ax_bar2.text(-0.11, 1.05, labels[idx], transform=ax_bar2.transAxes,
             fontsize=10, fontweight='bold', va='top', ha='left')

if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)
    stem = os.path.join(SAVE_DIR, 'draw_node_scale_extended')
    fig.savefig(stem + '.png', dpi=300)
    fig.savefig(stem + '.pdf', dpi=300)
    plt.show()
    print(f"Saved: {stem}.png / .pdf")