import os
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import warnings
import pickle
import hashlib
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, 'data', 'fig6_result')
SAVE_DIR   = SCRIPT_DIR

LAYOUT_CACHE_DIR = os.path.join(DATA_ROOT, '.layout_cache')
SEED = 42

METHOD = 'GNN_RL'

SCENARIO_CSVS = [
    ('COVID-19', 'covid19/covid19_results.csv'),
    ('Invasive Species', 'invasive_species/invasive_results.csv'),
    ('Flood', 'flood/flood_results.csv'),
    ('Smuggling', 'smuggling/smuggling_results.csv'),
    ('Socialbot', 'socialbot/socialbot_results.csv'),
    ('Fugitive Chase', 'fugitive_chase/fugitive_results.csv'),
]

BAR_LAMBDA = 20

BAR_COLORS = ['#2166AC', '#1B7837', '#762A83', '#4393C3', '#5AAE61', '#9970AB']


ROWS = [
    {
        'label': 'COVID-19',
        'graphml': 'covid19/covid19_network.graphml',
        'removed_txt': 'covid19/removed_{method}.txt',
        'targets_txt': 'covid19/final_targets_{method}.txt',
    },
    {
        'label': 'Invasive Species',
        'graphml': 'invasive_species/invasive_network.graphml',
        'removed_txt': 'invasive_species/removed_{method}.txt',
        'targets_txt': 'invasive_species/final_targets_{method}.txt',
    },
    {
        'label': 'Flood',
        'graphml': 'flood/flood_network.graphml',
        'geo_graphml': 'flood/flood_network_with_coords.graphml',
        'removed_txt': 'flood/removed_{method}.txt',
        'targets_txt': 'flood/final_targets_{method}.txt',
    },
    {
        'label': 'Smuggling',
        'graphml': 'smuggling/smuggling_network.graphml',
        'removed_txt': 'smuggling/removed_{method}.txt',
        'targets_txt': 'smuggling/final_targets_{method}.txt',
        'node_size_override': 1.5,
    },
    {
        'label': 'Socialbot',
        'graphml': 'socialbot/socialbot_network.graphml',
        'removed_txt': 'socialbot/removed_{method}.txt',
        'targets_txt': 'socialbot/final_targets_{method}.txt',
    },
    {
        'label': 'Fugitive Chase',
        'graphml': 'fugitive_chase/fugitive_network.graphml',
        'removed_txt': 'fugitive_chase/removed_{method}.txt',
        'targets_txt': 'fugitive_chase/final_targets_{method}.txt',
    },
]


NODE_COLOR_NORMAL = '#AAAAAA'
NODE_EDGE_NORMAL = 'black'
NODE_LW_NORMAL = 0.4

NODE_COLOR_REMOVED = '#F39C12'
NODE_EDGE_REMOVED = '#E67E22'
NODE_LW_REMOVED = 0.4

NODE_COLOR_TARGET = '#E74C3C'
NODE_EDGE_TARGET = 'white'
NODE_LW_TARGET = 0.3

EDGE_COLOR = '#DDDDDD'
EDGE_WIDTH = 0.15
EDGE_ALPHA = 0.3

PANEL_LABELS = list('abcdefgh')


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
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})



def _load_txt(path: str) -> set:
    if not os.path.exists(path):
        print(f"  Warning: file not found: {path}")
        return set()
    nodes = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                nodes.add(line)
    return nodes


def draw_terminal_network(ax, graphml_path: str,
                          removed_path: str, targets_path: str,
                          seed: int, method: str,
                          geo_graphml: str = None,
                          node_size_override: float = None):
    G = nx.read_graphml(graphml_path)
    G.remove_edges_from(nx.selfloop_edges(G))
    G_str = nx.relabel_nodes(G, {nd: str(nd) for nd in G.nodes()})
    n = G_str.number_of_nodes()

    removed_set = _load_txt(removed_path)
    target_set = _load_txt(targets_path)
    all_nodes = set(G_str.nodes())
    removed_set = removed_set & all_nodes
    target_set = target_set & all_nodes

    pos = None

    if geo_graphml and os.path.exists(geo_graphml):
        G_geo = nx.read_graphml(geo_graphml)
        coord_lookup = {}
        for nd, d in G_geo.nodes(data=True):
            x = d.get('x_plot') or d.get('x') or d.get('lon')
            y = d.get('y_plot') or d.get('y') or d.get('lat')
            if x is not None and y is not None:
                coord_lookup[str(nd)] = (float(x), float(y))
        coverage = sum(1 for nd in G_str.nodes() if nd in coord_lookup)
        if coverage >= 0.8 * n:
            pos = {nd: coord_lookup.get(nd, (0.0, 0.0)) for nd in G_str.nodes()}
            print(f"  Using geographic layout ({coverage}/{n} nodes)")

    if pos is None:
        edge_sig = hashlib.md5(str(sorted(G_str.edges())).encode()).hexdigest()[:12]
        k_val = max(0.8, 3.5 / max(1, n ** 0.45))
        cache_key = f'spring_n{n}_s{seed}_k{k_val:.4f}_e{edge_sig}'
        cache_file = os.path.join(LAYOUT_CACHE_DIR, cache_key + '.pkl')
        os.makedirs(LAYOUT_CACHE_DIR, exist_ok=True)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as _f:
                pos = pickle.load(_f)
            print(f"  Spring layout loaded from cache")
        else:
            np.random.seed(seed)
            pos = nx.spring_layout(G_str, seed=seed, iterations=50, k=k_val)
            with open(cache_file, 'wb') as _f:
                pickle.dump(pos, _f)
            print(f"  Spring layout computed and cached")

    xs = np.array([pos[nd][0] for nd in pos])
    ys = np.array([pos[nd][1] for nd in pos])
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    xs -= cx
    ys -= cy
    max_span = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    if max_span > 0:
        xs /= max_span
        ys /= max_span
    for i, nd in enumerate(pos.keys()):
        pos[nd] = (xs[i], ys[i])

    if node_size_override is not None:
        node_size = node_size_override
    else:
        node_size = max(0.3, min(6.0, 4800 / max(1, n)))

    nodes_all = list(G_str.nodes())
    normal_nodes = [nd for nd in nodes_all if nd not in removed_set and nd not in target_set]
    removed_nodes = [nd for nd in nodes_all if nd in removed_set]
    target_nodes = [nd for nd in nodes_all if nd in target_set]

    from matplotlib.collections import LineCollection
    segments = [(pos[u], pos[v]) for u, v in G_str.edges() if u in pos and v in pos]
    lc = LineCollection(segments, colors=EDGE_COLOR, linewidths=EDGE_WIDTH,
                        alpha=EDGE_ALPHA, zorder=1)
    ax.add_collection(lc)

    if normal_nodes:
        nx.draw_networkx_nodes(
            G_str, pos, ax=ax, nodelist=normal_nodes,
            node_color=NODE_COLOR_NORMAL, node_size=node_size, linewidths=0,
        )

    if removed_nodes:
        nx.draw_networkx_nodes(
            G_str, pos, ax=ax, nodelist=removed_nodes,
            node_color=NODE_COLOR_REMOVED, node_size=node_size, linewidths=0,
        )

    if target_nodes:
        nx.draw_networkx_nodes(
            G_str, pos, ax=ax, nodelist=target_nodes,
            node_color=NODE_COLOR_TARGET, node_size=node_size, linewidths=0,
        )

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    print(f"  {n:>5} nodes | removed={len(removed_nodes):>4} | "
          f"targets={len(target_nodes):>3} | normal={len(normal_nodes):>5}")


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
    return None


def _annotate_sig(ax, x_tia, x_gnn, y_tia, y_gnn, e_tia, e_gnn, p):
    label = _pval_label(p)
    if label is None:
        return
    y_top = max(y_tia + e_tia, y_gnn + e_gnn)
    gap   = 0.050
    h     = 0.016
    y0    = y_top + gap
    y1    = y0 + h
    x_l, x_r = min(x_tia, x_gnn), max(x_tia, x_gnn)
    ax.plot([x_l, x_l, x_r, x_r], [y0, y1, y1, y0],
            lw=0.6, color='#333333', zorder=10, clip_on=False)
    ax.text((x_l+x_r)/2, y1+0.003, label,
            ha='center', va='bottom', fontsize=6,
            color='#333333', zorder=10, clip_on=False)


def draw_bar_row(ax, data_root, scenario_csvs, lambda_val, metric, ylabel):
    scenario_labels = [s for s, _ in scenario_csvs]
    n_scenarios = len(scenario_labels)
    x_pos = np.arange(n_scenarios)

    width = 0.3
    offsets = [-width / 2 - 0.02, width / 2 + 0.02]

    mpl.rcParams['hatch.linewidth'] = 0.6

    for s_idx, (s_label, csv_rel) in enumerate(scenario_csvs):
        csv_path = os.path.join(data_root, csv_rel)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)

        c = BAR_COLORS[s_idx % len(BAR_COLORS)]

        y_tia = e_tia = None
        y_gnn = e_gnn = None

        row_tia = df[(df['method'] == 'TIA') & (df['lambda_eff'] == lambda_val)]
        if not row_tia.empty:
            y_tia = float(row_tia.iloc[0][f'{metric}_mean'])
            e_tia = float(row_tia.iloc[0][f'{metric}_sem'])
            ax.bar(x_pos[s_idx] + offsets[0], y_tia, width=width,
                   facecolor='white', edgecolor=c, linestyle='--', linewidth=1.2,
                   hatch='////', zorder=4)
            ax.errorbar(x_pos[s_idx] + offsets[0], y_tia, yerr=e_tia,
                        fmt='none', ecolor=c, elinewidth=1.0, capsize=2.5, capthick=0.8, zorder=5)

        row_gnn = df[(df['method'] == 'GNN-RL') & (df['lambda_eff'] == lambda_val)]
        if not row_gnn.empty:
            y_gnn = float(row_gnn.iloc[0][f'{metric}_mean'])
            e_gnn = float(row_gnn.iloc[0][f'{metric}_sem'])
            ax.bar(x_pos[s_idx] + offsets[1], y_gnn, width=width,
                   facecolor=c, edgecolor=c, linestyle='-', linewidth=1.2, zorder=4)
            ax.errorbar(x_pos[s_idx] + offsets[1], y_gnn, yerr=e_gnn,
                        fmt='none', ecolor=c, elinewidth=1.0, capsize=2.5, capthick=0.8, zorder=5)

        if y_tia is not None and y_gnn is not None:
            std_col = f'{metric}_std'
            if std_col in df.columns and 'n_runs' in df.columns:
                try:
                    std_tia = float(row_tia.iloc[0][std_col])
                    n_tia   = int(row_tia.iloc[0]['n_runs'])
                    std_gnn = float(row_gnn.iloc[0][std_col])
                    n_gnn   = int(row_gnn.iloc[0]['n_runs'])
                    p = _welch_pvalue(y_tia, std_tia, n_tia,
                                      y_gnn, std_gnn, n_gnn)
                    _annotate_sig(ax,
                                  x_pos[s_idx] + offsets[0],
                                  x_pos[s_idx] + offsets[1],
                                  y_tia, y_gnn, e_tia, e_gnn, p)
                except Exception:
                    pass

    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_labels, rotation=0, ha='center', fontsize=7)
    ax.set_xlim(-0.5, n_scenarios - 0.5)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2)

    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelsize=7, pad=2)
    ax.tick_params(axis='y', which='both', direction='in', right=True, labelsize=7, pad=1.5)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, min_n_ticks=3))

    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.spines['top'].set_visible(True)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5, linestyle='--', zorder=1)
    ax.set_axisbelow(True)



def build_figure(data_root: str, save_dir: str, method: str = METHOD):
    N_COLS = 3
    FIG_W = 7.087

    W_avail = FIG_W * (0.99 - 0.06)
    W_cell = W_avail / (N_COLS + (N_COLS - 1) * 0.10)

    NET_H = W_cell
    NET_LEG = 0.24
    BAR_H = 1.20
    GAP_H = 0.35
    DOT_LEG_H = 0.22

    sum_h = 2 * NET_H + NET_LEG + 2 * BAR_H + GAP_H + DOT_LEG_H
    FIG_H = sum_h / (0.97 - 0.04)

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    gs = gridspec.GridSpec(
        7, N_COLS,
        figure=fig,
        height_ratios=[NET_H, NET_H, NET_LEG, BAR_H, GAP_H, BAR_H, DOT_LEG_H],
        hspace=0.0,
        wspace=0.10,
        left=0.06, right=0.99,
        top=0.97, bottom=0.04,
    )

    print(f"\nDrawing fig6 final  [method={method}]")
    print(f"{'─' * 60}")

    NET_ROW_MAP = [0, 0, 0, 1, 1, 1]
    for idx, row_cfg in enumerate(ROWS):
        gs_row = NET_ROW_MAP[idx]
        col = idx % N_COLS

        label_str = row_cfg['label']
        gml_path = os.path.join(data_root, row_cfg['graphml'])
        removed_path = os.path.join(data_root, row_cfg['removed_txt'].format(method=method))
        targets_path = os.path.join(data_root, row_cfg['targets_txt'].format(method=method))
        geo_gml = (os.path.join(data_root, row_cfg['geo_graphml']) if row_cfg.get('geo_graphml') else None)

        ax = fig.add_subplot(gs[gs_row, col])
        print(f"  [{PANEL_LABELS[idx]}] {label_str}")
        draw_terminal_network(ax, gml_path, removed_path, targets_path,
                              seed=SEED, method=method, geo_graphml=geo_gml,
                              node_size_override=row_cfg.get('node_size_override'))
        ax.set_title(label_str, fontsize=8, pad=1, loc='center', fontweight='bold', y=0.95)
        ax.text(-0.12, 1.06, PANEL_LABELS[idx],
                transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')

    ax_net_leg = fig.add_subplot(gs[2, :])
    ax_net_leg.set_axis_off()
    net_handles = [
        mlines.Line2D([], [], marker='o', color='w',
                      markerfacecolor=NODE_COLOR_NORMAL, markeredgecolor='none',
                      markeredgewidth=0, markersize=5, label='Normal node'),
        mlines.Line2D([], [], marker='o', color='w',
                      markerfacecolor=NODE_COLOR_REMOVED, markeredgecolor='none',
                      markeredgewidth=0, markersize=5, label='Removed node'),
        mlines.Line2D([], [], marker='o', color='w',
                      markerfacecolor=NODE_COLOR_TARGET, markeredgecolor='none',
                      markeredgewidth=0, markersize=5, label='Target node'),
    ]
    ax_net_leg.legend(
        handles=net_handles, loc='center', ncol=3, frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, 0.8), handlelength=1.2, handletextpad=0.4,
        columnspacing=1.2, borderpad=0,
    )

    ax_pc = fig.add_subplot(gs[3, :])
    draw_bar_row(ax_pc, data_root, SCENARIO_CSVS, BAR_LAMBDA, metric='pc', ylabel=r'$P_C$')
    ax_pc.set_ylim(0, ax_pc.get_ylim()[1] * 1.1)
    ax_pc.text(-0.04, 1.08, PANEL_LABELS[6], transform=ax_pc.transAxes,
               fontsize=10, fontweight='bold', va='top', ha='left')

    ax_anc = fig.add_subplot(gs[5, :])
    draw_bar_row(ax_anc, data_root, SCENARIO_CSVS, BAR_LAMBDA, metric='anc', ylabel='ANC')
    ax_anc.set_ylim(0, ax_anc.get_ylim()[1] * 1.1)
    ax_anc.text(-0.04, 1.08, PANEL_LABELS[7], transform=ax_anc.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')

    ax_mleg = fig.add_subplot(gs[6, :])
    ax_mleg.set_axis_off()

    method_handles = [
        mpatches.Patch(facecolor='white', edgecolor='#666666',
                       hatch='////', linestyle='--', linewidth=1.2,
                       label='Adaptive TIA'),
        mpatches.Patch(facecolor='#666666', edgecolor='#666666',
                       linestyle='-', linewidth=1.2,
                       label='MobileIsolator'),
    ]
    ax_mleg.legend(
        handles=method_handles, loc='center', ncol=2, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.25), handlelength=1.5, handleheight=0.9,
        handletextpad=0.5, columnspacing=2.0, borderpad=0,
    )

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, f'draw_fig6_final_{method}')
    fig.savefig(stem + '.pdf', dpi=300)
    fig.savefig(stem + '.png', dpi=300)
    print(f"\nSaved: {stem}.pdf / .png")
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    build_figure(DATA_ROOT, SAVE_DIR, METHOD)