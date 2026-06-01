import os, json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.legend_handler import HandlerPatch


SELECTED_GRAPH_TYPE  = 'WS'
SELECTED_TARGET_DIST = 'localized'

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(SCRIPT_DIR, 'data')
CSV_PATH         = os.path.join(DATA_DIR, 'results_n1024.csv')
SINGLE_GRAPH_DIR = os.path.join(DATA_DIR, 'single_graph')
OUT_DIR          = SCRIPT_DIR

CONFIGS = [
    ('BA', 'random'),
    ('BA', 'localized'),
    ('WS', 'random'),
    ('WS', 'localized'),
]

COL_TITLES = [
    'Heterogeneous-Random',
    'Heterogeneous-Localized',
    'Homogeneous-Random',
    'Homogeneous-Localized'
]

LAYOUT_SEED = 42
LAMBDA_LO   = 9
LAMBDA_HI   = 11

# S_LCC_MIN, S_LCC_MAX =  2.0,  8.0
S_LCC_MIN, S_LCC_MAX =  1.0,  4.0
# S_REM_MIN, S_REM_MAX =  5.0, 12.0
S_REM_MIN, S_REM_MAX =  3.0, 8.0
# S_TGT_MIN, S_TGT_MAX = 12.0, 30.0
S_TGT_MIN, S_TGT_MAX = 8.0, 15.0
# S_NRM_MIN, S_NRM_MAX =  5.0, 11.0
S_NRM_MIN, S_NRM_MAX =  2.0, 8.0


mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'lines.linewidth': 1.0,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

C_LCC            = '#7F8C8D'
C_REMOVED_BORDER = '#F39C12'
C_TARGET         = '#E74C3C'
C_NRM_OUT        = '#BDC3C7'
C_CONNECT        = '#5DADE2'
C_TRAP           = '#EBEBEB'

WONG = {
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
METHOD_ORDER    = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']
METHOD_FILE_KEY = {
    'Adaptive TD':   'Adaptive_TD',
    'Adaptive Katz': 'Adaptive_Katz',
    'Adaptive TIA':  'Adaptive_TIA',
    'GNN-RL':        'GNN_RL',
}
METHOD_DISPLAY = {
    'Adaptive TD':    'Adaptive TD',
    'Adaptive Katz':  'Adaptive T-Katz',
    'Adaptive TIA':   'Adaptive TIA',
    'GNN-RL':         'MobileIsolator',
}


class _CircleHandler_(HandlerPatch):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        r  = min(width, height) * 0.42
        cx, cy = (width-xd)/2, (height-yd)/2
        return [mpatches.Circle((cx,cy), r,
                facecolor=orig.get_facecolor(), edgecolor=orig.get_edgecolor(),
                linewidth=orig.get_linewidth(), linestyle=orig.get_linestyle(),
                transform=trans)]

CIRCLE_MAP = {mpatches.Circle: _CircleHandler_()}


def _read_ids(path):
    ids = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'): ids.append(s)
    return ids


def load_single_graph(single_graph_dir, graph_type, target_dist):
    d = os.path.join(single_graph_dir, f"{graph_type}-{target_dist}")
    G = nx.read_graphml(os.path.join(d, 'graph.graphml'))
    with open(os.path.join(d, 'summary.json')) as f:
        summary = json.load(f)
    removed, final_targets = {}, {}
    for method, fkey in METHOD_FILE_KEY.items():
        removed[method]       = set(_read_ids(os.path.join(d, f'removed_{fkey}.txt')))
        final_targets[method] = set(_read_ids(os.path.join(d, f'final_targets_{fkey}.txt')))
    return G, removed, final_targets, summary


def compute_lcc(G, removed_set):
    H = G.copy()
    for nd in removed_set:
        if H.has_node(nd): H.remove_node(nd)
    if H.number_of_nodes() == 0: return set()
    try: return max(nx.connected_components(H), key=len)
    except ValueError: return set()


def deg_size(nd, degrees, dmin, dmax, s_min, s_max):
    if dmax == dmin: return (s_min+s_max)/2
    return s_min + ((degrees[nd]-dmin)/(dmax-dmin))**0.5 * (s_max-s_min)


def classify_nodes(G, removed_set, target_set):
    all_nodes = list(G.nodes())
    lcc_set   = compute_lcc(G, removed_set)
    lcc_nodes = [nd for nd in all_nodes if nd in lcc_set and nd not in target_set]
    rem_nodes = [nd for nd in all_nodes if nd in removed_set]
    tgt_nodes = [nd for nd in all_nodes if nd in target_set]
    nrm_out   = [nd for nd in all_nodes
                 if nd not in removed_set and nd not in lcc_set and nd not in target_set]
    return lcc_nodes, rem_nodes, tgt_nodes, nrm_out


def _remove_overlaps_coords(coords, node_sizes_pt2, pts_per_unit,
                             padding=1.15, max_iter=300):
    from scipy.spatial import KDTree
    nodes = list(coords.keys())
    arr   = np.array([coords[nd] for nd in nodes], dtype=float)
    radii = np.array([np.sqrt(node_sizes_pt2.get(nd,4)/np.pi)/pts_per_unit*padding
                      for nd in nodes])
    max_r = radii.max()
    for _ in range(max_iter):
        moved = False
        tree  = KDTree(arr)
        for i, j in tree.query_pairs(r=max_r*4):
            dx, dy = arr[j,0]-arr[i,0], arr[j,1]-arr[i,1]
            dist   = np.hypot(dx, dy)
            min_d  = radii[i]+radii[j]
            if dist < min_d and dist > 1e-9:
                shift = (min_d-dist)/2
                arr[i] -= [dx/dist*shift, dy/dist*shift]
                arr[j] += [dx/dist*shift, dy/dist*shift]
                moved = True
        if not moved: break
    return {nodes[i]: (arr[i,0], arr[i,1]) for i in range(len(nodes))}


def categorical_layout(lcc_nodes, rem_nodes, tgt_nodes, nrm_out_nodes, seed=42):
    rng = np.random.RandomState(seed)
    PACK = 1.25
    R1 = PACK*np.sqrt(max(len(lcc_nodes),1)/np.pi)
    R2 = R1*1.06 + PACK*np.sqrt(max(len(rem_nodes),1)/np.pi)
    R3 = R2*1.04 + PACK*np.sqrt(max(len(tgt_nodes)+len(nrm_out_nodes),1)/np.pi)

    def sample_ann(n, rlo, rhi):
        r = np.sqrt(rng.uniform(rlo**2, rhi**2, n))
        t = rng.uniform(0, 2*np.pi, n)
        return r*np.cos(t), r*np.sin(t)

    coords = {}
    def assign(nodes, rlo, rhi):
        if not nodes: return
        xs, ys = sample_ann(len(nodes), rlo, rhi)
        for i, nd in enumerate(nodes): coords[nd] = (xs[i], ys[i])

    assign(lcc_nodes, 0, R1)
    assign(rem_nodes, R1*1.04, R2)
    outer = list(tgt_nodes)+list(nrm_out_nodes); rng.shuffle(outer)
    assign(outer, R2*1.03, R3)
    if not coords: return {}

    all_nds = list(coords.keys())
    arr = np.array([coords[nd] for nd in all_nds])
    arr -= arr.min(axis=0)
    arr /= np.maximum(arr.max(axis=0), 1e-9)
    coords = {nd: (arr[i,0], arr[i,1]) for i, nd in enumerate(all_nds)}

    size_map = {}
    for nd in lcc_nodes:     size_map[nd] = (S_LCC_MIN+S_LCC_MAX)/2
    for nd in rem_nodes:     size_map[nd] = (S_REM_MIN+S_REM_MAX)/2
    for nd in tgt_nodes:     size_map[nd] = (S_TGT_MIN+S_TGT_MAX)/2
    for nd in nrm_out_nodes: size_map[nd] = (S_NRM_MIN+S_NRM_MAX)/2
    return _remove_overlaps_coords(coords, size_map,
                                   pts_per_unit=175, padding=1.15, max_iter=300)


def draw_network_panel(ax, G, degrees, dmin, dmax,
                       removed_nodes, final_target_nodes,
                       method_name, seed=42, edge_alpha=0.05):
    removed_set = set(removed_nodes)
    target_set  = set(final_target_nodes)
    lcc_nodes, rem_nodes, tgt_nodes, nrm_out = classify_nodes(G, removed_set, target_set)
    pos = categorical_layout(lcc_nodes, rem_nodes, tgt_nodes, nrm_out, seed=seed)

    def xs_ys(lst):
        return ([pos[nd][0] for nd in lst if nd in pos],
                [pos[nd][1] for nd in lst if nd in pos])
    def sizes(lst, smin, smax):
        return [deg_size(nd,degrees,dmin,dmax,smin,smax) for nd in lst if nd in pos]

    if edge_alpha > 0:
        en = [(u,v) for u,v in G.edges() if u in pos and v in pos
              and u not in removed_set and v not in removed_set]
        er = [(u,v) for u,v in G.edges() if u in pos and v in pos
              and (u in removed_set or v in removed_set)]
        if en: nx.draw_networkx_edges(G,pos,ax=ax,edgelist=en,
                                      alpha=edge_alpha,width=0.25,edge_color='#BDC3C7')
        if er: nx.draw_networkx_edges(G,pos,ax=ax,edgelist=er,
                                      alpha=min(edge_alpha*1.5,1.0),
                                      width=0.25,edge_color='#BDC3C7',style='dashed')

    x, y = xs_ys(lcc_nodes)
    if x: ax.scatter(x,y,s=sizes(lcc_nodes,S_LCC_MIN,S_LCC_MAX),
                     facecolors=C_LCC,edgecolors='none',linewidths=0,zorder=2)

    x, y = xs_ys(rem_nodes)
    if x:
        sc = ax.scatter(x,y,s=sizes(rem_nodes,S_REM_MIN,S_REM_MAX),
                        facecolors='none',edgecolors=C_REMOVED_BORDER,linewidths=0.75,zorder=3)
        sc.set_linestyle((0,(1.2,0.8)))

    x, y = xs_ys(nrm_out)
    if x: ax.scatter(x,y,s=sizes(nrm_out,S_NRM_MIN,S_NRM_MAX),
                     facecolors=C_NRM_OUT,edgecolors='none',linewidths=0,zorder=4)

    x, y = xs_ys(tgt_nodes)
    if x: ax.scatter(x,y,s=sizes(tgt_nodes,S_TGT_MIN,S_TGT_MAX),
                     facecolors=C_TARGET,edgecolors='white',linewidths=0.35,zorder=5)

    all_pos = list(pos.values())
    if all_pos:
        arr = np.array(all_pos)
        px = max((arr[:,0].max()-arr[:,0].min())*0.02, 0.02)
        py = max((arr[:,1].max()-arr[:,1].min())*0.02, 0.02)
        ax.set_xlim(arr[:,0].min()-px, arr[:,0].max()+px)
        ax.set_ylim(arr[:,1].min()-py, arr[:,1].max()+py)

    # Method name as title
    ax.set_title(method_name, fontsize=7, pad=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')


def main():
    df = pd.read_csv(CSV_PATH)
    sel_config_key = f"{SELECTED_GRAPH_TYPE}-{SELECTED_TARGET_DIST}"
    sel_col = next(i for i,(gt,td) in enumerate(CONFIGS)
                   if gt==SELECTED_GRAPH_TYPE and td==SELECTED_TARGET_DIST)
    print(f"Selected config: {sel_config_key}  (curve column {sel_col})")

    G, removed, final_tgts, _ = load_single_graph(
        SINGLE_GRAPH_DIR, SELECTED_GRAPH_TYPE, SELECTED_TARGET_DIST)
    degrees = dict(G.degree())
    dmin, dmax = min(degrees.values()), max(degrees.values())
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    FIG_W   = 7.087
    SNAP_H  = 1.60
    NET_LEG = 0.40
    CURVE_H = 1.40
    CRV_LEG = 0.30
    HSPACE  = 0.28
    FIG_H   = SNAP_H + NET_LEG + 2*CURVE_H + CRV_LEG + HSPACE*CURVE_H + 0.18

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    outer = gridspec.GridSpec(
        7, 1, figure=fig,
        height_ratios=[SNAP_H, NET_LEG, CURVE_H, HSPACE*CURVE_H, CURVE_H, 0.15, CRV_LEG],
        hspace=0.0,
        left=0.08, right=0.99, top=0.96, bottom=0.05,
    )

    _snap_ss = outer[0]
    _snap_pos = _snap_ss.get_position(fig)
    snap_gs = gridspec.GridSpec(
        1, 4,
        left=0.04, right=0.99,
        top=_snap_pos.y1, bottom=_snap_pos.y0,
        wspace=0.04,
    )
    snap_axes = [fig.add_subplot(snap_gs[0,c]) for c in range(4)]

    ax_net_leg = fig.add_subplot(outer[1]); ax_net_leg.set_axis_off()

    pc_gs  = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec=outer[2], wspace=0.38)
    anc_gs = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec=outer[4], wspace=0.38)
    pc_axes  = [fig.add_subplot(pc_gs[0,c])  for c in range(4)]
    anc_axes = [fig.add_subplot(anc_gs[0,c]) for c in range(4)]

    ax_crv_leg = fig.add_subplot(outer[6]); ax_crv_leg.set_axis_off()

    print("Drawing snapshots…")
    for col, method in enumerate(METHOD_ORDER):
        draw_network_panel(
            snap_axes[col], G, degrees, dmin, dmax,
            removed[method], final_tgts[method],
            method_name=METHOD_DISPLAY[method], seed=LAYOUT_SEED,
        )

    config_keys = [f"{gt}-{td}" for gt,td in CONFIGS]
    for col, ck in enumerate(config_keys):
        sub = df[df['config']==ck]
        lambdas_all = sorted(sub['lambda_eff'].unique())
        for ax, metric, ylabel in [(pc_axes[col],'pc','$P_C$'),
                                   (anc_axes[col],'anc','ANC')]:
            for method in METHOD_ORDER:
                mdf = sub[sub['method']==method].sort_values('lambda_eff')
                if mdf.empty: continue
                lam  = mdf['lambda_eff'].values
                mean = mdf[f'{metric}_mean'].values
                sem  = mdf[f'{metric}_sem'].values
                c    = WONG[method]
                ax.plot(lam, mean, color=c, marker=MARKERS[method],
                        markersize=2.5, linewidth=0.8,
                        markeredgecolor='white', markeredgewidth=0.3)
                ax.fill_between(lam, mean-sem, mean+sem,
                                color=c, alpha=0.15, linewidth=0)

            if col == sel_col and metric == 'pc':
                ax.axvspan(LAMBDA_LO, LAMBDA_HI,
                           color=C_CONNECT, alpha=0.10, linewidth=0, zorder=0)
                for xv in (LAMBDA_LO, LAMBDA_HI):
                    ax.axvline(xv, color=C_CONNECT, linewidth=0.8,
                               linestyle=(0,(4,2)), zorder=0.5, alpha=0.85)

            ax.tick_params(which='both', direction='in',
                           top=True, right=True, labelsize=7, pad=2)
            ax.set_xlim(min(lambdas_all)-0.5, max(lambdas_all)+0.5)
            ax.set_xticks([1,5,10,15,20])
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            for sp in ax.spines.values(): sp.set_linewidth(0.5)

        pc_axes[col].set_title(COL_TITLES[col], fontsize=8, pad=3)

        pc_axes[col].set_xlabel(r'$\lambda$', fontsize=8, labelpad=2)
        anc_axes[col].set_xlabel(r'$\lambda$', fontsize=8, labelpad=2)

        pc_axes[col].set_ylabel('$P_C$', fontsize=8, labelpad=3)
        anc_axes[col].set_ylabel('ANC',  fontsize=8, labelpad=3)


    ax_net_leg.legend(
        handles=[
            mpatches.Circle((0,0),1,facecolor=C_LCC,edgecolor='none',
                             linewidth=0,label='LCC node'),
            mpatches.Circle((0,0),1,facecolor='none',edgecolor=C_REMOVED_BORDER,
                             linewidth=0.75,linestyle=(0,(1.2,0.8)),label='Removed node'),
            mpatches.Circle((0,0),1,facecolor=C_TARGET,edgecolor='white',
                             linewidth=0.35,label='Target node'),
            mpatches.Circle((0,0),1,facecolor=C_NRM_OUT,edgecolor='none',
                             linewidth=0,label='Normal node'),
        ],
        handler_map=CIRCLE_MAP,
        loc='center', ncol=4,
        frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, 0.6),
        handlelength=2.0, handletextpad=0.5,
        columnspacing=1.2, borderpad=0,
    )

    ax_crv_leg.legend(
        handles=[
            mlines.Line2D([],[],color=WONG[m],marker=MARKERS[m],
                          markersize=4,linewidth=0.8,
                          markerfacecolor=WONG[m],
                          markeredgecolor='white',markeredgewidth=0.3,label=METHOD_DISPLAY[m])
            for m in METHOD_ORDER
        ],
        loc='center', ncol=4,
        frameon=False, fontsize=7,
        bbox_to_anchor=(0.5, 0.3),
        handlelength=2.0, handletextpad=0.5,
        columnspacing=1.5, borderpad=0,
    )

    fig.canvas.draw()


    def data_x_to_fig(ax, xdata):
        pt = ax.transData.transform((xdata, 0))
        return fig.transFigure.inverted().transform(pt)[0]

    spos = [ax.get_position() for ax in snap_axes]
    sx0 = min(p.x0 for p in spos)
    sx1 = max(p.x1 for p in spos)
    sy0 = min(p.y0 for p in spos)
    sy1 = max(p.y1 for p in spos)
    PAD = 0.005


    snap_title_h = 0.042

    _xoffs = [-0.22, -0.22, -0.22, -0.22]
    _col_x = [pc_axes[c].get_position().x0
              + _xoffs[c] * pc_axes[c].get_position().width
              for c in range(4)]

    _y_snap = sy1 + snap_title_h + PAD - 0.025
    _y_pc   = pc_axes[0].get_position().y1 + 0.004
    _y_anc  = anc_axes[0].get_position().y1 + 0.004

    all_labels = [
        (_col_x[0], _y_snap, 'a'),
        (_col_x[1], _y_snap, 'b'),
        (_col_x[2], _y_snap, 'c'),
        (_col_x[3], _y_snap, 'd'),
        (_col_x[0], _y_pc,   'e'),
        (_col_x[1], _y_pc,   'f'),
        (_col_x[2], _y_pc,   'g'),
        (_col_x[3], _y_pc,   'h'),
        (_col_x[0], _y_anc,  'i'),
        (_col_x[1], _y_anc,  'j'),
        (_col_x[2], _y_anc,  'k'),
        (_col_x[3], _y_anc,  'l'),
    ]
    for txt in snap_axes[0].texts: txt.set_visible(False)
    for xx, yy, lb in all_labels:
        fig.text(xx, yy, lb,
                 fontsize=10, fontweight='bold',
                 va='bottom', ha='left',
                 transform=fig.transFigure, zorder=50)

    box_x0 = sx0 - PAD
    box_x1 = sx1 + PAD
    fig.add_artist(mpatches.FancyBboxPatch(
        (box_x0, sy0 - PAD),
        box_x1 - box_x0, sy1 - sy0 + 2*PAD + snap_title_h + 0.014,
        boxstyle='square,pad=0',
        linewidth=1.0, edgecolor=C_CONNECT, facecolor='none',
        linestyle=(0,(4,2)),
        transform=fig.transFigure, clip_on=False, zorder=30,
    ))

    sel_pc = pc_axes[sel_col]
    lx_lo  = data_x_to_fig(sel_pc, LAMBDA_LO)
    lx_hi  = data_x_to_fig(sel_pc, LAMBDA_HI)
    ly_top = sel_pc.get_position().y1
    ly_bot = sel_pc.get_position().y0

    fig.add_artist(mpatches.FancyBboxPatch(
        (lx_lo, ly_bot - PAD), lx_hi - lx_lo, ly_top - ly_bot + 2*PAD,
        boxstyle='square,pad=0',
        linewidth=1.0, edgecolor=C_CONNECT, facecolor='none',
        linestyle=(0,(4,2)),
        transform=fig.transFigure, clip_on=False, zorder=30,
    ))

    fig.add_artist(mpatches.Polygon(
        np.array([
            [lx_lo,  ly_top],
            [lx_hi,  ly_top],
            [box_x1, sy0 - PAD],
            [box_x0, sy0 - PAD],
        ]),
        closed=True, facecolor=C_TRAP, edgecolor='none',
        transform=fig.transFigure, clip_on=False, zorder=0,
    ))

    for x_snap, x_lam in [(box_x0, lx_lo), (box_x1, lx_hi)]:
        fig.add_artist(mlines.Line2D(
            [x_snap, x_lam], [sy0 - PAD, ly_top],
            color=C_CONNECT, linewidth=0.8,
            linestyle=(0,(4,2)),
            transform=fig.transFigure, clip_on=False, zorder=0,
        ))

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR,
           f'draw_fig3_final_{SELECTED_GRAPH_TYPE}_{SELECTED_TARGET_DIST}')
    fig.savefig(stem+'.pdf', dpi=300)
    fig.savefig(stem+'.png', dpi=300)
    print(f"Saved: {stem}.pdf / .png")
    plt.show()


if __name__ == '__main__':
    main()