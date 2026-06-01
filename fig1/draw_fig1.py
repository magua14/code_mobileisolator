import os
import re, random
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as MRect
from xml.etree import ElementTree as ET
import pandas as pd
import seaborn as sns
from typing import List
import matplotlib.patches as _mpatches_
from matplotlib.legend_handler import HandlerPatch as _HandlerPatch_, HandlerBase
from scipy.spatial import ConvexHull
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1 import make_axes_locatable


class _HandlerPointyArrow_(HandlerBase):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        cy = (height - yd) / 2
        arrow = mpatches.FancyArrowPatch(
            posA=(0, cy), posB=(width, cy),
            arrowstyle='->',
            color=orig.get_color(),
            linewidth=orig.get_linewidth(),
            linestyle=orig.get_linestyle(),
            mutation_scale=12, transform=trans)
        return [arrow]


class _HandlerCircle_(_HandlerPatch_):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        r  = min(width, height) * 0.6
        cx = (width - xd) / 2
        cy = (height - yd) / 2
        p  = _mpatches_.Circle(
            (cx, cy), r,
            facecolor=orig.get_facecolor(),
            edgecolor=orig.get_edgecolor(),
            linewidth=orig.get_linewidth(),
            linestyle=orig.get_linestyle(),
            transform=trans)
        return [p]

class _HandlerLargeCircle_(_HandlerPatch_):
    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        r = min(width, height) * 1
        cx = (width - xd) / 2
        cy = (height - yd) / 2
        p = _mpatches_.Circle(
            (cx, cy), r,
            facecolor=orig.get_facecolor(),
            edgecolor=orig.get_edgecolor(),
            linewidth=orig.get_linewidth(),
            linestyle=orig.get_linestyle(),
            transform=trans)
        return [p]


# Locate this script's directory so paths work regardless of the
# user's current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')

XML_PATH     = os.path.join(DATA_DIR, 'network.xml')
FIXED_CSV    = os.path.join(DATA_DIR, 'static_tia_fixed.csv')
RW_CSV       = os.path.join(DATA_DIR, 'static_tia_random_walk.csv')
HEATMAP_CSV  = os.path.join(DATA_DIR, 'pc_random_ws.csv')

N_TARGETS_SNAP       = 3
ATTACK_PER_STEP_SNAP = 1
MAX_PAIR_DIST        = 1
MAX_KB               = 5
LAYOUT_SEED          = 7

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

C_BLUE   = '#2471a3'
C_RED    = '#c0392b'
C_ORANGE = '#e67e22'
C_ARROW  = '#c0392b'


def tia_core(G: nx.Graph, target_nodes: List[int]):
    G = G.copy()
    T = set(target_nodes); K = set(); R = set(G.nodes()) - T
    for node in T:
        if node in G: K.update(G.neighbors(node))
    K -= T; R -= K
    neighbors_set = {node: set(G.neighbors(node)) for node in G.nodes()}
    Nmin_K = len(K)
    while True:
        changed = False
        DR = {node: len(neighbors_set[node] & R) for node in K}
        nodes_to_move = [node for node in K if DR[node] == 0]
        if nodes_to_move:
            for node in nodes_to_move: T.add(node); K.remove(node)
            changed = True
        if len(K) < Nmin_K: Nmin_K = len(K)
        DR = {node: len(neighbors_set[node] & R) for node in K}
        for r_node in R:
            if r_node not in neighbors_set: continue
            r_node_neighbors = neighbors_set[r_node]
            k_neighbors_of_r = r_node_neighbors & K
            if not any(DR.get(nbr, 0) == 1 for nbr in k_neighbors_of_r): continue
            K_temp = K | {r_node}; R_temp = R - {r_node}; T_temp = set(T)
            DR_temp = dict(DR)
            DR_temp[r_node] = len(r_node_neighbors & R_temp)
            for nbr in k_neighbors_of_r: DR_temp[nbr] -= 1
            for k_node in [k for k in K_temp if DR_temp.get(k, 0) == 0]:
                T_temp.add(k_node); K_temp.remove(k_node)
            if len(K_temp) < Nmin_K:
                K, T, R = K_temp, T_temp, R_temp
                Nmin_K = len(K); changed = True; break
        if not changed: break
    final_DR = {node: len(neighbors_set[node] & R) for node in K}
    return T, sorted(K, key=lambda x: final_DR[x], reverse=True), R


def extended_tia_attack_order(G: nx.Graph, target_nodes: List[int]) -> List[int]:
    original_target_set = set(target_nodes)
    T, K, R = tia_core(G, target_nodes)
    return list(K) + list(T - original_target_set) + list(R)


def load_911_network(xml_path: str, layout_seed: int = 7):
    with open(xml_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
    root    = ET.fromstring(content)
    ns      = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    keys    = {k.get('id'): k.get('attr.name') for k in root.findall('g:key', ns)}
    graph_el = root.find('g:graph', ns)

    G = nx.Graph(); hijacker_nodes = []; node_labels = {}
    for node_el in graph_el.findall('g:node', ns):
        nid  = int(node_el.get('id')[1:])
        data = {keys[d.get('key')]: d.text for d in node_el.findall('g:data', ns)}
        group = int(data.get('group', 4))
        name  = data.get('name', '').replace('_', ' ').strip()
        G.add_node(nid, group=group, name=name)
        node_labels[nid] = name
        if group in (0, 1, 2, 3): hijacker_nodes.append(nid)
    for edge_el in graph_el.findall('g:edge', ns):
        G.add_edge(int(edge_el.get('source')[1:]), int(edge_el.get('target')[1:]))

    n   = G.number_of_nodes()
    pos = nx.spring_layout(G, k=2.0/np.sqrt(n), iterations=600, seed=layout_seed)
    x_vals = [v[0] for v in pos.values()]; y_vals = [v[1] for v in pos.values()]
    scale  = 4.0 / max(max(x_vals)-min(x_vals), max(y_vals)-min(y_vals))
    pos    = {nd: (x*scale, y*scale) for nd, (x, y) in pos.items()}

    degrees = dict(G.degree())
    deg_arr = np.array(list(degrees.values()), dtype=float)
    dmin, dmax = deg_arr.min(), deg_arr.max()
    node_sizes = {nd: 20 + (degrees[nd]-dmin)/max(dmax-dmin, 1)*(45-20)
                  for nd in G.nodes()}

    PTS_PER_UNIT = 10.0; SAFETY = 3.71
    node_list = list(pos.keys())
    for _ in range(1000):
        moved = False
        for i, n1 in enumerate(node_list):
            x1, y1 = pos[n1]
            r1 = np.sqrt(node_sizes[n1]/np.pi)/PTS_PER_UNIT*SAFETY
            for n2 in node_list[i+1:]:
                x2, y2 = pos[n2]
                r2 = np.sqrt(node_sizes[n2]/np.pi)/PTS_PER_UNIT*SAFETY
                min_sep = r1 + r2
                dx, dy = x2-x1, y2-y1
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_sep and dist > 1e-9:
                    shift = (min_sep - dist) / 2
                    pos[n1] = (x1 - dx/dist*shift, y1 - dy/dist*shift)
                    pos[n2] = (x2 + dx/dist*shift, y2 + dy/dist*shift)
                    moved = True
        if not moved: break

    print(f"9/11 network: N={G.number_of_nodes()}, E={G.number_of_edges()}, "
          f"hijackers={len(hijacker_nodes)}")
    return G, pos, node_sizes, hijacker_nodes, node_labels


VISUAL_THRESH = 0.2

def _one_step(G, current, occupied_extra, forbidden_set, rng):
    new_pos = []
    occupied = set(current) | set(occupied_extra)
    moved = 0
    for t in current:
        nbrs = [nb for nb in G.neighbors(t)
                if nb not in occupied and nb not in forbidden_set]
        if nbrs:
            new_t = rng.choice(nbrs)
            new_pos.append(new_t)
            occupied.add(new_t)
            moved += 1
        else:
            new_pos.append(t)
            occupied.add(t)
    return new_pos, moved


def find_snapshot(G, pos, hijacker_nodes, n_targets=3, attack_per_step=1,
                  max_seed=8000, max_pair_dist=2, max_kb=6):
    xs_all = [v[0] for v in pos.values()]
    ys_all = [v[1] for v in pos.values()]
    layout_diam = ((max(xs_all)-min(xs_all))**2 + (max(ys_all)-min(ys_all))**2)**0.5
    max_spread  = VISUAL_THRESH * layout_diam

    best = None; best_score = -1e18
    all_nodes = list(G.nodes())

    for seed in range(max_seed):
        rng = random.Random(seed)
        center = rng.choice(hijacker_nodes if len(hijacker_nodes) >= n_targets
                            else all_nodes)

        pool = [nd for nd in all_nodes
                if nx.has_path(G, center, nd) and
                nx.shortest_path_length(G, center, nd) <= max_pair_dist]
        if len(pool) < n_targets: continue
        targets = rng.sample(pool, n_targets)
        too_far = any(
            not nx.has_path(G, targets[i], targets[j]) or
            nx.shortest_path_length(G, targets[i], targets[j]) > max_pair_dist
            for i in range(len(targets)) for j in range(i+1, len(targets)))
        if too_far: continue

        t_pos = [pos[t] for t in targets]
        max_geom = max(
            ((t_pos[i][0]-t_pos[j][0])**2 + (t_pos[i][1]-t_pos[j][1])**2)**0.5
            for i in range(len(targets)) for j in range(i+1, len(targets)))
        if max_geom > max_spread: continue

        _, K_before, _ = tia_core(G, targets)
        kb = len(K_before)
        if kb < 1 or kb > max_kb: continue
        K_before_set = set(K_before)

        targets_mid, moved1 = _one_step(G, targets, [], set(), rng)
        if moved1 < n_targets: continue
        if set(targets_mid) == set(targets): continue

        targets_after, moved2 = _one_step(G, targets_mid, [], K_before_set, rng)
        if moved2 < n_targets: continue
        if set(targets_after) == set(targets_mid): continue
        if set(targets_after) == set(targets): continue

        if len(set(targets)) != n_targets: continue

        if any(targets[i] == targets_after[i] for i in range(n_targets)): continue

        G_test = G.copy()
        for nd in K_before:
            if G_test.has_node(nd): G_test.remove_node(nd)
        try:
            lcc_a = max(nx.connected_components(G_test), key=len)
        except ValueError:
            continue
        if any(t in lcc_a for t in targets): continue

        try:
            lcc_c = max(nx.connected_components(G_test), key=len)
        except ValueError:
            continue
        if not all(t in lcc_c for t in targets_after): continue

        lcc_size_diff = abs(len(lcc_a) - len(lcc_c))
        score = (1000.0 / kb) - lcc_size_diff
        if score > best_score:
            best_score = score
            best = dict(
                seed=seed,
                targets_before=targets,
                targets_mid=targets_mid,
                targets_after=targets_after,
                K_before=K_before,
                k_before=kb,
            )

    if best is None:
        raise RuntimeError("No snapshot found; try relaxing max_pair_dist or max_kb.")
    print(f"Snapshot seed={best['seed']}: |K_before|={best['k_before']}, "
          f"targets: {best['targets_before']} → {best['targets_mid']} → {best['targets_after']}")
    return best


def draw_lcc_hull(ax, G_draw, pos, exclude_nodes=None, color='#9B59B6', alpha=0.22):
    try:
        from shapely.geometry import Point
        from shapely.ops import unary_union
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath
        HAS_SHAPELY = True
    except ImportError:
        HAS_SHAPELY = False

    G_eff = G_draw.copy()
    if exclude_nodes:
        for nd in exclude_nodes:
            if G_eff.has_node(nd): G_eff.remove_node(nd)
    if G_eff.number_of_nodes() == 0: return
    try:
        lcc_nodes = max(nx.connected_components(G_eff), key=len)
    except ValueError: return
    lcc_pts = [(pos[n][0], pos[n][1]) for n in lcc_nodes if n in pos]
    if not lcc_pts: return

    if not HAS_SHAPELY:
        pts = np.array(lcc_pts)
        if len(pts) < 3: return
        try:
            hull  = ConvexHull(pts)
            hpts  = pts[hull.vertices]
            closed = np.vstack([hpts, hpts[0]])
            ax.fill(closed[:,0], closed[:,1], color=color, alpha=alpha,
                    zorder=0, linewidth=0)
        except Exception: pass
        return

    pts_arr = np.array(lcc_pts)
    if len(pts_arr) >= 2:
        from scipy.spatial import KDTree
        tree = KDTree(pts_arr)
        dists, _ = tree.query(pts_arr, k=min(4, len(pts_arr)))
        avg_nn = float(np.median(dists[:, 1:]))
        radius = avg_nn * 0.45
    else:
        radius = 0.15

    disks = [Point(x, y).buffer(radius, resolution=32) for x, y in lcc_pts]
    union = unary_union(disks)

    def _poly_to_path(poly):
        from matplotlib.path import Path as MPath
        verts, codes = [], []
        ext = list(poly.exterior.coords)
        verts += ext
        codes += [MPath.MOVETO] + [MPath.LINETO]*(len(ext)-2) + [MPath.CLOSEPOLY]
        for interior in poly.interiors:
            inn = list(interior.coords)
            verts += inn
            codes += [MPath.MOVETO] + [MPath.LINETO]*(len(inn)-2) + [MPath.CLOSEPOLY]
        return MPath(verts, codes)

    from matplotlib.patches import PathPatch
    geoms = list(union.geoms) if union.geom_type == 'MultiPolygon' else [union]
    for geom in geoms:
        if geom.is_empty: continue
        patch = PathPatch(_poly_to_path(geom), facecolor=color, edgecolor='none',
                          alpha=alpha, zorder=0, linewidth=0)
        ax.add_patch(patch)


def draw_panel(ax, G_full, G_draw, pos, node_sizes, targets, K_nodes,
               removed_nodes=None, dashed_K_nodes=None,
               lcc_hull=False, lcc_exclude_nodes=None,
               prev_target_nodes=None,
               arrow_pairs=None):
    target_set   = set(targets)
    k_set        = set(K_nodes)
    removed_set  = set(removed_nodes)    if removed_nodes    else set()
    dashed_K_set = set(dashed_K_nodes)   if dashed_K_nodes   else set()
    prev_tgt_set = set(prev_target_nodes) if prev_target_nodes else set()
    active       = list(G_draw.nodes())

    if lcc_hull:
        exc = set(lcc_exclude_nodes) if lcc_exclude_nodes is not None else dashed_K_set
        draw_lcc_hull(ax, G_draw, pos, exclude_nodes=exc)

    for u, v in G_full.edges():
        if (u in dashed_K_set or v in dashed_K_set) and u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#C8C8C8', lw=0.75, linestyle=(0, (2.5, 1.5)),
                    alpha=1.0, zorder=1)

    edges_normal = [(u, v) for u, v in G_draw.edges()
                    if u not in dashed_K_set and v not in dashed_K_set]
    nx.draw_networkx_edges(G_draw, pos, ax=ax, edgelist=edges_normal,
                           alpha=0.35, width=0.5, edge_color='black')

    if dashed_K_set:
        dk = [nd for nd in dashed_K_set if nd in pos]
        if dk:
            sc = ax.scatter([pos[nd][0] for nd in dk], [pos[nd][1] for nd in dk],
                            s=[node_sizes.get(nd, 55) for nd in dk],
                            facecolors='none', edgecolors='#444444',
                            linewidths=0.7, zorder=6)
            sc.set_linestyle((0, (1.5, 1.5)))

    pt_nodes     = [nd for nd in prev_tgt_set
                    if nd in pos and nd not in target_set and nd not in dashed_K_set]
    pt_nodes_set = set(pt_nodes)

    main_nodes = [nd for nd in active
                  if nd not in dashed_K_set and nd not in pt_nodes_set]
    colors, sizes, ecs, lws = [], [], [], []
    for nd in main_nodes:
        if nd in target_set:
            colors.append(C_RED);    ecs.append(C_RED);    lws.append(1.5)
        elif nd in k_set:
            colors.append(C_ORANGE); ecs.append(C_ORANGE); lws.append(1.5)
        else:
            colors.append('#BDC3C7'); ecs.append('black');  lws.append(0.5)
        sizes.append(node_sizes.get(nd, 55))
    if main_nodes:
        nx.draw_networkx_nodes(G_draw, pos, nodelist=main_nodes, ax=ax,
                               node_color=colors, node_size=sizes,
                               edgecolors=ecs, linewidths=lws)

    if pt_nodes:
        nx.draw_networkx_nodes(G_draw, pos, nodelist=pt_nodes, ax=ax,
                               node_color=['#BDC3C7'] * len(pt_nodes),
                               node_size=[node_sizes.get(nd, 55) for nd in pt_nodes],
                               edgecolors='none', linewidths=0)
        sc_prev = ax.scatter(
            [pos[nd][0] for nd in pt_nodes],
            [pos[nd][1] for nd in pt_nodes],
            s=[node_sizes.get(nd, 55) for nd in pt_nodes],
            facecolors='none', edgecolors=C_RED,
            linewidths=1.2, zorder=8)
        sc_prev.set_linestyle((0, (1.5, 1.3)))

    if arrow_pairs is not None:
        for t_old, t_new in arrow_pairs:
            if t_old == t_new or t_old not in pos or t_new not in pos: continue
            x0, y0 = pos[t_old]
            x1, y1 = pos[t_new]
            r_src_pt = np.sqrt(node_sizes.get(t_old, 55) / np.pi)
            r_dst_pt = np.sqrt(node_sizes.get(t_new, 55) / np.pi)
            ax.annotate(
                '',
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->',
                    color=C_ARROW,
                    lw=1.5,
                    mutation_scale=12,
                    shrinkA=r_src_pt + 0.5,
                    shrinkB=r_dst_pt + 0.5,
                ),
                zorder=9,
            )

    all_shown = active + list(removed_set) + list(dashed_K_set)
    xc = [pos[nd][0] for nd in all_shown if nd in pos]
    yc = [pos[nd][1] for nd in all_shown if nd in pos]
    if xc:
        ddx = max(xc)-min(xc); ddy = max(yc)-min(yc); m = 0.06
        ax.set_xlim(min(xc) - ddx*m, max(xc) + ddx*m)
        ax.set_ylim(min(yc) - ddy*m, max(yc) + ddy*m)
    ax.axis('off'); ax.set_aspect('equal', adjustable='datalim')


print("Loading curve data ...")
df_fixed = pd.read_csv(FIXED_CSV)
df_rw    = pd.read_csv(RW_CSV)

print("Loading heatmap data ...")
df_heat = pd.read_csv(HEATMAP_CSV, index_col=0)

print("Loading 9/11 terrorist network ...")
G, pos, node_sizes, hijacker_nodes, node_labels = load_911_network(
    XML_PATH, layout_seed=LAYOUT_SEED)

print("Searching for snapshot (two-step movement) ...")
snap = find_snapshot(G, pos, hijacker_nodes,
                     n_targets=N_TARGETS_SNAP,
                     attack_per_step=ATTACK_PER_STEP_SNAP,
                     max_pair_dist=MAX_PAIR_DIST,
                     max_kb=MAX_KB)


G_c = G.copy()
for nd in snap['K_before']:
    if G_c.has_node(nd): G_c.remove_node(nd)
print(f"G_c: removed {snap['k_before']} K_before nodes (same as panel a)")



fig = plt.figure(figsize=(7.087, 5.55))

top_gs = fig.add_gridspec(
    1, 3,
    left=0.06, right=0.96, top=0.92, bottom=0.52,
    width_ratios=[1, 1, 1], wspace=0.15
)
ax_a = fig.add_subplot(top_gs[0, 0])
ax_b = fig.add_subplot(top_gs[0, 1])
ax_c = fig.add_subplot(top_gs[0, 2])

leg_gs = fig.add_gridspec(
    1, 1,
    left=0.06, right=0.96, top=0.49, bottom=0.45
)
ax_leg = fig.add_subplot(leg_gs[0, 0])
ax_leg.set_axis_off()

bot_gs = fig.add_gridspec(
    1, 2,
    left=0.07, right=0.92,
    top=0.42, bottom=0.08,
    width_ratios=[1, 1.2], wspace=0.2
)
ax_d  = fig.add_subplot(bot_gs[0, 0])
ax_e  = fig.add_subplot(bot_gs[0, 1])


draw_panel(ax_a, G, G, pos, node_sizes,
           targets=snap['targets_before'],
           K_nodes=[],
           dashed_K_nodes=snap['K_before'],
           lcc_hull=True)


draw_panel(ax_b, G, G, pos, node_sizes,
           targets=snap['targets_after'],
           K_nodes=[],
           prev_target_nodes=snap['targets_before'],
           arrow_pairs=list(zip(snap['targets_before'], snap['targets_after'])))


draw_panel(ax_c, G, G_c, pos, node_sizes,
           targets=snap['targets_after'],
           K_nodes=[],
           dashed_K_nodes=snap['K_before'],
           lcc_hull=True)


h_tgt  = _mpatches_.Circle((0,0), 1, facecolor=C_RED, edgecolor=C_RED,
                             linewidth=0, label='Target node')
h_rem  = _mpatches_.Circle((0,0), 1, facecolor='none', edgecolor='#444444',
                             linewidth=0.7, linestyle=(0,(1.5,1.5)),
                             label='Removed node')
h_prev = _mpatches_.Circle((0,0), 1, facecolor='#BDC3C7', edgecolor=C_RED,
                             linewidth=0.9, linestyle=(0,(2.0,1.5)),
                             label='Former target')
h_arr  = Line2D([0],[0], color=C_ARROW, lw=1.5,
                marker='>', markersize=4, markerfacecolor=C_ARROW,
                linestyle='-', label='Target movement')
h_lcc  = _mpatches_.Circle((0,0), 1, facecolor='#9B59B6', edgecolor='none',
                               linewidth=0, alpha=0.22, label='LCC')

ax_leg.legend(
    handles=[h_tgt, h_rem, h_prev, h_arr, h_lcc],
    handler_map={_mpatches_.Circle: _HandlerCircle_(),
                 h_lcc: _HandlerLargeCircle_(),
                 h_arr: _HandlerPointyArrow_()},
    loc='center',
    ncol=5, fontsize=7, frameon=False,
    handlelength=3.0, handletextpad=0.5,
    columnspacing=1.0, borderpad=0,
)


fig.canvas.draw()

ba = ax_a.get_position()
bb = ax_b.get_position()
bc = ax_c.get_position()
PAD = 0.001

BOX_STYLE = 'round,pad=0.01,rounding_size=0.02'
BOX_COLOR = '#555555'
BOX_LW    = 0.75

rect_a = mpatches.FancyBboxPatch(
    (ba.x0-PAD, ba.y0-PAD), ba.width+2*PAD, ba.height+2*PAD,
    boxstyle=BOX_STYLE,
    linewidth=BOX_LW, edgecolor=BOX_COLOR, facecolor='none',
    transform=fig.transFigure, clip_on=False, zorder=20)
fig.add_artist(rect_a)
fig.text((ba.x0+ba.x1)/2, ba.y1+PAD+0.012,
         'Static targets', ha='center', va='bottom',
         fontsize=8, transform=fig.transFigure)

x0_mv = bb.x0 - PAD
y0_mv = min(bb.y0, bc.y0) - PAD
w_mv  = bc.x1 - bb.x0 + 2*PAD
h_mv  = max(bb.y1, bc.y1) - min(bb.y0, bc.y0) + 2*PAD
rect_mv = mpatches.FancyBboxPatch(
    (x0_mv, y0_mv), w_mv, h_mv,
    boxstyle=BOX_STYLE,
    linewidth=BOX_LW, edgecolor=BOX_COLOR, facecolor='none',
    transform=fig.transFigure, clip_on=False, zorder=20)
fig.add_artist(rect_mv)
fig.text((bb.x0+bc.x1)/2, max(bb.y1, bc.y1)+PAD+0.012,
         'Moving targets', ha='center', va='bottom',
         fontsize=8, transform=fig.transFigure)


ax_d.plot(df_fixed['P'], df_fixed['S_T'],  color=C_BLUE, lw=1.5,
          label='Static targets')
ax_d.plot(df_rw['P'], df_rw['S_T_mean'],   color=C_RED,  lw=1.5,
          label='Moving targets')
ax_d.fill_between(df_rw['P'],
                  np.array(df_rw['S_T_mean']) - np.array(df_rw['S_T_sem']),
                  np.array(df_rw['S_T_mean']) + np.array(df_rw['S_T_sem']),
                  color=C_RED, alpha=0.15)
ax_d.set_xlabel(r'$P$'); ax_d.set_ylabel(r'$S_T$')
ax_d.set_xlim(0, 1); ax_d.set_ylim(-0.02, 1.05)
ax_d.tick_params(which='both', direction='in', top=True, right=True)
ax_d.legend(frameon=False, edgecolor='black', fancybox=False, loc='upper right')


def fmt_col(s):
    try:
        v = float(s)
        return f"{v:.2f}".rstrip('0').rstrip('.')
    except ValueError:
        return s

clean_cols = [fmt_col(c) for c in df_heat.columns]
show_cols  = [i for i, c in enumerate(df_heat.columns)
              if (lambda v: v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))(
                  float(c) if c.replace('.','').isnumeric() else -1)]

atk_vals  = df_heat.index.tolist()
show_rows = sorted(set(
    i for i, v in enumerate(atk_vals)
    if v % 10 == 0 or v == atk_vals[0] or v == atk_vals[-1]))

divider = make_axes_locatable(ax_e)
cax_e = divider.append_axes("right", size="3%", pad=0.4)

sns.heatmap(df_heat.astype(float), ax=ax_e, cbar_ax=cax_e,
            cmap='coolwarm', vmin=0.3, vmax=0.6,
            cbar_kws={'ticks': [0.3, 0.4, 0.5, 0.6]},
            xticklabels=False, yticklabels=False)
ax_e.set_xticks([i + 0.5 for i in show_cols])
ax_e.set_xticklabels([clean_cols[i] for i in show_cols], rotation=0, ha='center')
ax_e.set_yticks([i + 0.5 for i in show_rows])
ax_e.set_yticklabels([str(atk_vals[i]) for i in show_rows], rotation=0)
ax_e.invert_yaxis()
ax_e.tick_params(which='both', length=0)
ax_e.set_title('')
ax_e.set_ylabel('Attack per step', fontsize=8, labelpad=3)
ax_e.yaxis.set_label_position('left'); ax_e.yaxis.tick_left()
ax_e.set_xlabel('Move probability')

cbar = ax_e.collections[0].colorbar
cbar.ax.tick_params(labelsize=7)
cbar.set_label(r'$P_c$', fontsize=8)


x_col1 = 0.024
x_col2 = 0.348
y_row1 = ax_a.get_position().y1 + 0.03
y_row2 = ax_d.get_position().y1 + 0.02

fig.text(x_col1, y_row1, 'a', fontsize=10, fontweight='bold', va='top')
fig.text(x_col2, y_row1, 'b', fontsize=10, fontweight='bold', va='top')
fig.text(x_col1, y_row2, 'c', fontsize=10, fontweight='bold', va='top')
fig.text(0.445,  y_row2, 'd', fontsize=10, fontweight='bold', va='top')

# ── Save ─────────────────────────────────────────────────────────────────────
fig.savefig(os.path.join(SCRIPT_DIR, 'fig1.pdf'))
fig.savefig(os.path.join(SCRIPT_DIR, 'fig1.png'))
plt.show()
print("Done. Saved figure_combined.pdf / .png")