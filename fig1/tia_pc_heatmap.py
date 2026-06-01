import matplotlib
matplotlib.use('Agg')

import os
import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# SAVE_DIR = "/share/home/yuhuajian/Project_RL/comparison_rl/fig1/test"
# os.makedirs(SAVE_DIR, exist_ok=True)

def tia_core(G: nx.Graph, target_nodes: List[int]):
    G = G.copy()
    T = set(target_nodes)
    K = set()
    R = set(G.nodes()) - T
    for node in T:
        if node in G:
            K.update(G.neighbors(node))
    K -= T
    R -= K
    neighbors_set = {node: set(G.neighbors(node)) for node in G.nodes()}
    Nmin_K = len(K)
    while True:
        changed = False
        DR = {node: len(neighbors_set[node] & R) for node in K}
        nodes_to_move = [node for node in K if DR[node] == 0]
        if nodes_to_move:
            for node in nodes_to_move:
                T.add(node); K.remove(node)
            changed = True
        if len(K) < Nmin_K:
            Nmin_K = len(K)
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
    K_sorted = sorted(K, key=lambda x: final_DR[x], reverse=True)
    return T, K_sorted, R


def extended_tia_attack_order(G: nx.Graph, target_nodes: List[int]) -> List[int]:
    original_target_set = set(target_nodes)
    T, K, R = tia_core(G, target_nodes)
    return list(K) + list(T - original_target_set) + list(R)


def generate_random_targets(G: nx.Graph, target_ratio: float,
                            seed: Optional[int] = None) -> List[int]:
    if seed is not None:
        random.seed(seed)
    n = G.number_of_nodes()
    target_num = max(1, int(n * target_ratio))
    return random.sample(list(G.nodes()), target_num)


def generate_localized_targets(G: nx.Graph, target_ratio: float,
                               seed: Optional[int] = None,
                               radius: float = 0.05,
                               max_radius: float = 1.0,
                               step: float = 0.05) -> List[int]:
    rng = np.random.RandomState(seed)
    pos = nx.spring_layout(G, seed=seed)
    nodes = list(G.nodes())
    n_targets = max(1, int(len(nodes) * target_ratio))
    center_node = rng.choice(nodes)
    cx, cy = pos[center_node]
    current_radius = radius
    localized = []
    while current_radius < max_radius:
        localized = [n for n, (x, y) in pos.items()
                     if (x - cx) ** 2 + (y - cy) ** 2 < current_radius ** 2]
        if len(localized) >= n_targets: break
        current_radius += step
    if len(localized) >= n_targets:
        idx = rng.choice(len(localized), size=n_targets, replace=False)
        return [localized[i] for i in idx]
    else:
        remaining = n_targets - len(localized)
        others = list(set(nodes) - set(localized))
        if remaining > 0 and others:
            extra_idx = rng.choice(len(others), size=min(remaining, len(others)), replace=False)
            return localized + [others[i] for i in extra_idx]
        return localized


def simulate_adaptive(G, strategy_fn, target_nodes_init,
                      move_prob, attack_per_step, seed=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    n = G.number_of_nodes()
    n_targets = len(target_nodes_init)
    n_non_targets = n - n_targets
    max_steps = n_non_targets // attack_per_step
    G_sim = G.copy()
    targets = list(target_nodes_init)
    attacked = set()
    p_vals, st_vals = [], []

    def is_isolated(G_sub, tgts):
        if G_sub.number_of_nodes() == 0: return True
        lcc = max(nx.connected_components(G_sub), key=len)
        return all(t not in lcc for t in tgts)

    for _ in range(max_steps):
        new_targets = [None] * n_targets
        occupied = set()
        for i, node in enumerate(targets):
            if random.random() < move_prob:
                nbrs = [nb for nb in G_sim.neighbors(node)
                        if nb not in attacked and nb not in targets and nb not in occupied]
                new_targets[i] = random.choice(nbrs) if nbrs else node
            else:
                new_targets[i] = node
            occupied.add(new_targets[i])
        targets = new_targets

        attack_order = strategy_fn(G_sim, list(targets))
        attack_order = [nd for nd in attack_order if nd not in targets]
        count, idx = 0, 0
        while count < attack_per_step and idx < len(attack_order):
            c = attack_order[idx]; idx += 1
            if c not in targets and c not in attacked and G_sim.has_node(c):
                G_sim.remove_node(c); attacked.add(c); count += 1
                if is_isolated(G_sim, targets): break
        while count < attack_per_step:
            cands = [nd for nd in G_sim.nodes if nd not in targets and nd not in attacked]
            if not cands: break
            nd = random.choice(cands)
            G_sim.remove_node(nd); attacked.add(nd); count += 1
            if is_isolated(G_sim, targets): break

        if G_sim.number_of_nodes() == 0:
            ratio = 0.0
        else:
            lcc = max(nx.connected_components(G_sim), key=len)
            ratio = len([t for t in targets if t in lcc]) / n_targets
        p_vals.append(len(attacked) / n_non_targets)
        st_vals.append(ratio)

    return p_vals, st_vals


def simulate_multiple_runs(G, strategy_fn, target_nodes,
                           move_prob, attack_per_step, runs=10, seed=42):
    all_ratios = []
    p_ref = None
    current_seed = seed
    for _ in range(runs):
        p, ratios = simulate_adaptive(G, strategy_fn, target_nodes,
                                      move_prob, attack_per_step, seed=current_seed)
        current_seed += 1
        all_ratios.append(ratios)
        if p_ref is None: p_ref = p
    max_len = max(len(r) for r in all_ratios)
    for i in range(len(all_ratios)):
        if len(all_ratios[i]) < max_len:
            all_ratios[i] += [0.0] * (max_len - len(all_ratios[i]))
    return p_ref, np.mean(all_ratios, axis=0).tolist()


def get_pc(p_list, ratios, threshold=0.01):
    for p_val, ratio in zip(p_list, ratios):
        if ratio <= threshold: return p_val
    return 1.0

def _compute_one_cell(args):
    nodes, edges, target_nodes, mp, atk, runs, seed = args
    G_worker = nx.Graph()
    G_worker.add_nodes_from(nodes)
    G_worker.add_edges_from(edges)

    p_ref, mean_ratios = simulate_multiple_runs(
        G_worker, extended_tia_attack_order, target_nodes,
        move_prob=mp, attack_per_step=atk, runs=runs, seed=seed)
    pc = get_pc(p_ref, mean_ratios)
    return (atk, mp, pc)


def compute_pc_grid_parallel(G, target_nodes, move_probs, attack_per_steps,
                              runs=10, seed=42, n_workers=8):
    nodes = list(G.nodes())
    edges = list(G.edges())

    tasks = [
        (nodes, edges, target_nodes, mp, atk, runs, seed)
        for atk in attack_per_steps
        for mp in move_probs
    ]
    total = len(tasks)

    results = {}
    t0 = time.time()
    done = 0

    print(f"[{time.strftime('%H:%M:%S')}] Starting parallel sweep: "
          f"{total} cells, {n_workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(_compute_one_cell, t): t for t in tasks}

        for future in as_completed(future_to_task):
            atk, mp, pc = future.result()
            results[(atk, mp)] = pc
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"[{done}/{total}] atk={atk}, p={mp:.2f}, pc={pc:.3f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    n_rows, n_cols = len(attack_per_steps), len(move_probs)
    pc_matrix = np.zeros((n_rows, n_cols))
    for i, atk in enumerate(attack_per_steps):
        for j, mp in enumerate(move_probs):
            pc_matrix[i, j] = results[(atk, mp)]
    return pc_matrix


GRAPH_TYPE       = 'WS'
N                = 1000
TARGET_RATIO     = 0.05
TARGET_MODE      = 'random'
SEED             = 42
# MOVE_PROBS       = np.sort(np.append(np.linspace(0.0, 1.0, 11), 0.01))
MOVE_PROBS       = np.linspace(0.0, 1.0, 51)
# ATTACK_PER_STEPS = np.arange(1, 101, 5)
# ATTACK_PER_STEPS = np.sort(np.append(np.linspace(2, 100, 50).astype(int), 1))
ATTACK_PER_STEPS = np.arange(1, 51,1)
N_RUNS           = 100

N_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

print(f"[{time.strftime('%H:%M:%S')}] Building graph ...", flush=True)
if GRAPH_TYPE == 'WS':
    G = nx.watts_strogatz_graph(N, 10, 0.1, seed=SEED)
elif GRAPH_TYPE == 'BA':
    G = nx.barabasi_albert_graph(N, 4, seed=SEED)
else:
    raise ValueError(f"Unknown GRAPH_TYPE: {GRAPH_TYPE}")

if TARGET_MODE == 'random':
    target_nodes = generate_random_targets(G, TARGET_RATIO, seed=SEED)
else:
    target_nodes = generate_localized_targets(G, TARGET_RATIO, seed=SEED)

print(f"Network : {GRAPH_TYPE}, N={G.number_of_nodes()}, E={G.number_of_edges()}")
print(f"Targets : {len(target_nodes)} ({TARGET_MODE}), ratio={TARGET_RATIO}")
print(f"Sweep   : {len(MOVE_PROBS)} move_probs × "
      f"{len(ATTACK_PER_STEPS)} attack_per_steps, {N_RUNS} runs each")
print(f"Workers : {N_WORKERS}", flush=True)

if __name__ == '__main__':
    t_start = time.time()
    pc_matrix = compute_pc_grid_parallel(
        G, target_nodes,
        MOVE_PROBS, ATTACK_PER_STEPS,
        runs=N_RUNS, seed=SEED, n_workers=N_WORKERS)
    print(f"\nSweep done in {(time.time()-t_start)/60:.1f} min", flush=True)

    col_names = [f"{mp:.3f}" for mp in MOVE_PROBS]
    tag       = f"pc_{TARGET_MODE}_{GRAPH_TYPE.lower()}"
    df        = pd.DataFrame(pc_matrix, index=ATTACK_PER_STEPS, columns=col_names)
    df.index.name = "attack_per_step"

    csv_path = os.path.join(DATA_DIR, f"{tag}.csv")
    df.to_csv(csv_path)
    # csv_path = os.path.join(SAVE_DIR, f"{tag}.csv")
    # df.to_csv(csv_path)
    print(f"CSV saved : {csv_path}", flush=True)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(df.astype(float), annot=False, cmap='coolwarm',
                vmin=0, vmax=1, center=0.5,
                linewidths=0.3, linecolor='grey',
                cbar_kws={'label': r'$P_c$'}, ax=ax)
    ax.set_xlabel('Move Probability')
    ax.set_ylabel('Attack per Step')
    ax.invert_yaxis()
    ax.tick_params(direction='in')
    fig.tight_layout()

    fig_path = os.path.join(SAVE_DIR, f"{tag}_heatmap")
    fig.savefig(f"{fig_path}.pdf")
    fig.savefig(f"{fig_path}.png")
    plt.close(fig)
    print(f"Figures saved : {fig_path}.pdf / .png", flush=True)