import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

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
                T.add(node)
                K.remove(node)
            changed = True

        if len(K) < Nmin_K:
            Nmin_K = len(K)

        DR = {node: len(neighbors_set[node] & R) for node in K}

        for r_node in R:
            if r_node not in neighbors_set:
                continue
            r_node_neighbors = neighbors_set[r_node]
            k_neighbors_of_r = r_node_neighbors & K
            has_critical_neighbor = any(DR.get(nbr, 0) == 1 for nbr in k_neighbors_of_r)
            if not has_critical_neighbor:
                continue

            K_temp = K | {r_node}
            R_temp = R - {r_node}
            T_temp = set(T)

            DR_temp = dict(DR)
            DR_temp[r_node] = len(r_node_neighbors & R_temp)
            for nbr in k_neighbors_of_r:
                DR_temp[nbr] -= 1

            nodes_to_move_temp = [k for k in K_temp if DR_temp.get(k, 0) == 0]
            for k_node in nodes_to_move_temp:
                T_temp.add(k_node)
                K_temp.remove(k_node)

            if len(K_temp) < Nmin_K:
                K, T, R = K_temp, T_temp, R_temp
                Nmin_K = len(K)
                changed = True
                break

        if not changed:
            break

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
                               radius: float = 0.05, max_radius: float = 1.0,
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
        if len(localized) >= n_targets:
            break
        current_radius += step

    if len(localized) >= n_targets:
        idx = rng.choice(len(localized), size=n_targets, replace=False)
        return [localized[i] for i in idx]
    else:
        remaining = n_targets - len(localized)
        others = list(set(nodes) - set(localized))
        if remaining > 0 and others:
            extra_idx = rng.choice(len(others),
                                   size=min(remaining, len(others)), replace=False)
            return localized + [others[i] for i in extra_idx]
        return localized


def simulate_fixed(G: nx.Graph, attack_order: List[int],
                   target_nodes: List[int]):
    n = G.number_of_nodes()
    target_set = set(target_nodes)
    n_non_targets = n - len(target_set)
    order = [nd for nd in attack_order if nd not in target_set]

    G_sim = G.copy()
    p_vals, st_vals = [], []

    for i in range(n_non_targets):
        if i < len(order) and G_sim.has_node(order[i]):
            G_sim.remove_node(order[i])
        if G_sim.number_of_nodes() == 0:
            ratio = 0.0
        else:
            lcc = max(nx.connected_components(G_sim), key=len)
            targets_in_lcc = [t for t in target_set if t in lcc]
            ratio = 0.0 if all(nd in target_set for nd in lcc) \
                else len(targets_in_lcc) / len(target_set)
        p_vals.append((i + 1) / n_non_targets)
        st_vals.append(ratio)

    return p_vals, st_vals


def simulate_random_walk(G: nx.Graph, attack_order: List[int],
                         target_nodes_init: List[int],
                         move_prob: float, attack_per_step: int = 1,
                         seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = G.number_of_nodes()
    n_targets = len(target_nodes_init)
    n_non_targets = n - n_targets
    max_steps = n_non_targets // attack_per_step

    G_sim = G.copy()
    targets = list(target_nodes_init)
    attacked = set()
    p_vals, st_vals = [], []

    def is_isolated(G_sub, tgts):
        if G_sub.number_of_nodes() == 0:
            return True
        lcc = max(nx.connected_components(G_sub), key=len)
        return all(t not in lcc for t in tgts)

    for _ in range(max_steps):
        new_targets = [None] * n_targets
        occupied = set()
        for i, node in enumerate(targets):
            if random.random() < move_prob:
                nbrs = [nb for nb in G_sim.neighbors(node)
                        if nb not in attacked and nb not in targets
                        and nb not in occupied]
                new_targets[i] = random.choice(nbrs) if nbrs else node
            else:
                new_targets[i] = node
            occupied.add(new_targets[i])
        targets = new_targets

        order = [nd for nd in attack_order if nd not in targets]
        count, idx = 0, 0
        while count < attack_per_step and idx < len(order):
            c = order[idx]; idx += 1
            if c not in targets and c not in attacked and G_sim.has_node(c):
                G_sim.remove_node(c); attacked.add(c); count += 1
                if is_isolated(G_sim, targets):
                    break

        while count < attack_per_step:
            cands = [nd for nd in G_sim.nodes
                     if nd not in targets and nd not in attacked]
            if not cands:
                break
            nd = random.choice(cands)
            G_sim.remove_node(nd); attacked.add(nd); count += 1
            if is_isolated(G_sim, targets):
                break

        if G_sim.number_of_nodes() == 0:
            ratio = 0.0
        else:
            lcc = max(nx.connected_components(G_sim), key=len)
            targets_in_lcc = [t for t in targets if t in lcc]
            ratio = 0.0 if all(nd in set(targets) for nd in lcc) \
                else len(targets_in_lcc) / n_targets
        p_vals.append(len(attacked) / n_non_targets)
        st_vals.append(ratio)

    return p_vals, st_vals


def simulate_multiple_runs(G, attack_order, target_nodes,
                           move_prob, attack_per_step=1, runs=100):
    all_ratios = []
    p_ref = None
    for i in range(runs):
        p, ratios = simulate_random_walk(
            G, attack_order, target_nodes, move_prob, attack_per_step)
        all_ratios.append(ratios)
        if p_ref is None:
            p_ref = p

    min_len = min(len(r) for r in all_ratios)
    all_ratios = [r[:min_len] for r in all_ratios]
    p_ref = p_ref[:min_len]
    mean_ratios = np.mean(all_ratios, axis=0).tolist()
    sem_ratios = (np.std(all_ratios, axis=0) / np.sqrt(runs)).tolist()
    return p_ref, mean_ratios, sem_ratios


N = 1000
M = 5
G = nx.watts_strogatz_graph(N, 2 * M, 0.1, seed=42)

TARGET_RATIO = 0.05
TARGET_MODE = 'random'
SEED = 42

if TARGET_MODE == 'random':
    target_nodes = generate_random_targets(G, TARGET_RATIO, seed=SEED)
elif TARGET_MODE == 'localized':
    target_nodes = generate_localized_targets(G, TARGET_RATIO, seed=SEED)
else:
    raise ValueError(f"Unknown TARGET_MODE: {TARGET_MODE}")

MOVE_PROB = 0.05
ATTACK_PER_STEP = 1
N_RUNS = 100


print(f"Network: N={G.number_of_nodes()}, E={G.number_of_edges()}")
print(f"Targets: {len(target_nodes)} ({TARGET_MODE}), move_prob={MOVE_PROB}")
attack_order = extended_tia_attack_order(G, target_nodes)

p_fixed, st_fixed = simulate_fixed(G, attack_order, target_nodes)

p_rw, st_rw_mean, st_rw_sem = simulate_multiple_runs(
    G, attack_order, target_nodes, MOVE_PROB, ATTACK_PER_STEP, N_RUNS)


df_fixed = pd.DataFrame({'P': p_fixed, 'S_T': st_fixed})
df_fixed.to_csv(os.path.join(DATA_DIR, 'static_tia_fixed.csv'), index=False)
# df_fixed.to_csv('static_tia_fixed.csv', index=False)

df_rw = pd.DataFrame({'P': p_rw, 'S_T_mean': st_rw_mean, 'S_T_sem': st_rw_sem})
df_rw.to_csv(os.path.join(DATA_DIR, 'static_tia_random_walk.csv'), index=False)
# df_rw.to_csv('static_tia_random_walk.csv', index=False)


plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(4.5, 3.5))

ax.plot(p_fixed, st_fixed,
        color='#1f77b4', linestyle='-', label='Fixed targets')
ax.plot(p_rw, st_rw_mean,
        color='#d62728', linestyle='-', label='Random walk')
ax.fill_between(p_rw,
                np.array(st_rw_mean) - np.array(st_rw_sem),
                np.array(st_rw_mean) + np.array(st_rw_sem),
                color='#d62728', alpha=0.15)

ax.set_xlabel(r'$P$')
ax.set_ylabel(r'$S_T$')
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.05)
ax.legend(frameon=True, edgecolor='black', fancybox=False)
ax.tick_params(direction='in', top=True, right=True)

fig.tight_layout()
fig.savefig('static_tia_vs_moving.pdf')
fig.savefig('static_tia_vs_moving.png')
plt.show()