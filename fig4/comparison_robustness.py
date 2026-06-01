import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Tuple, List, Dict, Optional, Callable
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Batch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_FIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_FIG_DIR, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))

from comparison_method import (
    generate_graph,
    generate_random_targets,
    generate_localized_targets,
    td_method_adaptive,
    katz_method_adaptive,
    extended_tia_method_adaptive,
    RLAgentWrapper,
    IncrementalPyGBuilder,
    GNN_AVAILABLE,
)


N_NODES          = 1024
TARGET_RATIO     = 0.05

MOVE_PROB        = 1.0
ATTACK_RATIO     = 0.1

TAU_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

SIMULATION_TIMES = 100
SEED             = 42

MODEL_PATH = os.path.join(_REPO_ROOT, 'train',
                          'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                          'model.pt')
SAVE_DIR   = os.path.join(_FIG_DIR, 'data')

N_WORKERS    = 63
N_WORKERS_RL = 63

GRAPH_PARAMS = {
    'ba_m_range':    (2, 4),
    'ws_k_range':    (4, 8),
    'ws_beta_range': (0.1, 0.3),
}

CONFIGS = [
    ('BA', 'random'),
    ('BA', 'localized'),
    ('WS', 'random'),
    ('WS', 'localized'),
]


mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,  'ps.fonttype': 42,
    'font.size':         6,
    'axes.labelsize':    7,
    'axes.titlesize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size':  2.5, 'ytick.major.size':  2.5,
    'xtick.direction':   'in', 'ytick.direction':   'in',
    'figure.dpi':        300,  'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'figure.facecolor':  'white', 'axes.facecolor': 'white',
})

METHOD_COLORS = {
    'Adaptive TD':   '#0072B2',
    'Adaptive Katz': '#D55E00',
    'Adaptive TIA':  '#009E73',
    'GNN-RL':        '#CC79A7',
}
METHOD_MARKERS = {
    'Adaptive TD':   'o',
    'Adaptive Katz': 's',
    'Adaptive TIA':  '^',
    'GNN-RL':        'D',
}
PANEL_LABELS = list('abcdefgh')


class BiasedMovingTargetSimulator:

    def __init__(self, G: nx.Graph, target_nodes: List[int],
                 move_prob: float, attack_per_step: int,
                 tau: float = 2.0, seed: Optional[int] = None):
        self.original_graph   = G.copy()
        self.current_graph    = G.copy()
        self.target_nodes     = set(target_nodes)
        self.removed_nodes    = set()
        self.move_prob        = move_prob
        self.attack_per_step  = attack_per_step
        self.tau              = tau
        self.n_non_targets    = sum(1 for n in G.nodes() if n not in target_nodes)
        self.initial_lcc_size = self._get_lcc_size()
        self.cumulative_anc   = 0.0
        self.anc_count        = 0
        self.attacks_in_current_round = 0
        self.rng = np.random.RandomState(seed)

    def _get_lcc_size(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        comps = list(nx.connected_components(self.current_graph))
        return len(max(comps, key=len)) if comps else 0

    def _count_targets_in_lcc(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        comps = list(nx.connected_components(self.current_graph))
        if not comps:
            return 0
        lcc = max(comps, key=len)
        return sum(1 for t in self.target_nodes if t in lcc)

    def _softmax_choice(self, neighbors: List[int]) -> int:
        cur_deg = dict(self.current_graph.degree())
        degs = [cur_deg.get(nb, 1) for nb in neighbors]
        max_d = max(degs)
        exp_v = [math.exp((d - max_d) / self.tau) for d in degs]
        total = sum(exp_v)
        probs = [e / total for e in exp_v]
        return self.rng.choice(neighbors, p=probs)

    def _execute_target_movement(self):
        if self.move_prob <= 0:
            return
        movements = []
        claimed   = set()
        for t in [t for t in self.target_nodes if t in self.current_graph]:
            if self.rng.random() > self.move_prob:
                continue
            valid = [n for n in self.current_graph.neighbors(t)
                     if n not in self.removed_nodes
                     and n not in self.target_nodes
                     and n not in claimed]
            if valid:
                new_pos = self._softmax_choice(valid)
                movements.append((t, new_pos))
                claimed.add(new_pos)
        for old, new in movements:
            self.target_nodes.discard(old)
            self.target_nodes.add(new)

    def is_done(self) -> bool:
        return self._count_targets_in_lcc() == 0

    def remove_single_node(self, node: int) -> bool:
        if node in self.target_nodes or node in self.removed_nodes \
                or node not in self.current_graph:
            return self.is_done()
        self.removed_nodes.add(node)
        self.current_graph.remove_node(node)
        self.attacks_in_current_round += 1
        self.cumulative_anc += self._get_lcc_size() / max(1, self.initial_lcc_size)
        self.anc_count += 1
        return self.is_done()

    def remove_nodes_batch(self, nodes_to_remove: List[int]) -> bool:
        for node in nodes_to_remove:
            if node in self.target_nodes or node in self.removed_nodes \
                    or node not in self.current_graph:
                continue
            self.removed_nodes.add(node)
            self.current_graph.remove_node(node)
            self.attacks_in_current_round += 1
            self.cumulative_anc += self._get_lcc_size() / max(1, self.initial_lcc_size)
            self.anc_count += 1
            if self.is_done():
                return True
        return self.is_done()

    def execute_movement_if_round_complete(self):
        if self.attacks_in_current_round >= self.attack_per_step:
            if not self.is_done():
                self._execute_target_movement()
            self.attacks_in_current_round = 0

    def get_metrics(self) -> Tuple[float, float]:
        pc  = len(self.removed_nodes) / max(1, self.n_non_targets)
        anc = self.cumulative_anc / self.anc_count if self.anc_count > 0 else 1.0
        return pc, anc



def simulate_biased_heuristic(G, target_nodes, move_prob, attack_per_step,
                               method_func, tau, seed=None,
                               batch_mode=False) -> Tuple[float, float, bool]:
    sim       = BiasedMovingTargetSimulator(G, target_nodes, move_prob,
                                            attack_per_step, tau, seed)
    max_steps = sum(1 for n in G.nodes() if n not in target_nodes)

    if batch_mode:
        while not sim.is_done():
            quota = sim.attack_per_step - sim.attacks_in_current_round
            if quota <= 0:
                sim.execute_movement_if_round_complete()
                quota = sim.attack_per_step
            ranking = method_func(sim.current_graph, list(sim.target_nodes))
            batch   = []
            for node in ranking:
                if node not in sim.removed_nodes and node not in sim.target_nodes:
                    batch.append(node)
                    if len(batch) >= quota:
                        break
            if not batch:
                break
            if sim.remove_nodes_batch(batch):
                break
            sim.execute_movement_if_round_complete()
    else:
        for _ in range(max_steps):
            if sim.is_done():
                break
            order = method_func(sim.current_graph, list(sim.target_nodes))
            order = [n for n in order
                     if n not in sim.removed_nodes and n not in sim.target_nodes]
            if not order:
                break
            if sim.remove_single_node(order[0]):
                break
            sim.execute_movement_if_round_complete()

    pc, anc = sim.get_metrics()
    return pc, anc, sim.is_done()


def simulate_biased_rl(G, target_nodes, rl_agent, move_prob, attack_per_step,
                       tau, seed=None) -> Tuple[float, float, bool]:
    sim     = BiasedMovingTargetSimulator(G, target_nodes, move_prob,
                                          attack_per_step, tau, seed)
    builder = IncrementalPyGBuilder(sim.original_graph, sim.target_nodes)
    max_steps = sum(1 for n in G.nodes() if n not in target_nodes)

    for _ in range(max_steps):
        if sim.is_done():
            break
        data, nodes = builder.build_data(
            sim.current_graph, sim.target_nodes, sim.removed_nodes,
            move_prob, sim.attacks_in_current_round, attack_per_step)
        action_idx = rl_agent.get_action_deterministic(data)
        node       = nodes[action_idx]
        if sim.remove_single_node(node):
            break
        sim.execute_movement_if_round_complete()

    pc, anc = sim.get_metrics()
    return pc, anc, sim.is_done()


_worker_rl_agent = None


def _init_rl_worker(model_path: str):
    global _worker_rl_agent
    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path) and GNN_AVAILABLE:
        try:
            _worker_rl_agent = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            print(f"  [RL worker] load failed: {e}")
            _worker_rl_agent = None
    else:
        _worker_rl_agent = None


def _run_heuristic_task(args):
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     method, move_prob, attack_per_step, tau, sim_seed) = args

    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    targets = (generate_random_targets(G, target_ratio, sim_seed)
               if target_dist == 'random'
               else generate_localized_targets(G, target_ratio, sim_seed))

    if method == 'Adaptive TD':
        func, batch = td_method_adaptive, True
    elif method == 'Adaptive TIA':
        func, batch = extended_tia_method_adaptive, True
    else:
        func, batch = katz_method_adaptive, False

    pc, anc, success = simulate_biased_heuristic(
        G, targets, move_prob, attack_per_step, func, tau, sim_seed, batch)
    return {'pc': pc, 'anc': anc, 'success': success, 'method': method, 'tau': tau}


def _run_rl_task(args):
    global _worker_rl_agent
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     move_prob, attack_per_step, tau, sim_seed) = args

    if _worker_rl_agent is None:
        return {'pc': None, 'anc': None, 'success': False, 'tau': tau}

    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    targets = (generate_random_targets(G, target_ratio, sim_seed)
               if target_dist == 'random'
               else generate_localized_targets(G, target_ratio, sim_seed))

    pc, anc, success = simulate_biased_rl(
        G, targets, _worker_rl_agent, move_prob, attack_per_step, tau, sim_seed)
    return {'pc': pc, 'anc': anc, 'success': success, 'tau': tau}



def run_robustness_experiment(
        graph_type:  str,
        target_dist: str,
        move_prob:   float,
        attack_ratio: float,
        tau_values:  List[float],
        sim_times:   int,
        graph_params: Dict,
        model_path:  Optional[str],
        seed:        int = 42,
        n_workers:   int = None,
        n_workers_rl: int = None,
) -> Dict:

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    use_rl      = bool(model_path and os.path.exists(model_path) and GNN_AVAILABLE)
    h_methods   = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA']
    all_methods = h_methods + (['GNN-RL'] if use_rl else [])
    results     = {m: defaultdict(lambda: {'pc': [], 'anc': [], 'success': []})
                   for m in all_methods}

    rng           = np.random.RandomState(seed)
    h_tasks: List = []
    rl_tasks: List = []

    print(f"\n  Generating tasks  ({graph_type}-{target_dist})  ...")
    print(f"    move_prob={move_prob:.3f},  attack_ratio={attack_ratio:.3f}")
    print(f"    GNN-RL: {'Enabled' if use_rl else 'Disabled'}")

    for tau in tau_values:
        for _ in range(sim_times):
            sim_seed = int(rng.randint(0, int(1e9)))

            for method in h_methods:
                h_tasks.append((
                    graph_type, N_NODES, graph_params, target_dist, TARGET_RATIO,
                    method, move_prob, attack_ratio, tau, sim_seed
                ))
            if use_rl:
                rl_tasks.append((
                    graph_type, N_NODES, graph_params, target_dist, TARGET_RATIO,
                    move_prob, attack_ratio, tau, sim_seed
                ))

    print(f"  Heuristic tasks: {len(h_tasks)}  workers: {n_workers}")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in tqdm(
                ex.map(_run_heuristic_task_v2, h_tasks),
                desc=f"  {graph_type}-{target_dist} heuristic",
                total=len(h_tasks)):
            if res['pc'] is not None:
                results[res['method']][res['tau']]['pc'].append(res['pc'])
                results[res['method']][res['tau']]['anc'].append(res['anc'])
                results[res['method']][res['tau']]['success'].append(
                    float(res['success']))

    if use_rl and rl_tasks:
        print(f"  GNN-RL tasks:   {len(rl_tasks)}  workers: {n_workers_rl}")
        with ProcessPoolExecutor(
                max_workers=n_workers_rl,
                initializer=_init_rl_worker,
                initargs=(model_path,)
        ) as ex:
            for res in tqdm(
                    ex.map(_run_rl_task_v2, rl_tasks),
                    desc=f"  {graph_type}-{target_dist} GNN-RL",
                    total=len(rl_tasks)):
                if res['pc'] is not None:
                    results['GNN-RL'][res['tau']]['pc'].append(res['pc'])
                    results['GNN-RL'][res['tau']]['anc'].append(res['anc'])
                    results['GNN-RL'][res['tau']]['success'].append(
                        float(res['success']))

    return results


def _run_heuristic_task_v2(args):
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     method, move_prob, attack_ratio, tau, sim_seed) = args

    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    targets = (generate_random_targets(G, target_ratio, sim_seed)
               if target_dist == 'random'
               else generate_localized_targets(G, target_ratio, sim_seed))

    attack_per_step = max(1, int(attack_ratio * len(targets)))

    if method == 'Adaptive TD':
        func, batch = td_method_adaptive, True
    elif method == 'Adaptive TIA':
        func, batch = extended_tia_method_adaptive, True
    else:
        func, batch = katz_method_adaptive, False

    pc, anc, success = simulate_biased_heuristic(
        G, targets, move_prob, attack_per_step, func, tau, sim_seed, batch)
    return {'pc': pc, 'anc': anc, 'success': success, 'method': method, 'tau': tau}


def _run_rl_task_v2(args):
    global _worker_rl_agent
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     move_prob, attack_ratio, tau, sim_seed) = args

    if _worker_rl_agent is None:
        return {'pc': None, 'anc': None, 'success': False, 'tau': tau}

    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    targets = (generate_random_targets(G, target_ratio, sim_seed)
               if target_dist == 'random'
               else generate_localized_targets(G, target_ratio, sim_seed))

    attack_per_step = max(1, int(attack_ratio * len(targets)))

    pc, anc, success = simulate_biased_rl(
        G, targets, _worker_rl_agent, move_prob, attack_per_step, tau, sim_seed)
    return {'pc': pc, 'anc': anc, 'success': success, 'tau': tau}



def save_results_to_csv(all_results: Dict, tau_values: List[float],
                        save_dir: str) -> str:
    rows = []
    for config, res in all_results.items():
        for method, tau_dict in res.items():
            for tau in sorted(tau_values):
                pc_vals  = tau_dict[tau]['pc']
                anc_vals = tau_dict[tau]['anc']
                suc_vals = tau_dict[tau]['success']
                if not pc_vals:
                    continue
                n = len(pc_vals)
                rows.append({
                    'config':       config,
                    'method':       method,
                    'tau':          tau,
                    'n_runs':       n,
                    'pc_mean':      float(np.mean(pc_vals)),
                    'pc_std':       float(np.std(pc_vals)),
                    'pc_sem':       float(np.std(pc_vals) / np.sqrt(n)),
                    'anc_mean':     float(np.mean(anc_vals)),
                    'anc_std':      float(np.std(anc_vals)),
                    'anc_sem':      float(np.std(anc_vals) / np.sqrt(n)),
                    'success_rate': float(np.mean(suc_vals)),
                })
    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'robustness_n{N_NODES}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}  ({df.shape[0]} rows)")
    return csv_path



def plot_robustness(all_results: Dict, tau_values: List[float],
                    save_dir: Optional[str] = None, show_figure: bool = False):
    col_titles  = ['BA-Random', 'BA-Localized', 'WS-Random', 'WS-Localized']
    config_keys = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
    methods     = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']

    FIG_W = 7.087
    PAN_H = 1.55
    LEG_H = 0.46
    TOP_H = 0.05
    FIG_H = 2 * PAN_H + LEG_H + TOP_H

    fig, axes = plt.subplots(
        2, 4, figsize=(FIG_W, FIG_H),
        gridspec_kw={
            'hspace':  0.52, 'wspace':  0.40,
            'left':    0.085, 'right': 0.995,
            'top':     1.0 - TOP_H / FIG_H,
            'bottom':  LEG_H / FIG_H,
        },
    )

    panel_idx = 0
    for row_i, (metric, ylabel) in enumerate([('pc', r'$P_C$'), ('anc', 'ANC')]):
        for col_i, config in enumerate(config_keys):
            ax  = axes[row_i][col_i]
            res = all_results.get(config, {})

            for method in methods:
                if method not in res:
                    continue
                taus, means, errs = [], [], []
                for tau in sorted(tau_values):
                    vals = res[method][tau][metric]
                    if vals:
                        taus.append(tau)
                        means.append(np.mean(vals))
                        errs.append(np.std(vals) / np.sqrt(len(vals)))

                if not taus:
                    continue
                taus_a  = np.array(taus)
                means_a = np.array(means)
                errs_a  = np.array(errs)
                c  = METHOD_COLORS.get(method, '#333')
                mk = METHOD_MARKERS.get(method, 'o')
                ax.plot(taus_a, means_a, color=c, marker=mk,
                        markersize=2.5, linewidth=0.8,
                        markeredgecolor='white', markeredgewidth=0.3, zorder=3)
                ax.fill_between(taus_a, means_a - errs_a, means_a + errs_a,
                                color=c, alpha=0.12, linewidth=0, zorder=2)

            ax.set_xscale('log')
            ax.set_xticks([0.1, 1, 10, 100])
            ax.get_xaxis().set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.set_xticks([], minor=True)
            ax.set_xlim(min(tau_values) * 0.8, max(tau_values) * 1.25)
            ax.tick_params(which='both', direction='in',
                           top=True, right=True, labelsize=6, pad=2)
            ax.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=4, min_n_ticks=3))
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

            ax.set_xlabel(r'$\tau$', fontsize=7, labelpad=2)
            ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=7, pad=4)

            ax.text(-0.02, 1.10,
                    f'({PANEL_LABELS[panel_idx]})',
                    transform=ax.transAxes,
                    fontsize=7, fontweight='bold',
                    ha='left', va='bottom')
            panel_idx += 1

    handles = [
        plt.Line2D([], [], color=METHOD_COLORS.get(m, '#333'),
                   marker=METHOD_MARKERS.get(m, 'o'), markersize=3.5,
                   linewidth=0.8, markeredgecolor='white',
                   markeredgewidth=0.3,
                   label=m.replace('Adaptive ', ''))
        for m in methods
    ]
    legend_y = (LEG_H * 0.42) / FIG_H
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.54, legend_y), frameon=False,
               fontsize=6, handlelength=2.0, handletextpad=0.45,
               columnspacing=1.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stem = os.path.join(save_dir, f'robustness_n{N_NODES}')
        fig.savefig(stem + '.pdf', dpi=300)
        fig.savefig(stem + '.png', dpi=300)
        print(f"  Figure saved: {stem}.pdf / .png")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)



if __name__ == '__main__':
    print("Robustness Test: Degree-Biased Movement vs Trained GNN-RL")
    print("=" * 60)
    print(f"  N_nodes        : {N_NODES}")
    print(f"  Target ratio   : {TARGET_RATIO}")
    print(f"  move_prob  (p) : {MOVE_PROB}")
    print(f"  attack_ratio (r): {ATTACK_RATIO}")
    print(f"  τ sweep        : {TAU_VALUES}")
    print(f"  Sim times      : {SIMULATION_TIMES}")
    print(f"  GNN-RL         : {'Enabled' if (MODEL_PATH and os.path.exists(MODEL_PATH) and GNN_AVAILABLE) else 'Disabled'}")
    print(f"  Save dir       : {SAVE_DIR}")
    print("=" * 60)

    all_results: Dict = {}

    for graph_type, target_dist in CONFIGS:
        config_key = f"{graph_type}-{target_dist}"
        print(f"\n{'=' * 60}")
        print(f"Config: {config_key}")
        print(f"{'=' * 60}")

        all_results[config_key] = run_robustness_experiment(
            graph_type=graph_type,
            target_dist=target_dist,
            move_prob=MOVE_PROB,
            attack_ratio=ATTACK_RATIO,
            tau_values=TAU_VALUES,
            sim_times=SIMULATION_TIMES,
            graph_params=GRAPH_PARAMS,
            model_path=MODEL_PATH,
            seed=SEED,
            n_workers=N_WORKERS,
            n_workers_rl=N_WORKERS_RL,
        )

    save_results_to_csv(all_results, TAU_VALUES, SAVE_DIR)
    plot_robustness(all_results, TAU_VALUES, SAVE_DIR, show_figure=False)
    print("\nDone.")