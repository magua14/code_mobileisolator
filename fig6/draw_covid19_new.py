import datetime
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
from collections import defaultdict, deque
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings
import torch

warnings.filterwarnings('ignore')

_FIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_FIG_DIR, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))

from comparison_method import (
    td_method_adaptive,
    katz_method_adaptive,
    extended_tia_method_adaptive,
    simulate_heuristic,
    simulate_rl,
    RLAgentWrapper,
    GNN_AVAILABLE,
    compute_params_from_lambda,
    MovingTargetSimulator,
    IncrementalPyGBuilder,
)

print("✓ Successfully imported from comparison_method")

_worker_graph    = None
_worker_targets  = None
_worker_rl_agent = None


class FootprintSimulator(MovingTargetSimulator):
    def __init__(self, G, target_nodes, move_prob, attack_per_step, seed=None):
        super().__init__(G, target_nodes, move_prob, attack_per_step, seed)
        self.colonized_nodes      = set(target_nodes)
        self.original_total_nodes = G.number_of_nodes()

    def _execute_target_movement(self):
        if self.move_prob <= 0:
            return
        movements = []
        claimed   = set()
        for target in [t for t in self.target_nodes if t in self.current_graph]:
            if self.rng.random() > self.move_prob:
                continue
            valid_neighbors = [
                n for n in self.current_graph.neighbors(target)
                if n not in self.removed_nodes
                and n not in self.target_nodes
                and n not in claimed
            ]
            if valid_neighbors:
                new_pos = self.rng.choice(valid_neighbors)
                movements.append((target, new_pos))
                claimed.add(new_pos)

        for old_pos, new_pos in movements:
            self.target_nodes.remove(old_pos)
            self.target_nodes.add(new_pos)
            self.colonized_nodes.add(old_pos)
            self.colonized_nodes.add(new_pos)

    def get_footprint(self):
        return len(self.colonized_nodes) / max(1, self.original_total_nodes)


class SnapshotSimulator(FootprintSimulator):
    def __init__(self, G, target_nodes, move_prob, attack_per_step, seed=None):
        super().__init__(G, target_nodes, move_prob, attack_per_step, seed)
        self.removal_order: List[int] = []

    def remove_single_node(self, node: int) -> bool:
        done = super().remove_single_node(node)
        if node in self.removed_nodes:
            self.removal_order.append(node)
        return done

    def remove_nodes_batch(self, nodes: List[int]) -> bool:
        done = super().remove_nodes_batch(nodes)
        for nd in nodes:
            if nd in self.removed_nodes and nd not in self.removal_order:
                self.removal_order.append(nd)
        return done

    def get_final_targets(self) -> List[int]:
        return list(self.target_nodes)

    def get_removed_nodes(self) -> List[int]:
        return list(self.removal_order)


def _run_snapshot_heuristic(G, target_nodes, move_prob, attack_per_step,
                             method_func, seed, batch_mode=False):
    sim = SnapshotSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in set(target_nodes)])

    if batch_mode:
        while not sim.is_done():
            quota = sim.attack_per_step - sim.attacks_in_current_round
            if quota <= 0:
                sim.execute_movement_if_round_complete()
                quota = sim.attack_per_step
            ranking    = method_func(sim.current_graph, list(sim.target_nodes))
            valid_batch = []
            for nd in ranking:
                if nd not in sim.removed_nodes and nd not in sim.target_nodes:
                    valid_batch.append(nd)
                    if len(valid_batch) >= quota:
                        break
            if not valid_batch:
                break
            if sim.remove_nodes_batch(valid_batch):
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
    return (pc, anc, sim.get_footprint(), sim.is_done(),
            sim.get_removed_nodes(), sim.get_final_targets())


def _run_snapshot_rl(G, target_nodes, rl_agent, move_prob, attack_per_step, seed):
    sim     = SnapshotSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    builder = IncrementalPyGBuilder(sim.original_graph, sim.target_nodes)
    max_steps = len([n for n in G.nodes() if n not in set(target_nodes)])

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
    return (pc, anc, sim.get_footprint(), sim.is_done(),
            sim.get_removed_nodes(), sim.get_final_targets())


def save_terminal_snapshot(
        G:            nx.Graph,
        target_nodes: List[int],
        lambda_eff:   float,
        results:      Dict,
        model_path:   Optional[str],
        save_dir:     str,
) -> None:
    rl_agent = None
    use_rl = bool(model_path and os.path.exists(model_path) and GNN_AVAILABLE)
    move_prob, attack_ratio = compute_params_from_lambda(lambda_eff)
    attack_per_step         = max(1, int(attack_ratio * len(target_nodes)))

    print(f"\n  Snapshot phase: λ={lambda_eff}, "
          f"p={move_prob:.3f}, k={attack_per_step}")

    tia_data = results.get('TIA', {}).get(lambda_eff, {})
    rl_data  = results.get('GNN-RL', {}).get(lambda_eff, {})

    tia_pcs   = tia_data.get('pc',    [])
    tia_seeds = tia_data.get('seeds', [])

    if not tia_pcs or not tia_seeds:
        print(f"  No TIA results at λ={lambda_eff}. Snapshot skipped.")
        return

    if use_rl and rl_data.get('pc') and rl_data.get('seeds'):
        rl_pcs   = rl_data['pc']
        rl_seeds = rl_data['seeds']
        seed_to_rl  = {s: pc for s, pc in zip(rl_seeds, rl_pcs) if s is not None}
        advantages  = []
        for i, s in enumerate(tia_seeds):
            if s is not None and s in seed_to_rl:
                advantages.append((seed_to_rl[s] - tia_pcs[i], s, i))
        if not advantages:
            print("  Could not align GNN-RL and TIA seeds. Falling back to TIA.")
            use_rl = False
        else:
            advantages.sort(reverse=True)
            best_adv, best_seed, best_idx = advantages[0]
            print(f"  Best seed idx={best_idx}, seed={best_seed}, "
                  f"advantage(GNN-RL−TIA)={best_adv:.4f}  "
                  f"(GNN-RL pc={seed_to_rl[best_seed]:.4f}, "
                  f"TIA pc={tia_pcs[best_idx]:.4f})")
            try:
                rl_agent = RLAgentWrapper(model_path, device='cpu')
                print("  RL agent loaded for snapshot.")
            except Exception as e:
                print(f"  RL agent load failed: {e}")
                use_rl = False

    if not use_rl:
        best_idx  = int(np.argmax(tia_pcs))
        best_seed = tia_seeds[best_idx]
        print(f"  No GNN-RL; best TIA seed idx={best_idx}, "
              f"seed={best_seed}, pc_tia={tia_pcs[best_idx]:.4f}")

    methods_cfg = [
        ('TD',  lambda g, t, p, k, s: _run_snapshot_heuristic(
            g, t, p, k, td_method_adaptive, s, batch_mode=True)),
        ('TIA', lambda g, t, p, k, s: _run_snapshot_heuristic(
            g, t, p, k, extended_tia_method_adaptive, s, batch_mode=True)),
        ('Katz', lambda g, t, p, k, s: _run_snapshot_heuristic(
            g, t, p, k, katz_method_adaptive, s, batch_mode=False)),
    ]
    if use_rl:
        methods_cfg.append((
            'GNN_RL',
            lambda g, t, p, k, s: _run_snapshot_rl(g, t, rl_agent, p, k, s)
        ))

    os.makedirs(save_dir, exist_ok=True)
    for method_name, run_fn in methods_cfg:
        pc, anc, fp, success, removed, final_tgts = run_fn(
            G.copy(), list(target_nodes), move_prob, attack_per_step, best_seed)
        print(f"  {method_name:<8} pc={pc:.4f}  anc={anc:.4f}  "
              f"removed={len(removed)}  final_targets={len(final_tgts)}")

        _save_node_list(
            removed, os.path.join(save_dir, f'removed_{method_name}.txt'),
            header=f'removed nodes — method={method_name} λ={lambda_eff} seed={best_seed}')
        _save_node_list(
            final_tgts, os.path.join(save_dir, f'final_targets_{method_name}.txt'),
            header=f'final target nodes — method={method_name} λ={lambda_eff} seed={best_seed}')

    print(f"  Snapshot files saved to: {save_dir}")


def simulate_heuristic_fp(G, target_nodes, move_prob, attack_per_step,
                           method_func, seed=None, batch_mode=False):
    sim       = FootprintSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in set(target_nodes)])

    if batch_mode:
        while not sim.is_done():
            quota = sim.attack_per_step - sim.attacks_in_current_round
            if quota <= 0:
                sim.execute_movement_if_round_complete()
                quota = sim.attack_per_step
            ranking     = method_func(sim.current_graph, list(sim.target_nodes))
            valid_batch = []
            for nd in ranking:
                if nd not in sim.removed_nodes and nd not in sim.target_nodes:
                    valid_batch.append(nd)
                    if len(valid_batch) >= quota:
                        break
            if not valid_batch:
                break
            if sim.remove_nodes_batch(valid_batch):
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
    return pc, anc, sim.get_footprint(), sim.is_done()


def simulate_rl_fp(G, target_nodes, rl_agent, move_prob, attack_per_step, seed=None):
    sim       = FootprintSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in set(target_nodes)])
    builder   = IncrementalPyGBuilder(sim.original_graph, sim.target_nodes)

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
    return pc, anc, sim.get_footprint(), sim.is_done()


def _init_heuristic_worker(graph_nodes, graph_edges, target_nodes):
    global _worker_graph, _worker_targets
    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)
    _worker_graph   = G
    _worker_targets = target_nodes


def _init_rl_worker(model_path, graph_nodes, graph_edges, target_nodes):
    global _worker_rl_agent, _worker_graph, _worker_targets
    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path) and GNN_AVAILABLE:
        try:
            _worker_rl_agent = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            _worker_rl_agent = None
    else:
        _worker_rl_agent = None
    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)
    _worker_graph   = G
    _worker_targets = target_nodes


def _run_td_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args
    pc, anc, fp, success = simulate_heuristic_fp(
        _worker_graph.copy(), list(_worker_targets),
        move_prob, attack_per_step, td_method_adaptive, sim_seed, batch_mode=True)
    return {'pc': pc, 'anc': anc, 'fp': fp, 'success': success,
            'method': 'TD', 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def _run_tia_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args
    pc, anc, fp, success = simulate_heuristic_fp(
        _worker_graph.copy(), list(_worker_targets),
        move_prob, attack_per_step, extended_tia_method_adaptive, sim_seed, batch_mode=True)
    return {'pc': pc, 'anc': anc, 'fp': fp, 'success': success,
            'method': 'TIA', 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def _run_katz_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args
    pc, anc, fp, success = simulate_heuristic_fp(
        _worker_graph.copy(), list(_worker_targets),
        move_prob, attack_per_step, katz_method_adaptive, sim_seed, batch_mode=False)
    return {'pc': pc, 'anc': anc, 'fp': fp, 'success': success,
            'method': 'Katz', 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def _run_rl_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args
    if _worker_rl_agent is None:
        return {'pc': None, 'anc': None, 'fp': None,
                'success': False, 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}
    pc, anc, fp, success = simulate_rl_fp(
        _worker_graph.copy(), list(_worker_targets),
        _worker_rl_agent, move_prob, attack_per_step, sim_seed)
    return {'pc': pc, 'anc': anc, 'fp': fp, 'success': success,
            'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def run_covid_experiment(
        G, target_nodes, lambda_values, simulation_times=100,
        model_path=None, save_dir=None, show_figure=False,
        seed=42, n_workers=None, n_workers_rl=None,
) -> Dict:
    if n_workers is None:
        n_workers    = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    use_rl      = bool(model_path and os.path.exists(model_path) and GNN_AVAILABLE)
    all_methods = ['TD', 'TIA', 'Katz'] + (['GNN-RL'] if use_rl else [])

    print(f"\n{'=' * 70}")
    print(f"COVID-19 Network: Full {len(all_methods)}-Method Comparison")
    print(f"{'=' * 70}")
    print(f"  Network  : N={G.number_of_nodes()}, E={G.number_of_edges()}")
    print(f"  Targets  : {len(target_nodes)} ({len(target_nodes)/G.number_of_nodes():.2%})")
    print(f"  λ values : {lambda_values}")
    print(f"  Sims/λ   : {simulation_times}")
    print(f"{'=' * 70}")

    results = {m: defaultdict(lambda: {'pc': [], 'anc': [], 'fp': [],
                                        'success': [], 'seeds': []})
               for m in all_methods}

    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    n_targets    = len(target_nodes)
    rng          = np.random.RandomState(seed)

    td_tasks, tia_tasks, katz_tasks, rl_tasks = [], [], [], []
    for lambda_eff in lambda_values:
        move_prob, attack_ratio = compute_params_from_lambda(lambda_eff)
        attack_per_step         = max(1, int(attack_ratio * n_targets))
        for _ in range(simulation_times):
            sim_seed = int(rng.randint(0, int(1e9)))
            task     = (move_prob, attack_per_step, sim_seed, lambda_eff)
            td_tasks.append(task); tia_tasks.append(task)
            katz_tasks.append(task)
            if use_rl: rl_tasks.append(task)

    def _collect(results_dict, method_key, result_iter, total):
        for r in tqdm(result_iter, desc=method_key, total=total):
            lam = r['lambda_eff']
            results_dict[method_key][lam]['pc'].append(r['pc'])
            results_dict[method_key][lam]['anc'].append(r['anc'])
            results_dict[method_key][lam]['fp'].append(r['fp'])
            results_dict[method_key][lam]['success'].append(float(r['success']))
            results_dict[method_key][lam]['seeds'].append(r.get('sim_seed'))

    print(f"\n[1/{len(all_methods)}] TD...")
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_heuristic_worker,
                             initargs=(static_nodes, static_edges, target_nodes)) as ex:
        _collect(results, 'TD', ex.map(_run_td_task, td_tasks), len(td_tasks))

    print(f"[2/{len(all_methods)}] TIA...")
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_heuristic_worker,
                             initargs=(static_nodes, static_edges, target_nodes)) as ex:
        _collect(results, 'TIA', ex.map(_run_tia_task, tia_tasks), len(tia_tasks))

    print(f"[3/{len(all_methods)}] Katz...")
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_heuristic_worker,
                             initargs=(static_nodes, static_edges, target_nodes)) as ex:
        _collect(results, 'Katz', ex.map(_run_katz_task, katz_tasks), len(katz_tasks))

    if use_rl and rl_tasks:
        print(f"[4/4] GNN-RL...")
        with ProcessPoolExecutor(max_workers=n_workers_rl,
                                 initializer=_init_rl_worker,
                                 initargs=(model_path, static_nodes,
                                           static_edges, target_nodes)) as ex:
            for r in tqdm(ex.map(_run_rl_task, rl_tasks),
                          desc="GNN-RL", total=len(rl_tasks)):
                if r['pc'] is not None:
                    lam = r['lambda_eff']
                    results['GNN-RL'][lam]['pc'].append(r['pc'])
                    results['GNN-RL'][lam]['anc'].append(r['anc'])
                    results['GNN-RL'][lam]['fp'].append(r['fp'])
                    results['GNN-RL'][lam]['success'].append(float(r['success']))
                    results['GNN-RL'][lam]['seeds'].append(r.get('sim_seed'))

    print_results_summary(results, lambda_values)
    if save_dir:
        save_results_to_csv(results, lambda_values, G.number_of_nodes(),
                            n_targets, save_dir)
    plot_results(results, lambda_values, G.number_of_nodes(), n_targets,
                 save_dir, show_figure)
    return results


def save_results_to_csv(results, lambda_values, n_nodes, n_targets, save_dir):
    rows = []
    for method, lam_dict in results.items():
        for lam in sorted(lambda_values):
            d = lam_dict.get(lam, {})
            pc_vals  = d.get('pc',      [])
            anc_vals = d.get('anc',     [])
            fp_vals  = d.get('fp',      [])
            suc_vals = d.get('success', [])
            if not pc_vals: continue
            n = len(pc_vals)
            def _s(vals):
                a = np.array(vals, dtype=float)
                return float(np.mean(a)), float(np.std(a)), float(np.std(a)/np.sqrt(n))
            pm, ps, pe   = _s(pc_vals)
            am, as_, ae  = _s(anc_vals)
            fm, fs, fe   = _s(fp_vals) if fp_vals else (0., 0., 0.)
            rows.append({
                'method': method, 'lambda_eff': lam, 'n_runs': n,
                'pc_mean': pm,  'pc_std': ps,  'pc_sem': pe,
                'anc_mean': am, 'anc_std': as_, 'anc_sem': ae,
                'invasion_mean': fm, 'invasion_std': fs, 'invasion_sem': fe,
                'success_rate': float(np.mean(suc_vals)),
            })
    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f'covid19_results_{ts}.csv')
    df.to_csv(path, index=False)
    print(f"  CSV saved: {path}")
    return path


def print_results_summary(results, lambda_values):
    print(f"\n{'=' * 90}\nResults Summary\n{'=' * 90}")
    for lam in lambda_values:
        print(f"\nλ = {lam}")
        for method in results:
            d = results[method][lam]
            if d['pc']:
                print(f"  {method:<8} pc={np.mean(d['pc']):.4f}  "
                      f"anc={np.mean(d['anc']):.4f}  "
                      f"fp={np.mean(d['fp']):.4f}  "
                      f"success={np.mean(d['success']):.2%}")


def plot_results(results, lambda_values, n_nodes, n_targets,
                 save_dir=None, show_figure=True):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors  = {'TD': '#0072B2', 'TIA': '#009E73',
               'Katz': '#D55E00', 'GNN-RL': '#CC79A7'}
    markers = {'TD': 'o', 'TIA': '^', 'Katz': 's', 'GNN-RL': 'D'}
    panels  = [(axes[0], 'pc', r'$P_C$'), (axes[1], 'anc', 'ANC'),
               (axes[2], 'fp', 'Footprint')]
    for ax, metric, ylabel in panels:
        for method in results:
            xs, means, errs = [], [], []
            for lam in lambda_values:
                vals = results[method][lam].get(metric, [])
                if vals:
                    xs.append(lam); means.append(np.mean(vals))
                    errs.append(np.std(vals)/np.sqrt(len(vals)))
            if xs:
                m = np.array(means); e = np.array(errs)
                ax.plot(xs, m, color=colors.get(method, '#000'),
                        marker=markers.get(method, 'o'), label=method,
                        linewidth=2, markersize=7)
                ax.fill_between(xs, m-e, m+e, alpha=0.15,
                                color=colors.get(method, '#000'))
        ax.set_xlabel(r'$\lambda_{\rm eff}$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f'covid19_results_{ts}.png')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.show() if show_figure else plt.close(fig)
    return fig


def _save_node_list(nodes: List[int], path: str, header: str = '') -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        for nd in nodes:
            f.write(f"{nd}\n")


def load_network(graphml_path):
    """Load the released COVID-19 network (topology only) from GraphML."""
    G = nx.read_graphml(graphml_path)
    G = nx.relabel_nodes(G, int)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(f"  Loaded network: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def generate_cluster_targets(
        G:            nx.Graph,
        target_ratio: float,
        n_clusters:   int,
        k_hops:       int  = 5,
        seed:         int  = 42,
) -> List[int]:
    rng       = np.random.RandomState(seed)
    all_nodes = list(G.nodes())
    n_total   = len(all_nodes)
    n_targets = max(n_clusters, round(n_total * target_ratio))

    base_quota = n_targets // n_clusters
    remainder  = n_targets - base_quota * n_clusters
    quotas     = [base_quota + (1 if i < remainder else 0)
                  for i in range(n_clusters)]

    selected = []
    claimed  = set()

    for cluster_idx, quota in enumerate(quotas):
        unclaimed = [n for n in all_nodes if n not in claimed]
        if not unclaimed:
            break
        seed_node = int(rng.choice(unclaimed))
        claimed.add(seed_node)

        pool    = [seed_node]
        visited = {seed_node}
        frontier = [seed_node]
        for hop in range(k_hops):
            if not frontier:
                break
            next_frontier = []
            for cur in frontier:
                for nb in G.neighbors(cur):
                    if nb not in visited and nb not in claimed:
                        visited.add(nb)
                        pool.append(nb)
                        next_frontier.append(nb)
            frontier = next_frontier

        sample_size = min(quota, len(pool))
        chosen_idx  = rng.choice(len(pool), size=sample_size, replace=False)
        chosen      = [pool[i] for i in chosen_idx]
        for nd in chosen:
            claimed.add(nd)
        selected.extend(chosen)

    print(f"  K-hop cluster targets: {len(selected)} nodes "
          f"across {n_clusters} clusters (K={k_hops})")
    return selected



if __name__ == "__main__":
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

    INPUT_GRAPHML = os.path.join(SCRIPT_DIR, 'data', 'fig6_data',
                                 'covid19_network.graphml')

    MODEL_PATH = os.path.join(REPO_ROOT, 'train',
                              'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                              'model.pt')
    SAVE_DIR   = os.path.join(SCRIPT_DIR, 'data', 'fig6_result', 'covid19')

    SNAPSHOT_LAMBDA      = 20

    TARGET_RATIO = 0.05
    N_CLUSTERS   = 4
    K_HOPS       = 5
    CLUSTER_SEED = 42

    LAMBDA_VALUES    = [1] + list(range(2, 22, 2))
    SIMULATION_TIMES = 10
    N_WORKERS        = 55
    N_WORKERS_RL     = 55

    print("Loading COVID-19 community network (topology only)...")
    G = load_network(INPUT_GRAPHML)
    if G is None:
        print("Failed to load network."); sys.exit(1)

    print("Generating k-hop cluster targets...")
    target_nodes = generate_cluster_targets(
        G, TARGET_RATIO, N_CLUSTERS, K_HOPS, CLUSTER_SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)

    results = run_covid_experiment(
        G=G, target_nodes=target_nodes,
        lambda_values=LAMBDA_VALUES,
        simulation_times=SIMULATION_TIMES,
        model_path=MODEL_PATH,
        save_dir=SAVE_DIR,
        show_figure=False,
        seed=42,
        n_workers=N_WORKERS,
        n_workers_rl=N_WORKERS_RL,
    )

    print(f"\nSaving terminal snapshot at λ={SNAPSHOT_LAMBDA} ...")
    save_terminal_snapshot(
        G=G,
        target_nodes=target_nodes,
        lambda_eff=SNAPSHOT_LAMBDA,
        results=results,
        model_path=MODEL_PATH,
        save_dir=SAVE_DIR,
    )

    print("\nExperiment completed!")