import datetime
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable, Set
from tqdm import tqdm
from collections import defaultdict, deque
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings
import math

warnings.filterwarnings('ignore')

_FIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_FIG_DIR, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))

from comparison_method import (
    td_method_adaptive,
    katz_method_adaptive,
    extended_tia_method_adaptive,
    RLAgentWrapper,
    GNN_AVAILABLE,
    compute_params_from_lambda,
)

import torch
from torch_geometric.data import Data, Batch

_worker_graph = None
_worker_targets = None
_worker_rl_agent = None


def load_social_network(file_path: str) -> Optional[nx.Graph]:
    print(f"Loading social network from: {os.path.basename(file_path)}")

    try:
        G = nx.read_graphml(file_path)

        if G.is_directed():
            G = G.to_undirected()

        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            G.remove_edges_from(self_loops)
            print(f"   Removed {len(self_loops)} self-loops")

        isolated = list(nx.isolates(G))
        if isolated:
            G.remove_nodes_from(isolated)
            print(f"   Removed {len(isolated)} isolated nodes")

        print(f"   Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        return G

    except Exception as e:
        print(f"   Error loading file: {e}")
        return None


def build_socialbot_network(
        data_file: str,
        target_ratio: float = 0.05,
        seed: int = 42
) -> Tuple[Optional[nx.Graph], Optional[List[int]]]:
    print("=" * 60)
    print("Building Social Network for Socialbot Infiltration")
    print("=" * 60)

    print("\n1. Loading network data...")
    if not os.path.exists(data_file):
        print(f"   Error: File not found: {data_file}")
        return None, None

    G = load_social_network(data_file)
    if G is None:
        return None, None

    print("\n2. Extracting largest connected component...")
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"   Found {len(components)} components")
        largest_cc = max(components, key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"   LCC size: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print(f"   Graph is already connected ({G.number_of_nodes()} nodes)")

    print("\n3. Relabeling nodes...")
    G = nx.convert_node_labels_to_integers(G)
    print(f"   Nodes relabeled to 0-{G.number_of_nodes() - 1}")

    n_total = G.number_of_nodes()
    n_targets = max(1, int(n_total * target_ratio))
    print(f"\n4. Target count: {n_targets} ({target_ratio:.1%} of {n_total} total nodes)")

    print(f"\n5. Computing clustering coefficients...")
    clustering = nx.clustering(G)

    sorted_by_cc = sorted(clustering.items(), key=lambda x: x[1], reverse=True)

    candidates_with_cc = [(n, cc) for n, cc in sorted_by_cc if cc > 0]
    print(f"   Nodes with CC > 0: {len(candidates_with_cc)} ({len(candidates_with_cc) / n_total:.1%})")

    if len(candidates_with_cc) < n_targets:
        print(f"   Warning: Not enough high-CC nodes, using top nodes by degree as supplement")
        degrees = dict(G.degree())
        remaining = n_targets - len(candidates_with_cc)
        existing_nodes = set(n for n, _ in candidates_with_cc)
        degree_sorted = sorted(
            [(n, d) for n, d in degrees.items() if n not in existing_nodes],
            key=lambda x: x[1], reverse=True
        )
        candidates_with_cc.extend([(n, 0) for n, d in degree_sorted[:remaining]])

    print(f"\n6. Selecting initial bot entry points from high clustering coefficient nodes...")
    rng = np.random.RandomState(seed)

    pool_size = min(len(candidates_with_cc), max(n_targets * 4, int(n_total * 0.2)))
    candidate_pool = [n for n, cc in candidates_with_cc[:pool_size]]

    target_nodes = rng.choice(candidate_pool, size=min(n_targets, len(candidate_pool)),
                              replace=False).tolist()

    avg_target_cc = np.mean([clustering[n] for n in target_nodes])
    avg_target_degree = np.mean([G.degree(n) for n in target_nodes])
    print(f"   Selected {len(target_nodes)} initial socialbots")
    print(f"   Average clustering coefficient: {avg_target_cc:.4f}")
    print(f"   Average degree of entry nodes: {avg_target_degree:.2f}")

    all_cc = list(clustering.values())
    degrees = dict(G.degree())
    all_degrees = list(degrees.values())

    print(f"\n{'=' * 60}")
    print(f"Social Network Summary (soc-advogato):")
    print(f"  Nodes (accounts): {G.number_of_nodes()}")
    print(f"  Edges (trust links): {G.number_of_edges()}")
    print(f"  Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"  Degree range: {min(all_degrees)} - {max(all_degrees)}")
    print(f"  Avg clustering coefficient: {np.mean(all_cc):.4f}")
    print(f"  Transitivity: {nx.transitivity(G):.4f}")
    print(f"  Initial socialbots: {len(target_nodes)} ({len(target_nodes) / n_total:.2%})")
    print(f"{'=' * 60}")

    return G, target_nodes


class SocialbotSimulator:

    def __init__(self, G: nx.Graph, target_nodes: List[int], move_prob: float,
                 attack_per_step: int, seed: Optional[int] = None):
        self.original_graph = G.copy()
        self.current_graph = G.copy()
        self.target_nodes = set(target_nodes)
        self.removed_nodes = set()
        self.visited_nodes = set(target_nodes)
        self.move_prob = move_prob
        self.attack_per_step = attack_per_step
        self.n_non_targets = len([n for n in G.nodes() if n not in target_nodes])
        self.initial_lcc_size = self._get_lcc_size()
        self.cumulative_anc = 0.0
        self.anc_count = 0
        self.attacks_in_current_round = 0
        self.rng = np.random.RandomState(seed)

        self.original_total_nodes = G.number_of_nodes()

    def _get_lcc_size(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        components = list(nx.connected_components(self.current_graph))
        if not components:
            return 0
        return len(max(components, key=len))

    def _count_targets_in_lcc(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        components = list(nx.connected_components(self.current_graph))
        if not components:
            return 0
        lcc = max(components, key=len)
        return sum(1 for t in self.target_nodes if t in lcc)

    def _execute_bot_movement(self):
        if self.move_prob <= 0:
            return

        current_bots = list(self.target_nodes)
        self.rng.shuffle(current_bots)

        new_bot_positions = set()
        claimed_positions = set()

        for bot_node in current_bots:
            if bot_node not in self.current_graph:
                continue

            neighbors = list(self.current_graph.neighbors(bot_node))
            valid_neighbors = []

            for nb in neighbors:
                if (nb not in self.removed_nodes and
                        nb not in self.target_nodes and
                        nb not in claimed_positions):
                    valid_neighbors.append(nb)

            if valid_neighbors and self.rng.random() < self.move_prob:
                new_pos = self.rng.choice(valid_neighbors)

                claimed_positions.add(new_pos)
                new_bot_positions.add(new_pos)

                self.visited_nodes.add(new_pos)

            else:
                new_bot_positions.add(bot_node)
                claimed_positions.add(bot_node)

        self.target_nodes = new_bot_positions

    def is_done(self) -> bool:
        return self._count_targets_in_lcc() == 0

    def remove_single_node(self, node: int) -> bool:
        if node in self.target_nodes or node in self.removed_nodes or node not in self.current_graph:
            return self.is_done()

        self.removed_nodes.add(node)
        self.current_graph.remove_node(node)
        self.attacks_in_current_round += 1

        current_lcc_ratio = self._get_lcc_size() / max(1, self.initial_lcc_size)
        self.cumulative_anc += current_lcc_ratio
        self.anc_count += 1

        return self.is_done()

    def remove_nodes_batch(self, nodes_to_remove: List[int]) -> bool:
        for node in nodes_to_remove:
            if node in self.target_nodes or node in self.removed_nodes or node not in self.current_graph:
                continue

            self.removed_nodes.add(node)
            self.current_graph.remove_node(node)
            self.attacks_in_current_round += 1

            current_lcc_ratio = self._get_lcc_size() / max(1, self.initial_lcc_size)
            self.cumulative_anc += current_lcc_ratio
            self.anc_count += 1

            if self.is_done():
                return True

        return self.is_done()

    def execute_spread_if_round_complete(self):
        if self.attacks_in_current_round >= self.attack_per_step:
            if not self.is_done():
                self._execute_bot_movement()
            self.attacks_in_current_round = 0

    def get_metrics(self) -> Tuple[float, float, float]:
        pc = len(self.removed_nodes) / max(1, self.n_non_targets)
        anc = self.cumulative_anc / max(1, self.anc_count) if self.anc_count > 0 else 1.0
        infiltration_ratio = len(self.visited_nodes) / max(1, self.original_total_nodes)
        return pc, anc, infiltration_ratio


def simulate_socialbot_heuristic(G: nx.Graph, target_nodes: List[int], move_prob: float,
                                 attack_per_step: int, method_func: Callable,
                                 seed: Optional[int] = None,
                                 batch_mode: bool = False) -> Tuple[float, float, float, bool]:
    simulator = SocialbotSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in target_nodes])

    if batch_mode:
        while not simulator.is_done():
            current_quota = simulator.attack_per_step - simulator.attacks_in_current_round
            if current_quota <= 0:
                simulator.execute_spread_if_round_complete()
                current_quota = simulator.attack_per_step

            attack_ranking = method_func(simulator.current_graph, list(simulator.target_nodes))

            valid_batch = []
            for node in attack_ranking:
                if (node not in simulator.removed_nodes and
                        node not in simulator.target_nodes and
                        node in simulator.current_graph):
                    valid_batch.append(node)
                    if len(valid_batch) >= current_quota:
                        break

            if not valid_batch:
                break

            if simulator.remove_nodes_batch(valid_batch):
                break

            simulator.execute_spread_if_round_complete()
    else:
        for _ in range(max_steps):
            if simulator.is_done():
                break

            attack_order = method_func(simulator.current_graph, list(simulator.target_nodes))
            attack_order = [n for n in attack_order
                            if n not in simulator.removed_nodes
                            and n not in simulator.target_nodes
                            and n in simulator.current_graph]

            if not attack_order:
                break

            if simulator.remove_single_node(attack_order[0]):
                break

            simulator.execute_spread_if_round_complete()

    pc, anc, infiltration_ratio = simulator.get_metrics()
    return pc, anc, infiltration_ratio, simulator.is_done()


def compute_distances_to_targets(current_graph: nx.Graph, target_nodes: set) -> Dict[int, float]:
    distances = {node: float('inf') for node in current_graph.nodes()}
    queue = deque()

    for target in target_nodes:
        if target in current_graph:
            distances[target] = 0
            queue.append(target)

    while queue:
        current = queue.popleft()
        current_dist = distances[current]

        for neighbor in current_graph.neighbors(current):
            if distances[neighbor] > current_dist + 1:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

    return distances


def create_socialbot_pyg_data(G: nx.Graph, target_nodes: set, removed_nodes: set,
                              move_prob: float, attacks_in_round: int,
                              attack_per_step: int) -> Tuple[Data, List[int]]:
    all_invalid = removed_nodes
    original_nodes = sorted(set(G.nodes()) | all_invalid)
    n_nodes = len(original_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(original_nodes)}
    idx_to_node = {idx: node for idx, node in enumerate(original_nodes)}

    current_graph = G.copy()
    for node in all_invalid:
        if node in current_graph:
            current_graph.remove_node(node)

    distances = compute_distances_to_targets(current_graph, target_nodes)

    if current_graph.number_of_nodes() > 0:
        degrees = dict(current_graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
    else:
        max_degree = 1
        degrees = {}
    log_max_degree = math.log(max_degree + 1)

    edges = []
    for u, v in current_graph.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append([node_to_idx[u], node_to_idx[v]])
            edges.append([node_to_idx[v], node_to_idx[u]])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    local_features = np.zeros((n_nodes, 3), dtype=np.float32)

    for idx in range(n_nodes):
        node_id = idx_to_node[idx]

        local_features[idx, 0] = 1.0 if node_id in target_nodes else 0.0

        if node_id in all_invalid or node_id not in current_graph:
            local_features[idx, 1] = 0.0
            local_features[idx, 2] = 0.0
        else:
            dist = distances.get(node_id, float('inf'))
            local_features[idx, 1] = 1.0 / (dist + 1.0)

            degree = degrees.get(node_id, 0)
            log_degree = math.log(degree + 1)
            local_features[idx, 2] = log_degree / max(log_max_degree, 1e-6)

    action_mask = np.array([
        1.0 if (node not in target_nodes and
                node not in all_invalid and
                node in current_graph) else 0.0
        for node in original_nodes
    ], dtype=np.float32)

    urgency = attacks_in_round / max(1, attack_per_step)
    global_features = np.array([[move_prob, urgency]], dtype=np.float32)

    data = Data(
        x=torch.tensor(local_features, dtype=torch.float32),
        global_x=torch.tensor(global_features, dtype=torch.float32),
        edge_index=edge_index,
        action_mask=torch.tensor(action_mask, dtype=torch.float32),
        num_nodes=n_nodes
    )

    return data, original_nodes


def simulate_socialbot_rl(G: nx.Graph, target_nodes: List[int], rl_agent,
                          move_prob: float, attack_per_step: int,
                          seed: Optional[int] = None) -> Tuple[float, float, float, bool]:
    simulator = SocialbotSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in target_nodes])

    for _ in range(max_steps):
        if simulator.is_done():
            break

        data, nodes = create_socialbot_pyg_data(
            simulator.original_graph,
            simulator.target_nodes,
            simulator.removed_nodes,
            move_prob,
            simulator.attacks_in_current_round,
            attack_per_step
        )

        action_idx = rl_agent.get_action_deterministic(data)
        node = nodes[action_idx]

        if simulator.remove_single_node(node):
            break

        simulator.execute_spread_if_round_complete()

    pc, anc, infiltration_ratio = simulator.get_metrics()
    return pc, anc, infiltration_ratio, simulator.is_done()


def _init_heuristic_worker(graph_nodes, graph_edges, target_nodes):
    global _worker_graph, _worker_targets

    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)

    _worker_graph = G
    _worker_targets = target_nodes


def _init_rl_worker(model_path, graph_nodes, graph_edges, target_nodes):
    global _worker_rl_agent, _worker_graph, _worker_targets

    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path) and GNN_AVAILABLE:
        try:
            _worker_rl_agent = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            _worker_rl_agent = None
            print(f"Init RL Model Error: {e}")
    else:
        _worker_rl_agent = None

    G = nx.Graph()
    G.add_nodes_from(graph_nodes)
    G.add_edges_from(graph_edges)

    _worker_graph = G
    _worker_targets = target_nodes


def _run_td_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args
    G = _worker_graph.copy()
    targets = list(_worker_targets)

    pc, anc, infiltration, success = simulate_socialbot_heuristic(
        G, targets, move_prob, attack_per_step,
        td_method_adaptive, sim_seed, batch_mode=True
    )
    return {'pc': pc, 'anc': anc, 'infiltration': infiltration,
            'method': 'TD', 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def _run_tia_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args
    G = _worker_graph.copy()
    targets = list(_worker_targets)

    pc, anc, infiltration, success = simulate_socialbot_heuristic(
        G, targets, move_prob, attack_per_step,
        extended_tia_method_adaptive, sim_seed, batch_mode=True
    )
    return {'pc': pc, 'anc': anc, 'infiltration': infiltration,
            'method': 'TIA', 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def _run_katz_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args
    G = _worker_graph.copy()
    targets = list(_worker_targets)

    pc, anc, infiltration, success = simulate_socialbot_heuristic(
        G, targets, move_prob, attack_per_step,
        katz_method_adaptive, sim_seed, batch_mode=False
    )
    return {'pc': pc, 'anc': anc, 'infiltration': infiltration,
            'method': 'Katz', 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def _run_rl_task(args):
    move_prob, attack_per_step, sim_seed, lambda_eff = args

    if _worker_rl_agent is None:
        return {'pc': None, 'anc': None, 'infiltration': None, 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}

    G = _worker_graph.copy()
    targets = list(_worker_targets)

    pc, anc, infiltration, success = simulate_socialbot_rl(
        G, targets, _worker_rl_agent, move_prob, attack_per_step, sim_seed
    )
    return {'pc': pc, 'anc': anc, 'infiltration': infiltration, 'lambda_eff': lambda_eff, 'sim_seed': sim_seed}


def run_socialbot_experiment(
        G: nx.Graph,
        target_nodes: List[int],
        lambda_values: List[float],
        simulation_times: int = 100,
        model_path: Optional[str] = None,
        save_dir: Optional[str] = None,
        show_figure: bool = False,
        seed: int = 42,
        n_workers: int = None,
        n_workers_rl: int = None
) -> Dict:
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    print(f"\n{'=' * 70}")
    print(f" Socialbot Infiltration: Full 4-Method Comparison")
    print(f"{'=' * 70}")
    print(f"Network: N={G.number_of_nodes()}, E={G.number_of_edges()}")
    print(f"Model: Recoverable Random Walk (no stifler, no permanent marking)")
    print(f"Initial socialbots: {len(target_nodes)}")
    print(f"Initial ratio: {len(target_nodes) / G.number_of_nodes():.2%}")
    print(f"Lambda values: {lambda_values}")
    print(f"Simulations per lambda: {simulation_times}")

    use_rl = model_path and os.path.exists(model_path) and GNN_AVAILABLE

    all_methods = ['TD', 'TIA', 'Katz']
    if use_rl:
        all_methods.append('GNN-RL')

    print(f"\nMethods ({len(all_methods)} total):")
    print(f"  1. TD   - Batch Adaptive (fast)")
    print(f"  2. TIA  - Batch Adaptive (isolation-optimized)")
    print(f"  3. Katz - Fully Adaptive (best quality)")
    if use_rl:
        print(f"  4. GNN-RL - Graph Neural Network + RL")
        print(f"     Model: {os.path.basename(model_path)}")
    print(f"{'=' * 70}")

    results = {m: defaultdict(lambda: {'pc': [], 'anc': [], 'infiltration': [], 'success': [], 'seeds': []}) for m in all_methods}

    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    n_targets = len(target_nodes)

    rng = np.random.RandomState(seed)

    td_tasks, tia_tasks, katz_tasks, rl_tasks = [], [], [], []

    print(f"\nPreparing tasks...")
    for lambda_eff in lambda_values:
        move_prob, attack_ratio = compute_params_from_lambda(lambda_eff)
        attack_per_step = max(1, int(attack_ratio * n_targets))
        print(f"  λ={lambda_eff} → move_prob={move_prob:.3f}, attack_per_step={attack_per_step}")

        for sim_idx in range(simulation_times):
            sim_seed = int(rng.randint(0, 1e9))
            task_params = (move_prob, attack_per_step, sim_seed, lambda_eff)
            td_tasks.append(task_params)
            tia_tasks.append(task_params)
            katz_tasks.append(task_params)
            if use_rl:
                rl_tasks.append(task_params)

    print(f"Total tasks: {len(td_tasks)} per method × {len(all_methods)} methods\n")

    print(f"[1/{'4' if use_rl else '3'}] Running {len(td_tasks)} TD tasks...")
    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_heuristic_worker,
            initargs=(static_nodes, static_edges, target_nodes)
    ) as executor:
        for result in tqdm(executor.map(_run_td_task, td_tasks),
                           desc="TD", total=len(td_tasks)):
            results['TD'][result['lambda_eff']]['pc'].append(result['pc'])
            results['TD'][result['lambda_eff']]['anc'].append(result['anc'])
            results['TD'][result['lambda_eff']]['infiltration'].append(result['infiltration'])
            results['TD'][result['lambda_eff']]['success'].append(float(result.get('success', False)))
            results['TD'][result['lambda_eff']]['seeds'].append(result.get('sim_seed'))

    print(f"[2/{'4' if use_rl else '3'}] Running {len(tia_tasks)} TIA tasks...")
    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_heuristic_worker,
            initargs=(static_nodes, static_edges, target_nodes)
    ) as executor:
        for result in tqdm(executor.map(_run_tia_task, tia_tasks),
                           desc="TIA", total=len(tia_tasks)):
            results['TIA'][result['lambda_eff']]['pc'].append(result['pc'])
            results['TIA'][result['lambda_eff']]['anc'].append(result['anc'])
            results['TIA'][result['lambda_eff']]['infiltration'].append(result['infiltration'])
            results['TIA'][result['lambda_eff']]['success'].append(float(result.get('success', False)))
            results['TIA'][result['lambda_eff']]['seeds'].append(result.get('sim_seed'))

    print(f"[3/{'4' if use_rl else '3'}] Running {len(katz_tasks)} Katz tasks...")
    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_heuristic_worker,
            initargs=(static_nodes, static_edges, target_nodes)
    ) as executor:
        for result in tqdm(executor.map(_run_katz_task, katz_tasks),
                           desc="Katz", total=len(katz_tasks)):
            results['Katz'][result['lambda_eff']]['pc'].append(result['pc'])
            results['Katz'][result['lambda_eff']]['anc'].append(result['anc'])
            results['Katz'][result['lambda_eff']]['infiltration'].append(result['infiltration'])
            results['Katz'][result['lambda_eff']]['success'].append(float(result.get('success', False)))
            results['Katz'][result['lambda_eff']]['seeds'].append(result.get('sim_seed'))

    if use_rl and rl_tasks:
        print(f"[4/4] Running {len(rl_tasks)} GNN-RL tasks...")
        with ProcessPoolExecutor(
                max_workers=n_workers_rl,
                initializer=_init_rl_worker,
                initargs=(model_path, static_nodes, static_edges, target_nodes)
        ) as executor:
            for result in tqdm(executor.map(_run_rl_task, rl_tasks),
                               desc="GNN-RL", total=len(rl_tasks)):
                if result['pc'] is not None:
                    results['GNN-RL'][result['lambda_eff']]['pc'].append(result['pc'])
                    results['GNN-RL'][result['lambda_eff']]['anc'].append(result['anc'])
                    results['GNN-RL'][result['lambda_eff']]['infiltration'].append(result['infiltration'])
                    results['GNN-RL'][result['lambda_eff']]['success'].append(float(result.get('success', False)))
                    results['GNN-RL'][result['lambda_eff']]['seeds'].append(result.get('sim_seed'))

    print_results_summary(results, lambda_values)

    if save_dir:
        save_results_to_csv(results, lambda_values,
                            G.number_of_nodes(), n_targets, save_dir)

    plot_results(results, lambda_values, G.number_of_nodes(), len(target_nodes),
                 save_dir, show_figure)

    return results


def save_network_with_targets(
        G:            nx.Graph,
        target_nodes: List[int],
        save_dir:     str,
        graphml_name: str = 'socialbot_network.graphml',
        txt_name:     str = 'socialbot_target_nodes.txt',
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    target_set = set(target_nodes)

    G_out = G.copy()
    for nd in G_out.nodes():
        G_out.nodes[nd]['is_target'] = int(nd in target_set)

    graphml_path = os.path.join(save_dir, graphml_name)
    nx.write_graphml(G_out, graphml_path)
    print(f'  Network saved : {graphml_path}')

    txt_path = os.path.join(save_dir, txt_name)
    ratio = len(target_nodes) / max(1, G.number_of_nodes())
    with open(txt_path, 'w') as fh:
        fh.write('# Socialbot infiltration (soc-advogato) -- initial bot positions\n')
        fh.write(
            f'# total={len(target_nodes)}, '
            f'network_nodes={G.number_of_nodes()}, '
            f'ratio={ratio:.4f}\n'
        )
        for nd in target_nodes:
            fh.write(f'{nd}\n')
    print(f'  Targets saved : {txt_path}  ({len(target_nodes)} nodes)')


def save_results_to_csv(
        results:       Dict,
        lambda_values: List[float],
        n_nodes:       int,
        n_targets:     int,
        save_dir:      str,
) -> str:
    rows = []
    for method, lam_dict in results.items():
        for lam in sorted(lambda_values):
            d         = lam_dict.get(lam, {})
            pc_vals   = d.get('pc',            [])
            anc_vals  = d.get('anc',           [])
            inf_vals  = d.get('infiltration',  [])
            suc_vals  = d.get('success',       [])
            if not pc_vals:
                continue
            n = len(pc_vals)

            def _s(vals):
                a = np.array(vals, dtype=float)
                return float(np.mean(a)), float(np.std(a)), float(np.std(a) / np.sqrt(n))

            pm, ps, pe  = _s(pc_vals)
            am, as_, ae = _s(anc_vals)
            im, is_, ie = _s(inf_vals) if inf_vals else (0., 0., 0.)

            rows.append({
                'method':               method,
                'lambda_eff':           lam,
                'n_runs':               n,
                'pc_mean':              pm,  'pc_std':            ps,  'pc_sem':            pe,
                'anc_mean':             am,  'anc_std':           as_, 'anc_sem':           ae,
                'infiltration_mean':    im,  'infiltration_std':  is_, 'infiltration_sem':  ie,
                'success_rate':         float(np.mean(suc_vals)) if suc_vals else 0.0,
            })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(save_dir, f'socialbot_results_{ts}.csv')
    df.to_csv(csv_path, index=False)
    print(f'  CSV saved     : {csv_path}  ({df.shape[0]} rows x {df.shape[1]} cols)')
    return csv_path


def print_results_summary(results: Dict, lambda_values: List[float]):
    print(f"\n{'=' * 100}")
    print(f"Results Summary")
    print(f"{'=' * 100}")

    for lambda_eff in lambda_values:
        print(f"\nλ = {lambda_eff}")
        print(f"{'-' * 90}")
        print(f"{'Method':<15} {'PC (mean±std)':<22} {'ANC (mean±std)':<22} {'Infiltration (mean±std)':<22}")
        print(f"{'-' * 90}")

        for method in results:
            if results[method][lambda_eff]['pc']:
                pc_mean = np.mean(results[method][lambda_eff]['pc'])
                pc_std = np.std(results[method][lambda_eff]['pc'])
                anc_mean = np.mean(results[method][lambda_eff]['anc'])
                anc_std = np.std(results[method][lambda_eff]['anc'])
                inf_mean = np.mean(results[method][lambda_eff]['infiltration'])
                inf_std = np.std(results[method][lambda_eff]['infiltration'])
                print(
                    f"{method:<15} {pc_mean:.4f}±{pc_std:.4f}          {anc_mean:.4f}±{anc_std:.4f}          {inf_mean:.4f}±{inf_std:.4f}")

    print(f"\n{'=' * 100}")


def plot_results(
        results: Dict,
        lambda_values: List[float],
        n_nodes: int,
        n_initial_bots: int,
        save_dir: Optional[str] = None,
        show_figure: bool = True
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'TD': '#3498db', 'TIA': '#2ecc71', 'Katz': '#e74c3c', 'GNN-RL': '#9b59b6'}
    markers = {'TD': 'o', 'TIA': 's', 'Katz': '^', 'GNN-RL': 'D'}
    methods = list(results.keys())

    ax = axes[0]
    for method in methods:
        lambdas, means, sems = [], [], []
        for lam in lambda_values:
            vals = results[method][lam]['pc']
            if vals:
                lambdas.append(lam)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
        if lambdas:
            ax.plot(lambdas, means, color=colors.get(method, '#000'),
                    marker=markers.get(method, 'o'), label=method, linewidth=2, markersize=8)
            ax.fill_between(lambdas, np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            alpha=0.15, color=colors.get(method, '#000'))
    ax.set_xlabel('$\\lambda_{eff}$', fontsize=12)
    ax.set_ylabel('PC (Ban Cost)', fontsize=12)
    ax.set_title('Protection Cost (PC)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax = axes[1]
    for method in methods:
        lambdas, means, sems = [], [], []
        for lam in lambda_values:
            vals = results[method][lam]['anc']
            if vals:
                lambdas.append(lam)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
        if lambdas:
            ax.plot(lambdas, means, color=colors.get(method, '#000'),
                    marker=markers.get(method, 'o'), label=method, linewidth=2, markersize=8)
            ax.fill_between(lambdas, np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            alpha=0.15, color=colors.get(method, '#000'))
    ax.set_xlabel('$\\lambda_{eff}$', fontsize=12)
    ax.set_ylabel('ANC (Social Network Preserved)', fontsize=12)
    ax.set_title('Average Normalized Connectivity (ANC)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax = axes[2]
    for method in methods:
        lambdas, means, sems = [], [], []
        for lam in lambda_values:
            vals = results[method][lam]['infiltration']
            if vals:
                lambdas.append(lam)
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
        if lambdas:
            ax.plot(lambdas, means, color=colors.get(method, '#000'),
                    marker=markers.get(method, 'o'), label=method, linewidth=2, markersize=8)
            ax.fill_between(lambdas, np.array(means) - np.array(sems),
                            np.array(means) + np.array(sems),
                            alpha=0.15, color=colors.get(method, '#000'))
    initial_ratio = n_initial_bots / n_nodes
    ax.axhline(y=initial_ratio, color='gray', linestyle='--', linewidth=1.5,
               label=f'Initial bots ({initial_ratio:.2%})')
    ax.set_xlabel('$\\lambda_{eff}$', fontsize=12)
    ax.set_ylabel('Infiltration Ratio', fontsize=12)
    ax.set_title('Infiltration Footprint (Unique Accounts Visited)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    target_ratio = n_initial_bots / n_nodes
    n_methods = len(methods)
    fig.suptitle(
        f'Socialbot Infiltration: {n_methods}-Method Comparison | '
        f'N={n_nodes}, Bots={n_initial_bots} ({target_ratio:.1%}), Recoverable Walk',
        fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'socialbot_infiltration_results_{timestamp}.png'
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig


class SocialbotSnapshotSimulator(SocialbotSimulator):
    def __init__(self, G, target_nodes, move_prob, attack_per_step, seed=None):
        super().__init__(G, target_nodes, move_prob, attack_per_step, seed)
        self.removal_order: List[int] = []

    def remove_single_node(self, node: int) -> bool:
        done = super().remove_single_node(node)
        if node in self.removed_nodes and node not in self.removal_order:
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


def _save_node_list(nodes: List[int], path: str, header: str = '') -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        for nd in nodes:
            f.write(f"{nd}\n")


def _run_snapshot_socialbot_heuristic(G, target_nodes, move_prob, attack_per_step,
                                       method_func, seed, batch_mode=False):
    sim = SocialbotSnapshotSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in set(target_nodes)])

    if batch_mode:
        while not sim.is_done():
            quota = sim.attack_per_step - sim.attacks_in_current_round
            if quota <= 0:
                sim.execute_spread_if_round_complete()
                quota = sim.attack_per_step
            ranking = method_func(sim.current_graph, list(sim.target_nodes))
            valid_batch = [n for n in ranking
                           if n not in sim.removed_nodes
                           and n not in sim.target_nodes
                           and n in sim.current_graph][:quota]
            if not valid_batch:
                break
            if sim.remove_nodes_batch(valid_batch):
                break
            sim.execute_spread_if_round_complete()
    else:
        for _ in range(max_steps):
            if sim.is_done():
                break
            order = method_func(sim.current_graph, list(sim.target_nodes))
            order = [n for n in order
                     if n not in sim.removed_nodes
                     and n not in sim.target_nodes
                     and n in sim.current_graph]
            if not order:
                break
            if sim.remove_single_node(order[0]):
                break
            sim.execute_spread_if_round_complete()

    pc, anc, infiltration = sim.get_metrics()
    return pc, anc, infiltration, sim.is_done(), sim.get_removed_nodes(), sim.get_final_targets()


def _run_snapshot_socialbot_rl(G, target_nodes, rl_agent, move_prob,
                                attack_per_step, seed):
    sim = SocialbotSnapshotSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in set(target_nodes)])

    for _ in range(max_steps):
        if sim.is_done():
            break
        data, nodes = create_socialbot_pyg_data(
            sim.original_graph, sim.target_nodes, sim.removed_nodes,
            move_prob, sim.attacks_in_current_round, attack_per_step)
        action_idx = rl_agent.get_action_deterministic(data)
        node = nodes[action_idx]
        if sim.remove_single_node(node):
            break
        sim.execute_spread_if_round_complete()

    pc, anc, infiltration = sim.get_metrics()
    return pc, anc, infiltration, sim.is_done(), sim.get_removed_nodes(), sim.get_final_targets()


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
          f"move_prob={move_prob:.3f}, attack_per_step={attack_per_step}")

    tia_data  = results.get('TIA', {}).get(lambda_eff, {})
    rl_data   = results.get('GNN-RL', {}).get(lambda_eff, {})
    tia_pcs   = tia_data.get('pc',    [])
    tia_seeds = tia_data.get('seeds', [])

    if not tia_pcs or not tia_seeds:
        print(f"  No TIA results at λ={lambda_eff}. Snapshot skipped.")
        return

    if use_rl and rl_data.get('pc') and rl_data.get('seeds'):
        rl_pcs   = rl_data['pc']
        rl_seeds = rl_data['seeds']
        seed_to_rl = {s: pc for s, pc in zip(rl_seeds, rl_pcs) if s is not None}
        advantages = []
        for i, s in enumerate(tia_seeds):
            if s is not None and s in seed_to_rl:
                advantages.append((seed_to_rl[s] - tia_pcs[i], s, i))
        if not advantages:
            print("  Could not align GNN-RL and TIA seeds. Falling back to TIA.")
            use_rl = False
        else:
            advantages.sort(reverse=True)
            best_adv, best_seed, best_idx = advantages[0]
            print(f"  Best seed={best_seed}, "
                  f"advantage(GNN-RL−TIA)={best_adv:.4f}  "
                  f"(GNN-RL pc={seed_to_rl[best_seed]:.4f}, "
                  f"TIA pc={tia_pcs[best_idx]:.4f})")
            try:
                rl_agent = RLAgentWrapper(model_path, device='cpu')
            except Exception as e:
                print(f"  RL agent load failed: {e}")
                use_rl = False

    if not use_rl:
        best_idx  = int(np.argmax(tia_pcs))
        best_seed = tia_seeds[best_idx]
        print(f"  No GNN-RL; best TIA seed={best_seed}, "
              f"pc_tia={tia_pcs[best_idx]:.4f}")

    methods_cfg = [
        ('TD',   lambda g, t, p, k, s: _run_snapshot_socialbot_heuristic(
            g, t, p, k, td_method_adaptive, s, batch_mode=True)),
        ('TIA',  lambda g, t, p, k, s: _run_snapshot_socialbot_heuristic(
            g, t, p, k, extended_tia_method_adaptive, s, batch_mode=True)),
        ('Katz', lambda g, t, p, k, s: _run_snapshot_socialbot_heuristic(
            g, t, p, k, katz_method_adaptive, s, batch_mode=False)),
    ]
    if use_rl and rl_agent is not None:
        methods_cfg.append((
            'GNN_RL',
            lambda g, t, p, k, s: _run_snapshot_socialbot_rl(g, t, rl_agent, p, k, s)
        ))

    os.makedirs(save_dir, exist_ok=True)
    for method_name, run_fn in methods_cfg:
        pc, anc, infiltration, success, removed, final_tgts = run_fn(
            G.copy(), list(target_nodes), move_prob, attack_per_step, best_seed)
        print(f"  {method_name:<8} pc={pc:.4f}  anc={anc:.4f}  "
              f"removed={len(removed)}  final_targets={len(final_tgts)}")
        _save_node_list(
            removed,
            os.path.join(save_dir, f'removed_{method_name}.txt'),
            header=f'removed nodes — method={method_name} λ={lambda_eff} seed={best_seed}')
        _save_node_list(
            final_tgts,
            os.path.join(save_dir, f'final_targets_{method_name}.txt'),
            header=f'final target nodes — method={method_name} λ={lambda_eff} seed={best_seed}')

    print(f"  Snapshot files saved to: {save_dir}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    DATA_FILE = os.path.join(SCRIPT_DIR, "data", "fig6_data", "socialbot_network.graphml")

    MODEL_PATH = os.path.join(REPO_ROOT, "train",
                              "gnn_ppo_dual_stream_n50-100__seed42__1769653775",
                              "model.pt")

    SAVE_DIR = os.path.join(SCRIPT_DIR, "data", "fig6_result", "socialbot")

    TARGET_RATIO = 0.05

    SNAPSHOT_LAMBDA = 20

    LAMBDA_VALUES = [1] + list(range(2, 22, 2))
    SIMULATION_TIMES = 10

    N_WORKERS = 55
    N_WORKERS_RL = 55

    print("Building Social Network for Socialbot Infiltration...")
    G, target_nodes = build_socialbot_network(
        data_file=DATA_FILE,
        target_ratio=TARGET_RATIO,
        seed=42
    )

    if G is None or target_nodes is None:
        print("Failed to build network. Please check data files.")
        sys.exit(1)

    print(f"\nNetwork loaded successfully:")
    print(f"  - Accounts: {G.number_of_nodes()}")
    print(f"  - Trust links: {G.number_of_edges()}")
    print(f"  - Initial socialbots: {len(target_nodes)} ({len(target_nodes) / G.number_of_nodes():.2%})")

    save_network_with_targets(
        G=G,
        target_nodes=target_nodes,
        save_dir=SAVE_DIR,
        graphml_name='socialbot_network.graphml',
        txt_name='socialbot_target_nodes.txt',
    )

    results = run_socialbot_experiment(
        G=G,
        target_nodes=target_nodes,
        lambda_values=LAMBDA_VALUES,
        simulation_times=SIMULATION_TIMES,
        model_path=MODEL_PATH,
        save_dir=SAVE_DIR,
        show_figure=False,
        seed=42,
        n_workers=N_WORKERS,
        n_workers_rl=N_WORKERS_RL
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

    print("\n Socialbot Infiltration Experiment Completed!")