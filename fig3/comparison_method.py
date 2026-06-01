import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from typing import Tuple, List, Dict, Optional, Callable
from tqdm import tqdm
from collections import defaultdict, deque
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import warnings
import torch
from torch_geometric.data import Data, Batch

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_FIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_FIG_DIR, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))

GNN_AVAILABLE = False

try:
    from ppo_gnn_moving_gat_dualstream_final import DualStreamAgent as GNNAgent
    GNN_AVAILABLE = True
    print("GNN-RL Agent loaded successfully")
except ImportError as e:
    print(f"GNN-RL Agent not found: {e}")

LOCAL_DIM = 3
GLOBAL_DIM = 2

_rl_agent = None


def compute_params_from_lambda(lambda_eff: float) -> Tuple[float, float]:
    if lambda_eff == 0:
        return 0.0, 1.0
    p = 1.0
    r = 1.0 / lambda_eff
    return p, r


def generate_random_targets(G: nx.Graph, target_ratio: float,
                            seed: Optional[int] = None) -> List[int]:
    if seed is not None:
        random.seed(seed)
    n = G.number_of_nodes()
    target_num = max(1, int(n * target_ratio))
    return random.sample(list(G.nodes()), target_num)


def generate_localized_targets(
        G: nx.Graph, target_ratio: float,
        seed: Optional[int] = None,
        radius: float = 0.05, max_radius: float = 1.0, step: float = 0.05
) -> List[int]:
    rng = np.random.RandomState(seed)
    pos = nx.spring_layout(G, seed=seed)
    n_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    n_targets = max(1, int(n_nodes * target_ratio))
    center_node = rng.choice(nodes)
    center_x, center_y = pos[center_node]
    current_radius = radius
    localized_candidate = []
    while current_radius < max_radius:
        localized_candidate = [
            node for node, (x, y) in pos.items()
            if (x - center_x) ** 2 + (y - center_y) ** 2 < current_radius ** 2
        ]
        if len(localized_candidate) >= n_targets:
            break
        current_radius += step
    if len(localized_candidate) >= n_targets:
        indices = rng.choice(len(localized_candidate), size=n_targets, replace=False)
        return [localized_candidate[i] for i in indices]
    else:
        remaining = n_targets - len(localized_candidate)
        other_nodes = list(set(nodes) - set(localized_candidate))
        if remaining > 0 and other_nodes:
            extra_indices = rng.choice(len(other_nodes),
                                       size=min(remaining, len(other_nodes)),
                                       replace=False)
            extra = [other_nodes[i] for i in extra_indices]
            return localized_candidate + extra
        return localized_candidate


def td_method_adaptive(G: nx.Graph, target_nodes: List[int]) -> List[int]:
    target_set = set(target_nodes)
    td_value = {}
    for node in G.nodes():
        if node not in target_set:
            td_value[node] = sum(1 for nb in G.neighbors(node) if nb in target_set)
    return [n for n, _ in sorted(td_value.items(), key=lambda x: x[1], reverse=True)]


def katz_method_adaptive(G: nx.Graph, target_nodes: List[int],
                         max_iter: int = 50, tol: float = 1e-6) -> List[int]:
    target_set = set(target_nodes)
    n = G.number_of_nodes()
    if n == 0:
        return []
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    A = nx.to_scipy_sparse_array(G, nodelist=node_list, format='csr', dtype=float)
    f = np.zeros(n)
    for node in target_nodes:
        if node in node_to_idx:
            f[node_to_idx[node]] = 1.0
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    for _ in range(20):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-10:
            break
        v = v_new / norm
    lambda_max = max(1e-10, np.abs(v @ (A @ v)))
    epsilon = 0.5 / lambda_max
    x = f.copy()
    for _ in range(max_iter):
        x_new = epsilon * (A @ x) + f
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    KatzT = x / epsilon - f / epsilon
    sorted_indices = np.argsort(-KatzT)
    return [node_list[idx] for idx in sorted_indices if node_list[idx] not in target_set]


def tia_core_adaptive(G, target_nodes):
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


def extended_tia_method_adaptive(G: nx.Graph, target_nodes: List[int]) -> List[int]:
    original_target_set = set(target_nodes)
    T, K, R = tia_core_adaptive(G, target_nodes)
    return list(K) + list(T - original_target_set) + list(R)


class MovingTargetSimulator:
    def __init__(self, G: nx.Graph, target_nodes: List[int], move_prob: float,
                 attack_per_step: int, seed: Optional[int] = None):
        self.original_graph = G.copy()
        self.current_graph = G.copy()
        self.target_nodes = set(target_nodes)
        self.removed_nodes = set()
        self.move_prob = move_prob
        self.attack_per_step = attack_per_step
        self.n_non_targets = len([n for n in G.nodes() if n not in target_nodes])
        self.initial_lcc_size = self._get_lcc_size()
        self.cumulative_anc = 0.0
        self.anc_count = 0
        self.attacks_in_current_round = 0
        self.rng = np.random.RandomState(seed)

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

    def _execute_target_movement(self):
        if self.move_prob <= 0:
            return
        movements = []
        claimed_positions = set()
        for target in [t for t in self.target_nodes if t in self.current_graph]:
            if self.rng.random() > self.move_prob:
                continue
            valid_neighbors = [
                n for n in self.current_graph.neighbors(target)
                if n not in self.removed_nodes
                and n not in self.target_nodes
                and n not in claimed_positions
            ]
            if valid_neighbors:
                new_pos = self.rng.choice(valid_neighbors)
                movements.append((target, new_pos))
                claimed_positions.add(new_pos)
        for old_pos, new_pos in movements:
            self.target_nodes.remove(old_pos)
            self.target_nodes.add(new_pos)

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

    def execute_movement_if_round_complete(self):
        if self.attacks_in_current_round >= self.attack_per_step:
            if not self.is_done():
                self._execute_target_movement()
            self.attacks_in_current_round = 0

    def get_metrics(self) -> Tuple[float, float]:
        pc = len(self.removed_nodes) / max(1, self.n_non_targets)
        anc = self.cumulative_anc / max(1, self.anc_count) if self.anc_count > 0 else 1.0
        return pc, anc


def simulate_heuristic(G, target_nodes, move_prob, attack_per_step,
                       method_func, seed=None, batch_mode=False):
    simulator = MovingTargetSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in target_nodes])

    if batch_mode:
        while not simulator.is_done():
            current_quota = simulator.attack_per_step - simulator.attacks_in_current_round
            if current_quota <= 0:
                simulator.execute_movement_if_round_complete()
                current_quota = simulator.attack_per_step
            attack_ranking = method_func(simulator.current_graph, list(simulator.target_nodes))
            valid_batch = []
            for node in attack_ranking:
                if node not in simulator.removed_nodes and node not in simulator.target_nodes:
                    valid_batch.append(node)
                    if len(valid_batch) >= current_quota:
                        break
            if not valid_batch:
                break
            if simulator.remove_nodes_batch(valid_batch):
                break
            simulator.execute_movement_if_round_complete()
    else:
        for _ in range(max_steps):
            if simulator.is_done():
                break
            attack_order = method_func(simulator.current_graph, list(simulator.target_nodes))
            attack_order = [n for n in attack_order
                            if n not in simulator.removed_nodes and n not in simulator.target_nodes]
            if not attack_order:
                break
            if simulator.remove_single_node(attack_order[0]):
                break
            simulator.execute_movement_if_round_complete()

    pc, anc = simulator.get_metrics()
    return pc, anc, simulator.is_done()


class IncrementalPyGBuilder:
    def __init__(self, G: nx.Graph, target_nodes: set):
        self.original_nodes = sorted(list(G.nodes()))
        self.n_nodes = len(self.original_nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.original_nodes)}
        edges = []
        for u, v in G.edges():
            if u in self.node_to_idx and v in self.node_to_idx:
                u_idx, v_idx = self.node_to_idx[u], self.node_to_idx[v]
                edges.append([u_idx, v_idx])
                edges.append([v_idx, u_idx])
        if edges:
            self.full_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            self.full_edge_index = torch.zeros((2, 0), dtype=torch.long)
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        self.log_max_degree = math.log(max_deg + 1)
        self.node_valid_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        self._processed_removed_nodes = set()

    def update_removed_nodes(self, removed_nodes: set):
        newly_removed = removed_nodes - self._processed_removed_nodes
        if not newly_removed:
            return
        indices_to_remove = torch.tensor(
            [self.node_to_idx[n] for n in newly_removed if n in self.node_to_idx],
            dtype=torch.long
        )
        if indices_to_remove.numel() > 0:
            self.node_valid_mask[indices_to_remove] = False
        self._processed_removed_nodes.update(newly_removed)

    def compute_distances_to_targets(self, current_graph, target_nodes):
        distances = {node: float('inf') for node in self.original_nodes}
        queue = deque()
        for t in target_nodes:
            if t in current_graph:
                distances[t] = 0
                queue.append(t)
        while queue:
            cur = queue.popleft()
            for nb in current_graph.neighbors(cur):
                if distances[nb] > distances[cur] + 1:
                    distances[nb] = distances[cur] + 1
                    queue.append(nb)
        return distances

    def build_data(self, current_graph, target_nodes, removed_nodes,
                   move_prob, attacks_in_round, attack_per_step):
        self.update_removed_nodes(removed_nodes)
        row_mask = self.node_valid_mask[self.full_edge_index[0]]
        col_mask = self.node_valid_mask[self.full_edge_index[1]]
        edge_mask = row_mask & col_mask
        current_edge_index = self.full_edge_index[:, edge_mask]
        if current_edge_index.numel() > 0:
            current_degrees = torch.bincount(current_edge_index[0], minlength=self.n_nodes).float()
        else:
            current_degrees = torch.zeros(self.n_nodes)
        if current_degrees.numel() > 0:
            current_max_deg = current_degrees.max().item()
        else:
            current_max_deg = 1.0
        current_log_max_degree = math.log(current_max_deg + 1)
        distances = self._bfs_from_edge_index(current_edge_index, target_nodes)
        x = torch.zeros((self.n_nodes, 3), dtype=torch.float32)
        target_indices = [self.node_to_idx[t] for t in target_nodes if t in self.node_to_idx]
        if target_indices:
            x[target_indices, 0] = 1.0
        dists = torch.tensor([distances[node] for node in self.original_nodes], dtype=torch.float32)
        x[:, 1] = 1.0 / (dists + 1.0)
        x[:, 2] = torch.log(current_degrees + 1) / max(current_log_max_degree, 1e-6)
        x[~self.node_valid_mask, 1:] = 0.0
        action_mask = torch.zeros(self.n_nodes, dtype=torch.float32)
        action_mask[self.node_valid_mask] = 1.0
        if target_indices:
            action_mask[target_indices] = 0.0
        urgency = attacks_in_round / max(1, attack_per_step)
        global_x = torch.tensor([[move_prob, urgency]], dtype=torch.float32)
        data = Data(
            x=x,
            edge_index=current_edge_index,
            global_x=global_x,
            action_mask=action_mask,
            num_nodes=self.n_nodes
        )
        return data, self.original_nodes

    def reset(self):
        self.node_valid_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        self._processed_removed_nodes = set()

    def _bfs_from_edge_index(self, current_edge_index: torch.Tensor,
                              target_nodes: set) -> dict:
        dist_arr = np.full(self.n_nodes, np.inf, dtype=np.float64)

        if current_edge_index.numel() == 0:
            for t in target_nodes:
                if t in self.node_to_idx:
                    dist_arr[self.node_to_idx[t]] = 0.0
            return {node: dist_arr[self.node_to_idx[node]]
                    for node in self.original_nodes}

        src = current_edge_index[0].numpy()
        dst = current_edge_index[1].numpy()

        order   = np.argsort(src, kind='stable')
        indices = dst[order]

        indptr  = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(indptr[1:], src, 1)
        np.cumsum(indptr, out=indptr)

        queue = deque()
        for t in target_nodes:
            if t in self.node_to_idx:
                idx = self.node_to_idx[t]
                if self.node_valid_mask[idx]:
                    dist_arr[idx] = 0.0
                    queue.append(idx)

        while queue:
            u = queue.popleft()
            d = dist_arr[u]
            for v in indices[indptr[u]:indptr[u + 1]]:
                if dist_arr[v] == np.inf:
                    dist_arr[v] = d + 1.0
                    queue.append(int(v))

        return {node: dist_arr[self.node_to_idx[node]]
                for node in self.original_nodes}


class RLAgentWrapper:
    def __init__(self, model_path: str, device: str = 'cpu'):
        if not GNN_AVAILABLE:
            raise RuntimeError("GNN Agent not available.")
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        args_dict = checkpoint.get('args', {})
        self.agent = GNNAgent(
            local_dim=args_dict.get('local_dim', LOCAL_DIM),
            global_dim=args_dict.get('global_dim', GLOBAL_DIM),
            hidden_dim=args_dict.get('hidden_dim', 128),
            context_dim=args_dict.get('context_dim', 32),
            num_gnn_layers=args_dict.get('num_gnn_layers', 3),
            gat_heads=args_dict.get('gat_heads', 4)
        ).to(self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.agent.eval()

    def get_action_deterministic(self, data: Data) -> int:
        batch_data = Batch.from_data_list([data]).to(self.device)
        with torch.no_grad():
            logits, _ = self.agent.forward(batch_data)
            logits = logits.squeeze(-1)
            action_mask = batch_data.action_mask
            masked_logits = logits.clone()
            masked_logits[action_mask == 0] = float('-inf')
            action = masked_logits.argmax().item()
        return action


def simulate_rl(G, target_nodes, rl_agent, move_prob, attack_per_step, seed=None):
    simulator = MovingTargetSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = len([n for n in G.nodes() if n not in target_nodes])
    pyg_builder = IncrementalPyGBuilder(simulator.original_graph, simulator.target_nodes)
    for _ in range(max_steps):
        if simulator.is_done():
            break
        data, nodes = pyg_builder.build_data(
            simulator.current_graph, simulator.target_nodes, simulator.removed_nodes,
            move_prob, simulator.attacks_in_current_round, attack_per_step
        )
        action_idx = rl_agent.get_action_deterministic(data)
        node = nodes[action_idx]
        if simulator.remove_single_node(node):
            break
        simulator.execute_movement_if_round_complete()
    pc, anc = simulator.get_metrics()
    return pc, anc, simulator.is_done()


def generate_graph(graph_type: str, n_nodes: int, params: Dict, seed: int) -> nx.Graph:
    rng = np.random.RandomState(seed)
    if graph_type == 'BA':
        m = rng.randint(params['ba_m_range'][0], params['ba_m_range'][1] + 1)
        G = nx.barabasi_albert_graph(n_nodes, min(m, n_nodes - 1), seed=seed)
    else:
        k = rng.randint(params['ws_k_range'][0] // 2, params['ws_k_range'][1] // 2 + 1) * 2
        k = max(2, min(k, n_nodes - 1))
        beta = rng.uniform(*params['ws_beta_range'])
        G = nx.watts_strogatz_graph(n_nodes, k, beta, seed=seed)
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = nx.convert_node_labels_to_integers(G.subgraph(lcc).copy())
    return G


def _run_heuristic_task(args):
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     method, move_prob, attack_ratio, sim_seed, lambda_eff) = args
    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    if target_dist == 'random':
        targets = generate_random_targets(G, target_ratio, sim_seed)
    else:
        targets = generate_localized_targets(G, target_ratio, sim_seed)
    n_targets = len(targets)
    attack_per_step = max(1, int(attack_ratio * n_targets))
    if method == 'Adaptive TD':
        func = td_method_adaptive
        batch_mode = True
    elif method == 'Adaptive TIA':
        func = extended_tia_method_adaptive
        batch_mode = True
    else:
        func = katz_method_adaptive
        batch_mode = False
    pc, anc, success = simulate_heuristic(G, targets, move_prob, attack_per_step,
                                          func, sim_seed, batch_mode)
    return {'pc': pc, 'anc': anc, 'success': success, 'method': method, 'lambda_eff': lambda_eff}


def _init_rl_worker(model_path):
    global _rl_agent
    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path) and GNN_AVAILABLE:
        try:
            _rl_agent = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            _rl_agent = None
            print(f"Init RL Model Error: {e}")
    else:
        _rl_agent = None


def _run_rl_task(args):
    global _rl_agent
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     move_prob, attack_ratio, sim_seed, lambda_eff) = args
    if _rl_agent is None:
        return {'pc': None, 'anc': None, 'success': False, 'lambda_eff': lambda_eff}
    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    if target_dist == 'random':
        targets = generate_random_targets(G, target_ratio, sim_seed)
    else:
        targets = generate_localized_targets(G, target_ratio, sim_seed)
    n_targets = len(targets)
    attack_per_step = max(1, int(attack_ratio * n_targets))
    pc, anc, success = simulate_rl(G, targets, _rl_agent, move_prob, attack_per_step, sim_seed)
    return {'pc': pc, 'anc': anc, 'success': success, 'lambda_eff': lambda_eff}


def run_phase_transition_experiments(
        graph_type, target_dist, n_nodes, target_ratio,
        lambda_eff_values, simulation_times, graph_params,
        model_path, seed=42, n_workers=None, n_workers_rl=None):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    heuristic_methods = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA']
    use_rl = model_path and os.path.exists(model_path) and GNN_AVAILABLE
    all_methods = heuristic_methods.copy()
    if use_rl:
        all_methods.append('GNN-RL')

    results = {m: defaultdict(lambda: {'pc': [], 'anc': [], 'success': []}) for m in all_methods}
    rng = np.random.RandomState(seed)
    heuristic_tasks = []
    rl_tasks = []

    print(f"\n  Generating task parameters for {graph_type}-{target_dist}...")
    print(f"    GNN-RL Model: {'Enabled' if use_rl else 'Disabled'}")
    print(f"    Strategy: TD/TIA=Batch Adaptive, Katz/RL=Fully Adaptive")

    for lambda_eff in lambda_eff_values:
        move_prob, attack_ratio = compute_params_from_lambda(lambda_eff)
        print(f"    λ={lambda_eff:.1f} → p={move_prob:.3f}, r={attack_ratio:.3f}")
        for sim_idx in range(simulation_times):
            sim_seed = int(rng.randint(0, 1e9))
            for method in heuristic_methods:
                heuristic_tasks.append((
                    graph_type, n_nodes, graph_params, target_dist, target_ratio,
                    method, move_prob, attack_ratio, sim_seed, lambda_eff
                ))
            if use_rl:
                rl_tasks.append((
                    graph_type, n_nodes, graph_params, target_dist, target_ratio,
                    move_prob, attack_ratio, sim_seed, lambda_eff
                ))

    print(f"  Running {len(heuristic_tasks)} heuristic tasks with {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(_run_heuristic_task, heuristic_tasks),
                           desc=f"{graph_type}-{target_dist} (Heuristic)",
                           total=len(heuristic_tasks)):
            results[result['method']][result['lambda_eff']]['pc'].append(result['pc'])
            results[result['method']][result['lambda_eff']]['anc'].append(result['anc'])
            results[result['method']][result['lambda_eff']]['success'].append(float(result['success']))

    if use_rl and rl_tasks:
        print(f"  Running {len(rl_tasks)} RL tasks with {n_workers_rl} workers...")
        with ProcessPoolExecutor(
                max_workers=n_workers_rl,
                initializer=_init_rl_worker,
                initargs=(model_path,)
        ) as executor:
            for result in tqdm(executor.map(_run_rl_task, rl_tasks),
                               desc=f"{graph_type}-{target_dist} (GNN-RL)",
                               total=len(rl_tasks)):
                if result['pc'] is not None:
                    results['GNN-RL'][result['lambda_eff']]['pc'].append(result['pc'])
                    results['GNN-RL'][result['lambda_eff']]['anc'].append(result['anc'])
                    results['GNN-RL'][result['lambda_eff']]['success'].append(float(result['success']))

    return results


def save_results_to_csv(all_results: Dict, lambda_eff_values: List[float],
                        n_nodes: int, target_ratio: float,
                        save_dir: str) -> str:
    rows = []
    for config, res in all_results.items():
        for method, lam_dict in res.items():
            for lam in sorted(lambda_eff_values):
                pc_vals  = lam_dict[lam]['pc']
                anc_vals = lam_dict[lam]['anc']
                suc_vals = lam_dict[lam]['success']
                if not pc_vals:
                    continue
                n_runs = len(pc_vals)
                rows.append({
                    'config':       config,
                    'method':       method,
                    'lambda_eff':   lam,
                    'n_runs':       n_runs,
                    # PC
                    'pc_mean':      float(np.mean(pc_vals)),
                    'pc_std':       float(np.std(pc_vals)),
                    'pc_sem':       float(np.std(pc_vals) / np.sqrt(n_runs)),  # SEM
                    # ANC
                    'anc_mean':     float(np.mean(anc_vals)),
                    'anc_std':      float(np.std(anc_vals)),
                    'anc_sem':      float(np.std(anc_vals) / np.sqrt(n_runs)),  # SEM
                    # Success rate
                    'success_rate': float(np.mean(suc_vals)),
                })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'results_n{n_nodes}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Results saved to: {csv_path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    return csv_path


def plot_comparison(all_results: Dict, lambda_eff_values: List[float],
                    n_nodes: int, target_ratio: float,
                    save_dir: Optional[str] = None, show_figure: bool = True):
    configs       = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
    config_labels = ['BA-Random', 'BA-Localized', 'WS-Random', 'WS-Localized']

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    colors  = {'Adaptive TD': '#3498db', 'Adaptive Katz': '#e74c3c',
               'Adaptive TIA': '#2ecc71', 'GNN-RL': '#9b59b6'}
    markers = {'Adaptive TD': 'o', 'Adaptive Katz': 's',
               'Adaptive TIA': '^', 'GNN-RL': 'D'}

    for col, config in enumerate(configs):
        ax_pc  = axes[0, col]
        ax_anc = axes[1, col]

        if config not in all_results:
            ax_pc.set_visible(False)
            ax_anc.set_visible(False)
            continue

        res = all_results[config]

        for method in res:
            lambdas, means, errs = [], [], []
            for lam in sorted(lambda_eff_values):
                vals = res[method][lam]['pc']
                if vals:
                    n = len(vals)
                    lambdas.append(lam)
                    means.append(np.mean(vals))
                    errs.append(np.std(vals) / np.sqrt(n))
            if lambdas:
                means_arr = np.array(means)
                errs_arr  = np.array(errs)
                ax_pc.plot(lambdas, means_arr,
                           color=colors.get(method, '#000'),
                           marker=markers.get(method, 'o'),
                           label=method, linewidth=2, markersize=6)
                ax_pc.fill_between(lambdas,
                                   means_arr - errs_arr,
                                   means_arr + errs_arr,
                                   alpha=0.15, color=colors.get(method, '#000'))

        ax_pc.set_xlabel(r'$\lambda_{eff}$', fontsize=12)
        ax_pc.set_ylabel('PC', fontsize=12)
        ax_pc.set_title(config_labels[col], fontsize=13, fontweight='bold')
        if col == 0:
            ax_pc.legend(loc='best', fontsize=8)
        ax_pc.grid(True, alpha=0.3, linestyle='--', which='both')

        for method in res:
            lambdas, means, errs = [], [], []
            for lam in sorted(lambda_eff_values):
                vals = res[method][lam]['anc']
                if vals:
                    n = len(vals)
                    lambdas.append(lam)
                    means.append(np.mean(vals))
                    errs.append(np.std(vals) / np.sqrt(n))
            if lambdas:
                means_arr = np.array(means)
                errs_arr  = np.array(errs)
                ax_anc.plot(lambdas, means_arr,
                            color=colors.get(method, '#000'),
                            marker=markers.get(method, 'o'),
                            label=method, linewidth=2, markersize=6)
                ax_anc.fill_between(lambdas,
                                    means_arr - errs_arr,
                                    means_arr + errs_arr,
                                    alpha=0.15, color=colors.get(method, '#000'))

        ax_anc.set_xlabel(r'$\lambda_{eff}$', fontsize=12)
        ax_anc.set_ylabel('ANC', fontsize=12)
        if col == 0:
            ax_anc.legend(loc='best', fontsize=8)
        ax_anc.grid(True, alpha=0.3, linestyle='--', which='both')

    fig.suptitle(
        f'Comparison: Adaptive Heuristics vs GNN-RL | N={n_nodes}, '
        f'Target Ratio={target_ratio}\n'
        f'(TD/TIA: Batch Adaptive, Katz/RL: Fully Adaptive)  '
        f'[Shaded band = ±1 SEM]',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'comparison_n{n_nodes}.png')
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {filepath}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig


def comprehensive_comparison(
        n_nodes: int = 128, target_ratio: float = 0.05,
        lambda_eff_values: List[float] = None, simulation_times: int = 100,
        model_path: Optional[str] = None,
        ba_m_range=(2, 4), ws_k_range=(4, 8), ws_beta_range=(0.1, 0.3),
        save_dir: Optional[str] = None, show_figure: bool = False,
        seed: int = 42, n_workers: int = None, n_workers_rl: int = None) -> Dict:

    if lambda_eff_values is None:
        lambda_eff_values = [1] + list(range(2, 22, 2))
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    graph_params = {
        'ba_m_range':   ba_m_range,
        'ws_k_range':   ws_k_range,
        'ws_beta_range': ws_beta_range,
    }
    configs = [('BA', 'random'), ('BA', 'localized'), ('WS', 'random'), ('WS', 'localized')]
    all_results = {}

    for graph_type, target_dist in configs:
        config_name = f"{graph_type}-{target_dist}"
        print(f"\n{'=' * 60}")
        print(f"Running: {config_name}")
        print(f"{'=' * 60}")
        all_results[config_name] = run_phase_transition_experiments(
            graph_type=graph_type, target_dist=target_dist, n_nodes=n_nodes,
            target_ratio=target_ratio, lambda_eff_values=lambda_eff_values,
            simulation_times=simulation_times, graph_params=graph_params,
            model_path=model_path, seed=seed,
            n_workers=n_workers, n_workers_rl=n_workers_rl
        )

    if save_dir:
        save_results_to_csv(all_results, lambda_eff_values, n_nodes,
                            target_ratio, save_dir)

    plot_comparison(all_results, lambda_eff_values, n_nodes, target_ratio,
                    save_dir, show_figure)
    return all_results


if __name__ == "__main__":
    N_NODES          = 1024
    TARGET_RATIO     = 0.05
    LAMBDA_EFF_VALUES = [1] + list(range(2, 22, 2))
    SIMULATION_TIMES = 100

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    MODEL_PATH = os.path.join(REPO_ROOT, 'train',
                              'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                              'model.pt')
    SAVE_DIR   = os.path.join(SCRIPT_DIR, 'data')

    N_WORKERS    = 31
    N_WORKERS_RL = 31

    results = comprehensive_comparison(
        n_nodes=N_NODES, target_ratio=TARGET_RATIO,
        lambda_eff_values=LAMBDA_EFF_VALUES, simulation_times=SIMULATION_TIMES,
        model_path=MODEL_PATH, save_dir=SAVE_DIR, show_figure=False,
        seed=42, n_workers=N_WORKERS, n_workers_rl=N_WORKERS_RL
    )