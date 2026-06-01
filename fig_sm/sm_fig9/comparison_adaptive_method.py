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
from typing import Tuple, List, Dict, Optional
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Batch

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))

GNN_AVAILABLE = False
try:
    from ppo_gnn_moving_gat_dualstream_final import DualStreamAgent as GNNAgent
    GNN_AVAILABLE = True
    print("GNN-RL Agent loaded successfully.")
except ImportError as e:
    print(f"GNN-RL Agent not found: {e}")

LOCAL_DIM  = 3
GLOBAL_DIM = 2

_rl_agent_worker = None


N_NODES         = 1024
TARGET_RATIO    = 0.05
LAMBDA_VALUES   = [1] + list(range(2, 22, 2))
SIMULATION_TIMES = 100
SEED            = 42

MODEL_PATH = os.path.join(_REPO_ROOT, 'train',
                          'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                          'model.pt')
SAVE_DIR   = os.path.join(_SCRIPT_DIR, 'data')

_DEFAULT_WORKERS = max(1, mp.cpu_count() - 1)
N_WORKERS    = min(60, _DEFAULT_WORKERS) if sys.platform.startswith('win') else _DEFAULT_WORKERS
N_WORKERS_RL = min(8, max(1, mp.cpu_count() // 2))

GRAPH_PARAMS = {
    'ba_m_range':    [2, 4],
    'ws_k_range':    [4, 8],
    'ws_beta_range': [0.1, 0.3],
}

CONFIGS = [
    ('BA', 'random'),
    ('BA', 'localized'),
    ('WS', 'random'),
    ('WS', 'localized'),
]

BASE_METHODS  = ['TD', 'Katz', 'TIA', 'GNN-RL']
STRATEGIES    = ['Fully', 'Batch']


mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size':         6,  'axes.labelsize': 7,  'axes.titlesize': 7,
    'xtick.labelsize':   6,  'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size':  2.5, 'ytick.major.size':  2.5,
    'xtick.direction':   'in', 'ytick.direction': 'in',
    'figure.dpi':        300, 'savefig.dpi': 300,
    'savefig.bbox':      'tight',
    'figure.facecolor':  'white', 'axes.facecolor': 'white',
})

METHOD_COLORS = {
    'TD':     '#0072B2',
    'Katz':   '#D55E00',
    'TIA':    '#009E73',
    'GNN-RL': '#CC79A7',
}
METHOD_MARKERS = {
    'TD':     'o',
    'Katz':   's',
    'TIA':    '^',
    'GNN-RL': 'D',
}
STRATEGY_LINESTYLE = {'Fully': '-', 'Batch': '--'}


def compute_params_from_lambda(lambda_eff: float) -> Tuple[float, float]:
    if lambda_eff == 0:
        return 0.0, 1.0
    return 1.0, 1.0 / lambda_eff



def generate_graph(graph_type: str, n_nodes: int, params: Dict, seed: int) -> nx.Graph:
    rng = np.random.RandomState(seed)
    if graph_type == 'BA':
        m = rng.randint(params['ba_m_range'][0], params['ba_m_range'][1] + 1)
        G = nx.barabasi_albert_graph(n_nodes, min(m, n_nodes - 1), seed=seed)
    else:
        k = rng.randint(params['ws_k_range'][0] // 2,
                        params['ws_k_range'][1] // 2 + 1) * 2
        k = max(2, min(k, n_nodes - 1))
        beta = rng.uniform(*params['ws_beta_range'])
        G = nx.watts_strogatz_graph(n_nodes, k, beta, seed=seed)
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = nx.convert_node_labels_to_integers(G.subgraph(lcc).copy())
    return G


def generate_random_targets(G: nx.Graph, target_ratio: float,
                             seed: Optional[int] = None) -> List[int]:
    import random as _random
    if seed is not None:
        _random.seed(seed)
    n = G.number_of_nodes()
    k = max(1, int(n * target_ratio))
    return _random.sample(list(G.nodes()), k)


def generate_localized_targets(G: nx.Graph, target_ratio: float,
                                seed: Optional[int] = None,
                                radius: float = 0.05,
                                max_radius: float = 1.0,
                                step: float = 0.05) -> List[int]:
    rng = np.random.RandomState(seed)
    pos = nx.spring_layout(G, seed=seed)
    n = G.number_of_nodes()
    k = max(1, int(n * target_ratio))
    nodes = list(G.nodes())
    center = rng.choice(nodes)
    cx, cy = pos[center]
    r = radius
    candidates = []
    while r < max_radius:
        candidates = [nd for nd, (x, y) in pos.items()
                      if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2]
        if len(candidates) >= k:
            break
        r += step
    if len(candidates) >= k:
        idx = rng.choice(len(candidates), size=k, replace=False)
        return [candidates[i] for i in idx]
    remaining = k - len(candidates)
    others = list(set(nodes) - set(candidates))
    if remaining > 0 and others:
        extra_idx = rng.choice(len(others),
                               size=min(remaining, len(others)), replace=False)
        return candidates + [others[i] for i in extra_idx]
    return candidates



def td_method_adaptive(G: nx.Graph, target_nodes: List[int]) -> List[int]:
    target_set = set(target_nodes)
    td = {nd: sum(1 for nb in G.neighbors(nd) if nb in target_set)
          for nd in G.nodes() if nd not in target_set}
    return [n for n, _ in sorted(td.items(), key=lambda x: x[1], reverse=True)]


def katz_method_adaptive(G: nx.Graph, target_nodes: List[int],
                          max_iter: int = 50, tol: float = 1e-6) -> List[int]:
    from scipy.sparse import issparse
    target_set = set(target_nodes)
    n = G.number_of_nodes()
    if n == 0:
        return []
    node_list = list(G.nodes())
    node_to_idx = {nd: i for i, nd in enumerate(node_list)}
    A = nx.to_scipy_sparse_array(G, nodelist=node_list, format='csr', dtype=float)
    f = np.zeros(n)
    for nd in target_nodes:
        if nd in node_to_idx:
            f[node_to_idx[nd]] = 1.0
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    for _ in range(20):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-10:
            break
        v = v_new / norm
    lambda_max = max(1e-10, abs(float(v @ (A @ v))))
    epsilon = 0.5 / lambda_max
    x = f.copy()
    for _ in range(max_iter):
        x_new = epsilon * (A @ x) + f
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    KatzT = x / epsilon - f / epsilon
    return [node_list[i] for i in np.argsort(-KatzT)
            if node_list[i] not in target_set]


def tia_core_adaptive(G: nx.Graph, target_nodes: List[int]):
    G = G.copy()
    T = set(target_nodes)
    K: set = set()
    R = set(G.nodes()) - T
    for nd in T:
        if nd in G:
            K.update(G.neighbors(nd))
    K -= T
    R -= K
    nbrs = {nd: set(G.neighbors(nd)) for nd in G.nodes()}
    Nmin_K = len(K)
    while True:
        changed = False
        DR = {nd: len(nbrs[nd] & R) for nd in K}
        to_move = [nd for nd in K if DR[nd] == 0]
        if to_move:
            for nd in to_move:
                T.add(nd); K.remove(nd)
            changed = True
        if len(K) < Nmin_K:
            Nmin_K = len(K)
        DR = {nd: len(nbrs[nd] & R) for nd in K}
        for r_node in R:
            if r_node not in nbrs:
                continue
            rn = nbrs[r_node]
            k_nbrs = rn & K
            if not any(DR.get(nb, 0) == 1 for nb in k_nbrs):
                continue
            K_t = K | {r_node}
            R_t = R - {r_node}
            T_t = set(T)
            DR_t = dict(DR)
            DR_t[r_node] = len(rn & R_t)
            for nb in k_nbrs:
                DR_t[nb] -= 1
            for kk in [kk for kk in K_t if DR_t.get(kk, 0) == 0]:
                T_t.add(kk); K_t.remove(kk)
            if len(K_t) < Nmin_K:
                K, T, R = K_t, T_t, R_t
                Nmin_K = len(K)
                changed = True
                break
        if not changed:
            break
    final_DR = {nd: len(nbrs[nd] & R) for nd in K}
    K_sorted = sorted(K, key=lambda x: final_DR[x], reverse=True)
    return T, K_sorted, R


def extended_tia_method_adaptive(G: nx.Graph, target_nodes: List[int]) -> List[int]:
    orig = set(target_nodes)
    T, K, R = tia_core_adaptive(G, target_nodes)
    return list(K) + list(T - orig) + list(R)



class MovingTargetSimulator:
    def __init__(self, G: nx.Graph, target_nodes: List[int],
                 move_prob: float, attack_per_step: int,
                 seed: Optional[int] = None):
        self.original_graph       = G.copy()
        self.current_graph        = G.copy()
        self.target_nodes         = set(target_nodes)
        self.removed_nodes        = set()
        self.move_prob            = move_prob
        self.attack_per_step      = attack_per_step
        self.n_non_targets        = sum(1 for n in G.nodes() if n not in target_nodes)
        self.initial_lcc_size     = self._get_lcc_size()
        self.cumulative_anc       = 0.0
        self.anc_count            = 0
        self.attacks_in_round     = 0
        self.rng                  = np.random.RandomState(seed)

    def _get_lcc_size(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        comps = list(nx.connected_components(self.current_graph))
        return len(max(comps, key=len)) if comps else 0

    def _targets_in_lcc(self) -> int:
        if self.current_graph.number_of_nodes() == 0:
            return 0
        comps = list(nx.connected_components(self.current_graph))
        if not comps:
            return 0
        lcc = max(comps, key=len)
        return sum(1 for t in self.target_nodes if t in lcc)

    def _move_targets(self):
        if self.move_prob <= 0:
            return
        movements, claimed = [], set()
        for t in [t for t in self.target_nodes if t in self.current_graph]:
            if self.rng.random() > self.move_prob:
                continue
            valid = [nb for nb in self.current_graph.neighbors(t)
                     if nb not in self.removed_nodes
                     and nb not in self.target_nodes
                     and nb not in claimed]
            if valid:
                new_pos = self.rng.choice(valid)
                movements.append((t, new_pos))
                claimed.add(new_pos)
        for old, new in movements:
            self.target_nodes.discard(old)
            self.target_nodes.add(new)

    def is_done(self) -> bool:
        return self._targets_in_lcc() == 0

    def _record_anc(self):
        ratio = self._get_lcc_size() / max(1, self.initial_lcc_size)
        self.cumulative_anc += ratio
        self.anc_count      += 1

    def remove_single_node(self, node) -> bool:
        if node in self.target_nodes or node in self.removed_nodes \
                or node not in self.current_graph:
            return self.is_done()
        self.removed_nodes.add(node)
        self.current_graph.remove_node(node)
        self.attacks_in_round += 1
        self._record_anc()
        return self.is_done()

    def remove_nodes_batch(self, nodes: List) -> bool:
        for nd in nodes:
            if nd in self.target_nodes or nd in self.removed_nodes \
                    or nd not in self.current_graph:
                continue
            self.removed_nodes.add(nd)
            self.current_graph.remove_node(nd)
            self.attacks_in_round += 1
            self._record_anc()
            if self.is_done():
                return True
        return self.is_done()

    def tick_round(self):
        if self.attacks_in_round >= self.attack_per_step:
            if not self.is_done():
                self._move_targets()
            self.attacks_in_round = 0

    def get_metrics(self) -> Tuple[float, float]:
        pc  = len(self.removed_nodes) / max(1, self.n_non_targets)
        anc = (self.cumulative_anc / self.anc_count
               if self.anc_count > 0 else 1.0)
        return pc, anc



def simulate_heuristic(G, target_nodes, move_prob, attack_per_step,
                       method_func, seed=None, batch_mode=False):
    sim = MovingTargetSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    max_steps = sum(1 for n in G.nodes() if n not in target_nodes)

    if batch_mode:
        while not sim.is_done():
            quota = sim.attack_per_step - sim.attacks_in_round
            if quota <= 0:
                sim.tick_round()
                quota = sim.attack_per_step
            ranking = method_func(sim.current_graph, list(sim.target_nodes))
            batch = [n for n in ranking
                     if n not in sim.removed_nodes
                     and n not in sim.target_nodes][:quota]
            if not batch:
                break
            if sim.remove_nodes_batch(batch):
                break
            sim.tick_round()
    else:
        for _ in range(max_steps):
            if sim.is_done():
                break
            order = method_func(sim.current_graph, list(sim.target_nodes))
            order = [n for n in order
                     if n not in sim.removed_nodes
                     and n not in sim.target_nodes]
            if not order:
                break
            if sim.remove_single_node(order[0]):
                break
            sim.tick_round()

    pc, anc = sim.get_metrics()
    return pc, anc, sim.is_done()



class IncrementalPyGBuilder:
    def __init__(self, G: nx.Graph, target_nodes: set):
        self.original_nodes = sorted(G.nodes())
        self.n_nodes        = len(self.original_nodes)
        self.node_to_idx    = {nd: i for i, nd in enumerate(self.original_nodes)}
        edges = []
        for u, v in G.edges():
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            edges += [[ui, vi], [vi, ui]]
        self.full_edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                                if edges else torch.zeros((2, 0), dtype=torch.long))
        degs = dict(G.degree())
        self.log_max_degree = math.log(max(degs.values(), default=1) + 1)
        self.node_valid_mask = torch.ones(self.n_nodes, dtype=torch.bool)
        self._done_removed   = set()

    def update_removed(self, removed: set):
        new = removed - self._done_removed
        if not new:
            return
        idx = torch.tensor([self.node_to_idx[n] for n in new
                             if n in self.node_to_idx], dtype=torch.long)
        if idx.numel() > 0:
            self.node_valid_mask[idx] = False
        self._done_removed.update(new)

    def _bfs(self, edge_index: torch.Tensor, target_nodes: set) -> dict:
        dist = np.full(self.n_nodes, np.inf)
        if edge_index.numel() == 0:
            for t in target_nodes:
                if t in self.node_to_idx:
                    dist[self.node_to_idx[t]] = 0.0
            return {nd: dist[self.node_to_idx[nd]] for nd in self.original_nodes}
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        order   = np.argsort(src, kind='stable')
        indices = dst[order]
        indptr  = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.add.at(indptr[1:], src, 1)
        np.cumsum(indptr, out=indptr)
        q = deque()
        for t in target_nodes:
            if t in self.node_to_idx:
                i = self.node_to_idx[t]
                if self.node_valid_mask[i]:
                    dist[i] = 0.0
                    q.append(i)
        while q:
            u = q.popleft()
            for v in indices[indptr[u]:indptr[u+1]]:
                if dist[v] == np.inf:
                    dist[v] = dist[u] + 1.0
                    q.append(int(v))
        return {nd: dist[self.node_to_idx[nd]] for nd in self.original_nodes}

    def build_data(self, current_graph, target_nodes, removed_nodes,
                   move_prob, attacks_in_round, attack_per_step):
        self.update_removed(removed_nodes)
        rm = self.node_valid_mask[self.full_edge_index[0]]
        cm = self.node_valid_mask[self.full_edge_index[1]]
        ei = self.full_edge_index[:, rm & cm]
        degs = torch.bincount(ei[0], minlength=self.n_nodes).float() \
               if ei.numel() > 0 else torch.zeros(self.n_nodes)
        log_max = math.log(degs.max().item() + 1) if degs.numel() > 0 else 1.0
        dists_d = self._bfs(ei, target_nodes)
        x = torch.zeros((self.n_nodes, 3))
        t_idx = [self.node_to_idx[t] for t in target_nodes if t in self.node_to_idx]
        if t_idx:
            x[t_idx, 0] = 1.0
        dists_t = torch.tensor(
            [dists_d[nd] for nd in self.original_nodes], dtype=torch.float32)
        x[:, 1] = 1.0 / (dists_t + 1.0)
        x[:, 2] = torch.log(degs + 1) / max(log_max, 1e-6)
        x[~self.node_valid_mask, 1:] = 0.0
        mask = self.node_valid_mask.float()
        if t_idx:
            mask[t_idx] = 0.0
        global_x = torch.tensor([[move_prob,
                                   attacks_in_round / max(1, attack_per_step)]],
                                 dtype=torch.float32)
        return Data(x=x, edge_index=ei, global_x=global_x,
                    action_mask=mask, num_nodes=self.n_nodes), self.original_nodes

    def reset(self):
        self.node_valid_mask[:] = True
        self._done_removed.clear()


class RLAgentWrapper:
    def __init__(self, model_path: str, device: str = 'cpu'):
        if not GNN_AVAILABLE:
            raise RuntimeError("GNN Agent not available.")
        self.device = torch.device(device)
        ck = torch.load(model_path, map_location=self.device, weights_only=False)
        a  = ck.get('args', {})
        self.agent = GNNAgent(
            local_dim      = a.get('local_dim',      LOCAL_DIM),
            global_dim     = a.get('global_dim',     GLOBAL_DIM),
            hidden_dim     = a.get('hidden_dim',     128),
            context_dim    = a.get('context_dim',    32),
            num_gnn_layers = a.get('num_gnn_layers', 3),
            gat_heads      = a.get('gat_heads',      4),
        ).to(self.device)
        self.agent.load_state_dict(ck['model_state_dict'])
        self.agent.eval()

    def get_action_deterministic(self, data: Data) -> int:
        bd = Batch.from_data_list([data]).to(self.device)
        with torch.no_grad():
            logits, _ = self.agent.forward(bd)
            logits = logits.squeeze(-1)
            logits[bd.action_mask == 0] = float('-inf')
            return int(logits.argmax().item())


def simulate_rl_fully(G, target_nodes, rl_agent, move_prob,
                      attack_per_step, seed=None):
    sim = MovingTargetSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    builder = IncrementalPyGBuilder(sim.original_graph, sim.target_nodes)
    max_steps = sum(1 for n in G.nodes() if n not in target_nodes)
    for _ in range(max_steps):
        if sim.is_done():
            break
        data, nodes = builder.build_data(
            sim.current_graph, sim.target_nodes, sim.removed_nodes,
            move_prob, sim.attacks_in_round, attack_per_step)
        node = nodes[rl_agent.get_action_deterministic(data)]
        if sim.remove_single_node(node):
            break
        sim.tick_round()
    pc, anc = sim.get_metrics()
    return pc, anc, sim.is_done()


def simulate_rl_batch(G, target_nodes, rl_agent, move_prob,
                      attack_per_step, seed=None):
    sim = MovingTargetSimulator(G, target_nodes, move_prob, attack_per_step, seed)
    builder = IncrementalPyGBuilder(sim.original_graph, sim.target_nodes)
    while not sim.is_done():
        quota = sim.attack_per_step - sim.attacks_in_round
        if quota <= 0:
            sim.tick_round()
            quota = sim.attack_per_step
        data, nodes = builder.build_data(
            sim.current_graph, sim.target_nodes, sim.removed_nodes,
            move_prob, sim.attacks_in_round, attack_per_step)
        batch = []
        visited = set(sim.removed_nodes) | set(sim.target_nodes)
        tmp_data = data
        tmp_nodes = list(nodes)
        while len(batch) < quota:
            action_idx = rl_agent.get_action_deterministic(tmp_data)
            node = tmp_nodes[action_idx]
            if node in visited:
                break
            batch.append(node)
            visited.add(node)
            tmp_data.action_mask[action_idx] = 0.0
        if not batch:
            break
        if sim.remove_nodes_batch(batch):
            break
        sim.tick_round()
    pc, anc = sim.get_metrics()
    return pc, anc, sim.is_done()



def _run_heuristic_task(args):
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     base_method, strategy, move_prob, attack_ratio, sim_seed, lambda_eff) = args

    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    if target_dist == 'random':
        targets = generate_random_targets(G, target_ratio, sim_seed)
    else:
        targets = generate_localized_targets(G, target_ratio, sim_seed)

    n_targets       = len(targets)
    attack_per_step = max(1, int(attack_ratio * n_targets))
    batch_mode      = (strategy == 'Batch')

    if base_method == 'TD':
        func = td_method_adaptive
    elif base_method == 'TIA':
        func = extended_tia_method_adaptive
    else:
        func = katz_method_adaptive

    pc, anc, success = simulate_heuristic(
        G, targets, move_prob, attack_per_step, func, sim_seed, batch_mode)

    return {'base_method': base_method, 'strategy': strategy,
            'lambda_eff': lambda_eff, 'pc': pc, 'anc': anc, 'success': success}


def _init_rl_worker(model_path: str):
    global _rl_agent_worker
    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path) and GNN_AVAILABLE:
        try:
            _rl_agent_worker = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            print(f"[RL worker] load failed: {e}")
            _rl_agent_worker = None
    else:
        _rl_agent_worker = None


def _run_rl_task(args):
    global _rl_agent_worker
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     strategy, move_prob, attack_ratio, sim_seed, lambda_eff) = args

    if _rl_agent_worker is None:
        return {'base_method': 'GNN-RL', 'strategy': strategy,
                'lambda_eff': lambda_eff, 'pc': None, 'anc': None, 'success': False}

    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)
    if target_dist == 'random':
        targets = generate_random_targets(G, target_ratio, sim_seed)
    else:
        targets = generate_localized_targets(G, target_ratio, sim_seed)

    n_targets       = len(targets)
    attack_per_step = max(1, int(attack_ratio * n_targets))

    if strategy == 'Fully':
        pc, anc, success = simulate_rl_fully(
            G, targets, _rl_agent_worker, move_prob, attack_per_step, sim_seed)
    else:
        pc, anc, success = simulate_rl_batch(
            G, targets, _rl_agent_worker, move_prob, attack_per_step, sim_seed)

    return {'base_method': 'GNN-RL', 'strategy': strategy,
            'lambda_eff': lambda_eff, 'pc': pc, 'anc': anc, 'success': success}



def run_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)
    use_rl = bool(MODEL_PATH and os.path.exists(MODEL_PATH) and GNN_AVAILABLE)
    rng = np.random.RandomState(SEED)

    all_results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {'pc': [], 'anc': []}))))

    csv_rows = []

    for graph_type, target_dist in CONFIGS:
        config_key = f"{graph_type}-{target_dist}"
        print(f"\n{'=' * 60}")
        print(f"  Config: {config_key}")
        print(f"{'=' * 60}")

        heuristic_tasks, rl_tasks = [], []

        for lam in LAMBDA_VALUES:
            move_prob, attack_ratio = compute_params_from_lambda(lam)
            for _ in range(SIMULATION_TIMES):
                sim_seed = int(rng.randint(0, int(1e9)))
                for base_method in ['TD', 'Katz', 'TIA']:
                    for strategy in STRATEGIES:
                        heuristic_tasks.append((
                            graph_type, N_NODES, GRAPH_PARAMS,
                            target_dist, TARGET_RATIO,
                            base_method, strategy,
                            move_prob, attack_ratio, sim_seed, lam
                        ))
                if use_rl:
                    for strategy in STRATEGIES:
                        rl_tasks.append((
                            graph_type, N_NODES, GRAPH_PARAMS,
                            target_dist, TARGET_RATIO,
                            strategy, move_prob, attack_ratio, sim_seed, lam
                        ))

        print(f"  Heuristic tasks: {len(heuristic_tasks)}")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            for res in tqdm(ex.map(_run_heuristic_task, heuristic_tasks),
                            desc=f"  {config_key} heuristic",
                            total=len(heuristic_tasks)):
                if res['pc'] is not None:
                    all_results[config_key][res['base_method']][res['strategy']][res['lambda_eff']]['pc'].append(res['pc'])
                    all_results[config_key][res['base_method']][res['strategy']][res['lambda_eff']]['anc'].append(res['anc'])

        if use_rl and rl_tasks:
            print(f"  RL tasks: {len(rl_tasks)}")
            with ProcessPoolExecutor(
                    max_workers=N_WORKERS_RL,
                    initializer=_init_rl_worker,
                    initargs=(MODEL_PATH,)) as ex:
                for res in tqdm(ex.map(_run_rl_task, rl_tasks),
                                desc=f"  {config_key} GNN-RL",
                                total=len(rl_tasks)):
                    if res['pc'] is not None:
                        all_results[config_key]['GNN-RL'][res['strategy']][res['lambda_eff']]['pc'].append(res['pc'])
                        all_results[config_key]['GNN-RL'][res['strategy']][res['lambda_eff']]['anc'].append(res['anc'])

        active = ['TD', 'Katz', 'TIA'] + (['GNN-RL'] if use_rl else [])
        for bm in active:
            for st in STRATEGIES:
                for lam in LAMBDA_VALUES:
                    d = all_results[config_key][bm][st][lam]
                    n = len(d['pc'])
                    if n == 0:
                        continue
                    pc_a  = np.array(d['pc'],  float)
                    anc_a = np.array(d['anc'], float)
                    csv_rows.append({
                        'config': config_key, 'base_method': bm,
                        'strategy': st, 'lambda_eff': lam,
                        'n_sims': n,
                        'pc_mean':  float(pc_a.mean()),
                        'pc_sem':   float(pc_a.std() / np.sqrt(n)),
                        'anc_mean': float(anc_a.mean()),
                        'anc_sem':  float(anc_a.std() / np.sqrt(n)),
                    })

    df = pd.DataFrame(csv_rows)
    df.to_csv(os.path.join(SAVE_DIR, 'comparison_adaptive_results.csv'), index=False)
    print(f"\nCSV saved: {os.path.join(SAVE_DIR, 'comparison_adaptive_results.csv')}")
    return all_results, df



def plot_results(all_results, df, save_dir):
    n_rows = 2
    n_cols = len(CONFIGS)
    col_titles = [f"{g}-{d.capitalize()}" for g, d in CONFIGS]
    row_ylabels = [r'$P_C$', 'ANC']
    row_metrics = ['pc', 'anc']
    X_TICKS = [1, 5, 10, 15, 20]

    FIG_W = 7.087
    PAN_H = 1.50
    LEG_H = 0.36
    FIG_H = n_rows * PAN_H + LEG_H + 0.20

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_W, FIG_H),
        gridspec_kw={'hspace': 0.45, 'wspace': 0.38,
                     'left': 0.09, 'right': 0.99,
                     'top':  0.94, 'bottom': 0.14},
    )

    for row_i, (metric, ylabel) in enumerate(zip(row_metrics, row_ylabels)):
        for col_i, (graph_type, target_dist) in enumerate(CONFIGS):
            config_key = f"{graph_type}-{target_dist}"
            ax = axes[row_i][col_i]

            for bm in BASE_METHODS:
                c  = METHOD_COLORS.get(bm, '#333333')
                mk = METHOD_MARKERS.get(bm, 'o')
                sub = df[(df['config']      == config_key) &
                         (df['base_method'] == bm)].sort_values('lambda_eff')

                for strategy in STRATEGIES:
                    s = sub[sub['strategy'] == strategy]
                    if s.empty:
                        continue
                    xs   = s['lambda_eff'].values
                    ys   = s[f'{metric}_mean'].values
                    errs = s[f'{metric}_sem'].values
                    ls   = STRATEGY_LINESTYLE[strategy]
                    ax.plot(xs, ys, color=c, marker=mk, markersize=2.5,
                            linewidth=0.8, linestyle=ls,
                            markeredgecolor='white', markeredgewidth=0.3,
                            zorder=3)
                    ax.fill_between(xs, ys - errs, ys + errs,
                                    color=c, alpha=0.12, linewidth=0, zorder=2)

            ax.set_xticks(X_TICKS)
            ax.set_xticklabels([str(v) for v in X_TICKS])
            ax.set_xlim(0, max(LAMBDA_VALUES) + 1)
            ax.tick_params(which='both', direction='in',
                           top=True, right=True, labelsize=6, pad=2)
            ax.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=4, min_n_ticks=3))
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=7, pad=3)
            if row_i == n_rows - 1:
                ax.set_xlabel(r'$\lambda_{eff}$', fontsize=7, labelpad=2)
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=7, labelpad=3)

    fig.suptitle(
        f'Fully vs Batch Adaptive  (N={N_NODES}, ratio={TARGET_RATIO})',
        fontsize=8, fontweight='bold', y=0.975)

    handles = []
    for bm in BASE_METHODS:
        c  = METHOD_COLORS.get(bm, '#333')
        mk = METHOD_MARKERS.get(bm, 'o')
        for st in STRATEGIES:
            ls = STRATEGY_LINESTYLE[st]
            handles.append(
                plt.Line2D([], [], color=c, marker=mk, markersize=3.5,
                           linewidth=0.8, linestyle=ls,
                           markeredgecolor='white', markeredgewidth=0.3,
                           label=f'{bm} ({st})'))
    fig.legend(
        handles=handles,
        loc='lower center', ncol=4,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False, fontsize=5.5,
        handlelength=1.8, handletextpad=0.4,
        columnspacing=0.8,
    )

    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.join(save_dir, 'comparison_adaptive')
    fig.savefig(stem + '.pdf', dpi=300)
    fig.savefig(stem + '.png', dpi=300)
    print(f"  Saved: {stem}.pdf / .png")
    plt.close(fig)



if __name__ == '__main__':
    use_rl = bool(MODEL_PATH and os.path.exists(MODEL_PATH) and GNN_AVAILABLE)
    print("Fully vs Batch Adaptive Strategy Comparison")
    print("=" * 60)
    print(f"  N_nodes        : {N_NODES}")
    print(f"  Target ratio   : {TARGET_RATIO}")
    print(f"  λ values       : {LAMBDA_VALUES}")
    print(f"  Sim times      : {SIMULATION_TIMES}")
    print(f"  GNN-RL         : {'Enabled' if use_rl else 'Disabled'}")
    print(f"  Save dir       : {SAVE_DIR}")
    print("=" * 60)

    all_results, df = run_experiment()
    plot_results(all_results, df, SAVE_DIR)
    print("\nDone.")