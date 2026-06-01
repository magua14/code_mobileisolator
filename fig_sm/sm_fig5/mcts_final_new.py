import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
import math
import time
from collections import deque


class TargetIsolationGame:
    def __init__(self, G, target_nodes, move_prob, attack_per_step, seed=None):
        self.original_graph  = G.copy()
        self.n_nodes         = G.number_of_nodes()
        self.n_targets       = len(target_nodes)
        self.n_non_targets   = self.n_nodes - self.n_targets
        self.move_prob       = move_prob
        self.attack_per_step = attack_per_step
        self.rng = np.random.RandomState(seed)
        self.current_graph        = G.copy()
        self.target_nodes         = set(target_nodes)
        self.removed_nodes        = set()
        self.current_attack_count = 0
        self.round_idx            = 0
        self.initial_lcc_size = self._get_lcc_size()
        self.cumulative_anc   = 0.0
        self.anc_count        = 0

    def clone(self):
        new = TargetIsolationGame.__new__(TargetIsolationGame)
        new.original_graph      = self.original_graph
        new.n_nodes             = self.n_nodes
        new.n_targets           = self.n_targets
        new.n_non_targets       = self.n_non_targets
        new.move_prob           = self.move_prob
        new.attack_per_step     = self.attack_per_step
        new.rng = np.random.RandomState(self.rng.randint(0, 2**31))
        new.current_graph         = self.current_graph.copy()
        new.target_nodes          = self.target_nodes.copy()
        new.removed_nodes         = self.removed_nodes.copy()
        new.current_attack_count  = self.current_attack_count
        new.round_idx             = self.round_idx
        new.initial_lcc_size = self.initial_lcc_size
        new.cumulative_anc   = self.cumulative_anc
        new.anc_count        = self.anc_count
        return new

    def get_valid_actions(self):
        return [n for n in self.current_graph.nodes()
                if n not in self.target_nodes and n not in self.removed_nodes]

    def step(self, action):
        if (action in self.target_nodes or action in self.removed_nodes
                or action not in self.current_graph):
            return 0.0, False
        self.removed_nodes.add(action)
        self.current_graph.remove_node(action)
        self.current_attack_count += 1
        lcc_ratio = self._get_lcc_size() / max(1, self.initial_lcc_size)
        self.cumulative_anc += lcc_ratio
        self.anc_count      += 1
        if self._count_targets_in_lcc() == 0:
            return self._calculate_reward(), True
        if self.current_attack_count >= self.attack_per_step:
            self._move_targets()
            self.current_attack_count = 0
            self.round_idx += 1
        return 0.0, False

    def _move_targets(self):
        if self.move_prob <= 0:
            return
        movements, claimed = [], set()
        for t in [t for t in self.target_nodes if t in self.current_graph]:
            if self.rng.random() > self.move_prob:
                continue
            valid = [n for n in self.current_graph.neighbors(t)
                     if n not in self.removed_nodes
                     and n not in self.target_nodes
                     and n not in claimed]
            if valid:
                new_pos = self.rng.choice(valid)
                movements.append((t, new_pos))
                claimed.add(new_pos)
        for old, new in movements:
            self.target_nodes.discard(old)
            self.target_nodes.add(new)

    def _get_lcc_size(self):
        if self.current_graph.number_of_nodes() == 0:
            return 0
        try:
            return len(max(nx.connected_components(self.current_graph), key=len))
        except ValueError:
            return 0

    def _count_targets_in_lcc(self):
        if self.current_graph.number_of_nodes() == 0:
            return 0
        try:
            lcc = max(nx.connected_components(self.current_graph), key=len)
            return sum(1 for t in self.target_nodes if t in lcc)
        except ValueError:
            return 0

    def _calculate_reward(self):
        pc  = len(self.removed_nodes) / max(1, self.n_non_targets)
        anc = (self.cumulative_anc / self.anc_count) if self.anc_count > 0 else 1.0
        return (1.0 - pc) + anc

    def is_terminal(self):
        return self._count_targets_in_lcc() == 0

    def get_metrics(self):
        pc  = len(self.removed_nodes) / max(1, self.n_non_targets)
        anc = (self.cumulative_anc / self.anc_count) if self.anc_count > 0 else 1.0
        return pc, anc

    def _compute_distances_to_targets(self):
        distances = {node: float('inf') for node in self.current_graph.nodes()}
        queue = deque()
        for t in self.target_nodes:
            if t in self.current_graph:
                distances[t] = 0
                queue.append(t)
        while queue:
            cur = queue.popleft()
            for nb in self.current_graph.neighbors(cur):
                if distances[nb] > distances[cur] + 1:
                    distances[nb] = distances[cur] + 1
                    queue.append(nb)
        return distances


class MCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent   = parent
        self.action   = action
        self.children = {}
        self.visits      = 0
        self.total_value = 0.0
        self.untried_actions = None
        self.is_terminal = False

    @property
    def q_value(self):
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.q_value + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c=1.4):
        return max(self.children.values(), key=lambda n: n.ucb1(c))

    def most_visited_child(self):
        return max(self.children.values(), key=lambda n: n.visits)


def _rl_score(game, node):
    if node not in game.current_graph or node in game.target_nodes:
        return -float('inf')
    distances  = game._compute_distances_to_targets()
    degrees    = dict(game.current_graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    proximity  = 1.0 / (distances.get(node, float('inf')) + 1.0)
    importance = math.log(degrees.get(node, 0) + 1) / max(math.log(max_degree + 1), 1e-6)
    return proximity + importance


def _get_top_k_actions(game, k=15):
    valid = game.get_valid_actions()
    if len(valid) <= k:
        return valid
    scores = [_rl_score(game, a) for a in valid]
    top_k  = np.argsort(scores)[-k:]
    return [valid[i] for i in top_k]


def _rl_rollout(game, rng, max_steps=None):
    if max_steps is None:
        max_steps = game.n_nodes
    for _ in range(max_steps):
        if game.is_terminal():
            break
        valid = game.get_valid_actions()
        if not valid:
            break
        if rng.random() < 0.8:
            scores = [_rl_score(game, a) for a in valid]
            action = valid[int(np.argmax(scores))]
        else:
            action = valid[rng.randint(len(valid))]
        _, done = game.step(action)
        if done:
            break
    return game._calculate_reward()


class ImprovedMCTS:

    def __init__(self,
                 num_simulations=2000,
                 exploration_weight=1.4,
                 use_progressive_widening=True,
                 top_k=15,
                 subtree_reuse=True,
                 reuse_discount=0.7,
                 use_determinization=True,
                 seed=None):

        self.num_simulations          = num_simulations
        self.exploration_weight       = exploration_weight
        self.use_progressive_widening = use_progressive_widening
        self.top_k                    = top_k
        self.subtree_reuse            = subtree_reuse
        self.reuse_discount           = reuse_discount
        self.use_determinization      = use_determinization
        self.rng = np.random.RandomState(seed)

        self._root        = None
        self._sim_offset  = 0

    def _try_reuse_root(self, last_action, game):
        if (self.subtree_reuse
                and self._root is not None
                and last_action is not None
                and last_action in self._root.children):

            new_root = self._root.children[last_action]
            new_root.parent = None
            if self.reuse_discount < 1.0:
                self._discount_subtree(new_root, self.reuse_discount)
            return new_root

        root = MCTSNode()
        root.untried_actions = (
            _get_top_k_actions(game, self.top_k)
            if self.use_progressive_widening
            else game.get_valid_actions()
        )
        return root

    def _discount_subtree(self, node, gamma):
        node.visits      = int(node.visits * gamma)
        node.total_value = node.total_value * gamma
        for child in node.children.values():
            self._discount_subtree(child, gamma)

    def search(self, game, last_action=None):
        root = self._try_reuse_root(last_action, game)

        if root.untried_actions is None and not root.children:
            root.untried_actions = (
                _get_top_k_actions(game, self.top_k)
                if self.use_progressive_widening
                else game.get_valid_actions()
            )

        for sim_i in range(self.num_simulations):
            sim_game = game.clone()

            if self.use_determinization:
                det_seed     = (self._sim_offset + sim_i) & 0x7FFFFFFF
                sim_game.rng = np.random.RandomState(det_seed)
                rollout_rng  = np.random.RandomState((det_seed + 1) & 0x7FFFFFFF)
            else:
                rollout_rng  = self.rng

            node = self._select(root, sim_game)

            if node.untried_actions and not node.is_terminal:
                node = self._expand(node, sim_game)

            reward = _rl_rollout(sim_game, rollout_rng)
            self._backpropagate(node, reward)

        self._sim_offset += self.num_simulations
        self._root = root

        if not root.children:
            valid = game.get_valid_actions()
            return valid[0] if valid else -1
        return root.most_visited_child().action

    def _select(self, node, game):
        while True:
            if node.is_terminal or node.untried_actions:
                return node
            if not node.children:
                node.is_terminal = True
                return node
            node = node.best_child(self.exploration_weight)
            _, done = game.step(node.action)
            if done:
                node.is_terminal = True

    def _expand(self, node, game):
        idx    = self.rng.randint(len(node.untried_actions))
        action = node.untried_actions.pop(idx)
        _, done = game.step(action)
        child = MCTSNode(parent=node, action=action)
        child.untried_actions = (
            _get_top_k_actions(game, self.top_k)
            if self.use_progressive_widening
            else game.get_valid_actions()
        )
        child.is_terminal     = done
        node.children[action] = child
        return child

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits      += 1
            node.total_value += reward
            node = node.parent

    def reset(self):
        self._root       = None
        self._sim_offset = 0


def solve_with_improved_mcts(G, target_nodes, move_prob, attack_per_step,
                             num_simulations=2000,
                             use_progressive_widening=True,
                             top_k=15,
                             subtree_reuse=True,
                             reuse_discount=0.7,
                             use_determinization=True,
                             seed=None,
                             verbose=False):
    start = time.time()
    game  = TargetIsolationGame(G, target_nodes, move_prob, attack_per_step, seed)

    mcts = ImprovedMCTS(
        num_simulations=num_simulations,
        use_progressive_widening=use_progressive_widening,
        top_k=top_k,
        subtree_reuse=subtree_reuse,
        reuse_discount=reuse_discount,
        use_determinization=use_determinization,
        seed=seed,
    )

    removed_nodes = []
    max_steps     = G.number_of_nodes()
    last_action   = None

    for step in range(max_steps):
        if game.is_terminal():
            break

        action = mcts.search(game, last_action=last_action)
        if action == -1:
            break

        removed_nodes.append(action)
        _, done = game.step(action)
        last_action = action

        if verbose and (step + 1) % 10 == 0:
            pc, anc = game.get_metrics()
            print(f"  Step {step+1}: removed={len(removed_nodes)}, "
                  f"PC={pc:.3f}, ANC={anc:.3f}")
        if done:
            break

    pc, anc = game.get_metrics()
    elapsed = time.time() - start

    if verbose:
        print(f"\n✓ MCTS (v2) finished in {elapsed:.2f}s | "
              f"PC={pc:.4f}, ANC={anc:.4f}, success={game.is_terminal()}")

    return {
        'pc':            pc,
        'anc':           anc,
        'success':       game.is_terminal(),
        'removed_nodes': removed_nodes,
        'time':          elapsed,
    }