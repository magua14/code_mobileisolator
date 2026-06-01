import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm
from torch_geometric.data import Batch

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig6'))

GNN_AVAILABLE = False
try:
    from ppo_gnn_moving_gat_dualstream_final import DualStreamAgent as GNNAgent
    GNN_AVAILABLE = True
    print("GNN Agent (final) loaded successfully.")
except ImportError as e:
    print(f"GNN Agent not found: {e}")
    sys.exit(1)

LOCAL_DIM  = 3
GLOBAL_DIM = 2


import draw_covid19_new            as covid_mod
import draw_city_flood_new         as flood_mod
import draw_invasive_species_new   as invasive_mod
import draw_smuggling_new          as smuggling_mod
import draw_socialbot_new          as socialbot_mod
import draw_fugitive_chasing_new   as fugitive_mod

from comparison_method import compute_params_from_lambda


LAMBDA_VALUES    = [1, 10, 20]
SIMULATION_TIMES = 50
SEED             = 14

N_WORKERS        = 60 if sys.platform.startswith('win') else 128
DEVICE           = 'cpu'

SAVE_DIR = os.path.join(_SCRIPT_DIR, 'data')

MODEL_PATHS = {
    'Full':       os.path.join(_REPO_ROOT, 'train',
                               'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                               'model.pt'),
    'No-Global':  os.path.join(_SCRIPT_DIR,
                               'gnn_ppo_dual_stream_n50-100_no_global_stream__seed42__1776256198',
                               'model.pt'),
    'No-Attn':    os.path.join(_SCRIPT_DIR,
                               'gnn_ppo_dual_stream_n50-100_no_attention__seed42__1776302534',
                               'model.pt'),
    'No-VNode':   os.path.join(_SCRIPT_DIR,
                               'gnn_ppo_dual_stream_n50-100_no_virtual_node__seed42__1776317806',
                               'model.pt'),
}


class AblationAgentWrapper:

    def __init__(self, model_path: str, device: str = 'cpu'):
        if not GNN_AVAILABLE:
            raise RuntimeError("GNN Agent not available.")
        self.device = torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device,
                          weights_only=False)
        args_dict = ckpt.get('args', {})

        self.agent = GNNAgent(
            local_dim          = args_dict.get('local_dim',          LOCAL_DIM),
            global_dim         = args_dict.get('global_dim',         GLOBAL_DIM),
            hidden_dim         = args_dict.get('hidden_dim',         128),
            context_dim        = args_dict.get('context_dim',        32),
            num_gnn_layers     = args_dict.get('num_gnn_layers',     3),
            gat_heads          = args_dict.get('gat_heads',          4),
            use_global_stream  = args_dict.get('use_global_stream',  True),
            use_attention      = args_dict.get('use_attention',      True),
            use_virtual_node   = args_dict.get('use_virtual_node',   True),
        ).to(self.device)
        self.agent.load_state_dict(ckpt['model_state_dict'])
        self.agent.eval()

    def get_action_deterministic(self, data) -> int:
        batch_data = Batch.from_data_list([data]).to(self.device)
        with torch.no_grad():
            logits, _ = self.agent.forward(batch_data)
            logits = logits.squeeze(-1)
            mask = batch_data.action_mask
            logits[mask == 0] = float('-inf')
            return logits.argmax().item()



def _setup_covid():
    GRAPHML_PATH = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_data',
                                'covid19_network.graphml')

    G = covid_mod.load_network(GRAPHML_PATH)

    target_nodes = covid_mod.generate_cluster_targets(
        G, target_ratio=0.05, n_clusters=4, k_hops=5, seed=42)

    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    init_args = (static_nodes, static_edges, target_nodes)
    return G, target_nodes, init_args


def _setup_flood():
    GRAPHML_FILE = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_data',
                                'flood_network.graphml')
    G, target_nodes, elevations = flood_mod.build_flood_network(
        graphml_file=GRAPHML_FILE,
        elevation_min=50.0,
        elevation_max=100.0,
        target_ratio=0.05,
        elevation_attr='elevation',
        seed=42,
    )
    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    init_args = (static_nodes, static_edges, target_nodes, elevations)
    return G, target_nodes, init_args


def _setup_invasive():
    DATA_FILE = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_data',
                             'invasive_network.graphml')
    G, target_nodes = invasive_mod.build_invasive_network(
        data_file=DATA_FILE,
        target_ratio=0.05,
        seed=42,
    )
    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    init_args = (static_nodes, static_edges, target_nodes)
    return G, target_nodes, init_args


def _setup_smuggling():
    DATA_FILE = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_data',
                             'smuggling_network.graphml')
    G, target_nodes = smuggling_mod.build_smuggling_network(
        data_file=DATA_FILE,
        target_ratio=0.05,
        max_peripheral_degree=3,
        seed=42,
    )
    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    TAU = 2.0
    init_args = (static_nodes, static_edges, target_nodes, TAU)
    return G, target_nodes, init_args


def _setup_socialbot():
    DATA_FILE = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_data',
                             'socialbot_network.graphml')
    G, target_nodes = socialbot_mod.build_socialbot_network(
        data_file=DATA_FILE,
        target_ratio=0.05,
        seed=42,
    )
    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    init_args = (static_nodes, static_edges, target_nodes)
    return G, target_nodes, init_args


def _setup_fugitive():
    DATA_FILE = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_data',
                             'fugitive_network.graphml')
    G, target_nodes = fugitive_mod.build_fugitive_network(
        data_file=DATA_FILE,
        target_ratio=0.03,
        seed=42,
    )
    static_nodes = list(G.nodes())
    static_edges = list(G.edges())
    init_args = (static_nodes, static_edges, target_nodes)
    return G, target_nodes, init_args


SCENARIOS = [
    ('covid19',   _setup_covid,     covid_mod),
    ('flood',     _setup_flood,     flood_mod),
    ('invasive',  _setup_invasive,  invasive_mod),
    ('smuggling', _setup_smuggling, smuggling_mod),
    ('socialbot', _setup_socialbot, socialbot_mod),
    ('fugitive',  _setup_fugitive,  fugitive_mod),
]

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"Real-world ablation experiment")
    print(f"{'='*72}")
    print(f"  Scenarios          : {[name for name, _, _ in SCENARIOS]}")
    print(f"  Variants           : {list(MODEL_PATHS.keys())}")
    print(f"  Lambdas            : {LAMBDA_VALUES}")
    print(f"  Seeds per setting  : {SIMULATION_TIMES}")
    print(f"  Workers per pool   : {N_WORKERS}")
    print(f"  Output             : {SAVE_DIR}")
    print(f"{'='*72}\n")

    active_variants = {}
    for name, path in MODEL_PATHS.items():
        if path and os.path.exists(path):
            active_variants[name] = path
        else:
            print(f"  [warn] model not found, skipping variant '{name}': {path}")
    if not active_variants:
        raise RuntimeError("No valid model paths found.")
    print(f"Active variants: {list(active_variants.keys())}\n")

    long_rows = []

    for scenario_name, setup_fn, scenario_mod in SCENARIOS:
        print(f"\n>>> Setting up scenario: {scenario_name}")
        t0 = time.time()
        G, target_nodes, init_args = setup_fn()
        n_nodes = G.number_of_nodes()
        n_targets = len(target_nodes)
        print(f"    network: N={n_nodes}, E={G.number_of_edges()}, "
              f"|T|={n_targets} ({n_targets/n_nodes:.2%})  "
              f"setup time: {time.time()-t0:.1f}s")

        rng = np.random.RandomState(SEED)
        tasks = []
        for lam in LAMBDA_VALUES:
            move_prob, attack_ratio = compute_params_from_lambda(lam)
            attack_per_step = max(1, int(attack_ratio * n_targets))
            for _ in range(SIMULATION_TIMES):
                sim_seed = int(rng.randint(0, 1_000_000_000))
                tasks.append(
                    (move_prob, attack_per_step, sim_seed, lam)
                )
        print(f"    total tasks per variant: {len(tasks)}")

        run_rl_task   = scenario_mod._run_rl_task
        init_rl_worker = scenario_mod._init_rl_worker

        scenario_mod.RLAgentWrapper = AblationAgentWrapper

        for variant_name, model_path in active_variants.items():
            print(f"    ── variant: {variant_name}")
            t_var = time.time()

            with ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=init_rl_worker,
                initargs=(model_path, *init_args),
            ) as executor:
                pbar = tqdm(executor.map(run_rl_task, tasks),
                            total=len(tasks),
                            desc=f"{scenario_name}/{variant_name}",
                            leave=False)
                for r in pbar:
                    if r is None or r.get('pc') is None:
                        continue
                    long_rows.append({
                        'scenario':   scenario_name,
                        'variant':    variant_name,
                        'lambda_eff': r['lambda_eff'],
                        'sim_seed':   r.get('sim_seed'),
                        'pc':         r['pc'],
                        'anc':        r['anc'],
                    })

            print(f"       done in {time.time()-t_var:.1f}s")

    df_long = pd.DataFrame(long_rows)
    long_csv = os.path.join(SAVE_DIR, 'results_ablation_realworld.csv')
    df_long.to_csv(long_csv, index=False)
    print(f"\n✓ Long-format results saved: {long_csv} ({len(df_long)} rows)")

    summary_rows = []
    for (scen, var, lam), grp in df_long.groupby(['scenario', 'variant', 'lambda_eff']):
        n = len(grp)
        if n == 0:
            continue
        summary_rows.append({
            'scenario':   scen,
            'variant':    var,
            'lambda_eff': lam,
            'n_runs':     n,
            'pc_mean':    float(np.mean(grp['pc'])),
            'pc_sem':     float(np.std(grp['pc']) / np.sqrt(n)),
            'anc_mean':   float(np.mean(grp['anc'])),
            'anc_sem':    float(np.std(grp['anc']) / np.sqrt(n)),
        })
    df_summary = pd.DataFrame(summary_rows).sort_values(
        ['scenario', 'lambda_eff', 'variant']
    )
    summary_csv = os.path.join(SAVE_DIR, 'summary_ablation_realworld.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"✓ Summary saved: {summary_csv} ({len(df_summary)} rows)\n")

    print("=" * 72)
    print("Summary (mean ± SEM)")
    print("=" * 72)
    for scen in df_summary['scenario'].unique():
        print(f"\n[{scen}]")
        sub = df_summary[df_summary['scenario'] == scen]
        for lam in sorted(sub['lambda_eff'].unique()):
            print(f"  λ = {lam}")
            for _, row in sub[sub['lambda_eff'] == lam].iterrows():
                print(f"    {row['variant']:<10}  "
                      f"PC = {row['pc_mean']:.3f} ± {row['pc_sem']:.3f}   "
                      f"ANC = {row['anc_mean']:.3f} ± {row['anc_sem']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()