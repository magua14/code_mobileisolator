import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import torch
import networkx as nx

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_FIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_FIG_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))

from comparison_method import (
    RLAgentWrapper,
    GNN_AVAILABLE,
    td_method_adaptive,
    katz_method_adaptive,
    extended_tia_method_adaptive,
    simulate_heuristic,
    simulate_rl,
    generate_graph,
    generate_random_targets,
    generate_localized_targets,
    compute_params_from_lambda,
)

print("✓ Successfully imported from comparison_method")

_rl_agent = None



def _run_heuristic_task(args):
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     method, move_prob, attack_ratio, sim_seed) = args

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

    pc, anc, success = simulate_heuristic(
        G, targets, move_prob, attack_per_step, func, sim_seed, batch_mode
    )
    return {'pc': pc, 'anc': anc, 'success': success,
            'method': method, 'target_ratio': target_ratio}


def _init_rl_worker(model_path: str):
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
     move_prob, attack_ratio, sim_seed) = args

    if _rl_agent is None:
        return {'pc': None, 'anc': None, 'success': False,
                'target_ratio': target_ratio}

    G = generate_graph(graph_type, n_nodes, graph_params, sim_seed)

    if target_dist == 'random':
        targets = generate_random_targets(G, target_ratio, sim_seed)
    else:
        targets = generate_localized_targets(G, target_ratio, sim_seed)

    n_targets = len(targets)
    attack_per_step = max(1, int(attack_ratio * n_targets))

    pc, anc, success = simulate_rl(
        G, targets, _rl_agent, move_prob, attack_per_step, sim_seed
    )
    return {'pc': pc, 'anc': anc, 'success': success,
            'target_ratio': target_ratio}



def run_target_ratio_experiments(
        graph_type: str,
        target_dist: str,
        n_nodes: int,
        lambda_eff: float,
        target_ratio_values: List[float],
        simulation_times: int,
        graph_params: Dict,
        model_path: Optional[str],
        seed: int = 42,
        n_workers: int = None,
        n_workers_rl: int = None,
) -> Dict:
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    heuristic_methods = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA']
    use_rl = bool(model_path and os.path.exists(model_path) and GNN_AVAILABLE)
    all_methods = heuristic_methods + (['GNN-RL'] if use_rl else [])

    results = {m: defaultdict(lambda: {'pc': [], 'anc': [], 'success': []})
               for m in all_methods}

    rng = np.random.RandomState(seed)
    move_prob, attack_ratio = compute_params_from_lambda(lambda_eff)

    print(f"\n  Config: {graph_type}-{target_dist}")
    print(f"    λ_eff={lambda_eff} → move_prob={move_prob:.3f}, "
          f"attack_ratio={attack_ratio:.4f}")
    print(f"    RL: {'Enabled' if use_rl else 'Disabled'}")
    print(f"    Strategy: TD/TIA=Batch Adaptive, Katz/RL=Fully Adaptive")

    heuristic_tasks, rl_tasks = [], []
    for target_ratio in target_ratio_values:
        for _ in range(simulation_times):
            sim_seed = int(rng.randint(0, int(1e9)))
            for method in heuristic_methods:
                heuristic_tasks.append((
                    graph_type, n_nodes, graph_params, target_dist, target_ratio,
                    method, move_prob, attack_ratio, sim_seed
                ))
            if use_rl:
                rl_tasks.append((
                    graph_type, n_nodes, graph_params, target_dist, target_ratio,
                    move_prob, attack_ratio, sim_seed
                ))

    print(f"  Running {len(heuristic_tasks)} heuristic tasks "
          f"with {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(
                executor.map(_run_heuristic_task, heuristic_tasks),
                desc=f"{graph_type}-{target_dist} (Heuristic)",
                total=len(heuristic_tasks)):
            tr = result['target_ratio']
            results[result['method']][tr]['pc'].append(result['pc'])
            results[result['method']][tr]['anc'].append(result['anc'])
            results[result['method']][tr]['success'].append(float(result['success']))

    if use_rl and rl_tasks:
        print(f"  Running {len(rl_tasks)} RL tasks "
              f"with {n_workers_rl} workers...")
        with ProcessPoolExecutor(
                max_workers=n_workers_rl,
                initializer=_init_rl_worker,
                initargs=(model_path,)
        ) as executor:
            for result in tqdm(
                    executor.map(_run_rl_task, rl_tasks),
                    desc=f"{graph_type}-{target_dist} (GNN-RL)",
                    total=len(rl_tasks)):
                if result['pc'] is not None:
                    tr = result['target_ratio']
                    results['GNN-RL'][tr]['pc'].append(result['pc'])
                    results['GNN-RL'][tr]['anc'].append(result['anc'])
                    results['GNN-RL'][tr]['success'].append(float(result['success']))

    return {m: dict(v) for m, v in results.items()}



def save_results_to_csv(
        all_results: Dict,
        target_ratio_values: List[float],
        n_nodes: int,
        lambda_eff: float,
        save_dir: str,
) -> str:
    rows = []
    for config, res in all_results.items():
        for method, tr_dict in res.items():
            for tr in sorted(target_ratio_values):
                pc_vals  = tr_dict.get(tr, {}).get('pc', [])
                anc_vals = tr_dict.get(tr, {}).get('anc', [])
                suc_vals = tr_dict.get(tr, {}).get('success', [])
                if not pc_vals:
                    continue
                n_runs = len(pc_vals)
                rows.append({
                    'config':       config,
                    'method':       method,
                    'target_ratio': tr,
                    'n_runs':       n_runs,
                    'pc_mean':      float(np.mean(pc_vals)),
                    'pc_std':       float(np.std(pc_vals)),
                    'pc_sem':       float(np.std(pc_vals) / np.sqrt(n_runs)),
                    'anc_mean':     float(np.mean(anc_vals)),
                    'anc_std':      float(np.std(anc_vals)),
                    'anc_sem':      float(np.std(anc_vals) / np.sqrt(n_runs)),
                    'success_rate': float(np.mean(suc_vals)),
                })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(
        save_dir, f'results_target_ratio_n{n_nodes}_lambda{lambda_eff}_generalization.csv'
    )
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}  "
          f"({df.shape[0]} rows × {df.shape[1]} cols)")
    return csv_path



def plot_target_ratio_comparison(
        all_results: Dict,
        target_ratio_values: List[float],
        n_nodes: int,
        lambda_eff: float,
        save_dir: Optional[str] = None,
        show_figure: bool = False,
):
    configs       = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
    config_labels = ['BA-Random', 'BA-Localized', 'WS-Random', 'WS-Localized']

    colors  = {'Adaptive TD':   '#0072B2',
               'Adaptive Katz': '#D55E00',
               'Adaptive TIA':  '#009E73',
               'GNN-RL':        '#CC79A7'}
    markers = {'Adaptive TD': 'o', 'Adaptive Katz': 's',
               'Adaptive TIA': '^', 'GNN-RL': 'D'}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    tr_pct = [tr * 100 for tr in target_ratio_values]

    for col, config in enumerate(configs):
        ax_pc  = axes[0, col]
        ax_anc = axes[1, col]

        if config not in all_results:
            ax_pc.set_visible(False); ax_anc.set_visible(False)
            continue

        res = all_results[config]

        for metric, ax, ylabel in [('pc', ax_pc, 'PC'), ('anc', ax_anc, 'ANC')]:
            for method in res:
                xs, means, errs = [], [], []
                for tr in sorted(target_ratio_values):
                    vals = res[method].get(tr, {}).get(metric, [])
                    if vals:
                        xs.append(tr * 100)
                        means.append(np.mean(vals))
                        errs.append(np.std(vals) / np.sqrt(len(vals)))  # SEM
                if xs:
                    m = np.array(means); e = np.array(errs)
                    ax.plot(xs, m,
                            color=colors.get(method, '#000'),
                            marker=markers.get(method, 'o'),
                            label=method, linewidth=2, markersize=6)
                    ax.fill_between(xs, m - e, m + e,
                                    alpha=0.15,
                                    color=colors.get(method, '#000'))

            ax.set_xlabel('Target Ratio (%)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            if col == 0:
                ax.set_title(config_labels[col], fontsize=13, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
            else:
                ax.set_title(config_labels[col], fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle(
        f'Target Ratio Sensitivity | N={n_nodes}, λ_eff={lambda_eff}  '
        f'[Shaded band = ±1 SEM]',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(
            save_dir,
            f'target_ratio_sensitivity_n{n_nodes}_lambda{lambda_eff}.png'
        )
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {fig_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig



def comprehensive_target_ratio_analysis(
        n_nodes: int = 1024,
        lambda_eff: float = 10.0,
        target_ratio_values: List[float] = None,
        simulation_times: int = 100,
        model_path: Optional[str] = None,
        ba_m_range: Tuple[int, int] = (2, 4),
        ws_k_range: Tuple[int, int] = (4, 8),
        ws_beta_range: Tuple[float, float] = (0.1, 0.3),
        save_dir: Optional[str] = None,
        show_figure: bool = False,
        seed: int = 42,
        n_workers: int = None,
        n_workers_rl: int = None,
) -> Dict:

    if target_ratio_values is None:
        target_ratio_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    graph_params = {
        'ba_m_range':    ba_m_range,
        'ws_k_range':    ws_k_range,
        'ws_beta_range': ws_beta_range,
    }
    configs    = [('BA', 'random'), ('BA', 'localized'),
                  ('WS', 'random'), ('WS', 'localized')]
    all_results = {}

    print("=" * 70)
    print("Target Ratio Sensitivity Analysis")
    print("=" * 70)
    print(f"  N_NODES        : {n_nodes}")
    print(f"  LAMBDA_EFF     : {lambda_eff}")
    print(f"  TARGET_RATIOS  : {[f'{tr:.0%}' for tr in target_ratio_values]}")
    print(f"  SIMULATIONS    : {simulation_times}")
    print(f"  WORKERS        : {n_workers} (heuristic), {n_workers_rl} (RL)")
    print("=" * 70)

    t0 = time.time()
    for graph_type, target_dist in configs:
        config_name = f"{graph_type}-{target_dist}"
        print(f"\n{'=' * 60}")
        print(f"Running: {config_name}")
        print(f"{'=' * 60}")
        all_results[config_name] = run_target_ratio_experiments(
            graph_type=graph_type,
            target_dist=target_dist,
            n_nodes=n_nodes,
            lambda_eff=lambda_eff,
            target_ratio_values=target_ratio_values,
            simulation_times=simulation_times,
            graph_params=graph_params,
            model_path=model_path,
            seed=seed,
            n_workers=n_workers,
            n_workers_rl=n_workers_rl,
        )

    print(f"\nAll configs done in {(time.time() - t0) / 60:.1f} min")

    if save_dir:
        save_results_to_csv(
            all_results, target_ratio_values, n_nodes, lambda_eff, save_dir
        )

    plot_target_ratio_comparison(
        all_results, target_ratio_values, n_nodes, lambda_eff,
        save_dir, show_figure
    )

    print("\n" + "=" * 70)
    print("Target Ratio Analysis Completed!")
    print("=" * 70)
    return all_results



if __name__ == "__main__":
    N_NODES      = 1024
    LAMBDA_EFF   = 20.0

    # TARGET_RATIO_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05,0.06, 0.07, 0.08, 0.09, 0.10]
    # TARGET_RATIO_VALUES = [round(i * 0.02, 2) for i in range(1, 11)]
    TARGET_RATIO_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                           0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]

    SIMULATION_TIMES = 100

    MODEL_PATH = os.path.join(_REPO_ROOT, 'train',
                              'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                              'model.pt')

    SAVE_DIR = os.path.join(_FIG_DIR, 'data')

    N_WORKERS    = 60 if sys.platform.startswith('win') else 63
    N_WORKERS_RL = 60 if sys.platform.startswith('win') else 63

    results = comprehensive_target_ratio_analysis(
        n_nodes=N_NODES,
        lambda_eff=LAMBDA_EFF,
        target_ratio_values=TARGET_RATIO_VALUES,
        simulation_times=SIMULATION_TIMES,
        model_path=MODEL_PATH,
        save_dir=SAVE_DIR,
        show_figure=False,
        seed=42,
        n_workers=N_WORKERS,
        n_workers_rl=N_WORKERS_RL,
    )