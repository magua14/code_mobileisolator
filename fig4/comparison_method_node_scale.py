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
from concurrent.futures import ProcessPoolExecutor
import warnings
import torch

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_FIG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_FIG_DIR, '..'))
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


N_NODES_DEFAULT: List[int] = [2 ** i for i in range(5, 15)]

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
            'method': method, 'n_nodes': n_nodes}


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
        return {'pc': None, 'anc': None, 'success': False, 'n_nodes': n_nodes}

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
    return {'pc': pc, 'anc': anc, 'success': success, 'n_nodes': n_nodes}



def run_node_scale_experiments(
        graph_type: str,
        target_dist: str,
        n_nodes_values: List[int],
        target_ratio: float,
        lambda_eff: float,
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
    print(f"    target_ratio={target_ratio:.2%}, λ_eff={lambda_eff}")
    print(f"    → move_prob={move_prob:.3f}, attack_ratio={attack_ratio:.4f}")
    print(f"    RL: {'Enabled' if use_rl else 'Disabled'}")
    print(f"    Strategy: TD/TIA=Batch Adaptive, Katz/RL=Fully Adaptive")
    print(f"    N sequence: {n_nodes_values}")

    heuristic_tasks, rl_tasks = [], []
    for n_nodes in n_nodes_values:
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
            n = result['n_nodes']
            results[result['method']][n]['pc'].append(result['pc'])
            results[result['method']][n]['anc'].append(result['anc'])
            results[result['method']][n]['success'].append(float(result['success']))

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
                    n = result['n_nodes']
                    results['GNN-RL'][n]['pc'].append(result['pc'])
                    results['GNN-RL'][n]['anc'].append(result['anc'])
                    results['GNN-RL'][n]['success'].append(float(result['success']))

    return {m: dict(v) for m, v in results.items()}



def compute_relative_improvement(
        all_results: Dict,
        n_nodes_values: List[int],
) -> Dict:
    heuristic_methods = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA']
    improvement_results = {}

    for config, res in all_results.items():
        if 'GNN-RL' not in res:
            continue

        improvement_results[config] = {}
        for heuristic in heuristic_methods:
            pc_imp, anc_imp = {}, {}
            for n in n_nodes_values:
                rl_pc  = res['GNN-RL'].get(n, {}).get('pc', [])
                h_pc   = res[heuristic].get(n, {}).get('pc', [])
                rl_anc = res['GNN-RL'].get(n, {}).get('anc', [])
                h_anc  = res[heuristic].get(n, {}).get('anc', [])

                if rl_pc and h_pc:
                    h_mean = np.mean(h_pc)
                    pc_imp[n] = (h_mean - np.mean(rl_pc)) / h_mean * 100 \
                                if h_mean > 0 else 0.0

                if rl_anc and h_anc:
                    h_mean = np.mean(h_anc)
                    anc_imp[n] = (np.mean(rl_anc) - h_mean) / h_mean * 100 \
                                 if h_mean > 0 else 0.0

            improvement_results[config][f'vs_{heuristic}'] = {
                'pc_improvement':  pc_imp,
                'anc_improvement': anc_imp,
            }

    return improvement_results



def save_results_to_csv(
        all_results: Dict,
        n_nodes_values: List[int],
        target_ratio: float,
        lambda_eff: float,
        save_dir: str,
) -> str:
    rows = []
    for config, res in all_results.items():
        for method, n_dict in res.items():
            for n in sorted(n_nodes_values):
                pc_vals  = n_dict.get(n, {}).get('pc', [])
                anc_vals = n_dict.get(n, {}).get('anc', [])
                suc_vals = n_dict.get(n, {}).get('success', [])
                if not pc_vals:
                    continue
                n_runs = len(pc_vals)
                rows.append({
                    'config':       config,
                    'method':       method,
                    'n_nodes':      n,
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
        save_dir,
        f'results_node_scale_tr{int(target_ratio * 100)}_lambda{lambda_eff}.csv'
    )
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}  "
          f"({df.shape[0]} rows × {df.shape[1]} cols)")
    return csv_path


def save_improvement_to_csv(
        improvement_results: Dict,
        n_nodes_values: List[int],
        target_ratio: float,
        lambda_eff: float,
        save_dir: str,
) -> str:
    rows = []
    for config, comp_dict in improvement_results.items():
        for comparison, data in comp_dict.items():
            for n in sorted(n_nodes_values):
                rows.append({
                    'config':          config,
                    'comparison':      comparison,
                    'n_nodes':         n,
                    'pc_improvement':  data['pc_improvement'].get(n, float('nan')),
                    'anc_improvement': data['anc_improvement'].get(n, float('nan')),
                })

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(
        save_dir,
        f'improvement_node_scale_tr{int(target_ratio * 100)}_lambda{lambda_eff}.csv'
    )
    df.to_csv(csv_path, index=False)
    print(f"  Improvement CSV saved: {csv_path}  "
          f"({df.shape[0]} rows × {df.shape[1]} cols)")
    return csv_path


def plot_node_scale_comparison(
        all_results: Dict,
        n_nodes_values: List[int],
        target_ratio: float,
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

    for col, config in enumerate(configs):
        ax_pc  = axes[0, col]
        ax_anc = axes[1, col]

        if config not in all_results:
            ax_pc.set_visible(False); ax_anc.set_visible(False)
            continue

        res = all_results[config]

        for metric, ax, ylabel in [
            ('pc',  ax_pc,  'PC (Lower is Better)'),
            ('anc', ax_anc, 'ANC (Higher is Better)'),
        ]:
            for method in res:
                xs, means, errs = [], [], []
                for n in sorted(n_nodes_values):
                    vals = res[method].get(n, {}).get(metric, [])
                    if vals:
                        xs.append(n)
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

            ax.set_xscale('log', base=2)
            ax.set_xticks(n_nodes_values)
            ax.set_xticklabels([str(n) for n in n_nodes_values],
                               rotation=45, ha='right', fontsize=8)
            ax.set_xlabel('Number of Nodes (N)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            if col == 0:
                ax.set_title(config_labels[col], fontsize=13, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
            else:
                ax.set_title(config_labels[col], fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', which='both')

    fig.suptitle(
        f'Node Scale Sensitivity | Target Ratio={target_ratio:.0%}, '
        f'λ_eff={lambda_eff}  [Shaded band = ±1 SEM]',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(
            save_dir,
            f'node_scale_sensitivity_tr{int(target_ratio * 100)}_lambda{lambda_eff}.png'
        )
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {fig_path}")

    plt.show() if show_figure else plt.close(fig)
    return fig



def plot_relative_improvement(
        improvement_results: Dict,
        n_nodes_values: List[int],
        target_ratio: float,
        lambda_eff: float,
        save_dir: Optional[str] = None,
        show_figure: bool = False,
):
    configs       = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
    config_labels = ['BA-Random', 'BA-Localized', 'WS-Random', 'WS-Localized']

    colors  = {'vs_Adaptive TD':   '#0072B2',
               'vs_Adaptive Katz': '#D55E00',
               'vs_Adaptive TIA':  '#009E73'}
    markers = {'vs_Adaptive TD': 'o', 'vs_Adaptive Katz': 's', 'vs_Adaptive TIA': '^'}
    labels  = {'vs_Adaptive TD':   'vs Adaptive TD',
               'vs_Adaptive Katz': 'vs Adaptive Katz',
               'vs_Adaptive TIA':  'vs Adaptive TIA'}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, config in enumerate(configs):
        ax_pc  = axes[0, col]
        ax_anc = axes[1, col]

        if config not in improvement_results:
            ax_pc.set_visible(False); ax_anc.set_visible(False)
            continue

        res = improvement_results[config]

        for comparison in res:
            nodes = sorted(res[comparison]['pc_improvement'].keys())
            imps  = [res[comparison]['pc_improvement'][n] for n in nodes]
            if nodes:
                ax_pc.plot(nodes, imps,
                           color=colors.get(comparison, '#000'),
                           marker=markers.get(comparison, 'o'),
                           label=labels.get(comparison, comparison),
                           linewidth=2, markersize=6)

            nodes = sorted(res[comparison]['anc_improvement'].keys())
            imps  = [res[comparison]['anc_improvement'][n] for n in nodes]
            if nodes:
                ax_anc.plot(nodes, imps,
                            color=colors.get(comparison, '#000'),
                            marker=markers.get(comparison, 'o'),
                            label=labels.get(comparison, comparison),
                            linewidth=2, markersize=6)

        for ax, ylabel in [
            (ax_pc,  'PC Improvement (%)'),
            (ax_anc, 'ANC Improvement (%)'),
        ]:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xscale('log', base=2)
            ax.set_xticks(n_nodes_values)
            ax.set_xticklabels([str(n) for n in n_nodes_values],
                               rotation=45, ha='right', fontsize=8)
            ax.set_xlabel('Number of Nodes (N)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--', which='both')
            if col == 0:
                ax.set_title(config_labels[col], fontsize=13, fontweight='bold')
                ax.legend(loc='best', fontsize=9)
            else:
                ax.set_title(config_labels[col], fontsize=13, fontweight='bold')

    fig.suptitle(
        f'GNN-RL Relative Improvement | '
        f'Target Ratio={target_ratio:.0%}, λ_eff={lambda_eff}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(
            save_dir,
            f'rl_improvement_tr{int(target_ratio * 100)}_lambda{lambda_eff}.png'
        )
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved: {fig_path}")

    plt.show() if show_figure else plt.close(fig)
    return fig


def comprehensive_node_scale_analysis(
        n_nodes_values: List[int] = None,
        target_ratio: float = 0.05,
        lambda_eff: float = 10.0,
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
) -> Tuple[Dict, Dict]:

    if n_nodes_values is None:
        n_nodes_values = N_NODES_DEFAULT
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers_rl is None:
        n_workers_rl = min(8, max(1, mp.cpu_count() // 2))

    graph_params = {
        'ba_m_range':    ba_m_range,
        'ws_k_range':    ws_k_range,
        'ws_beta_range': ws_beta_range,
    }
    configs = [('BA', 'random'), ('BA', 'localized'),
               ('WS', 'random'), ('WS', 'localized')]
    all_results = {}

    print("=" * 70)
    print("Node Scale Sensitivity Analysis (Generalization Test)")
    print("=" * 70)
    print(f"  TARGET_RATIO   : {target_ratio:.2%}")
    print(f"  LAMBDA_EFF     : {lambda_eff}")
    print(f"  N_NODES_VALUES : {n_nodes_values}")
    print(f"  SIMULATIONS    : {simulation_times}")
    print(f"  WORKERS        : {n_workers} (heuristic), {n_workers_rl} (RL)")
    print("=" * 70)

    t0 = time.time()
    for graph_type, target_dist in configs:
        config_name = f"{graph_type}-{target_dist}"
        print(f"\n{'=' * 60}")
        print(f"Running: {config_name}")
        print(f"{'=' * 60}")
        all_results[config_name] = run_node_scale_experiments(
            graph_type=graph_type,
            target_dist=target_dist,
            n_nodes_values=n_nodes_values,
            target_ratio=target_ratio,
            lambda_eff=lambda_eff,
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
            all_results, n_nodes_values, target_ratio, lambda_eff, save_dir
        )

    plot_node_scale_comparison(
        all_results, n_nodes_values, target_ratio, lambda_eff,
        save_dir, show_figure
    )

    improvement_results = compute_relative_improvement(all_results, n_nodes_values)

    if save_dir:
        save_improvement_to_csv(
            improvement_results, n_nodes_values, target_ratio, lambda_eff, save_dir
        )

    plot_relative_improvement(
        improvement_results, n_nodes_values, target_ratio, lambda_eff,
        save_dir, show_figure
    )

    print("\n" + "=" * 70)
    print("GNN-RL Improvement Summary (PC, lower is better)")
    print("=" * 70)
    for config in improvement_results:
        print(f"\n{config}:")
        for comparison, data in improvement_results[config].items():
            imp_vals = list(data['pc_improvement'].values())
            if imp_vals:
                print(f"  {comparison}: {np.mean(imp_vals):+.2f}% average improvement")

    print("\n" + "=" * 70)
    print("Node Scale Analysis Completed!")
    print("=" * 70)

    return all_results, improvement_results


if __name__ == "__main__":
    N_NODES_VALUES = [2 ** i for i in range(5, 15)]
    # = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    TARGET_RATIO = 0.05
    LAMBDA_EFF   = 20.0

    SIMULATION_TIMES = 100

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    MODEL_PATH = os.path.join(REPO_ROOT, 'train',
                              'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                              'model.pt')
    SAVE_DIR   = os.path.join(SCRIPT_DIR, 'data')

    N_WORKERS    = 78
    N_WORKERS_RL = 78

    results, improvements = comprehensive_node_scale_analysis(
        n_nodes_values=N_NODES_VALUES,
        target_ratio=TARGET_RATIO,
        lambda_eff=LAMBDA_EFF,
        simulation_times=SIMULATION_TIMES,
        model_path=MODEL_PATH,
        save_dir=SAVE_DIR,
        show_figure=False,
        seed=42,
        n_workers=N_WORKERS,
        n_workers_rl=N_WORKERS_RL,
    )