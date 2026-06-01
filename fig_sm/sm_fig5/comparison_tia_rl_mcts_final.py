import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch
import warnings

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
_REPO_ROOT = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))

try:
    from mcts_final_new import solve_with_improved_mcts
except ImportError:
    raise ImportError("mcts_final.py not found in the same directory.")

try:
    from comparison_method import (
        RLAgentWrapper,
        simulate_rl,
        simulate_heuristic,
        extended_tia_method_adaptive,
        generate_graph,
        generate_random_targets,
        generate_localized_targets,
        compute_params_from_lambda,
    )
except ImportError:
    raise ImportError("comparison_method.py not found in the same directory.")


class Config:
    N_NODES      = 50
    TARGET_RATIO = 0.05

    LAMBDA_LEVELS = [1,10,20]
    SIMULATION_TIMES = 100
    MCTS_SIMULATIONS = 20000
    TOP_K        = 47
    SUBTREE_REUSE      = False
    USE_DETERMINIZATION = False

    MODEL_PATH = os.path.join(_REPO_ROOT, 'train',
                              'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                              'model.pt')
    SAVE_DIR   = os.path.join(current_dir, 'data')
    N_WORKERS  = 60 if sys.platform.startswith('win') else 77

    GRAPH_PARAMS = {
        'ba_m_range':   (4, 4),
        'ws_k_range':   (8, 8),
        'ws_beta_range': (0.1, 0.1),
    }


_worker_rl_agent = None


def _init_worker(model_path: str):
    global _worker_rl_agent
    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path):
        try:
            _worker_rl_agent = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            print(f"Warning: RL Agent load failed: {e}")
            _worker_rl_agent = None
    else:
        _worker_rl_agent = None


def run_single_comparison(args):
    graph_type, target_dist, lam, seed = args

    move_prob, attack_ratio = compute_params_from_lambda(lam)

    G = generate_graph(graph_type, Config.N_NODES, Config.GRAPH_PARAMS, seed)

    if target_dist == 'random':
        targets = generate_random_targets(G, Config.TARGET_RATIO, seed)
    else:
        targets = generate_localized_targets(G, Config.TARGET_RATIO, seed)

    n_targets = len(targets)
    attack_per_step = max(1, int(attack_ratio * n_targets))

    results = {}

    pc_tia, anc_tia, succ_tia = simulate_heuristic(
        G.copy(), list(targets), move_prob, attack_per_step,
        extended_tia_method_adaptive, seed, batch_mode=True
    )
    if not succ_tia:
        pc_tia = 1.0
    results['TIA'] = {'pc': pc_tia, 'anc': anc_tia}

    global _worker_rl_agent
    if _worker_rl_agent is not None:
        pc_rl, anc_rl, succ_rl = simulate_rl(
            G.copy(), list(targets), _worker_rl_agent, move_prob, attack_per_step, seed
        )
        if not succ_rl:
            pc_rl = 1.0
        results['GNN-RL'] = {'pc': pc_rl, 'anc': anc_rl}
    else:
        results['GNN-RL'] = {'pc': 1.0, 'anc': 0.0}

    try:
        mcts_out = solve_with_improved_mcts(
            G, list(targets), move_prob, attack_per_step,
            num_simulations=Config.MCTS_SIMULATIONS,
            use_progressive_widening=True,
            top_k=Config.TOP_K,
            subtree_reuse=Config.SUBTREE_REUSE,
            use_determinization=Config.USE_DETERMINIZATION,
            seed=seed,
            verbose=False,
        )
        pc_mcts = mcts_out['pc'] if mcts_out['success'] else 1.0
        results['MCTS'] = {'pc': pc_mcts, 'anc': mcts_out['anc']}
    except Exception as e:
        print(f"MCTS failed (seed={seed}, λ={lam}): {e}")
        results['MCTS'] = {'pc': 1.0, 'anc': 0.0}

    return {
        'config':  f"{graph_type}-{target_dist}",
        'lambda':  lam,
        'results': results,
    }


def save_results_to_csv(df: pd.DataFrame, save_dir: str) -> str:
    summary_rows = []
    for (config, lam, method), grp in df.groupby(['Config', 'Lambda', 'Method']):
        pc_vals  = grp['PC'].values
        anc_vals = grp['ANC'].values
        n = len(pc_vals)
        summary_rows.append({
            'config':       config,
            'method':       method,
            'lambda_eff':   lam,
            'n_runs':       n,
            'pc_mean':      float(np.mean(pc_vals)),
            'pc_std':       float(np.std(pc_vals)),
            'pc_sem':       float(np.std(pc_vals) / np.sqrt(n)),
            'anc_mean':     float(np.mean(anc_vals)),
            'anc_std':      float(np.std(anc_vals)),
            'anc_sem':      float(np.std(anc_vals) / np.sqrt(n)),
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir,
        f'results_mcts_n{Config.N_NODES}_sim{Config.MCTS_SIMULATIONS}.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Summary CSV saved: {csv_path}  ({summary_df.shape[0]} rows)")
    return csv_path


def plot_results(df: pd.DataFrame, save_dir: str):
    configs       = ['BA-random', 'BA-localized', 'WS-random', 'WS-localized']
    config_titles = ['BA-Random', 'BA-Localized', 'WS-Random', 'WS-Localized']
    methods       = ['MCTS', 'GNN-RL', 'TIA']

    colors  = {'MCTS': '#34495e', 'GNN-RL': '#e74c3c', 'TIA': '#bdc3c7'}
    markers = {'MCTS': 's',       'GNN-RL': 'D',        'TIA': 'o'}

    lambda_vals = sorted(df['Lambda'].unique())

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, (config, title) in enumerate(zip(configs, config_titles)):
        subset = df[df['Config'] == config]

        for row, metric in enumerate(['PC', 'ANC']):
            ax = axes[row, col]
            for method in methods:
                means, errs, lams = [], [], []
                for lam in lambda_vals:
                    vals = subset[(subset['Lambda'] == lam) &
                                  (subset['Method'] == method)][metric].values
                    if len(vals) == 0:
                        continue
                    lams.append(lam)
                    means.append(np.mean(vals))
                    errs.append(np.std(vals) / np.sqrt(len(vals)))

                if not lams:
                    continue
                lams_arr  = np.array(lams)
                means_arr = np.array(means)
                errs_arr  = np.array(errs)

                ax.plot(lams_arr, means_arr,
                        color=colors[method], marker=markers[method],
                        label=method, linewidth=2, markersize=6)
                ax.fill_between(lams_arr,
                                means_arr - errs_arr,
                                means_arr + errs_arr,
                                alpha=0.15, color=colors[method])

            if row == 0:
                ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel(r'$\lambda_{eff}$', fontsize=11)
            ax.set_ylabel('PC' if row == 0 else 'ANC', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            if col == 0:
                ax.legend(loc='best', fontsize=9)

    fig.suptitle(
        f'RL-Informed MCTS vs GNN-RL vs Adaptive TIA\n'
        f'N={Config.N_NODES}, Target Ratio={Config.TARGET_RATIO}, '
        f'MCTS Simulations={Config.MCTS_SIMULATIONS}  '
        f'[Shaded band = ±1 SEM]',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir,
        f'benchmark_mcts_n{Config.N_NODES}_sim{Config.MCTS_SIMULATIONS}.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figure saved: {fig_path}")


if __name__ == "__main__":
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    graph_configs = [
        ('BA', 'random'), ('BA', 'localized'),
        ('WS', 'random'), ('WS', 'localized'),
    ]
    tasks = []
    for graph_type, target_dist in graph_configs:
        for lam in Config.LAMBDA_LEVELS:
            for s in range(Config.SIMULATION_TIMES):
                seed = 42 + s * 1000 + int(lam)
                tasks.append((graph_type, target_dist, lam, seed))

    print(f"\nTotal tasks : {len(tasks)}")
    print(f"Workers     : {Config.N_WORKERS}")
    print(f"Lambda range: {Config.LAMBDA_LEVELS}")
    print(f"attack_per_step basis: n_targets")
    print("Starting parallel execution...\n")

    raw_rows = []
    with ProcessPoolExecutor(
        max_workers=Config.N_WORKERS,
        initializer=_init_worker,
        initargs=(Config.MODEL_PATH,)
    ) as executor:
        for res in tqdm(executor.map(run_single_comparison, tasks),
                        total=len(tasks), desc='Benchmarking'):
            for method, metrics in res['results'].items():
                raw_rows.append({
                    'Config': res['config'],
                    'Lambda': res['lambda'],
                    'Method': method,
                    'PC':     metrics['pc'],
                    'ANC':    metrics['anc'],
                })

    df = pd.DataFrame(raw_rows)
    raw_path = os.path.join(Config.SAVE_DIR,
        f'raw_data_mcts_n{Config.N_NODES}_sim{Config.MCTS_SIMULATIONS}.csv')
    df.to_csv(raw_path, index=False)
    print(f"\n✓ Raw data saved: {raw_path}")

    save_results_to_csv(df, Config.SAVE_DIR)
    plot_results(df, Config.SAVE_DIR)