import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'train'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'fig3'))

from comparison_method import (
    generate_graph,
    generate_random_targets,
    generate_localized_targets,
    simulate_heuristic,
    simulate_rl,
    td_method_adaptive,
    katz_method_adaptive,
    extended_tia_method_adaptive,
    RLAgentWrapper,
    GNN_AVAILABLE,
)


LAMBDA_VALUES = [1, 5, 10, 20]
R_VALUES = [round(i * 0.01, 4) for i in range(1, 6)]

N_NODES      = 1000
TARGET_RATIO = 0.10
SIM_TIMES    = 500
SEED         = 42

MODEL_PATH = os.path.join(_REPO_ROOT, 'train',
                          'gnn_ppo_dual_stream_n50-100__seed42__1769653775',
                          'model.pt')
SAVE_DIR   = os.path.join(_SCRIPT_DIR, 'data')

N_WORKERS    = 33
N_WORKERS_RL = 33

GRAPH_PARAMS = {
    'ba_m_range':    [4, 4],
    'ws_k_range':    [8, 8],
    'ws_beta_range': [0.1, 0.1],
}

CONFIGS = [
    ('BA', 'random'),
    ('BA', 'localized'),
    ('WS', 'random'),
    ('WS', 'localized'),
]

HEURISTIC_METHODS = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA']
METHOD_ORDER      = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']


mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.size':         6,
    'axes.labelsize':    7,
    'axes.titlesize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'legend.fontsize':   6,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size':  2.5, 'ytick.major.size':  2.5,
    'xtick.direction':   'in', 'ytick.direction': 'in',
    'figure.dpi':        300, 'savefig.dpi': 300,
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


_rl_agent_worker = None


def _init_rl_worker(model_path: str):
    global _rl_agent_worker
    import torch
    torch.set_num_threads(1)
    if model_path and os.path.exists(model_path) and GNN_AVAILABLE:
        try:
            _rl_agent_worker = RLAgentWrapper(model_path, device='cpu')
        except Exception as e:
            print(f"  [RL worker] load failed: {e}")
            _rl_agent_worker = None
    else:
        _rl_agent_worker = None



def _run_heuristic_task(args):
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     method, move_prob, attack_ratio, graph_seed, game_seed) = args

    G = generate_graph(graph_type, n_nodes, graph_params, graph_seed)
    if target_dist == 'random':
        targets = generate_random_targets(G, target_ratio, graph_seed)
    else:
        targets = generate_localized_targets(G, target_ratio, graph_seed)

    n_targets       = len(targets)
    attack_per_step = max(1, round(attack_ratio * n_targets))

    if method == 'Adaptive TD':
        func, batch = td_method_adaptive, True
    elif method == 'Adaptive TIA':
        func, batch = extended_tia_method_adaptive, True
    else:
        func, batch = katz_method_adaptive, False

    pc, anc, _ = simulate_heuristic(
        G, targets, move_prob, attack_per_step, func, game_seed, batch)

    return {'method': method, 'move_prob': move_prob,
            'attack_ratio': attack_ratio, 'pc': pc, 'anc': anc}


def _run_rl_task(args):
    global _rl_agent_worker
    (graph_type, n_nodes, graph_params, target_dist, target_ratio,
     move_prob, attack_ratio, graph_seed, game_seed) = args

    if _rl_agent_worker is None:
        return {'method': 'GNN-RL', 'move_prob': move_prob,
                'attack_ratio': attack_ratio, 'pc': None, 'anc': None}

    G = generate_graph(graph_type, n_nodes, graph_params, graph_seed)
    if target_dist == 'random':
        targets = generate_random_targets(G, target_ratio, graph_seed)
    else:
        targets = generate_localized_targets(G, target_ratio, graph_seed)

    n_targets       = len(targets)
    attack_per_step = max(1, round(attack_ratio * n_targets))

    pc, anc, _ = simulate_rl(
        G, targets, _rl_agent_worker, move_prob, attack_per_step, game_seed)

    return {'method': 'GNN-RL', 'move_prob': move_prob,
            'attack_ratio': attack_ratio, 'pc': pc, 'anc': anc}



def run_validity_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)

    use_rl = bool(MODEL_PATH and os.path.exists(MODEL_PATH) and GNN_AVAILABLE)
    active_methods = HEURISTIC_METHODS + (['GNN-RL'] if use_rl else [])

    all_results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {'pc': [], 'anc': []}))))

    rng      = np.random.RandomState(SEED)
    csv_rows = []

    for lam in LAMBDA_VALUES:
        print(f"\n{'=' * 60}")
        print(f"  λ_eff = {lam}   (p = λ × r, valid r where p ≤ 1)")
        print(f"{'=' * 60}")

        for graph_type, target_dist in CONFIGS:
            config_key = f"{graph_type}-{target_dist}"
            print(f"\n  Config: {config_key}")

            heuristic_tasks = []
            rl_tasks        = []

            valid_r = [(r, lam * r) for r in R_VALUES
                       if 0 < lam * r <= 1.0]
            print(f"    Valid (r, p) pairs: {[(r, round(p,4)) for r, p in valid_r]}")

            graph_seeds = [int(rng.randint(0, int(1e9))) for _ in range(SIM_TIMES)]

            for r_val, p in valid_r:
                for graph_seed in graph_seeds:
                    game_seed = int(rng.randint(0, int(1e9)))
                    for method in HEURISTIC_METHODS:
                        heuristic_tasks.append((
                            graph_type, N_NODES, GRAPH_PARAMS,
                            target_dist, TARGET_RATIO,
                            method, p, r_val, graph_seed, game_seed
                        ))
                    if use_rl:
                        rl_tasks.append((
                            graph_type, N_NODES, GRAPH_PARAMS,
                            target_dist, TARGET_RATIO,
                            p, r_val, graph_seed, game_seed
                        ))

            print(f"    Heuristic tasks: {len(heuristic_tasks)}")
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                for res in tqdm(
                        ex.map(_run_heuristic_task, heuristic_tasks),
                        desc=f"  {config_key} heuristic",
                        total=len(heuristic_tasks)):
                    r_key = round(res['attack_ratio'], 10)
                    m     = res['method']
                    if res['pc'] is not None:
                        all_results[lam][config_key][m][r_key]['pc'].append(res['pc'])
                        all_results[lam][config_key][m][r_key]['anc'].append(res['anc'])

            if use_rl and rl_tasks:
                print(f"    RL tasks: {len(rl_tasks)}")
                with ProcessPoolExecutor(
                        max_workers=N_WORKERS_RL,
                        initializer=_init_rl_worker,
                        initargs=(MODEL_PATH,)) as ex:
                    for res in tqdm(
                            ex.map(_run_rl_task, rl_tasks),
                            desc=f"  {config_key} GNN-RL",
                            total=len(rl_tasks)):
                        r_key = round(res['attack_ratio'], 10)
                        if res['pc'] is not None:
                            all_results[lam][config_key]['GNN-RL'][r_key]['pc'].append(res['pc'])
                            all_results[lam][config_key]['GNN-RL'][r_key]['anc'].append(res['anc'])

            valid_r_csv = [(r, lam * r) for r in R_VALUES
                           if 0 < lam * r <= 1.0]
            for method in active_methods:
                for r_val, p in valid_r_csv:
                    r_key = round(r_val, 10)
                    d     = all_results[lam][config_key][method][r_key]
                    n     = len(d['pc'])
                    if n == 0:
                        continue
                    pc_arr  = np.array(d['pc'],  dtype=float)
                    anc_arr = np.array(d['anc'], dtype=float)
                    csv_rows.append({
                        'lambda_eff':   lam,
                        'config':       config_key,
                        'method':       method,
                        'move_prob':    p,
                        'attack_ratio': r_val,
                        'n_sims':       n,
                        'pc_mean':      float(pc_arr.mean()),
                        'pc_std':       float(pc_arr.std()),
                        'pc_sem':       float(pc_arr.std() / np.sqrt(n)),
                        'anc_mean':     float(anc_arr.mean()),
                        'anc_std':      float(anc_arr.std()),
                        'anc_sem':      float(anc_arr.std() / np.sqrt(n)),
                    })

    df       = pd.DataFrame(csv_rows)
    csv_path = os.path.join(SAVE_DIR, 'lambda_validity_results_500.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}  ({len(df)} rows)")
    return all_results, df



def plot_validity(all_results, df, save_dir):
    n_rows = len(METHOD_ORDER)
    n_cols = len(CONFIGS)
    col_titles = [f"{g}-{d.capitalize()}" for g, d in CONFIGS]
    row_labels  = [m.replace('Adaptive ', '') for m in METHOD_ORDER]

    FIG_W = 7.087
    PAN_H = 1.25
    LEG_H = 0.28
    FIG_H = n_rows * PAN_H + LEG_H + 0.20

    for lam in LAMBDA_VALUES:
        lam_df = df[df['lambda_eff'] == lam]

        for metric, ylabel in [('pc', r'$P_C$'), ('anc', 'ANC')]:
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(FIG_W, FIG_H),
                gridspec_kw={
                    'hspace': 0.55, 'wspace': 0.40,
                    'left': 0.10, 'right': 0.99,
                    'top':  0.93, 'bottom': 0.11,
                },
            )

            for row_i, method in enumerate(METHOD_ORDER):
                for col_i, (graph_type, target_dist) in enumerate(CONFIGS):
                    config_key = f"{graph_type}-{target_dist}"
                    ax = axes[row_i][col_i]

                    sub = lam_df[
                        (lam_df['config']  == config_key) &
                        (lam_df['method']  == method)
                    ].sort_values('attack_ratio')

                    if not sub.empty:
                        xs   = sub['attack_ratio'].values
                        ys   = sub[f'{metric}_mean'].values
                        errs = sub[f'{metric}_sem'].values
                        c    = METHOD_COLORS.get(method, '#333333')
                        mk   = METHOD_MARKERS.get(method, 'o')

                        ax.plot(xs, ys,
                                color=c, marker=mk, markersize=2.5,
                                linewidth=0.8,
                                markeredgecolor='white', markeredgewidth=0.3,
                                zorder=3)
                        ax.fill_between(xs, ys - errs, ys + errs,
                                        color=c, alpha=0.15,
                                        linewidth=0, zorder=2)

                    ax.set_xlim(0.005, 0.055)
                    ax.set_xticks([0.01, 0.02, 0.03, 0.04, 0.05])
                    ax.tick_params(which='both', direction='in',
                                   top=True, right=True,
                                   labelsize=6, pad=2)
                    ax.yaxis.set_major_locator(
                        mticker.MaxNLocator(nbins=4, min_n_ticks=3))
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.5)

                    if row_i == 0:
                        ax.set_title(col_titles[col_i], fontsize=7, pad=3)

                    if row_i == n_rows - 1:
                        ax.set_xlabel('$r$', fontsize=7, labelpad=2)

                    if col_i == 0:
                        ax.set_ylabel(ylabel, fontsize=7, labelpad=3)

                        ax.annotate(
                            row_labels[row_i],
                            xy=(-0.30, 0.5), xycoords='axes fraction',
                            fontsize=6, fontweight='bold',
                            ha='right', va='center', rotation=90,
                        )

            fig.suptitle(
                f'λ_eff = {lam}   {ylabel} — collapse validation',
                fontsize=8, fontweight='bold', y=0.975,
            )


            handles = [
                plt.Line2D([], [],
                           color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                           markersize=4, linewidth=0.8,
                           markeredgecolor='white', markeredgewidth=0.3,
                           label=m.replace('Adaptive ', ''))
                for m in METHOD_ORDER if m in METHOD_COLORS
            ]
            fig.legend(
                handles=handles,
                loc='lower center', ncol=len(handles),
                bbox_to_anchor=(0.5, 0.01),
                frameon=False, fontsize=6,
                handlelength=1.6, handletextpad=0.4,
                columnspacing=1.0,
            )

            stem = os.path.join(save_dir, f'lambda{lam}_{metric}_collapse')
            fig.savefig(stem + '.pdf', dpi=300)
            fig.savefig(stem + '.png', dpi=300)
            print(f"  Saved: {stem}.pdf / .png")
            plt.close(fig)



if __name__ == '__main__':
    use_rl = bool(MODEL_PATH and os.path.exists(MODEL_PATH) and GNN_AVAILABLE)

    print("λ_eff Collapse Validity Experiment")
    print("=" * 60)
    print(f"  λ values     : {LAMBDA_VALUES}")
    print(f"  r sweep      : {R_VALUES}")
    print(f"  p = λ × r  (skipped when p > 1)")
    print(f"  r = λ / p   (derived per point)")
    print(f"  N_nodes      : {N_NODES}")
    print(f"  Target ratio : {TARGET_RATIO}")
    print(f"  Sim times    : {SIM_TIMES}")
    print(f"  GNN-RL       : {'Enabled' if use_rl else 'Disabled'}")
    print(f"  Save dir     : {SAVE_DIR}")
    print("=" * 60)

    all_results, df = run_validity_experiment()
    plot_validity(all_results, df, SAVE_DIR)
    print("\nDone.")