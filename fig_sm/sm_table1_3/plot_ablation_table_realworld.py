import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, 'data', 'summary_ablation_realworld.csv')
SAVE_DIR   = SCRIPT_DIR
LAMBDA_TARGETS = [1, 10, 20]

mpl.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'figure.dpi':  300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

SCENARIOS       = ['covid19', 'flood', 'invasive', 'smuggling', 'socialbot', 'fugitive']
SCENARIO_LABELS = [
    'COVID-19',
    'Flood',
    'Invasive Species',
    'Smuggling',
    'Socialbot',
    'Fugitive Chase',
]
VARIANTS    = ['Full', 'No-Global', 'No-Attn', 'No-VNode']
VAR_LABELS  = ['Full', 'w/o Global', 'w/o Attn', 'w/o VNode']

FIG_W = 7.087
FIG_H = 2.80

SCEN_X     = 0.080
VAR_COL_W  = 0.10125
PC_START   = 0.16
SEP_X      = 0.585
ANC_START  = 0.595

PC_XC  = [PC_START  + VAR_COL_W * (i + 0.5) for i in range(4)]
ANC_XC = [ANC_START + VAR_COL_W * (i + 0.5) for i in range(4)]

Y_TOP_LINE   = 0.97
Y_GRP_HEAD   = 0.90
Y_COL_HEAD   = 0.81
Y_THICK_LINE = 0.76

Y_DATA       = [0.685, 0.605, 0.525, 0.445, 0.365, 0.285]

Y_THIN_LINE  = 0.205
Y_AVG        = 0.115
Y_BOT_LINE   = 0.05

FS_GRP  = 7
FS_COL  = 6
FS_DATA = 6
FS_MARK = 5.5

C_BEST  = '#000000'
C_OTHER = '#333333'
C_SEP   = '#888888'
C_LINE  = '#222222'
C_FULL_HEADER = '#9A4080'



def fmt(mean: float, sem: float) -> str:
    return f"{mean:.3f} ± {sem:.3f}"


def fmt_avg(mean: float) -> str:
    return f"{mean:.3f}"


def is_best(val: float, best_val: float, decimals: int = 3) -> bool:
    return round(val, decimals) == round(best_val, decimals)


def render_table(ax: plt.Axes, df_lam: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    lkw = dict(transform=ax.transAxes, clip_on=False)

    def hline(y, lw, ls='-'):
        ax.plot([0.01, 0.99], [y, y], color=C_LINE, lw=lw,
                linestyle=ls, **lkw)

    hline(Y_TOP_LINE,   lw=1.0)
    hline(Y_THICK_LINE, lw=0.8)
    hline(Y_THIN_LINE,  lw=0.5, ls=(0, (4, 3)))
    hline(Y_BOT_LINE,   lw=0.8)

    ax.plot([SEP_X, SEP_X],
            [Y_TOP_LINE, Y_BOT_LINE],
            color=C_SEP, lw=0.5, **lkw)

    pc_mid  = PC_START  + 4 * VAR_COL_W * 0.5
    anc_mid = ANC_START + 4 * VAR_COL_W * 0.5

    for x, txt in [(pc_mid, r'$P_C$'),
                   (anc_mid, r'ANC')]:
        ax.text(x, Y_GRP_HEAD, txt, ha='center', va='center',
                fontsize=FS_GRP, fontweight='bold', color='#111111')

    ax.text(SCEN_X, Y_COL_HEAD, 'Scenario',
            ha='center', va='center', fontsize=FS_COL, fontstyle='italic')

    for xc, lbl in zip(PC_XC + ANC_XC, VAR_LABELS * 2):
        is_full = (lbl == 'Full')
        ax.text(xc, Y_COL_HEAD, lbl,
                ha='center', va='center', fontsize=FS_COL,
                fontweight='bold' if is_full else 'normal',
                color=C_FULL_HEADER if is_full else '#333333')

    all_pc:  dict[str, list] = {v: [] for v in VARIANTS}
    all_anc: dict[str, list] = {v: [] for v in VARIANTS}

    for row_i, (scen, scen_lbl) in enumerate(zip(SCENARIOS, SCENARIO_LABELS)):
        y_row = Y_DATA[row_i]
        sub   = df_lam[df_lam['scenario'] == scen]

        pc_row:  dict[str, tuple] = {}
        anc_row: dict[str, tuple] = {}
        for v in VARIANTS:
            r = sub[sub['variant'] == v]
            if not r.empty:
                pc_row[v]  = (float(r['pc_mean'].values[0]),
                              float(r['pc_sem'].values[0]))
                anc_row[v] = (float(r['anc_mean'].values[0]),
                              float(r['anc_sem'].values[0]))
                all_pc[v].append(pc_row[v][0])
                all_anc[v].append(anc_row[v][0])

        best_pc  = min(v[0] for v in pc_row.values())  if pc_row  else None
        best_anc = max(v[0] for v in anc_row.values()) if anc_row else None

        ax.text(SCEN_X, y_row, scen_lbl,
                ha='center', va='center', fontsize=FS_DATA, color='#111111')

        for xc, v in zip(PC_XC, VARIANTS):
            if v not in pc_row:
                continue
            mean, sem = pc_row[v]
            bold  = is_best(mean, best_pc)
            ax.text(xc, y_row, fmt(mean, sem),
                    ha='center', va='center', fontsize=FS_DATA,
                    fontweight='bold' if bold else 'normal',
                    color=C_BEST if bold else C_OTHER)

        for xc, v in zip(ANC_XC, VARIANTS):
            if v not in anc_row:
                continue
            mean, sem = anc_row[v]
            bold = is_best(mean, best_anc)
            ax.text(xc, y_row, fmt(mean, sem),
                    ha='center', va='center', fontsize=FS_DATA,
                    fontweight='bold' if bold else 'normal',
                    color=C_BEST if bold else C_OTHER)

    ax.text(SCEN_X, Y_AVG, 'Average',
            ha='center', va='center', fontsize=FS_DATA,
            fontstyle='italic', color='#111111')

    avg_pc  = {v: np.mean(all_pc[v])  for v in VARIANTS if all_pc[v]}
    avg_anc = {v: np.mean(all_anc[v]) for v in VARIANTS if all_anc[v]}

    best_avg_pc  = min(avg_pc.values())  if avg_pc  else None
    best_avg_anc = max(avg_anc.values()) if avg_anc else None

    for xc, v in zip(PC_XC, VARIANTS):
        if v not in avg_pc:
            continue
        mean = avg_pc[v]
        bold = round(mean, 3) == round(best_avg_pc, 3)
        ax.text(xc, Y_AVG, fmt_avg(mean),
                ha='center', va='center', fontsize=FS_DATA,
                fontweight='bold' if bold else 'normal',
                color=C_BEST if bold else C_OTHER)

    for xc, v in zip(ANC_XC, VARIANTS):
        if v not in avg_anc:
            continue
        mean = avg_anc[v]
        bold = round(mean, 3) == round(best_avg_anc, 3)
        ax.text(xc, Y_AVG, fmt_avg(mean),
                ha='center', va='center', fontsize=FS_DATA,
                fontweight='bold' if bold else 'normal',
                color=C_BEST if bold else C_OTHER)



def make_latex(df: pd.DataFrame, save_dir: str) -> None:
    lines = []
    for lam in LAMBDA_TARGETS:
        df_lam = df[df['lambda_eff'] == lam]
        lines.append(f"% ── λ = {lam} (real-world) ─────────────────")
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(
            r"\caption{Real-world ablation study ($\lambda_{eff}="
            + str(lam)
            + r"$). Best value per row is \textbf{bold}. "
            r"Values: mean $\pm$ SEM over 10 trials.}"
        )
        lines.append(f"\\label{{tab:ablation_realworld_lambda{lam}}}")
        lines.append(r"\resizebox{\linewidth}{!}{%")
        lines.append(r"\begin{tabular}{lcccc|cccc}")
        lines.append(r"\toprule")
        lines.append(
            r" & \multicolumn{4}{c|}{$P_C$ ($\downarrow$)} "
            r"& \multicolumn{4}{c}{ANC ($\uparrow$)} \\"
        )
        lines.append(r"\cmidrule(lr){2-5}\cmidrule(l){6-9}")
        lines.append("Scenario & Full & w/o Global & w/o Attn & w/o VNode"
                     " & Full & w/o Global & w/o Attn & w/o VNode \\\\")
        lines.append(r"\midrule")

        all_pc:  dict = {v: [] for v in VARIANTS}
        all_anc: dict = {v: [] for v in VARIANTS}

        for scen, scen_lbl in zip(SCENARIOS, SCENARIO_LABELS):
            sub = df_lam[df_lam['scenario'] == scen]
            pc_row, anc_row = {}, {}
            for v in VARIANTS:
                r = sub[sub['variant'] == v]
                if not r.empty:
                    pc_row[v]  = (float(r['pc_mean'].values[0]),
                                  float(r['pc_sem'].values[0]))
                    anc_row[v] = (float(r['anc_mean'].values[0]),
                                  float(r['anc_sem'].values[0]))
                    all_pc[v].append(pc_row[v][0])
                    all_anc[v].append(anc_row[v][0])

            best_pc  = min(v[0] for v in pc_row.values())  if pc_row  else None
            best_anc = max(v[0] for v in anc_row.values()) if anc_row else None

            row_cells = [scen_lbl]
            for v in VARIANTS:
                if v in pc_row:
                    mean, sem = pc_row[v]
                    s = f"{mean:.3f} $\\pm$ {sem:.3f}"
                    if is_best(mean, best_pc):
                        s = f"\\textbf{{{s}}}"
                    row_cells.append(s)
                else:
                    row_cells.append("--")
            for v in VARIANTS:
                if v in anc_row:
                    mean, sem = anc_row[v]
                    s = f"{mean:.3f} $\\pm$ {sem:.3f}"
                    if is_best(mean, best_anc):
                        s = f"\\textbf{{{s}}}"
                    row_cells.append(s)
                else:
                    row_cells.append("--")
            lines.append(" & ".join(row_cells) + r" \\")

        lines.append(r"\midrule")

        avg_pc  = {v: np.mean(all_pc[v])  for v in VARIANTS if all_pc[v]}
        avg_anc = {v: np.mean(all_anc[v]) for v in VARIANTS if all_anc[v]}
        best_avg_pc  = min(avg_pc.values())  if avg_pc  else None
        best_avg_anc = max(avg_anc.values()) if avg_anc else None

        avg_cells = ["Average"]
        for v in VARIANTS:
            if v in avg_pc:
                s = f"{avg_pc[v]:.3f}"
                if is_best(avg_pc[v], best_avg_pc):
                    s = f"\\textbf{{{s}}}"
                avg_cells.append(s)
            else:
                avg_cells.append("--")
        for v in VARIANTS:
            if v in avg_anc:
                s = f"{avg_anc[v]:.3f}"
                if is_best(avg_anc[v], best_avg_anc):
                    s = f"\\textbf{{{s}}}"
                avg_cells.append(s)
            else:
                avg_cells.append("--")
        lines.append(" & ".join(avg_cells) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table}")
        lines.append("")

    tex_path = os.path.join(save_dir, "ablation_realworld_tables.tex")
    os.makedirs(save_dir, exist_ok=True)
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX saved: {tex_path}")



def plot_all(csv_path: str = CSV_PATH, save_dir: str = SAVE_DIR) -> None:
    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    for lam in LAMBDA_TARGETS:
        df_lam = df[df['lambda_eff'] == lam].copy()
        if df_lam.empty:
            print(f"  [skip] λ={lam} not found.")
            continue

        fig = plt.figure(figsize=(FIG_W, FIG_H))
        ax  = fig.add_axes([0, 0, 1, 1])
        render_table(ax, df_lam)

        ax.text(0.98, Y_TOP_LINE + 0.01,
                f'$\\lambda_{{\\mathrm{{eff}}}}={lam}$',
                ha='right', va='bottom', fontsize=6,
                color='#555555', transform=ax.transAxes)

        stem = os.path.join(save_dir, f'ablation_realworld_lambda{lam}')
        fig.savefig(stem + '.pdf', dpi=300)
        fig.savefig(stem + '.png', dpi=300)
        plt.close(fig)
        print(f"  Saved: {stem}.pdf / .png")

    make_latex(df, save_dir)
    print("Done.")


if __name__ == '__main__':
    plot_all()