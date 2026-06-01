# Supplementary Figure 8: Real-world scenarios — full $P_C$ and ANC curves

This directory contains the script and shared-data reference 
required to reproduce **Supplementary Figure 8** of the paper 
*Containment of Escaping Targets in Complex Networks*.

Supplementary Figure 8 complements main-text **Figure 6** by 
showing, for each of the six real-world scenarios, the full 
$P_C(\lambda)$ and ANC$(\lambda)$ curves of all four methods 
across the evasion-factor sweep. The figure is organized as 
$3 \times 4 = 12$ panels in three domain blocks: public health 
(COVID-19 and invasive species), infrastructure protection (urban 
flood and smuggling), and security (socialbot and fugitive 
chase). In the main text, Figure 6 reports only $P_C$ and the 
scenario-specific metric at the slice $\lambda = 20$; this 
supplementary figure shows the full curves underlying that slice.


## Files

### Script

| Script | Role |
|--------|------|
| `draw_realworld_combined.py` | Renders Supplementary Figure 8 by reading the six per-scenario results CSVs that already exist under `fig6/data/fig6_result/`. |

### Data

**No new data is shipped with this directory.** Supplementary 
Figure 8 is rendered from the same per-scenario CSVs that produce 
main-text Figure 6:

```
../../fig6/data/fig6_result/{scenario}/{scenario}_results.csv
```

The six `{scenario}` subdirectories are `covid19`, 
`invasive_species`, `flood`, `smuggling`, `socialbot`, and 
`fugitive_chase`. See `fig6/README.md` for the per-scenario data 
details and how to regenerate the CSVs.


## Reproducing the figure

The repository ships with the pre-computed per-scenario data 
under `fig6/data/fig6_result/`, so the figure can be reproduced 
directly:

```bash
python draw_realworld_combined.py
```

This produces `draw_realworld_combined.pdf` and `.png` in this 
directory.

If `fig6/data/fig6_result/` is missing or incomplete, regenerate 
the per-scenario data first by running the data scripts in 
`fig6/` (see `fig6/README.md`), then re-run 
`draw_realworld_combined.py`.


## Path resolution

`draw_realworld_combined.py` locates the shared Fig. 6 results 
directory by walking two levels up from itself and into 
`fig6/data/fig6_result/`:

```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_ROOT  = os.path.join(_REPO_ROOT, 'fig6', 'data', 'fig6_result')
```

No manual configuration is needed as long as the directory 
structure of the repository is preserved 
(`fig_sm/sm_fig8/` next to `fig6/` at the repo root level).


## Requirements

This figure uses the project-wide Python environment specified in 
the top-level `requirements.txt`. The specific packages used here 
are:

- `numpy`, `pandas`
- `matplotlib`

The rendering script has no dependency on `torch`, 
`torch-geometric`, or `networkx` — it operates purely on the 
released CSV tables.


## Citation

If you use any of the code or data in this directory, please cite 
the main paper.
