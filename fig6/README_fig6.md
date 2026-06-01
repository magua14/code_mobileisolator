# Figure 6: Real-world isolation scenarios

This directory contains the scripts and data required to reproduce 
**Figure 6** of the paper *Containment of Escaping Targets in 
Complex Networks*.

Figure 6 evaluates MobileIsolator against the three adaptive 
heuristic baselines on six real-world isolation scenarios drawn 
from three application domains. Each scenario contributes one row 
of the figure: a network-visualization panel showing the final 
isolation pattern produced by a selected method, plus bar charts 
that compare $P_C$ and the scenario-specific containment metric 
across the four methods at $\lambda = 20$.

The six scenarios, in the order they appear in the figure (rows 
a–f) and in the main text, are:

| Order | Scenario | Domain | Data script |
|-------|----------|--------|-------------|
| a | COVID-19 community contact network | Public health | `draw_covid19_new.py` |
| b | Invasive species (global aviation) | Public health | `draw_invasive_species_new.py` |
| c | Urban flood (Haidian road network) | Infrastructure | `draw_city_flood_new.py` |
| d | Smuggling (European E-road network) | Infrastructure | `draw_smuggling_new.py` |
| e | Socialbot infiltration (Advogato) | Security | `draw_socialbot_new.py` |
| f | Fugitive chase (Rome transit) | Security | `draw_fugitive_chasing_new.py` |


## Data model

The six data scripts and the rendering script share a uniform 
convention: only **pure network topologies** are released, and 
target selection happens inside each scenario's data script.

- **`data/fig6_data/`** — Released input networks. Each file is a 
  pure GraphML topology (nodes and edges only, no target labels). 
  The flood network additionally carries a per-node `elevation` 
  attribute, which its data script uses for target selection.
- **`data/fig6_result/`** — Per-scenario experiment outputs, 
  produced by the data scripts. Each scenario's subdirectory 
  contains: the network re-saved with an `is_target` attribute, a 
  results CSV for the bar charts, and per-method text files listing 
  the removed-node sequence and the final target positions used 
  by the network panels.

The rendering script `draw_fig6.py` reads exclusively from 
`fig6_result/`.


## Platform note

All scripts in this directory — both the six data-generating 
scripts and the rendering script — run on Linux, macOS, and 
Windows. The data scripts use Python's `ProcessPoolExecutor` with 
per-worker `initializer` callbacks, so they do not depend on the 
Unix-only `fork` multiprocessing start method and work unchanged 
under the `spawn` start method that Windows uses by default.

In our own pipeline, the data scripts were run on a Linux cluster 
(for throughput) and the final figure was rendered on Windows; 
both steps can be performed on any platform.


## Files

### Data-generating scripts (one per scenario)

| Script | Scenario | Reads (from `data/fig6_data/`) | Writes (to `data/fig6_result/{subdir}/`) |
|--------|----------|--------------------------------|------------------------------------------|
| `draw_covid19_new.py` | COVID-19 | `covid19_network.graphml` | `covid19/` |
| `draw_invasive_species_new.py` | Invasive species | `invasive_network.graphml` | `invasive_species/` |
| `draw_city_flood_new.py` | Urban flood | `flood_network.graphml` (with `elevation`) | `flood/` |
| `draw_smuggling_new.py` | Smuggling | `smuggling_network.graphml` | `smuggling/` |
| `draw_socialbot_new.py` | Socialbot | `socialbot_network.graphml` | `socialbot/` |
| `draw_fugitive_chasing_new.py` | Fugitive chase | `fugitive_network.graphml` | `fugitive_chase/` |

Each data script: (i) loads its pure-topology GraphML from 
`fig6_data/`, (ii) selects target nodes using scenario-specific 
logic, (iii) writes a copy of the network annotated with the 
`is_target` attribute to its `fig6_result/` subdirectory, (iv) 
runs the four-method experiment across the $\lambda$ sweep, and 
(v) saves a terminal-state snapshot of the isolation episode, 
including a per-method list of removed nodes and final target 
positions.

### Rendering script (run on any platform)

| Script | Role |
|--------|------|
| `draw_fig6.py` | Combines all six scenarios into the final Figure 6. Reads each scenario's network GraphML, per-method removal/target text files, and results CSV from `data/fig6_result/`. |

### Data (`data/`)

```
data/
├── fig6_data/                       Released pure-topology networks
│   ├── covid19_network.graphml
│   ├── invasive_network.graphml
│   ├── flood_network.graphml        (with `elevation` attribute)
│   ├── smuggling_network.graphml
│   ├── socialbot_network.graphml
│   └── fugitive_network.graphml
│
└── fig6_result/                     Per-scenario experiment outputs
    ├── .layout_cache/               Pickled spring-layout coordinates
    │   ├── spring_n100_s42_*.pkl    (one .pkl per network panel)
    │   └── ...
    ├── covid19/
    │   ├── covid19_network.graphml          (with is_target)
    │   ├── covid19_target_nodes.txt
    │   ├── covid19_results.csv
    │   ├── removed_{method}.txt             (×4 methods)
    │   ├── final_targets_{method}.txt       (×4 methods)
    │   └── ... (other snapshot files)
    ├── invasive_species/                    (same layout)
    ├── flood/
    │   ├── flood_network.graphml
    │   ├── flood_network_with_coords.graphml  (geographic x/y)
    │   ├── ...
    ├── smuggling/
    ├── socialbot/
    └── fugitive_chase/
```

### About the `.layout_cache/` directory

The network panels in Figure 6 use a force-directed (spring) 
layout, which is computationally expensive and can produce 
slightly different coordinates across NetworkX versions and 
platforms even with a fixed random seed. To guarantee 
**pixel-level** reproduction of the published figure, computed 
node coordinates are cached as `.pkl` files in 
`data/fig6_result/.layout_cache/`. Each cache file's name encodes 
the graph size, seed, layout parameter `k`, and a hash of the 
edge structure, so it is robust to incidental file moves.

These cache files are **included** in the released data and 
should be kept. On first run, if no cache is found, the rendering 
script computes the layout and writes the cache file, so 
subsequent runs are fast and exactly reproduce the same figure.

The flood scenario is special: it uses **geographic coordinates** 
(from `flood_network_with_coords.graphml`) instead of a spring 
layout, so its panel is not cached.


## COVID-19 two-script split

Of the six data scripts, only the COVID-19 pipeline includes a 
separate **network-construction** step that is **not part of the 
released code**. The reason is that the COVID-19 community contact 
network is assembled from two raw text files (a community-to-id 
mapping and an edge list) that are not redistributed here; the 
other five scenarios consume pre-released GraphML networks 
directly.

A separate, **unreleased** script `build_covid19_network.py` 
performs this step: it reads the raw text inputs and writes 
`data/fig6_data/covid19_network.graphml` (pure topology, no 
target labels). The released `draw_covid19_new.py` then proceeds 
exactly like the other five scripts: load the pure-topology 
network, select targets, run the experiment.

If you only want to reproduce the figure, you do not need 
`build_covid19_network.py` — the resulting GraphML is already in 
`data/fig6_data/`.


## Dependencies

The six data scripts import shared utilities from the Fig. 3 
directory and the agent module from the repository-level `train/` 
directory:

```python
from comparison_method import (...)              # ../fig3/
from ppo_gnn_moving_gat_dualstream_final import DualStreamAgent  # ../train/
```

They also load the pre-trained policy weights from:

```
../train/gnn_ppo_dual_stream_n50-100__seed42__1769653775/model.pt
```

Path resolution is automatic; the scripts locate `fig3/` and 
`train/` relative to their own location. The rendering script 
`draw_fig6.py` has no such dependency — it only reads files 
from `data/fig6_result/`.


## Reproducing the figure

The repository ships with the pre-computed per-scenario data in 
`data/fig6_result/`, so the figure can be reproduced directly on 
any platform:

```bash
python draw_fig6.py
```

This produces `draw_fig6_final_GNN_RL.pdf` and `.png` in this 
directory. The `GNN_RL` suffix reflects the value of the `METHOD` 
variable at the top of the script; it selects which method's 
terminal state is shown in the network panels. To render the 
analogous figure for a baseline, set `METHOD` to `'TD'`, `'TIA'`, 
or `'Katz'`.

### Regenerating the per-scenario data from scratch

Each scenario is generated independently. Run the corresponding 
data script on any platform (Linux / macOS / Windows). For 
example, for the COVID-19 scenario:

```bash
python draw_covid19_new.py
```

and analogously for the other five scenarios. Each script writes 
its outputs (annotated GraphML, target list, results CSV, 
per-method removed/target text files) into its own 
`data/fig6_result/{subdir}/` subdirectory. The largest network 
(the aviation network for invasive species, with roughly 2,900 
nodes) dominates the runtime.

Once all six scenarios have been (re)generated:

```bash
python draw_fig6.py
```

Note that re-running the data scripts will overwrite the released 
CSVs and text files. If you want to preserve pixel-level 
reproducibility of the published figure, keep the shipped 
`.layout_cache/` directory; the rendering script will read 
cached layouts in preference to recomputing them.


## Configuration

- `draw_fig6.py`:
  - `METHOD = 'GNN_RL'` — which method's terminal isolation 
    pattern to render in the network panels. One of `'GNN_RL'`, 
    `'TD'`, `'TIA'`, `'Katz'`.
  - `BAR_LAMBDA = 20` — the $\lambda$ slice shown in the bar 
    charts
  - `SEED = 42` — layout seed (used only when no cached layout 
    is present)
- Each `draw_*_new.py` data script exposes, in its 
  `if __name__ == "__main__"` block:
  - `LAMBDA_VALUES` — the evasion-factor sweep grid
  - `SIMULATION_TIMES` — number of independent simulations per 
    condition
  - `SNAPSHOT_LAMBDA` — the $\lambda$ at which the terminal 
    isolation snapshot is recorded
  - scenario-specific target-selection parameters (e.g. 
    `TARGET_RATIO`, `MAX_PERIPHERAL_DEGREE`, `ELEVATION_MIN`, 
    `N_CLUSTERS`)


## Requirements

This figure uses the project-wide Python environment specified in 
the top-level `requirements.txt`. The specific packages used here 
are:

- `numpy`, `pandas`, `scipy`, `tqdm`
- `networkx`
- `matplotlib`
- `torch`, `torch-geometric` (data scripts only, for loading the 
  trained policy)


## Data sources

Each scenario's underlying network derives from one or more 
published sources. The reference list below mirrors exactly the 
citations used in the main text where the six scenarios are 
introduced.

### Public health

- **COVID-19 community contact network** —  
  Zhao, C., Zhang, J., Hou, X., Yeung, C. H., & Zeng, A. 
  A high-frequency mobility big-data reveals how COVID-19 spread 
  across professions, locations and age groups. *PLOS Computational 
  Biology* **19**, e1011083 (2023).

- **Invasive species (global aviation network)** —  
  OpenFlights. *Airports, airlines and routes database.* 
  <https://openflights.org/data.php> (accessed May 2025).

### Infrastructure protection

- **Urban flood (Haidian road network)** —  
  Haklay, M. & Weber, P. OpenStreetMap: User-generated street maps. 
  *IEEE Pervasive Computing* **7**, 12–18 (2008).  
  Farr, T. G. *et al.* The Shuttle Radar Topography Mission. 
  *Reviews of Geophysics* **45**, RG2004 (2007).

- **Smuggling (European E-road network)** —  
  Kunegis, J. KONECT: the Koblenz network collection. 
  *Proceedings of the 22nd International Conference on World Wide 
  Web*, 1343–1350 (2013).  
  Šubelj, L. & Bajec, M. Robust network community detection using 
  balanced propagation. *The European Physical Journal B* **81**, 
  353–362 (2011).

### Security

- **Socialbot infiltration (Advogato trust network)** —  
  Massa, P., Salvetti, M., & Tomasoni, D. Bowling alone and trust 
  decline in social network sites. *Eighth IEEE International 
  Conference on Dependable, Autonomic and Secure Computing*, 
  658–663 (2009).

- **Fugitive chase (Rome multimodal transit network)** —  
  Kujala, R., Weckström, C., Darst, R. K., Mladenović, M. N., & 
  Saramäki, J. A collection of public transport network data sets 
  for 25 cities. *Scientific Data* **5**, 180089 (2018).


## Citation

If you use any of the code or data in this directory, please cite 
the main paper and the per-scenario data sources listed above.
