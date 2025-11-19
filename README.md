# APLit

APLit is a Streamlit UI for browsing AlphaPulldown runs, visualising AF2 **and AF3** outputs, and (optionally) overlaying AlphaJudge interface scores.

## Highlights

- 🎨 Clean, responsive grid with instant search, sort, and ipTM / PAE sliders
- 📊 Built-in viewers for PAE heatmaps and predicted models
- 🔌 Optional AlphaJudge integration – drop `interfaces.csv` next to each job to unlock extra metrics
- 🧮 AlphaJudge filters are cumulative so you can combine multiple sliders at once
- 🔄 Auto-refresh to follow running AlphaPulldown scans
- 📥 Export any table view as CSV for downstream analysis

## Installation

Requirements: Python 3.8+

```bash
pip install git+ssh://git@github.com/KosinskiLab/aplit.git
```

## Running the app

```bash
# Local defaults: binds to localhost:8501 and opens a browser
aplit

# Provide a default predictions folder and custom port
aplit --directory /path/to/predictions --port 8502
```

Inside the UI open **Configuration ▸ Predictions Directory** and point it to the parent folder that contains your AlphaPulldown jobs (AF2 or AF3):

```
/path/to/predictions/
├── protein1_and_protein2/          # AF2 job (ranking_debug.json, pae_*.json, ranked_*.pdb/cif)
│   └── ...
├── protein3_and_protein4/          # AF3 job (ranking_scores.csv, *_summary_confidences.json, etc.)
│   └── ...
├── protein1_and_protein3/
│   └── ...
```

### AlphaJudge scores (optional)

- When an AlphaPulldown job directory includes AlphaJudge’s `interfaces.csv`, APLit automatically loads all numeric columns (e.g., `global_dockq`, `best_interface_ipsae`, `best_interface_lis`) and exposes them in the table, sort menu, and filter sliders.
- If **no** AlphaJudge file is present (or it is empty), the UI silently skips those columns—ipTM, pTM, and PAE views continue to work exactly as before, so you can safely mix jobs with and without AlphaJudge annotations.
- The AlphaJudge expander only appears when at least one numeric score is available.

### Filtering & sorting

- The ipTM and mean-PAE sliders are always active; search matches job names case-insensitively.
- AlphaJudge sliders apply **all at once** (logical AND). Shrinking multiple ranges narrows the table to entries that satisfy every slider you touched.
- You can sort by ipTM, ipTM+pTM, job name, or any AlphaJudge metric currently loaded.
- Click a job name in the table to jump directly to the detailed viewer for that run.

### Running on HPC via SSH

1. Launch APLit on the cluster/login node (headless, bound to localhost):

   ```bash
   aplit --directory /cluster/path/to/predictions --server-address localhost --no-browser
   ```

2. Create an SSH tunnel from your laptop to the cluster (example for EMBL login node):

   ```bash
   ssh -N -L 8501:localhost:8501 login1.cluster.embl.de
   ```

3. Visit `http://localhost:8501` in your local browser. The tunnel forwards traffic securely to the remote Streamlit session.

> Tip: keep the tunnel window open; closing it stops the port forward. Use a different local port (`-L 8502:localhost:8501`) if 8501 is already taken.

## Links

- [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown)
- [AlphaJudge](https://github.com/KosinskiLab/AlphaJudge)