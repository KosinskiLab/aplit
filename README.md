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

## Submission prototype

APLit can also prepare, validate, submit, track, resume, and cancel
AlphaPulldownSnakemake projects. Submission is disabled unless a site configuration is
provided, so the existing results-only command remains unchanged.

Start a per-user instance on the cluster login host:

```bash
aplit \
  --site-config /path/to/site-config.yaml \
  --server-address localhost \
  --no-browser
```

When `access_token_required` is enabled, the CLI prints a generated authenticated URL.
Forward the selected port through SSH and open that complete URL, including its token.
Each APLit process submits as the Unix account that started it.

This process-per-user setup is the first deployment profile, not a permanent service
constraint. A shared web service would need an authenticated delegation mechanism that
still submits under each user's cluster identity (for example, a site-provided submission
agent or SLURM REST integration); browser authentication alone cannot change the Unix
owner of local Snakemake and SLURM processes.

See [`examples/site-config.yaml`](examples/site-config.yaml) for the site schema. Shared
database, weights, container, profile, and scheduler settings are administrator-controlled;
the submission form exposes only backend selection and a small allowlist of run overrides.
Safe run-scoped cancellation uses the SLURM executor's `--slurm-jobname-prefix` option,
so the cluster Snakemake environment must include
[`snakemake-executor-plugin-slurm` 2.2.0 or newer](https://github.com/snakemake/snakemake-executor-plugin-slurm/blob/main/CHANGELOG.md).

Shared read-only Apptainer/Singularity image files and shared weights directories are
intentional. Concurrent runs can use the same image and weights paths; per-run writable
state remains inside each generated project and the configured Apptainer cache.

### Submission behavior

- Direct-complex and bait-prey modes are mutually exclusive.
- FASTA records may be uploaded or referenced through paths under `allowed_input_roots`.
- Uploaded multi-record FASTAs are split into safe aliases displayed by the form.
- Validation creates a frozen project and runs Snakemake with `--dry-run`.
- Reusing a matching project name resumes its frozen project. A changed input fingerprint
  is rejected instead of overwriting data.
- Cancellation stops the local controller and explicitly cancels all SLURM jobs carrying
  that run's unique prefix.

Generated projects contain their own workflow snapshot, resolved configuration, manifest,
status, event log, controller log, and output directory. No database or task queue is
required.

### Testing and GUI inspection

Create a development environment and run all service and Streamlit GUI tests:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest -q
```

The GUI suite uses Streamlit `AppTest` to exercise navigation, mutually exclusive forms,
validation, project generation, detached launch, and the Runs page. For visual inspection,
start APLit with a local/fake site configuration and inspect `?section=Results`,
`?section=Submit`, and `?section=Runs`; these query parameters only choose the initial page.

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
