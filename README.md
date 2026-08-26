# APLit

APLit is a Streamlit application for two connected tasks:

1. preparing and running reproducible AlphaPulldownSnakemake projects; and
2. browsing AlphaFold 2, AlphaFold 3, and optional AlphaJudge results.

Submission is optional. Without a site configuration, APLit remains a
results-only viewer.

## Install

```bash
pip install git+ssh://git@github.com/KosinskiLab/aplit.git
```

For development:

```bash
git clone git@github.com:KosinskiLab/aplit.git
cd aplit
uv venv .venv
uv pip install --python .venv/bin/python -e . pytest
```

## Browse existing results

Start APLit with the directory that contains the prediction folders:

```bash
aplit --directory /path/to/predictions
```

The directory can contain AF2 and AF3 jobs, with or without AlphaJudge output:

```text
predictions/
├── P69905+P68871/
│   ├── ranking_debug.json                 # AF2
│   ├── ranked_0.pdb
│   └── interfaces.csv                     # optional AlphaJudge
└── another_complex/
    ├── ranking_scores.csv                 # AF3
    ├── *_summary_confidences.json
    └── ...
```

Use the **Results** page to search and filter jobs, inspect confidence and PAE,
open structures, and export tables or models.

## Enable submission on a cluster

APLit does not contain its own scheduler. It creates a frozen
AlphaPulldownSnakemake project and launches the site's existing Snakemake command.
Snakemake chooses resources and submits jobs to SLURM.

Before starting APLit, a site administrator needs to provide:

- an AlphaPulldownSnakemake workflow checkout or release;
- a Snakemake environment with the SLURM executor plugin;
- CPU and GPU SLURM partitions and QoS;
- AlphaFold databases and optional precomputed features;
- shared AF2/AF3 weights and prediction containers; and
- a private project root for each Unix user.

Safe run-scoped cancellation uses `--slurm-jobname-prefix`, which requires
[`snakemake-executor-plugin-slurm` 2.2.0 or newer](https://github.com/snakemake/snakemake-executor-plugin-slurm/blob/main/CHANGELOG.md).

Copy [examples/site-config.yaml](examples/site-config.yaml) and replace its example
paths. The important structure is:

```yaml
schema_version: 1
name: My SLURM cluster

deployment:
  projects_root: "/scratch/{username}/aplit/projects"
  workflow_template: "/shared/workflows/AlphaPulldownSnakemake/2.7.0"
  snakemake_executable: "/shared/conda/envs/snakemake/bin/snakemake"
  project_mode: "0700"
  access_token_required: true

allowed_input_roots:
  - "/scratch/{username}"
  - "/g"

databases_directory: "/shared/alphafold/databases"
feature_directories:
  - "/shared/alphafold/features"

backends:
  af2:
    data_pipeline: alphafold2
    fold_backend: alphafold2
    prediction_container: "/shared/apptainer/alphafold2.sif"
    weights_directory: "/shared/alphafold/af2-weights"
    databases_directory: "/shared/alphafold/af2-databases"
    feature_directories: ["/shared/alphafold/af2-features"]
  af3:
    data_pipeline: alphafold3
    fold_backend: alphafold3
    prediction_container: "/shared/apptainer/alphafold3.sif"
    weights_directory: "/shared/alphafold/af3-weights"
    databases_directory: "/shared/alphafold/af3-databases"
    feature_directories: ["/shared/alphafold/af3-features"]

execution:
  executor: slurm
  profile_template: "config/profiles/slurm"
  max_parallel_jobs: 200
  retries: 3
  slurm_jobname_prefix: aplit

workflow_config:
  slurm_partition: "gpu-el8,gpu-training"
  slurm_partition_cpu: "htc-el8,htc-el10,bigmem"
  slurm_qos: normal
  structure_inference_gpus_per_task: 1
  structure_inference_max_runtime: 10080
  enable_structure_analysis: true
  analysis_container: "/shared/apptainer/alphajudge.sif"
```

`databases_directory` and `feature_directories` under a backend override the
site-wide defaults for that preset. This is needed when AF2 and AF3 use different
database releases or incompatible precomputed feature formats.

Start a per-user instance on a login node:

```bash
aplit \
  --site-config /path/to/site-config.yaml \
  --server-address localhost \
  --port 8501 \
  --no-browser
```

When token protection is enabled, the command prints an authenticated URL. From
your workstation, open an SSH tunnel and then open that complete URL, including
the token:

```bash
ssh -N -L 8501:localhost:8501 login.cluster.example.org
```

## Example 1: a simple AF2 dimer

Open **Submit** and enter:

- **Project name:** `af2-haemoglobin`
- **Input mode:** `Direct complexes`
- **Backend:** `af2`
- **Complexes:**

  ```text
  P69905+P68871
  ```

This example uses two UniProt accessions. Select **Validate** first. APLit creates
the project, runs `snakemake --dry-run`, and displays:

- requested and skipped folds;
- total DAG jobs;
- estimated CPU and GPU worker jobs; and
- the exact frozen Snakemake command.

If validation succeeds, select **Submit run**. Open **Runs** to see the controller
state, active SLURM partitions, requested CPU/GPU resources, memory, elapsed time,
time limit, and pending reason.

## Example 2: AF3 with an uploaded FASTA

Create a file named `example_chain.fasta`:

```fasta
>example_chain
MKTIIALSYIFCLVFADYKDDDDKGGGGSAAAA
```

Open **Submit** and enter:

- **Project name:** `af3-upload-example`
- **Input mode:** `Direct complexes`
- **Backend:** `af3`
- upload `example_chain.fasta`;
- **Complexes:**

  ```text
  example_chain+P68871
  ```

The uploaded filename stem becomes the alias shown by the form. A multi-record
FASTA is split into one safely named alias per record. The uploaded sequence is
copied into the generated project, checksummed, and referenced using its frozen
project path.

Validate, submit, and monitor the run as in the AF2 example. AF2 and AF3 runs may
use the same shared, read-only SIF and weights paths concurrently; writable state
is kept in the individual project and configured Apptainer cache.

## Example 3: a small bait–prey screen

Choose **Bait–prey screen** and enter:

```text
Baits
P69905
P02042
```

```text
Preys
P68871
P02144
```

This produces four combinations. Large Cartesian products are accepted but shown
with a warning before submission. Direct-complex and bait–prey inputs cannot be
mixed in one run.

## AlphaJudge score support

When a prediction directory contains a non-empty `interfaces.csv`, APLit ingests
every numeric AlphaJudge column rather than relying on a fixed score list. The current
AlphaJudge output includes:

- ipTM, pTM, confidence, pDockQ/mpDockQ, pDockQ2, ipSAE, LIS, cLIS, and iLIS;
- the AlphaJudge interface meta score;
- interface PAE, pLDDT, residue and contact counts;
- contact-probability maximum, top-10 mean, expected contacts, and CCC; and
- polar/hydrophobic/charged fractions, hydrogen bonds, salt bridges, disulfides,
  shape complementarity, buried area, and solvation energy.

The overview chooses the representative interface by `interface_meta_score`, falling
back through pDockQ2, interface score, ipTM, and ipTM+pTM for older CSVs. Every numeric
field is available for overview sorting and filtering. The selected-model page shows
the current score set and exposes the complete per-interface table for CSV download.
New numeric columns added by AlphaJudge are carried through automatically.

## CPU and GPU resource management

APLit deliberately does not reproduce AlphaPulldownSnakemake's resource formulas.
The frozen workflow currently applies the following policy:

| Stage | Resource behavior | Partition control |
| --- | --- | --- |
| Feature generation | CPU threads; length-aware linear RAM; RAM increases on retry | `slurm_partition_cpu` |
| Structure inference | GPU request; length-aware quadratic host RAM; retry scaling; wall-time cap; optional unified-memory spill | `slurm_partition` |
| GPU model/tier routing | Fixed GPU model or length-based VRAM tiers that exclude undersized nodes | `structure_inference_gpu_model` or `structure_inference_gpu_tiers` |
| Download, linking, and AlphaJudge analysis | Smaller CPU jobs | profile or cluster default partition |

Several comma-separated partitions mean “allow SLURM to select any eligible
partition”; they do not launch duplicate jobs. `max_parallel_jobs` caps Snakemake's
global in-flight job count. Per-user, per-partition, and aggregate resource limits
remain SLURM/QoS policy and should not be recreated in APLit.

Resource and partition keys live in the trusted `workflow_config` section of the
site file. They are not accepted through the user's advanced YAML box. Useful
administrator-controlled keys include:

```yaml
workflow_config:
  slurm_partition: "gpu-el8,gpu-training"
  slurm_partition_cpu: "htc-el8,htc-el10,bigmem,bigmem-el10"
  slurm_qos: normal
  feature_threads: 8
  structure_inference_gpus_per_task: 1
  structure_inference_gpu_model: ""       # empty allows any compatible GPU
  structure_inference_max_runtime: 10080
  structure_inference_unified_memory: true
  structure_inference_gpu_vram_headroom: 1.0
  structure_inference_gpu_tiers:
    - {min_vram_gb: 24, nodes: "gpu-small01,gpu-small02"}
    - {min_vram_gb: 48, nodes: "gpu-large01,gpu-large02"}
    - {min_vram_gb: 80, nodes: "gpu-h10001,gpu-h10002"}
```

GPU tier node names are site-specific. Test them with a small workflow before using
them for production submissions.

## Monitoring and cancellation

The **Runs** page obtains one user-scoped `squeue` snapshot per refresh and filters
it locally by each run's unique job-name prefix. For active jobs it shows:

- CPU versus GPU job counts;
- requested CPUs and GPUs;
- actual or eligible partition;
- requested memory and time limit;
- elapsed time and allocated node; and
- the scheduler's reason for a pending job, such as `Resources` or `Priority`.

This is live queue monitoring, not historical accounting. Completed jobs disappear
from `squeue`; the final workflow state, exit code, event log, and controller log stay
in the project. Historical CPU time, peak RSS, and GPU accounting can be added later
through a separately rate-limited `sacct` adapter.

Cancellation first signals the detached Snakemake controller, then queries and
explicitly cancels every current-user SLURM job whose name belongs to that run.

## Resume and project contents

Reusing a project name with the same scientific inputs resumes its frozen workflow
after showing a warning. If the backend, inputs, uploads, or allowlisted overrides
differ, APLit refuses the collision instead of overwriting the project. An active
project cannot be revalidated or launched a second time.

Each generated project contains:

```text
project/
├── workflow/                       # copied workflow snapshot
├── config/
│   ├── config.yaml                 # resolved site + scientific configuration
│   ├── profile/                    # copied Snakemake profile
│   └── sample_sheet.csv            # or baits.txt + preys.txt
├── input/uploads/                  # staged uploaded FASTAs
├── output/predictions/
└── .aplit/
    ├── run.json                    # immutable provenance and command
    ├── status.json                 # current state and scheduler snapshot
    ├── plan.json                   # dry-run job plan
    ├── events.jsonl
    └── controller.log
```

Select **View results** from Runs when predictions are available to open the existing
viewer at that project's `output/predictions` directory.

## Security and deployment model

The initial deployment profile is one APLit process per Unix identity. Projects are
private by default (`0700`), scheduler queries are restricted to the current user,
commands are argument arrays rather than shell strings, server FASTA paths must stay
inside configured roots, and resource/container/weights paths are administrator-owned.

This is not a permanent one-user application constraint. A shared service needs an
authenticated delegation mechanism that still starts Snakemake and SLURM submissions
under the real user's cluster identity. Browser authentication alone does not change
the Unix owner of local processes. MCP is not required for either deployment model;
the submission service is already independent of Streamlit and can receive another
adapter later.

## Test the prototype

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/aplit/submission src/aplit/views src/aplit/cli.py tests
```

The tests cover schemas, FASTA staging, path containment, collision-safe resume,
dry-run planning, detached lifecycle, scheduler scoping, CPU/GPU partition summaries,
cancellation, and Streamlit navigation/submission flows.

For visual inspection, start APLit with a local fake site and inspect the **Results**,
**Submit**, and **Runs** pages. Query parameters `?section=Results`, `?section=Submit`,
and `?section=Runs` select only the initial page.

## Links

- [AlphaPulldown](https://github.com/KosinskiLab/AlphaPulldown)
- [AlphaPulldownSnakemake](https://github.com/KosinskiLab/AlphaPulldownSnakemake)
- [AlphaJudge](https://github.com/KosinskiLab/AlphaJudge)
