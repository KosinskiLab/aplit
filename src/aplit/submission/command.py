"""Construct the one canonical Snakemake command for a generated project."""

from __future__ import annotations

from pathlib import Path

from .models import SiteConfig


def build_snakemake_command(
    site: SiteConfig, project_dir: Path, slurm_prefix: str
) -> list[str]:
    command = [
        str(site.snakemake_executable),
        "--directory",
        str(project_dir),
        "--snakefile",
        str(project_dir / "workflow" / "Snakefile"),
        "--profile",
        str(project_dir / "config" / "profile"),
        "--configfile",
        str(project_dir / "config" / "config.yaml"),
        "--jobs",
        str(site.max_parallel_jobs),
        "--retries",
        str(site.retries),
    ]
    if site.executor == "slurm":
        command.extend(["--slurm-jobname-prefix", slurm_prefix])
    return command
