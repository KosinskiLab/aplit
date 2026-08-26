"""Snakemake dry-run planning behind a replaceable adapter."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SubmissionError
from .models import RunState
from .registry import (
    atomic_write_json,
    metadata_dir,
    read_manifest,
    read_status,
    update_status,
)

JOB_ROW_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s+(\d+)\s*$")
GPU_RULES = {"structure_inference"}
AGGREGATE_RULES = {"all"}
PLANNABLE_STATES = {
    RunState.DRAFT.value,
    RunState.READY.value,
    RunState.FAILED.value,
    RunState.CANCELLED.value,
}


@dataclass(frozen=True)
class PlanResult:
    valid: bool
    command: tuple[str, ...]
    output: str
    requested_folds: int
    skipped_folds: int
    dag_jobs: int | None
    jobs_by_rule: dict[str, int]
    cpu_jobs: int
    gpu_jobs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "command": list(self.command),
            "output": self.output,
            "requested_folds": self.requested_folds,
            "skipped_folds": self.skipped_folds,
            "dag_jobs": self.dag_jobs,
            "jobs_by_rule": self.jobs_by_rule,
            "cpu_jobs": self.cpu_jobs,
            "gpu_jobs": self.gpu_jobs,
        }


def parse_job_stats(output: str) -> tuple[int | None, dict[str, int]]:
    jobs: dict[str, int] = {}
    in_stats = False
    for line in output.splitlines():
        if line.strip() == "Job stats:":
            in_stats = True
            continue
        if not in_stats:
            continue
        match = JOB_ROW_RE.match(line)
        if not match:
            if jobs and not line.strip():
                break
            continue
        name, count_raw = match.groups()
        if name.lower() in {"job", "count"}:
            continue
        jobs[name] = int(count_raw)
    total = jobs.pop("total", None)
    if total is None and jobs:
        total = sum(jobs.values())
    return total, jobs


def _requested_folds(manifest: dict[str, Any]) -> int:
    inputs = manifest.get("resolved_inputs") or {}
    if manifest.get("input_mode") == "bait_prey":
        return int(inputs.get("combination_count", 0))
    return len(inputs.get("direct_complexes") or ())


def _skipped_folds(project_dir: Path) -> int:
    path = project_dir / "output" / "skipped_folds.tsv"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    return max(0, len(lines) - 1)


def _resource_job_counts(jobs: dict[str, int]) -> tuple[int, int]:
    gpu_jobs = sum(count for rule, count in jobs.items() if rule in GPU_RULES)
    cpu_jobs = sum(
        count
        for rule, count in jobs.items()
        if rule not in GPU_RULES and rule not in AGGREGATE_RULES
    )
    return cpu_jobs, gpu_jobs


def plan_project(project_dir: Path, *, timeout: float = 300.0) -> PlanResult:
    manifest = read_manifest(project_dir)
    status = read_status(project_dir)
    if status.get("state") not in PLANNABLE_STATES:
        raise SubmissionError(
            f"Project cannot be validated from state {status.get('state')!r}."
        )
    command = tuple(str(item) for item in manifest.get("command") or ())
    if not command:
        raise SubmissionError("Run manifest has no Snakemake command")

    update_status(
        project_dir,
        state=RunState.VALIDATING.value,
        message="Running Snakemake dry-run",
    )
    dry_run_command = (*command, "--dry-run")
    try:
        completed = subprocess.run(
            dry_run_command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout + completed.stderr
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        output = stdout + stderr + f"\nDry-run timed out after {timeout:g} seconds.\n"
        return_code = 124
    except OSError as exc:
        update_status(
            project_dir,
            state=RunState.DRAFT.value,
            message="Could not start Snakemake dry-run",
        )
        raise SubmissionError(f"Could not start Snakemake dry-run: {exc}") from exc

    (metadata_dir(project_dir) / "validation.log").write_text(output, encoding="utf-8")
    total, jobs = parse_job_stats(output)
    cpu_jobs, gpu_jobs = _resource_job_counts(jobs)
    result = PlanResult(
        valid=return_code == 0,
        command=command,
        output=output,
        requested_folds=_requested_folds(manifest),
        skipped_folds=_skipped_folds(project_dir),
        dag_jobs=total,
        jobs_by_rule=jobs,
        cpu_jobs=cpu_jobs,
        gpu_jobs=gpu_jobs,
    )
    atomic_write_json(metadata_dir(project_dir) / "plan.json", result.to_dict())
    update_status(
        project_dir,
        state=RunState.READY.value if result.valid else RunState.DRAFT.value,
        message=None if result.valid else "Snakemake dry-run failed",
        plan={
            "requested_folds": result.requested_folds,
            "skipped_folds": result.skipped_folds,
            "dag_jobs": result.dag_jobs,
            "jobs_by_rule": result.jobs_by_rule,
            "cpu_jobs": result.cpu_jobs,
            "gpu_jobs": result.gpu_jobs,
        },
    )
    return result
