"""Narrow SLURM query/cancel adapter scoped to one run and Unix owner."""

from __future__ import annotations

import getpass
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SubmissionError


@dataclass(frozen=True)
class SchedulerJob:
    job_id: str
    name: str
    state: str

    def to_dict(self) -> dict[str, str]:
        return {"job_id": self.job_id, "name": self.name, "state": self.state}


def query_slurm_jobs(
    *,
    prefix: str,
    squeue_executable: Path,
    username: str | None = None,
    timeout: float = 10.0,
    strict: bool = False,
) -> tuple[SchedulerJob, ...]:
    username = username or getpass.getuser()
    try:
        completed = subprocess.run(
            [
                str(squeue_executable),
                "--noheader",
                "--user",
                username,
                "--format",
                "%i|%j|%T",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if strict:
            raise SubmissionError(f"Could not query SLURM jobs: {exc}") from exc
        return ()
    if completed.returncode != 0:
        if strict:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise SubmissionError(
                f"SLURM job query failed with exit code {completed.returncode}: {message}"
            )
        return ()
    jobs: list[SchedulerJob] = []
    expected = f"{prefix}_"
    for line in completed.stdout.splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) != 3:
            continue
        job_id, name, state = parts
        if name == prefix or name.startswith(expected):
            jobs.append(SchedulerJob(job_id, name, state))
    return tuple(jobs)


def cancel_slurm_jobs(
    jobs: tuple[SchedulerJob, ...], *, scancel_executable: Path, timeout: float = 15.0
) -> subprocess.CompletedProcess[str] | None:
    if not jobs:
        return None
    completed = subprocess.run(
        [str(scancel_executable), "--me", *[job.job_id for job in jobs]],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise SubmissionError(
            f"SLURM cancellation failed with exit code {completed.returncode}: {message}"
        )
    return completed


def summarize_jobs(jobs: tuple[SchedulerJob, ...]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job.state] = counts.get(job.state, 0) + 1
    return {"total": len(jobs), "by_state": counts}
