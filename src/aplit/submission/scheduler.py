"""Narrow SLURM query/cancel adapter scoped to one run and Unix owner."""

from __future__ import annotations

import getpass
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SubmissionError


@dataclass(frozen=True)
class SchedulerJob:
    job_id: str
    name: str
    state: str
    partition: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    time_limit: str | None = None
    time_used: str | None = None
    reason_or_nodes: str | None = None
    gres: str | None = None

    @property
    def is_gpu_job(self) -> bool:
        return "gpu" in (self.gres or "").lower()

    @property
    def gpu_count(self) -> int:
        if not self.is_gpu_job:
            return 0
        counts = re.findall(
            r"gpu(?::[^,:=()]+)?[:=](\d+)",
            self.gres or "",
            re.IGNORECASE,
        )
        return sum(int(value) for value in counts) if counts else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "state": self.state,
            "partition": self.partition,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "time_limit": self.time_limit,
            "time_used": self.time_used,
            "reason_or_nodes": self.reason_or_nodes,
            "gres": self.gres,
            "gpu_count": self.gpu_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SchedulerJob:
        return cls(
            job_id=str(value.get("job_id", "")),
            name=str(value.get("name", "")),
            state=str(value.get("state", "")),
            partition=_optional_text(str(value.get("partition") or "")),
            cpus=_optional_int(str(value.get("cpus") or "")),
            memory_mb=_optional_int(str(value.get("memory_mb") or "")),
            time_limit=_optional_text(str(value.get("time_limit") or "")),
            time_used=_optional_text(str(value.get("time_used") or "")),
            reason_or_nodes=_optional_text(str(value.get("reason_or_nodes") or "")),
            gres=_optional_text(str(value.get("gres") or "")),
        )


def _optional_text(value: str) -> str | None:
    rendered = value.strip()
    return None if rendered in {"", "N/A", "(null)", "null"} else rendered


def _optional_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def query_user_slurm_jobs(
    *,
    squeue_executable: Path,
    username: str | None = None,
    timeout: float = 10.0,
    strict: bool = False,
) -> tuple[SchedulerJob, ...]:
    """Fetch one queue snapshot for a user, including requested resources."""

    username = username or getpass.getuser()
    try:
        completed = subprocess.run(
            [
                str(squeue_executable),
                "--noheader",
                "--user",
                username,
                "--format",
                "%i|%j|%T|%P|%C|%m|%l|%M|%R|%b",
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
    for line in completed.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        parts.extend([""] * (10 - len(parts)))
        job_id, name, state, partition, cpus, memory, limit, used, reason, gres = parts[
            :10
        ]
        jobs.append(
            SchedulerJob(
                job_id=job_id.strip(),
                name=name.strip(),
                state=state.strip(),
                partition=_optional_text(partition),
                cpus=_optional_int(cpus),
                memory_mb=_optional_int(memory),
                time_limit=_optional_text(limit),
                time_used=_optional_text(used),
                reason_or_nodes=_optional_text(reason),
                gres=_optional_text(gres),
            )
        )
    return tuple(jobs)


def filter_slurm_jobs(
    jobs: tuple[SchedulerJob, ...], *, prefix: str
) -> tuple[SchedulerJob, ...]:
    expected = f"{prefix}_"
    return tuple(
        job for job in jobs if job.name == prefix or job.name.startswith(expected)
    )


def query_slurm_jobs(
    *,
    prefix: str,
    squeue_executable: Path,
    username: str | None = None,
    timeout: float = 10.0,
    strict: bool = False,
) -> tuple[SchedulerJob, ...]:
    jobs = query_user_slurm_jobs(
        squeue_executable=squeue_executable,
        username=username,
        timeout=timeout,
        strict=strict,
    )
    return filter_slurm_jobs(jobs, prefix=prefix)


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
    partitions: dict[str, dict[str, Any]] = {}
    for job in jobs:
        counts[job.state] = counts.get(job.state, 0) + 1
        partition = job.partition or "unknown"
        summary = partitions.setdefault(
            partition,
            {"jobs": 0, "cpus": 0, "gpu_jobs": 0, "gpus": 0, "by_state": {}},
        )
        summary["jobs"] += 1
        summary["cpus"] += job.cpus or 0
        summary["gpu_jobs"] += int(job.is_gpu_job)
        summary["gpus"] += job.gpu_count
        summary["by_state"][job.state] = summary["by_state"].get(job.state, 0) + 1
    gpu_jobs = sum(job.is_gpu_job for job in jobs)
    return {
        "total": len(jobs),
        "by_state": counts,
        "by_partition": partitions,
        "requested_cpus": sum(job.cpus or 0 for job in jobs),
        "gpu_jobs": gpu_jobs,
        "cpu_jobs": len(jobs) - gpu_jobs,
        "requested_gpus": sum(job.gpu_count for job in jobs),
    }
