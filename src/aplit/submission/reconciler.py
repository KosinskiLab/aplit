"""Reconcile persisted status with controller identity and the scheduler."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from .models import RunState
from .processes import process_matches
from .registry import read_manifest, read_status, update_status
from .scheduler import SchedulerJob, filter_slurm_jobs, query_slurm_jobs

ACTIVE_STATES = {
    RunState.SUBMITTING.value,
    RunState.QUEUED.value,
    RunState.RUNNING.value,
    RunState.CANCELLING.value,
}


def reconcile_project(
    project_dir: Path,
    *,
    scheduler_snapshot: tuple[SchedulerJob, ...] | None = None,
) -> dict[str, Any]:
    manifest = read_manifest(project_dir)
    status = read_status(project_dir)
    controller = status.get("controller") or {}
    same_host = controller.get("hostname") == socket.gethostname()
    controller_alive = same_host and process_matches(
        controller.get("pid"), controller.get("start_ticks")
    )

    site = manifest.get("site") or {}
    jobs = ()
    if site.get("executor") == "slurm":
        prefix = str(manifest["slurm_jobname_prefix"])
        if scheduler_snapshot is None:
            jobs = query_slurm_jobs(
                prefix=prefix,
                squeue_executable=Path(
                    str(site.get("squeue_executable", "/usr/bin/squeue"))
                ),
            )
        else:
            jobs = filter_slurm_jobs(scheduler_snapshot, prefix=prefix)

    changes: dict[str, Any] = {"scheduler_jobs": [job.to_dict() for job in jobs]}
    state = str(status.get("state"))
    if state in ACTIVE_STATES:
        running = any(job.state.upper() in {"RUNNING", "COMPLETING"} for job in jobs)
        pending = any(job.state.upper() in {"PENDING", "CONFIGURING"} for job in jobs)
        if status.get("cancellation_requested"):
            changes["state"] = (
                RunState.CANCELLING.value
                if jobs or controller_alive
                else RunState.CANCELLED.value
            )
        elif controller_alive and running:
            changes["state"] = RunState.RUNNING.value
        elif controller_alive and pending:
            changes["state"] = RunState.QUEUED.value
        elif controller_alive:
            changes["state"] = RunState.RUNNING.value
        elif jobs:
            changes["state"] = RunState.UNKNOWN.value
            changes["message"] = "SLURM jobs exist but the Snakemake controller is gone"
        elif status.get("exit_code") is None:
            changes["state"] = RunState.UNKNOWN.value
            changes["message"] = (
                "Controller is no longer running and no exit status was recorded"
            )

    if changes.get("state") == RunState.CANCELLED.value:
        from .registry import utc_now

        changes["finished_at"] = status.get("finished_at") or utc_now()
    return update_status(project_dir, **changes)
