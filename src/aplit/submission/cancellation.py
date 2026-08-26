"""Verified cancellation of both the local controller and run-scoped SLURM jobs."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

from .models import RunState
from .processes import process_matches, terminate_process_group
from .registry import read_manifest, read_status, update_status, utc_now
from .scheduler import cancel_slurm_jobs, query_slurm_jobs


def cancel_project(project_dir: Path, *, wait_timeout: float = 20.0) -> dict[str, Any]:
    manifest = read_manifest(project_dir)
    status = read_status(project_dir)
    controller = status.get("controller") or {}
    update_status(
        project_dir,
        state=RunState.CANCELLING.value,
        cancellation_requested=True,
        message="Stopping controller and scheduler jobs",
    )

    pid = controller.get("pid")
    controller_host = controller.get("hostname")
    remote_controller = bool(
        controller_host and controller_host != socket.gethostname()
    )
    if not remote_controller and process_matches(pid, controller.get("start_ticks")):
        terminate_process_group(int(pid))

    site = manifest.get("site") or {}
    scheduler_jobs = ()
    if site.get("executor") == "slurm":
        squeue = Path(str(site.get("squeue_executable", "/usr/bin/squeue")))
        scancel = Path(str(site.get("scancel_executable", "/usr/bin/scancel")))
        scheduler_jobs = query_slurm_jobs(
            prefix=str(manifest["slurm_jobname_prefix"]),
            squeue_executable=squeue,
            strict=True,
        )
        cancel_slurm_jobs(scheduler_jobs, scancel_executable=scancel)
        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            scheduler_jobs = query_slurm_jobs(
                prefix=str(manifest["slurm_jobname_prefix"]),
                squeue_executable=squeue,
                strict=True,
            )
            if not scheduler_jobs:
                break
            time.sleep(0.25)

    controller_alive = process_matches(pid, controller.get("start_ticks"))
    if not remote_controller and not controller_alive and not scheduler_jobs:
        return update_status(
            project_dir,
            state=RunState.CANCELLED.value,
            finished_at=utc_now(),
            scheduler_jobs=[],
            message=None,
        )
    return update_status(
        project_dir,
        state=RunState.CANCELLING.value,
        scheduler_jobs=[job.to_dict() for job in scheduler_jobs],
        message=(
            f"SLURM jobs were cancelled, but the controller is on host {controller_host}; "
            "stop it on that host before marking the run cancelled."
            if remote_controller
            else "Cancellation requested; waiting for remaining processes or jobs"
        ),
    )
