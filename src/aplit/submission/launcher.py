"""Launch a detached controller without depending on an interactive shell."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import ProjectBusyError, SubmissionError
from .models import RunState
from .processes import process_matches, process_start_ticks
from .registry import (
    append_event,
    atomic_write_json,
    log_path,
    project_lock,
    read_manifest,
    read_status,
    status_path,
    utc_now,
)


def launch_project(project_dir: Path) -> dict[str, Any]:
    manifest = read_manifest(project_dir)
    if not manifest.get("command"):
        raise SubmissionError("Project has no Snakemake command")

    with project_lock(project_dir):
        status = read_status(project_dir)
        controller = status.get("controller") or {}
        if process_matches(controller.get("pid"), controller.get("start_ticks")):
            raise ProjectBusyError("This project already has a running controller")
        if status.get("state") not in {
            RunState.READY.value,
            RunState.FAILED.value,
            RunState.CANCELLED.value,
        }:
            raise SubmissionError(
                f"Project cannot be submitted from state {status.get('state')!r}; "
                "validate it first."
            )

        status.update(
            {
                "state": RunState.SUBMITTING.value,
                "updated_at": utc_now(),
                "finished_at": None,
                "exit_code": None,
                "cancellation_requested": False,
                "message": None,
            }
        )
        atomic_write_json(status_path(project_dir), status)
        append_event(project_dir, "state_changed", {"to": RunState.SUBMITTING.value})

        path = log_path(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        controller_env = os.environ.copy()
        controller_env.pop("APLIT_ACCESS_TOKEN", None)
        with path.open("ab", buffering=0) as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "aplit.submission.controller",
                    str(project_dir),
                ],
                cwd=project_dir,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=controller_env,
            )

        start_ticks = None
        for _ in range(20):
            start_ticks = process_start_ticks(process.pid)
            if start_ticks is not None:
                break
            time.sleep(0.01)
        status["controller"] = {
            "pid": process.pid,
            "start_ticks": start_ticks,
            "hostname": socket.gethostname(),
        }
        status["updated_at"] = utc_now()
        atomic_write_json(status_path(project_dir), status)
        append_event(project_dir, "controller_started", status["controller"])
        return status
