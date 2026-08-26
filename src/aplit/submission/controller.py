"""Detached run controller invoked as ``python -m aplit.submission.controller``."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

from .models import RunState
from .registry import read_manifest, read_status, update_status, utc_now


def run_controller(project_dir: Path) -> int:
    manifest = read_manifest(project_dir)
    command = [str(item) for item in manifest.get("command") or ()]
    if not command:
        update_status(
            project_dir,
            state=RunState.FAILED.value,
            finished_at=utc_now(),
            exit_code=2,
            message="Run manifest has no Snakemake command",
        )
        return 2

    cancelled = False
    child: subprocess.Popen[bytes] | None = None

    def handle_termination(_signum: int, _frame: object) -> None:
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)

    update_status(
        project_dir,
        state=RunState.RUNNING.value,
        started_at=utc_now(),
        message=None,
    )
    try:
        child = subprocess.Popen(command, cwd=project_dir)
        return_code = child.wait()
    except OSError as exc:
        update_status(
            project_dir,
            state=RunState.FAILED.value,
            finished_at=utc_now(),
            exit_code=127,
            message=f"Could not launch Snakemake: {exc}",
        )
        return 127

    status = read_status(project_dir)
    was_cancelled = cancelled or bool(status.get("cancellation_requested"))
    if was_cancelled:
        state = RunState.CANCELLED
    elif return_code == 0:
        state = RunState.COMPLETED
    else:
        state = RunState.FAILED
    update_status(
        project_dir,
        state=state.value,
        finished_at=utc_now(),
        exit_code=return_code,
        message=None if return_code == 0 or was_cancelled else "Snakemake failed",
    )
    return return_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="APLit detached run controller")
    parser.add_argument("project", type=Path)
    args = parser.parse_args(argv)
    return run_controller(args.project.resolve())


if __name__ == "__main__":
    sys.exit(main())
