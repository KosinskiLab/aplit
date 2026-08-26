from __future__ import annotations

import time
from pathlib import Path

from aplit.submission.cancellation import cancel_project
from aplit.submission.launcher import launch_project
from aplit.submission.models import InputMode, RunSpec, SiteConfig
from aplit.submission.planner import plan_project
from aplit.submission.project import prepare_project
from aplit.submission.registry import read_status


def wait_for_state(project: Path, expected: str, timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    last = read_status(project)
    while time.monotonic() < deadline:
        last = read_status(project)
        if last["state"] == expected:
            return last
        time.sleep(0.05)
    raise AssertionError(f"Expected state {expected!r}; last status was {last!r}")


def prepare_ready_project(fake_site: SiteConfig, name: str) -> Path:
    prepared = prepare_project(
        RunSpec(
            project_name=name,
            input_mode=InputMode.DIRECT,
            backend="af3",
            direct_complexes=("P1+P2",),
        ),
        fake_site,
    )
    plan_project(prepared.path)
    return prepared.path


def test_detached_controller_records_completion(fake_site: SiteConfig) -> None:
    fake_site.snakemake_executable.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in *--dry-run*) echo 'Job stats:'; echo 'all 1'; exit 0;; esac\n"
        "echo running fake workflow\n"
        "exit 0\n",
        encoding="utf-8",
    )
    project = prepare_ready_project(fake_site, "complete")

    launched = launch_project(project)
    completed = wait_for_state(project, "completed")

    assert launched["controller"]["pid"] > 0
    assert completed["exit_code"] == 0


def test_cancel_stops_long_running_controller(fake_site: SiteConfig) -> None:
    fake_site.snakemake_executable.write_text(
        "#!/bin/sh\n"
        'case "$*" in *--dry-run*) exit 0;; esac\n'
        "trap 'exit 143' TERM INT\n"
        "sleep 30 & wait\n",
        encoding="utf-8",
    )
    project = prepare_ready_project(fake_site, "cancelled")
    launch_project(project)
    wait_for_state(project, "running")

    cancel_project(project, wait_timeout=2)
    cancelled = wait_for_state(project, "cancelled")

    assert cancelled["cancellation_requested"] is True
