import pytest

from aplit.submission.errors import SubmissionError
from aplit.submission.models import InputMode, RunSpec, RunState, SiteConfig
from aplit.submission.planner import parse_job_stats, plan_project
from aplit.submission.project import prepare_project
from aplit.submission.registry import read_status, update_status


def test_parse_job_stats() -> None:
    output = """Job stats:
job                   count
------------------  -------
all                        1
create_features            2
structure_inference        3
total                      6

Reasons:
"""

    total, jobs = parse_job_stats(output)

    assert total == 6
    assert jobs == {"all": 1, "create_features": 2, "structure_inference": 3}


def test_successful_dry_run_marks_project_ready(fake_site: SiteConfig) -> None:
    fake_site.snakemake_executable.write_text(
        "#!/bin/sh\n"
        "echo 'Job stats:'\n"
        "echo 'job count'\n"
        "echo 'all 1'\n"
        "echo 'structure_inference 2'\n"
        "echo 'total 3'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    prepared = prepare_project(
        RunSpec(
            project_name="planned",
            input_mode=InputMode.DIRECT,
            backend="af3",
            direct_complexes=("P1+P2", "P3+P4"),
        ),
        fake_site,
    )

    result = plan_project(prepared.path)

    assert result.valid
    assert result.requested_folds == 2
    assert result.dag_jobs == 3
    assert read_status(prepared.path)["state"] == "ready"


def test_dry_run_cannot_overwrite_active_state(fake_site: SiteConfig) -> None:
    prepared = prepare_project(
        RunSpec(
            project_name="active",
            input_mode=InputMode.DIRECT,
            backend="af3",
            direct_complexes=("P1+P2",),
        ),
        fake_site,
    )
    update_status(prepared.path, state=RunState.RUNNING.value)

    with pytest.raises(SubmissionError, match="cannot be validated"):
        plan_project(prepared.path)

    assert read_status(prepared.path)["state"] == RunState.RUNNING.value
