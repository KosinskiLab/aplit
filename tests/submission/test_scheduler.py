from pathlib import Path

import pytest

from aplit.submission.errors import SubmissionError
from aplit.submission.scheduler import query_slurm_jobs


def test_scheduler_query_is_scoped_to_exact_prefix(tmp_path: Path) -> None:
    squeue = tmp_path / "squeue"
    squeue.write_text(
        "#!/bin/sh\n"
        "echo '101|aplit_abc_uuid|RUNNING'\n"
        "echo '102|aplit_abcd_uuid|PENDING'\n"
        "echo '103|other|RUNNING'\n",
        encoding="utf-8",
    )
    squeue.chmod(0o755)

    jobs = query_slurm_jobs(prefix="aplit_abc", squeue_executable=squeue)

    assert [(job.job_id, job.state) for job in jobs] == [("101", "RUNNING")]


def test_strict_scheduler_query_does_not_treat_failure_as_empty(tmp_path: Path) -> None:
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/sh\necho unavailable >&2\nexit 1\n", encoding="utf-8")
    squeue.chmod(0o755)

    with pytest.raises(SubmissionError, match="unavailable"):
        query_slurm_jobs(prefix="aplit_abc", squeue_executable=squeue, strict=True)
