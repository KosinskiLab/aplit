from pathlib import Path

import pytest

from aplit.submission.errors import SubmissionError
from aplit.submission.scheduler import query_slurm_jobs, summarize_jobs


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


def test_scheduler_summarizes_cpu_and_gpu_partitions(tmp_path: Path) -> None:
    squeue = tmp_path / "squeue"
    squeue.write_text(
        "#!/bin/sh\n"
        "echo '201|aplit_run_a|RUNNING|gpu-el8|8|26279|03:00:00|00:12|gpu21|gpu:1'\n"
        "echo '202|aplit_run_b|PENDING|htc-el10|16|64000|1-00:00:00|0:00|(Resources)|(null)'\n"
        "echo '203|aplit_run_c|RUNNING|gpu-training|12|48000|06:00:00|01:02|gpu38|gpu:h100:2'\n",
        encoding="utf-8",
    )
    squeue.chmod(0o755)

    jobs = query_slurm_jobs(prefix="aplit_run", squeue_executable=squeue)
    summary = summarize_jobs(jobs)

    assert summary["total"] == 3
    assert summary["cpu_jobs"] == 1
    assert summary["gpu_jobs"] == 2
    assert summary["requested_cpus"] == 36
    assert summary["requested_gpus"] == 3
    assert summary["by_partition"]["gpu-el8"]["gpu_jobs"] == 1
    assert summary["by_partition"]["htc-el10"]["by_state"] == {"PENDING": 1}
    assert jobs[1].reason_or_nodes == "(Resources)"
