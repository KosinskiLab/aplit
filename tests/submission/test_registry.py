from pathlib import Path

from aplit.submission.registry import (
    initialize_status,
    read_status,
    tail_log,
    update_status,
)


def test_status_updates_atomically(tmp_path: Path) -> None:
    project = tmp_path / "project"
    initialize_status(project)

    update_status(project, state="running", started_at="now")

    assert read_status(project)["state"] == "running"
    events = (project / ".aplit" / "events.jsonl").read_text()
    assert '"to": "running"' in events


def test_tail_log_is_bounded(tmp_path: Path) -> None:
    project = tmp_path / "project"
    path = project / ".aplit" / "controller.log"
    path.parent.mkdir(parents=True)
    path.write_text("old line\n" + "x" * 200 + "\nlast line\n")

    result = tail_log(project, max_bytes=40)

    assert "old line" not in result
    assert result.endswith("last line\n")
