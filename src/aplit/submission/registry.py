"""Filesystem run registry with atomic status updates and an append-only event log."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ProjectBusyError, RunNotFoundError
from .models import RunState

METADATA_DIR = ".aplit"
MANIFEST_FILE = "run.json"
STATUS_FILE = "status.json"
EVENTS_FILE = "events.jsonl"
LOG_FILE = "controller.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_dir(project_dir: Path) -> Path:
    return project_dir / METADATA_DIR


def manifest_path(project_dir: Path) -> Path:
    return metadata_dir(project_dir) / MANIFEST_FILE


def status_path(project_dir: Path) -> Path:
    return metadata_dir(project_dir) / STATUS_FILE


def log_path(project_dir: Path) -> Path:
    return metadata_dir(project_dir) / LOG_FILE


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some network filesystems do not support directory fsync. The
            # same-directory rename remains the atomic publication mechanism.
            pass
    finally:
        temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(10):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            break
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 9:
                time.sleep(0.02 * (attempt + 1))
    else:
        raise RunNotFoundError(f"Run metadata is unavailable: {path}") from last_error
    if not isinstance(value, dict):
        raise RunNotFoundError(f"Run metadata is not an object: {path}")
    return value


def read_manifest(project_dir: Path) -> dict[str, Any]:
    return read_json(manifest_path(project_dir))


def read_status(project_dir: Path) -> dict[str, Any]:
    return read_json(status_path(project_dir))


def append_event(
    project_dir: Path, event: str, details: Mapping[str, Any] | None = None
) -> None:
    record = {
        "at": utc_now(),
        "event": event,
        "details": dict(details or {}),
    }
    path = metadata_dir(project_dir) / EVENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def project_lock(
    project_dir: Path, *, timeout: float = 5.0, stale_after: float = 300.0
) -> Iterator[None]:
    """Acquire a short mutation lock using atomic file creation."""

    lock_file = metadata_dir(project_dir) / "mutation.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    payload = json.dumps(
        {"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": utc_now()}
    ).encode("utf-8")

    while True:
        try:
            descriptor = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(descriptor, payload)
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_file.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after:
                try:
                    lock_file.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise ProjectBusyError(f"Project is busy: {project_dir}")
            time.sleep(0.05)

    try:
        yield
    finally:
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


def initialize_status(project_dir: Path) -> dict[str, Any]:
    now = utc_now()
    status = {
        "schema_version": 1,
        "state": RunState.DRAFT.value,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "controller": None,
        "cancellation_requested": False,
        "scheduler_jobs": [],
        "message": None,
    }
    atomic_write_json(status_path(project_dir), status)
    append_event(project_dir, "project_created", {"state": status["state"]})
    return status


def update_status(project_dir: Path, **changes: Any) -> dict[str, Any]:
    with project_lock(project_dir):
        status = read_status(project_dir)
        previous_state = status.get("state")
        status.update(changes)
        status["updated_at"] = utc_now()
        atomic_write_json(status_path(project_dir), status)
        if status.get("state") != previous_state:
            append_event(
                project_dir,
                "state_changed",
                {"from": previous_state, "to": status.get("state")},
            )
        return status


def list_projects(projects_root: Path) -> list[Path]:
    if not projects_root.exists():
        return []
    projects = [
        child
        for child in projects_root.iterdir()
        if child.is_dir() and manifest_path(child).is_file()
    ]
    return sorted(
        projects,
        key=lambda path: (
            status_path(path).stat().st_mtime if status_path(path).exists() else 0
        ),
        reverse=True,
    )


def tail_log(project_dir: Path, *, max_bytes: int = 64_000) -> str:
    path = log_path(project_dir)
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except FileNotFoundError:
        return ""
    if size > max_bytes:
        first_newline = data.find(b"\n")
        if first_newline >= 0:
            data = data[first_newline + 1 :]
    return data.decode("utf-8", "replace")
