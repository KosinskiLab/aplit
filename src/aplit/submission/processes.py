"""Process identity helpers that avoid treating a reused PID as our controller."""

from __future__ import annotations

import os
from pathlib import Path


def process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, IndexError, ValueError):
        return None


def process_matches(pid: int | None, start_ticks: int | None) -> bool:
    if not pid or start_ticks is None:
        return False
    current = process_start_ticks(int(pid))
    return current is not None and current == int(start_ticks)


def terminate_process_group(pid: int, *, force: bool = False) -> bool:
    try:
        os.killpg(pid, 9 if force else 15)
        return True
    except ProcessLookupError:
        return False
