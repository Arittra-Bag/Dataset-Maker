"""Temp-file registry so 'Clear all' genuinely frees disk, not just the UI.

Every PDF/ZIP scratch file goes through `new_temp`, which records the path. A
later `clear_all` unlinks every recorded file. Thread-safe (a min worker pool
on HF still shares this process). UI-free so `src/` stays testable.

Note: this clears the *file cache* we create. Gradio's own request queue is
per-request and transient (a handler can't flush other users' pending events),
and the priority queue in `queue_manager` is built and drained within a single
`process_pdf` call - neither leaves persistent state to clear.
"""
from __future__ import annotations

import os
import tempfile
import threading

_lock = threading.Lock()
_tracked: set[str] = set()


def new_temp(suffix: str = "") -> str:
    """Create a tracked temp file and return its path (handle closed)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with _lock:
        _tracked.add(path)
    return path


def register(path: str) -> None:
    """Track an externally created path so clear_all() will remove it."""
    with _lock:
        _tracked.add(path)


def discard(path: str) -> bool:
    """Unlink one tracked file early (e.g. an input PDF after it's loaded).

    Returns True if the file was removed. Untracks regardless so a vanished
    file doesn't linger in the registry.
    """
    with _lock:
        _tracked.discard(path)
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def clear_all() -> int:
    """Unlink every tracked temp file. Returns count actually removed."""
    removed = 0
    with _lock:
        # list() snapshot is required: we mutate _tracked (discard) in-loop.
        # Iterating the set directly -> "Set changed size during iteration".
        for path in list(_tracked):  # NOSONAR python:S7504 false positive
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
            except OSError:
                continue  # leave it tracked; retry on next clear
            _tracked.discard(path)
    return removed


def tracked_count() -> int:
    with _lock:
        return len(_tracked)
