"""Temp-file registry so 'Clear all' genuinely frees disk, not just the UI.

Every PDF/ZIP scratch file goes through `new_temp`, which records the path. A
later `clear_all` unlinks every recorded file. Thread-safe (a min worker pool
on HF still shares this process). UI-free so `src/` stays testable.

Note: this clears the *file cache* we create. Gradio's own request queue is
per-request and transient (a handler can't flush other users' pending events),
and the priority queue in `queue_manager` is built and drained within a single
`process_pdf` call — neither leaves persistent state to clear.
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


def clear_all() -> int:
    """Unlink every tracked temp file. Returns count actually removed."""
    removed = 0
    with _lock:
        for path in list(_tracked):
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
