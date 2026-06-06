"""Priority job queue for batch dataset generation.

Why a custom queue when Gradio already queues HTTP requests? Gradio's queue is
FIFO over *requests*. Here a single request can enqueue many *page jobs*, and we
want cheap jobs (few pieces / small pages) to clear first so the user sees early
progress and tail latency drops. That ordering is a priority queue's job.

Data structure: binary min-heap (heapq) keyed by a (priority, seq) tuple.
`seq` is a monotonic counter that breaks ties FIFO and keeps ordering stable
without comparing the payloads.

Complexity:
    push  : O(log n)          worst & average
    pop   : O(log n)          worst & average
    peek  : Theta(1)
    space : Theta(n)
Big-Omega: push/pop are Omega(1) in the best case (no sift needed); the heap can
never beat Omega(log n) amortized when both ops interleave under arbitrary keys.
"""
from __future__ import annotations

import heapq
import itertools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(order=False)
class Job:
    priority: float
    payload: Any
    seq: int = field(default=0)


class PriorityJobQueue:
    """Thread-safe min-heap priority queue. Lower priority value = runs first."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Job]] = []
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def push(self, priority: float, payload: Any) -> Job:
        """O(log n). Smaller `priority` is dequeued earlier; ties are FIFO."""
        with self._lock:
            seq = next(self._counter)
            job = Job(priority=priority, payload=payload, seq=seq)
            heapq.heappush(self._heap, (priority, seq, job))
            return job

    def pop(self) -> Job | None:
        """O(log n). Returns the highest-priority job, or None if empty."""
        with self._lock:
            if not self._heap:
                return None
            return heapq.heappop(self._heap)[2]

    def peek(self) -> Job | None:
        """Theta(1). Look at the next job without removing it."""
        with self._lock:
            return self._heap[0][2] if self._heap else None

    def drain(self, handler: Callable[[Job], Any]) -> list[Any]:
        """Pop every job in priority order, applying `handler`. Returns results."""
        results = []
        while True:
            job = self.pop()
            if job is None:
                break
            results.append(handler(job))
        return results


def page_priority(piece_count: int, page_index: int) -> float:
    """Cheaper pages first; page_index breaks ties to keep document order-ish.

    Cost grows with piece_count (more masks to extract), so use it as the key.
    """
    return float(piece_count) * 1000.0 + float(page_index)
