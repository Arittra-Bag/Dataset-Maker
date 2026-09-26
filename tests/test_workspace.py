"""Temp-file registry: 'Clear all' must genuinely unlink tracked files."""
import os

from src import workspace


def test_new_temp_tracked_then_cleared():
    before = workspace.tracked_count()
    p1 = workspace.new_temp(suffix=".pdf")
    p2 = workspace.new_temp(suffix=".zip")
    assert os.path.exists(p1) and os.path.exists(p2)
    assert workspace.tracked_count() == before + 2

    removed = workspace.clear_all()
    assert removed >= 2
    assert not os.path.exists(p1) and not os.path.exists(p2)
    assert workspace.tracked_count() == 0


def test_discard_unlinks_and_untracks():
    p = workspace.new_temp(suffix=".pdf")
    before = workspace.tracked_count()
    assert workspace.discard(p) is True
    assert not os.path.exists(p)
    assert workspace.tracked_count() == before - 1
    # idempotent: discarding again is harmless
    assert workspace.discard(p) is False


def test_clear_all_tolerates_missing_file():
    p = workspace.new_temp(suffix=".tmp")
    os.remove(p)                      # vanish underneath the registry
    workspace.clear_all()             # must not raise
    assert workspace.tracked_count() == 0


def test_clear_stale_keeps_fresh_files():
    import time

    old = workspace.new_temp(suffix=".zip")
    fresh = workspace.new_temp(suffix=".zip")
    past = time.time() - 7200
    os.utime(old, (past, past))
    workspace.clear_stale(3600)
    assert not os.path.exists(old)
    assert os.path.exists(fresh)          # e.g. another session's in-flight run
    assert workspace.discard(fresh) is True


def test_clear_stale_untracks_vanished_file():
    p = workspace.new_temp(suffix=".tmp")
    os.remove(p)
    before = workspace.tracked_count()
    workspace.clear_stale(3600)           # must not raise
    assert workspace.tracked_count() == before - 1
