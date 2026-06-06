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


def test_clear_all_tolerates_missing_file():
    p = workspace.new_temp(suffix=".tmp")
    os.remove(p)                      # vanish underneath the registry
    workspace.clear_all()             # must not raise
    assert workspace.tracked_count() == 0
