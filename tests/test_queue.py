"""Priority queue ordering + complexity-contract tests."""
from src.queue_manager import PriorityJobQueue, page_priority


def test_priority_order_cheap_first():
    q = PriorityJobQueue()
    q.push(page_priority(64, 0), "expensive")
    q.push(page_priority(4, 0), "cheap")
    q.push(page_priority(16, 0), "mid")
    order = [q.pop().payload for _ in range(3)]
    assert order == ["cheap", "mid", "expensive"]


def test_fifo_tie_break():
    q = PriorityJobQueue()
    for i in range(5):
        q.push(1.0, i)          # identical priority
    assert [q.pop().payload for _ in range(5)] == [0, 1, 2, 3, 4]


def test_pop_empty_returns_none():
    assert PriorityJobQueue().pop() is None


def test_len_and_peek():
    q = PriorityJobQueue()
    q.push(2.0, "b")
    q.push(1.0, "a")
    assert len(q) == 2
    assert q.peek().payload == "a"
    assert len(q) == 2          # peek does not consume
