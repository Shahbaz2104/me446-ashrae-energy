import numpy as np
import pandas as pd

from src.preprocessing import _allocate_proportionally


def test_proportional_allocation_respects_budget_and_minimum():
    sizes = pd.Series({"a": 50, "b": 30, "c": 20})
    allocation = _allocate_proportionally(sizes, total=10, minimum=1)
    assert sum(allocation.values()) == 10
    assert all(value >= 1 for value in allocation.values())
    assert allocation["a"] >= allocation["b"] >= allocation["c"]


def test_proportional_allocation_with_tight_budget():
    sizes = pd.Series({"a": 5, "b": 4, "c": 3})
    allocation = _allocate_proportionally(sizes, total=2, minimum=1)
    assert sum(allocation.values()) == 2
    assert set(allocation) == {"a", "b", "c"}
    assert all(value in (0, 1) for value in allocation.values())


def test_proportional_allocation_requires_room_for_minimum():
    sizes = pd.Series({"a": 1, "b": 1, "c": 1})
    allocation = _allocate_proportionally(sizes, total=5, minimum=1)
    assert sum(allocation.values()) == 5
