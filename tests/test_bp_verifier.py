"""Keep individual item IDs distinct from aggregate size-type indices."""
from __future__ import annotations

import pytest

from src.cg.bp.d_verifier import verify_solution


ITEM_DATA = {
    "bin_capacity": 10,
    "items": [{"id": 0, "size": 6}, {"id": 1, "size": 4}, {"id": 2, "size": 4}],
}


@pytest.mark.parametrize("bins", [
    [[0, 0], [1]],
    [{"contents": [{"item_id": 0, "count": 2}]}, {"contents": [{"item_id": 1, "count": 1}]}],
])
def test_individual_items_cannot_be_reinterpreted_as_size_types(bins):
    # As type indices, 0 and 1 mean sizes 4 and 6 and these bins appear valid.
    # As the actual item IDs, item 0 is repeated and item 2 is missing.
    result = verify_solution(ITEM_DATA, {"solution": {"bins": bins}})
    assert not result["ok"]
    assert result["objective_value"] is None


def test_capacity_is_checked_using_actual_item_sizes():
    data = {"bin_capacity": 10, "items": [
        {"id": 0, "size": 8}, {"id": 1, "size": 2}, {"id": 2, "size": 6}, {"id": 3, "size": 4},
    ]}
    # Sorting sizes would make both bins load 10, but actual items 0 and 3 load 12.
    result = verify_solution(data, {"solution": {"bins": [[0, 3], [1, 2]]}})
    assert not result["ok"]
    assert "exceeds capacity" in result["issues"][0]


@pytest.mark.parametrize("bins", [
    [[0, 1], [2]],
    [{"contents": [{"item_id": 0}, {"item_id": 1}]}, {"contents": [{"item_id": 2}]}],
])
def test_valid_individual_item_packing_still_passes(bins):
    result = verify_solution(ITEM_DATA, {"solution": {"bins": bins}})
    assert result["ok"], result["issues"]
    assert result["objective_value"] == 2


def test_individual_item_ids_need_not_be_contiguous():
    data = {"bin_capacity": 10, "items": [{"id": 17, "size": 6}, {"id": 4, "size": 4}]}
    result = verify_solution(data, {"solution": {"bins": [[17, 4]]}})
    assert result["ok"], result["issues"]
    assert result["objective_value"] == 1


def test_explicit_item_types_still_support_aggregate_patterns():
    data = {"bin_capacity": 10, "item_types": [
        {"id": 0, "size": 4, "demand": 2}, {"id": 1, "size": 6, "demand": 1},
    ]}
    bins = [{"contents": [{"item_id": 0, "count": 2}]}, {"contents": [{"item_id": 1}]}]
    result = verify_solution(data, {"solution": {"bins": bins}})
    assert result["ok"], result["issues"]
    assert result["objective_value"] == 2
