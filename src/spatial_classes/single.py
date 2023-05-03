from typing import Dict, Iterable, Tuple

from coordinates import Coordinates


def single_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(sparse_diff) == 1:
        return True, {"error_pattern": 0}
    else:
        return False, {}


def single_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(sparse_diff) == 1:
        return True, {"error_pattern": 0}
    else:
        return False, {}


def same_row_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Return True if a bullet Wake spatial distribution is recognized
    Same Row: multiple corrupted values lie in the same row (same feature map)
    """
    first_N = sparse_diff[0].N
    first_C = sparse_diff[0].C
    first_H = sparse_diff[0].H
    for coordinates in sparse_diff:
        if (
            coordinates.N != first_N
            or coordinates.C != first_C
            or coordinates.H != first_H
        ):
            return False, {}
    all_ws = [coord.W for coord in sparse_diff]
    min_w = min(all_ws)
    max_w_offset = max(all_ws) - min_w
    return True, {
        "error_pattern": tuple(sorted(coord.W - min_w for coord in sparse_diff)),
        "max_w_offset": max_w_offset,
        "MAX": [max_w_offset],
    }
