from typing import Dict, Iterable, Tuple

from coordinates import Coordinates


def same_column_pattern(
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
    first_W = sparse_diff[0].W
    for coordinates in sparse_diff:
        if (
            coordinates.N != first_N
            or coordinates.C != first_C
            or coordinates.W != first_W
        ):
            return False, {}
    all_hs = [coord.W for coord in sparse_diff]
    min_h = min(all_hs)
    max_h_offset = max(all_hs) - min_h
    return True, {
        "error_pattern": tuple(sorted(coord.W - min_h for coord in sparse_diff)),
        "max_h_offset": max_h_offset,
        "MAX": [max_h_offset],
    }
