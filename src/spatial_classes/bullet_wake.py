from typing import Dict, Iterable, Tuple

from coordinates import Coordinates

def bullet_wake_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Return True if a bullet Wake spatial distribution is recognized
    Bullet Wake: the same location is corrupted in all (or in multiple) feature maps
    """
    first_N = sparse_diff[0].N
    first_W = sparse_diff[0].W
    first_H = sparse_diff[0].H
    for coordinates in sparse_diff:
        if (
            coordinates.N != first_N
            or coordinates.H != first_H
            or coordinates.W != first_W
        ):
            return False, {}
    if len(sparse_diff) > 1:
        min_c = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - min_c
        return True, {
            "error_pattern": tuple(coord.C - min_c for coord in sparse_diff),
            "max_c_offset": max_c_offset,
            "MAX": [max_c_offset],
        }
    else:
        return False, {}