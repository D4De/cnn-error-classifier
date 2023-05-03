from collections import defaultdict
from operator import itemgetter
from typing import Dict, Iterable, Tuple

from coordinates import Coordinates


def shattered_glass_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Return True if a Shattered Glass spatial distribution is recognized.
    Shattered glass: like one or more Bullet wake errors, but in one or multiple feature maps the corruption spreads over a row (or part of the row)
    """
    # Common Row Index
    first_H = sparse_diff[0].H
    cols_by_channels = defaultdict(lambda: set())
    for coord in sparse_diff:
        # To be Shattered Glass all corruption must stay on the same row of different feature map
        if coord.H != first_H:
            return False, {}
        cols_by_channels[coord.C].add(coord.W)
    if len(cols_by_channels) == 0:
        return False, {}
    common_cols = cols_by_channels[list(cols_by_channels.keys())[0]]
    # Check if there is a common corrupted position in all corrupted feature maps
    for cols_set in cols_by_channels.values():
        common_cols &= cols_set
    if len(common_cols) > 0:
        common_element_col = next(iter(common_cols))
        smallest_chan = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - smallest_chan
        error_pattern = tuple(
            (
                chan - smallest_chan,
                tuple(col - common_element_col for col in sorted(cols)),
            )
            for chan, cols in sorted(cols_by_channels.items(), key=itemgetter(0))
        )
        min_w_offset = min(coord.W - common_element_col for coord in sparse_diff)
        max_w_offset = max(coord.W - common_element_col for coord in sparse_diff)
        feature_maps_count = len(cols_by_channels)
        return True, {
            "error_pattern": error_pattern,
            "min_w_offset": min_w_offset,
            "max_w_offset": max_w_offset,
            "max_c_offset": max_c_offset,
            "feature_maps_count": feature_maps_count,
            "MAX": [feature_maps_count, max_c_offset, min_w_offset, max_w_offset],
        }
    else:
        return False, {}
