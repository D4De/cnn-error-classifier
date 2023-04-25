from collections import defaultdict
from operator import itemgetter
from typing import Dict, Iterable, Set, Tuple

from coordinates import Coordinates

def quasi_shattered_glass_pattern(
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
    cols_by_channels : Dict[int, Set[int]] = defaultdict(lambda: set())
    channels_by_cols : Dict[int, Set[int]] = defaultdict(lambda: set())
    for coord in sparse_diff:
        # If the corruption expands to different rows, then it is not shattered glass
        if coord.H != first_H:
            return False, {}
        cols_by_channels[coord.C].add(coord.W)
        channels_by_cols[coord.W].add(coord.C)
    if len(cols_by_channels) == 0:
        return False, {}
    common_cols = {col : len(channel_set) for col, channel_set in channels_by_cols.items() if len(channel_set) >= 1}
    if len(common_cols) > 0:
        common_element_col, rows_in_common = max(common_cols.items(), key=itemgetter(1))
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
