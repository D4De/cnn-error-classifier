from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, identify_block_length, raveled_channel_index

import math

def full_single_channel_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(corr_channels) != 1:
        return False, {}
    chan_size = shape.W * shape.H
    corr_values = len(sparse_diff)
    corr_fraction = corr_values / chan_size
    if corr_fraction <= 0.5:
        return False, {}
    corr_pct = corr_fraction * 100
    corr_pct_rounded = int(math.ceil(corr_pct / 20) * 20)

    return True, {
        "error_pattern": corr_pct_rounded,
        "MAX": [],
    }
