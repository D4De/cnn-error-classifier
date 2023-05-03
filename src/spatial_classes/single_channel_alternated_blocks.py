from collections import defaultdict
from operator import itemgetter
from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, raveled_channel_index


def single_channel_alternated_blocks_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(corr_channels) != 1:
        return False, {}
    error_indexes = sorted(raveled_channel_index(shape, coord) for coord in sparse_diff)
    align = 32
    chan_size = shape.W * shape.H

    blocks_affected = sorted(set(idx // align for idx in error_indexes))
    if len(blocks_affected) < 2:
        return False, {}
    # Check that there aren't two consecutive blocks
    for i in range(1, len(blocks_affected)):
        if blocks_affected[i] - blocks_affected[i - 1] < 2:
            return False, {}
    # Check that every block has enough elements
    if chan_size % align != 0:
        # Last block is exempt from the check, if it is a remainder block
        exempt_channels = set([chan_size // align])
    else:
        exempt_channels = set()

    for block_id in set(blocks_affected) - exempt_channels:
        corrupted_elements = sum(1 for idx in error_indexes if idx // align == block_id)
        if corrupted_elements < align // 2:
            return False, {}

    min_block = blocks_affected[0]
    zero_idx = min_block * align
    max_idx = error_indexes[-1] - zero_idx
    error_pattern = (32, tuple(idx - zero_idx for idx in error_indexes))
    return True, {
        "error_pattern": error_pattern,
        "align": align,
        "MAX": [max_idx],
    }
