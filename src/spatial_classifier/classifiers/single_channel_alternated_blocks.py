from collections import defaultdict
from operator import itemgetter
from typing import Iterable, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.aggregators import MaxAggregator
from spatial_classifier.spatial_class import SpatialClass

def single_channel_alternated_blocks_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    if len(corr_channels) != 1:
        return None
    error_indexes = sorted(raveled_channel_index(shape, coord) for coord in sparse_diff)
    align = 32
    chan_size = shape.W * shape.H

    min_block_skip = 0
    max_block_skip = chan_size // align + 2

    blocks_affected = sorted(set(idx // align for idx in error_indexes))
    if len(blocks_affected) < 2:
        return None
    # Check that there aren't two consecutive blocks
    for i in range(1, len(blocks_affected)):
        block_skip = blocks_affected[i] - blocks_affected[i - 1]
        if block_skip < 2:
            return None
        min_block_skip = min(min_block_skip, block_skip)
        max_block_skip = max(max_block_skip, max_block_skip)
    # Check that every block has enough elements
    if chan_size % align != 0:
        # Last block is exempt from the check, if it is a remainder block
        exempt_channels = set([chan_size // align])
    else:
        exempt_channels = set()

    for block_id in set(blocks_affected) - exempt_channels:
        corrupted_elements = sum(1 for idx in error_indexes if idx // align == block_id)
        if corrupted_elements < align // 2:
            return None
    
    return SpatialClassParameters(
        SpatialClass.SINGLE_CHANNEL_ALTERNATED_BLOCKS,
        keys={
            "block_size": align,
            "min_block_skip": min_block_skip,
            "max_block_skip": max_block_skip
        },
        aggregate_values= {
            "max_feature_map_width": (shape.W, MaxAggregator()),
            "max_feature_map_height":  (shape.H, MaxAggregator())
        }
    )
