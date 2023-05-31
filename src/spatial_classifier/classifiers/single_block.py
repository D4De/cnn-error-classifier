import math
from typing import Iterable, Optional

from coordinates import Coordinates, identify_block_length, raveled_tensor_index
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass


def single_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    indexes = sorted(raveled_tensor_index(shape, coord) for coord in sparse_diff)
    block_begin = indexes[0]
    result = identify_block_length(indexes, min_block_size = 8, max_block_size=shape.W * shape.H)
    if result is None:
        return None
    block_length, aligment_offset, block_id = result
    block_corruption_pct = max(math.ceil(len(indexes) / (block_length * 20)) * 5, 100)
    return SpatialClassParameters(SpatialClass.SINGLE_BLOCK, 
        keys = {
            "block_length": block_length,
            "block_corruption_pct": block_corruption_pct
        }, 
        aggregate_values = {})