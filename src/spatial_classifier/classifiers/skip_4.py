from collections import defaultdict
import math
from operator import itemgetter
from typing import Dict, Iterable, Tuple, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass


def skip_4_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    """
    Return True if a Skip4 pattern is identified

    A skip 4 pattern happens when there are errors every 4 values (using raveled channel index) in each corrupeted channel.
    This algorithm finds the lowest raveled channel index of a corrupted values, called smallest_coordinate (the same for all channels).

    Errors are expected only in the values that have distance divisible by 4 from smallest_coordinate.

    To match there must be globally at least 3 raveled channel index that follow the criteria (good_positions), and there must not be more than 3
    positions that don't respect the criteria (bad_positions).

    The position 0 does not count either for the good_positions and for the wrong_positions
    """
    coordinates = set([raveled_channel_index(shape, coord) for coord in sparse_diff])
    smallest_coordinate = min(coordinates)
    # to be a skip 4 error it should be true that (raveled_channel_index(error) === smallest_coordinate mod 4)
    # 0 is always allowed inside the skip4 pattern
    candidate_positions = set(range(smallest_coordinate, shape.H * shape.W, 4))
    # All the errors in the tensor that have a distance multiple of 4
    good_positions = coordinates & candidate_positions
    # all the other positions
    wrong_positions = coordinates - candidate_positions

    if len(good_positions) >= 2 and len(wrong_positions) <= 1:
        # Generate the error_pattern
        min_c = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - min_c

        max_idx_offset = max(coordinates) - smallest_coordinate

        channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]  
        slots_corruption_pct = max(math.ceil(len(sparse_diff) / (len(good_positions) * len(corr_channels) * 20)) * 5, 100)

        return SpatialClassParameters(SpatialClass.SKIP_4, 
            keys = {
                "skip_amount": 4,
                "corrupted_channels": len(corr_channels),
                "min_channel_skip": min(channel_skips, default=1),
                "max_channel_skip": max(channel_skips, default=1),
                "value_offset": max_idx_offset,
                "corruption_pct": slots_corruption_pct,
                "channel_offset": max_c_offset
            },
            aggregate_values = {}
        )
    else:
        return None
