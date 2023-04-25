from collections import defaultdict
from operator import itemgetter
from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, raveled_channel_index

def skip_4_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
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
        indexes_by_channel = defaultdict(list)
        min_c = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - min_c
        min_index = smallest_coordinate
        for coord in sparse_diff:
            indexes_by_channel[coord.C - min_c].append(
                (raveled_channel_index(shape, coord) - min_index)
            )
            error_pattern = tuple(
                (chan, tuple(idx for idx in sorted(indexes)))
                for chan, indexes in sorted(
                    indexes_by_channel.items(), key=itemgetter(0)
                )
            )
        max_idx_offset = max(coordinates) - smallest_coordinate
        return True, {
            "error_pattern": error_pattern,
            "max_c_offset": max_c_offset,
            "max_idx_offset": max_idx_offset,
            "MAX": [max_c_offset, max_idx_offset],
        }
    else:
        return False, {}