from collections import defaultdict
from operator import itemgetter
from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, raveled_channel_index


def random_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    indexes_by_channel = defaultdict(list)
    min_c = min(coord.C for coord in sparse_diff)
    min_index = min(raveled_channel_index(shape, coord) for coord in sparse_diff)
    for coord in sparse_diff:
        indexes_by_channel[coord.C - min_c].append(
            raveled_channel_index(shape, coord) - min_index
        )
        error_pattern = tuple(
            (chan, tuple(idx for idx in sorted(indexes)))
            for chan, indexes in sorted(indexes_by_channel.items(), key=itemgetter(0))
        )
    return True, {"error_pattern": error_pattern}
