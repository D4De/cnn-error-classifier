from collections import defaultdict
from operator import itemgetter
from typing import Iterable, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass


def multiple_channels_uncategorized_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
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
    return SpatialClassParameters(SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED, keys = {
        "error_pattern": error_pattern
    }, aggregate_values = {})
