from collections import defaultdict
from typing import Dict, Iterable, Tuple, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass


def quasi_shattered_channel_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    if len(corr_channels) < 2:
        return None
    indexes_by_chan = {}
    chan_by_indexes = defaultdict(set)
    for chan in corr_channels:
        indexes_by_chan[chan] = set(
            raveled_channel_index(shape, coord)
            for coord in sparse_diff
            if coord.C == chan
        )
    for chan in indexes_by_chan:
        for idx in indexes_by_chan[chan]:
            chan_by_indexes[idx].add(chan)
    zero_index, common_channels = max(chan_by_indexes.items(), key=lambda s: len(s[1]))
    if len(common_channels) < 2:
        return None
    min_c = min(corr_channels)
    min_w_offset = min(idx - zero_index for idx in chan_by_indexes.keys())
    max_w_offset = max(idx - zero_index for idx in chan_by_indexes.keys())
    max_c_offset = max(corr_channels) - min_c
    feature_maps_count = len(corr_channels)
    error_pattern = tuple(
        (
            chan - min_c,
            tuple(error_idx - zero_index for error_idx in indexes_by_chan[chan]),
        )
        for chan in corr_channels
    )
    return SpatialClassParameters(SpatialClass.QUASI_SHATTERED_CHANNEL, 
        keys = {
            "error_pattern": error_pattern
        },
        aggregate_values = {
        }
    )