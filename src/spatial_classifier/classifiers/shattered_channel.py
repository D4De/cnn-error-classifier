from typing import Iterable, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass
from utils import quantize_percentage


def shattered_channel_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    if len(corr_channels) < 2:
        return None
    indexes_by_chan = {}
    for chan in corr_channels:
        indexes_by_chan[chan] = set(
            raveled_channel_index(shape, coord)
            for coord in sparse_diff
            if coord.C == chan
        )
    min_c = min(corr_channels)
    common_indexes = indexes_by_chan[min_c]
    all_indexes = set()
    for chan in corr_channels:
        common_indexes &= indexes_by_chan[chan]
        all_indexes |= indexes_by_chan[chan]
    if len(common_indexes) == 0:
        return None

    span_width_sum = 0
    max_span = -1
    for indexes in indexes_by_chan.values():
        min_idx = min(indexes)
        max_idx = max(indexes)
        span_width = max_idx - min_idx + 1
        span_width_sum += span_width
        max_span = max(max_span, span_width)
    avg_span_corruption_pct = quantize_percentage(len(sparse_diff) / span_width_sum)
    affected_channel_count = len(corr_channels)
    channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]      
    affected_channels_pct = quantize_percentage(affected_channel_count / shape.C)

    
    return SpatialClassParameters(SpatialClass.SHATTERED_CHANNEL, keys = {
        "avg_span_corruption_pct": avg_span_corruption_pct,
        "affected_channels_pct": affected_channels_pct,
    }, stats = {
        "max_corrupted_channels": (affected_channel_count, MaxAggregator()),
        "min_channel_skip": (min(channel_skips, default=1), MinAggregator()),
        "max_channel_skip": (max(channel_skips, default=1), MaxAggregator()),
        "min_span_width": (max_span, MinAggregator()),
        "max_span_width": (max_span, MaxAggregator())
    }) 