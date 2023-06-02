from collections import defaultdict
from typing import Dict, Iterable, Tuple, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass
from utils import quantize_percentage


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
    span_width_sum = 0
    min_span = shape.H * shape.W + 1
    max_span = -1

    for indexes in indexes_by_chan.values():
        min_idx = min(indexes)
        max_idx = max(indexes)
        span_width = max_idx - min_idx
        span_width_sum += span_width
        min_span = min(min_span, span_width)
        max_span = max(max_span, span_width)

    avg_span_corruption_pct = quantize_percentage(len(sparse_diff) / span_width_sum)
    affected_channel_count = len(corr_channels)
    channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]      
    affected_channels_pct = quantize_percentage(affected_channel_count / shape.C)

    
    return SpatialClassParameters(SpatialClass.QUASI_SHATTERED_CHANNEL, keys = {
        "avg_span_corruption_pct": avg_span_corruption_pct,
        "affected_channels_pct": affected_channels_pct
    }, stats = {
        "max_corrupted_channels": (affected_channel_count, MaxAggregator()),
        "min_channel_skip": (min(channel_skips, default=1), MinAggregator()),
        "max_channel_skip": (max(channel_skips, default=1), MaxAggregator()),
        "min_span_width": (min_span, MinAggregator()),
        "max_span_width": (max_span, MaxAggregator())
    }) 