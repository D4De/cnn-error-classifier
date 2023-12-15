from collections import defaultdict
from operator import itemgetter
from typing import Iterable, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass
from utils import quantize_percentage


def multiple_channels_uncategorized_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:

    if(len(corr_channels) < 2):
        return None

    errors_per_channel = defaultdict(int)

    for coord in sparse_diff:
        errors_per_channel[coord.C] += 1

    affected_channel_count = len(corr_channels)
    error_cardinality = len(sparse_diff)
    channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]      
    affected_channels_pct = quantize_percentage(affected_channel_count / shape.C)
    avg_channel_corruption_pct = quantize_percentage(error_cardinality / (affected_channel_count * shape.W * shape.H))



    return SpatialClassParameters(SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED, keys = {
        "avg_channel_corruption_pct": avg_channel_corruption_pct,
        "affected_channels_pct": affected_channels_pct
    }, 
    stats = {
        "max_corrupted_channels": (affected_channel_count, MaxAggregator()),
        "min_errors_per_channel": (min(errors_per_channel.values()), MinAggregator()),
        "max_errors_per_channel": (max(errors_per_channel.values()), MaxAggregator()),
        "min_channel_skip": (min(channel_skips, default=1), MinAggregator()),
        "max_channel_skip": (max(channel_skips, default=1), MaxAggregator())
    }) 
