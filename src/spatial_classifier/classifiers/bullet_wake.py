from typing import Dict, Iterable, Optional, Tuple

from coordinates import Coordinates
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass
from utils import quantize_percentage


def bullet_wake_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    """
    Return True if a bullet Wake spatial distribution is recognized
    Bullet Wake: the same location is corrupted in all (or in multiple) feature maps
    """
    first_N = sparse_diff[0].N
    first_W = sparse_diff[0].W
    first_H = sparse_diff[0].H
    for coordinates in sparse_diff:
        if (
            coordinates.N != first_N
            or coordinates.H != first_H
            or coordinates.W != first_W
        ):
            return None
    if len(sparse_diff) > 1:
        channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]  
        affected_channel_count = len(corr_channels)  

        # Distance from first to last corrupted channel
        channel_offset = max(corr_channels) - min(corr_channels)
        affected_channels_pct = quantize_percentage(affected_channel_count / shape.C)
        return SpatialClassParameters(SpatialClass.BULLET_WAKE, 
            keys = {
                "affected_channels_pct": affected_channels_pct
            },
            stats = {
                "max_corrupted_channels": (affected_channel_count, MaxAggregator()),
                "min_channel_skip": (min(channel_skips, default=1), MinAggregator()),
                "max_channel_skip": (max(channel_skips, default=1), MaxAggregator()),
            }
        )
    else:
        return None
