from typing import Dict, Iterable, Optional, Tuple

from coordinates import Coordinates
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass


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

        # Distance from first to last corrupted channel
        channel_offset = max(corr_channels) - min(corr_channels)
        
        return SpatialClassParameters(SpatialClass.BULLET_WAKE, 
            keys = {
                "corrupted_channels": len(corr_channels),
                "min_channel_skip": min(channel_skips),
                "max_channel_skip": max(channel_skips),
                "channel_offset": channel_offset
            },
            aggregate_values = {}
        )
    else:
        return None
