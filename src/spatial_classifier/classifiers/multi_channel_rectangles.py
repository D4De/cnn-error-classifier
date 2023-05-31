from collections import defaultdict
from typing import Iterable, Optional

from coordinates import Coordinates
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass


def rectangles_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    
    MAX_MISSING = 2
    
    min_h = min(coord.H for coord in sparse_diff)
    max_h = max(coord.H for coord in sparse_diff)
    min_w = min(coord.W for coord in sparse_diff)
    max_w = max(coord.W for coord in sparse_diff)

    if max_h - min_h < 2:
        return None
    if max_w - min_w < 2:
        return None
    if min_w == 0 and max_w == shape.W - 1:
        return None
    
    grid = defaultdict(bool)

    for coord in sparse_diff:
        grid[(coord.H, coord.W)] = True

    missing = 0
    
    for h in range(min_h, max_h + 1):
        for w in range(min_w, max_w + 1):
            if not grid[h, w]:
                missing += 1
                if missing > MAX_MISSING:
                    return None
    
    channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]     
    channel_offset = max(corr_channels) - min(corr_channels)

    return SpatialClassParameters(
        SpatialClass.RECTANGLES,
        keys = {
            "corrupted_channels": len(corr_channels),
            "rectangle_width": max_w - min_w,
            "rectangle_heigth": max_h - min_h,
            "min_channel_skip": min(channel_skips, default=1),
            "max_channel_skip": max(channel_skips, default=1),
            "channel_offset": channel_offset
        },
        aggregate_values= {}
    )
