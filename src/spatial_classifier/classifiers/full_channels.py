from typing import Iterable, Optional
from utils import count_by

from coordinates import Coordinates

import math
from spatial_classifier.spatial_class_parameters import SpatialClassParameters

from spatial_classifier.spatial_class import SpatialClass

def full_channels_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    """
    Return True if a Shattered Glass spatial distribution is recognized.
    Full Channels Pattern: All corrupted channels have at least 50% of faulty values
    """
    chan_size = shape.W * shape.H
    chan_pcts = {}
    for chan in corr_channels:
        # Count the number of faults 
        chan_fault_count = sum( 1
            for coord in sparse_diff
            if coord.C == chan
        )
        # All channels have at least more than 50% of their values corrupted
        corr_fraction = chan_fault_count / chan_size
        if corr_fraction <= 0.5:
            return None
        corr_pct = corr_fraction * 100
        # Round up the corruption % to the higher multiple of 5
        corr_pct_rounded = int(math.ceil(corr_pct / 20) * 20)
        chan_pcts[chan] = corr_pct_rounded


    channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]    
    pct_corrupt_keys_dict = count_by(chan_pcts.values(), key=lambda x: f'{x}%_corrupted_channels')
    
    # Distance from first to last corrupted channel
    channel_offset = max(corr_channels) - min(corr_channels)
        
    return SpatialClassParameters(SpatialClass.FULL_CHANNELS, 
        keys = {
            "corrupted_channels": len(corr_channels),
            "min_channel_skip": min(channel_skips, default=1),
            "max_channel_skip": max(channel_skips, default=1),
            "channel_offset": channel_offset,
            **pct_corrupt_keys_dict
        },
        aggregate_values = {}
    )
    

