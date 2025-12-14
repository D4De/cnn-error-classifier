import numpy as np

from typing import Iterable, Optional

from utils import quantize_percentage
from coordinates import Coordinates
from spatial_classifier.spatial_class import SpatialClass
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class_parameters import SpatialClassParameters


def full_channels_pattern(
    diff_mask: np.ndarray,
    shape: Coordinates,
    corrupted_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    """
    Return True if a Full Channels spatial distribution is recognized.
    Full Channels Pattern: All corrupted channels have at least 50% of faulty values.
    """
    total_corrupted_count = np.count_nonzero(diff_mask)
    corrupted_channel_count = len(corrupted_channels)
    chan_size = shape.W * shape.H

    # diff_mask is a boolean ndarray: a True element indicates that the error tensor differs from the golden tensor in
    # that spot. Start by counting the number of differing elements per channel. This is done by counting the nonzero
    # elements (i.e. the True ones), reducing over the last two dimensions (columns and rows of the tensor)
    chan_fault_counts = np.count_nonzero(diff_mask, axis=(-1,-2))

    # compute the corruption fraction of each channel by dividing the fault count by the channel size
    chan_fault_fractions: np.ndarray = chan_fault_counts / chan_size
    # if any channel is less than 50% corrupted, the pattern recognition fails
    if np.any(chan_fault_fractions <= 0.5):
        return None

    # chan_pcts = {}
    # for chan in corrupted_channels:
    #     # Count the number of faults 
    #     chan_fault_count = sum( 1
    #         for coord in sparse_diff
    #         if coord.C == chan
    #     )
    #     # All channels have at least more than 50% of their values corrupted
    #     corr_fraction = chan_fault_count / chan_size
    #     if corr_fraction <= 0.5:
    #         return None
    #     corr_pct = corr_fraction * 100
    #     # Round up the corruption % to the higher multiple of 5
    #     corr_pct_rounded = int(math.ceil(corr_pct / 20) * 20)
    #     chan_pcts[chan] = corr_pct_rounded

    # compute the skip distance for each sequential pair of corrupted channels
    channel_skips = [curr - prev for prev, curr in zip(corrupted_channels, corrupted_channels[1:])]    

    # total corrupted elements / total elements in all corrupted channels
    avg_chan_corruption     = total_corrupted_count / (corrupted_channel_count * chan_size)
    avg_chan_corruption_pct = quantize_percentage(avg_chan_corruption)

    # number of corrupted channels / number of channels
    corrupted_channels_pct  = quantize_percentage(corrupted_channel_count / shape.C)
    
    # Distance from first to last corrupted channel
    # channel_offset = max(corr_channels) - min(corr_channels)
        
    return SpatialClassParameters(SpatialClass.FULL_CHANNELS, 
        keys = {
            "avg_channel_corruption_pct" : avg_chan_corruption_pct,
            "affected_channels_pct"      : corrupted_channels_pct
        },
        stats = {
            "max_corrupted_channels" : (corrupted_channel_count, MaxAggregator()),
            "min_channel_skip"       : (min(channel_skips, default=1), MinAggregator()),
            "max_channel_skip"       : (max(channel_skips, default=1), MaxAggregator()),
        }
    )
    