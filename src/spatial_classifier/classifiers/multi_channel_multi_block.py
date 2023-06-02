from typing import Dict, Iterable, Optional, Tuple

from coordinates import Coordinates, identify_block_length, raveled_channel_index
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class import SpatialClass
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from utils import quantize_percentage

def multi_channel_multi_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    # There should be at least two channels involved
    if len(corr_channels) < 2:
        return None
    block_start = shape.H * shape.W
    # Stores the block with the biggest length
    max_block_length = -1
    # The "zero channel" that has the biggest block
    leader_chan = None
    # For each channel store the corrupted raveled channel indexes
    indexes_by_channel = {}
    for chan in corr_channels:
        # List containing all the corrupted raveled channel index of the current chan
        this_chan_indexes = sorted(
            raveled_channel_index(shape, coord)
            for coord in sparse_diff
            if coord.C == chan
        )
        first_chan_error = this_chan_indexes[0]
        indexes_by_channel[chan] = this_chan_indexes
        result = identify_block_length(this_chan_indexes, min_block_size = 8, max_block_size=shape.W * shape.H)
        # No block identified -> This could not be a leader channel
        if result is None:
            continue
        # Block identified, compare with the leader to check if this could be the new leader
        block_length, aligment_offset, block_id = result
        # If this channel has a longer block than the current leader, set it as new leader
        if block_length > max_block_length:
            max_block_length = block_length
            block_start = first_chan_error
            leader_chan = chan
        # In case of a tie in the block length, the leader is the one that has the first error as lowest index
        elif block_length == max_block_length and first_chan_error < block_start:
            block_start = first_chan_error
            leader_chan = chan
    if leader_chan is None:
        return None
    found_mismatch = False
    # To match with this pattern, in all corrupted tensors the errors must be contained in the same block
    # Scan all the blocks (except the leader) to check if there no are errors outside the block
    for chan in corr_channels:
        if chan == leader_chan:
            continue
        # Found errors 
        if not all(
            block_start <= error_idx <= block_start + max_block_length
            for error_idx in indexes_by_channel[chan]
        ):
            result = identify_block_length(indexes_by_channel[chan])
            if result is None:
                found_mismatch = True

    if found_mismatch:
        return None

    affected_channel_count = len(corr_channels)
    channel_skips = [curr - prev for prev, curr in zip(corr_channels, corr_channels[1:])]      
    affected_channels_pct = quantize_percentage(affected_channel_count / shape.C)
    avg_block_corruption_pct = quantize_percentage(len(sparse_diff) / (max_block_length * affected_channel_count))

    return SpatialClassParameters(
        SpatialClass.MULTI_CHANNEL_BLOCK,
        keys = {
            "block_size": max_block_length,
            "avg_block_corruption_pct": avg_block_corruption_pct,
            "affected_channels_pct": affected_channels_pct,
        },
        stats = {
            "max_corrupted_channels": (affected_channel_count, MaxAggregator()),
            "min_channel_skip": (min(channel_skips, default=1), MinAggregator()),
            "max_channel_skip": (max(channel_skips, default=1), MaxAggregator())
        }
    )
