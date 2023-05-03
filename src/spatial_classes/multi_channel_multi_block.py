from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, identify_block_length, raveled_channel_index


def multi_channel_multi_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(corr_channels) < 2:
        return False, {}
    block_start = shape.H * shape.W
    max_block_length = -1
    leader_chan = None
    indexes_by_channel = {}
    for chan in corr_channels:
        this_chan_indexes = sorted(
            raveled_channel_index(shape, coord)
            for coord in sparse_diff
            if coord.C == chan
        )
        first_chan_error = this_chan_indexes[0]
        indexes_by_channel[chan] = this_chan_indexes
        result = identify_block_length(this_chan_indexes)
        if result is None:
            continue
        block_length, aligment_offset, block_id = result
        if block_length > max_block_length:
            max_block_length = block_length
            block_start = first_chan_error
            leader_chan = chan
        elif block_length == max_block_length and first_chan_error < block_start:
            block_start = first_chan_error
            leader_chan = chan
    if leader_chan is None:
        return False, {}
    found_mismatch = False
    for chan in corr_channels:
        if chan == leader_chan:
            continue
        if not all(
            block_start <= error_idx <= block_start + max_block_length
            for error_idx in indexes_by_channel[chan]
        ):
            result = identify_block_length(indexes_by_channel[chan])
            if result is None:
                found_mismatch = True

    if found_mismatch:
        return False, {}
    corr_indexes = [
        raveled_channel_index(shape, coord) - block_start for coord in sparse_diff
    ]
    min_idx = min(corr_indexes)
    max_idx = max(corr_indexes)
    min_c = min(corr_channels)
    max_c = max(corr_channels)
    corrupted_feature_maps = len(corr_channels)

    error_pattern = (
        max_block_length,
        tuple(
            (
                chan - min_c,
                tuple(
                    error_idx - block_start for error_idx in indexes_by_channel[chan]
                ),
            )
            for chan in corr_channels
        ),
    )
    return True, {
        "error_pattern": error_pattern,
        "align": block_length,
        "MAX": [corrupted_feature_maps, max_c - min_c, min_idx, max_idx],
    }
