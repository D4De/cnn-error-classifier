from collections import defaultdict
from operator import itemgetter
from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, raveled_channel_index



def channel_aligned_same_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Returns True if a Channel Aligned Same Block spatial distribution is recognized.

    A block of errors is made of [max_align] errors that are contiguos in memory. max_aligns is also the aligment of the block and must be 16, 32 or 64.
    If the block is compatible with multiple aligments, only the biggest max_align is selected. The aligment is counted from the element 0,0 of each channel,
    in a row major order (like it is represented in memory).

    To match with this error there must be at least a channel (that will be called the leader channel) where there are at least [max_align - max_align >> 1 + 1]
    erroneous values, all contained inside a single block of dimension max_aligned aligned from the channel start.
    There must not be other erroneous values outside the the block.
    If there is more than one corrupted channel, the other corrupted channels must have errors ONLY inside the very same corrupted block.

    Example 1:
    Channel 0 (6x6) has errors in 10, 14
    Channel 1 (6x6) has errors in 8, 9, 11, 12, 13, 14, 15 (7/8 errors in erroneus values in block 1)
    Channel 2 (6x6) has an error in 13

    max_align is 8
    Channel 1 is the leader channel because has at least 8 - 1 errors inside block 1 (the one that goes from 8 to 15)
    Channel 0, 2 have errors inside that block
    This tensor will match with the pattern

    Example 2:
    Channel 0 and channel 1 (10x10) have all errors between 0 and 31 (32/32 errors in erroneus values in block 0)
    Channel 2 (10x10) has errors in 8, 9, 11, 12, 13, 14, 15, 31
    Channel 3 (10x10) has errors between 30 and 50

    max_align is 32
    One between channel 0 and 1 is the leader,
    Channel 2 has all his errors inside block 0
    However Channel 3 has errors also in block 1, so the tensor will not match with the pattern

    """
    # Scan all channels to elect the leader channels and determine the max_align
    leader_channel = None
    max_align = 0
    for channel in corr_channels:
        corrupted_items = sum(1 for coord in sparse_diff if coord.C == channel)
        # tolerance value
        if 10 <= corrupted_items <= 16:
            align = 16
        elif 17 <= corrupted_items <= 32:
            align = 32
        elif 34 <= corrupted_items <= 64:
            align = 64
        else:
            continue
        if align > max_align:
            leader_channel = channel
            max_align = align
    if leader_channel is None:
        return False, {}
    coordinates = [raveled_channel_index(shape, coord) for coord in sparse_diff]
    # Use integer division to calculate the block of the coordinate
    block_id = defaultdict(int)
    for coord in coordinates:
        block_id[coord//align] += 1
    # There must not be errors outside of the block

    if len(block_id) == 1:
        the_block_id = next(iter(block_id))
    elif len(block_id) == 2:
        the_block_id, occourrences = max(block_id.items(), key=itemgetter(1))
        dirty_block_id, dirty_occourrences = min(block_id.items(), key=itemgetter(1))
        if dirty_occourrences > 1 or abs(dirty_block_id - the_block_id) > 1:
            return False, {}
    else:
        return False, {}
    
    the_block_id = next(iter(block_id))
    indexes_by_channel = defaultdict(list)
    min_c = min(coord.C for coord in sparse_diff)
    # min_index is The starting position of the block
    min_index = the_block_id * max_align
    for coord in sparse_diff:
        indexes_by_channel[coord.C - min_c].append(
            raveled_channel_index(shape, coord) - min_index
        )
        error_pattern = (
            align,
            tuple(
                (chan, tuple(idx for idx in sorted(indexes)))
                for chan, indexes in sorted(
                    indexes_by_channel.items(), key=itemgetter(0)
                )
            ),
        )
    max_c_offset = max(coord.C for coord in sparse_diff) - min_c
    return True, {
        "error_pattern": error_pattern,
        "align": max_align,
        "MAX": [max_c_offset],
    }
