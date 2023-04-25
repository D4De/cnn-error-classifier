from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, raveled_channel_index

def shattered_channel_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(corr_channels) < 2:
        return False, {}
    indexes_by_chan = {}
    for chan in corr_channels:
        indexes_by_chan[chan] = set(raveled_channel_index(shape, coord) for coord in sparse_diff if coord.C == chan)
    min_c = min(corr_channels)
    common_indexes = indexes_by_chan[min_c]
    all_indexes = set()
    for chan in corr_channels:
        common_indexes &= indexes_by_chan[chan]
        all_indexes |= indexes_by_chan[chan]
    if len(common_indexes) == 0:
        return False, {}
    zero_index = next(iter(common_indexes))
    min_w_offset = min(idx - zero_index for idx in all_indexes)
    max_w_offset = max(idx - zero_index for idx in all_indexes)
    max_c_offset = max(corr_channels) - min_c
    feature_maps_count = len(corr_channels)
    error_pattern = tuple((chan - min_c, tuple(error_idx - zero_index for error_idx in indexes_by_chan[chan])) for chan in corr_channels)    

    return True, {
        "error_pattern": error_pattern,
        "min_w_offset": min_w_offset,
        "max_w_offset": max_w_offset,
        "max_c_offset": max_c_offset,
        "feature_maps_count": feature_maps_count,
        "MAX": [feature_maps_count, max_c_offset, min_w_offset, max_w_offset],
    }    

