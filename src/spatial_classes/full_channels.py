from operator import itemgetter
from typing import Dict, Iterable, Tuple

from coordinates import Coordinates

import math

def full_channels_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
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
            return False, {}
        corr_pct = corr_fraction * 100
        # Round up the corruption % to the higher multiple of 5
        corr_pct_rounded = int(math.ceil(corr_pct / 20) * 20)
        chan_pcts[chan] = corr_pct_rounded

    min_c = min(corr_channels)
    max_c = max(corr_channels)
    max_chan_offset = max_c - min_c
    max_sane_pct = 100 - min(chan_pcts.values())
    max_corrupted_pct = max(chan_pcts.values())
    pattern = tuple(
        (chan - min_c, chan_corr_pct)
        for chan, chan_corr_pct in sorted(chan_pcts.items(), key=itemgetter(0))
    )

    return True, {
        "error_pattern": pattern,
        "max_channels": len(corr_channels),
        "max_chan_offset": max_chan_offset,
        "max_sane_pct": max_sane_pct,
        "max_corrupted_pct": max_corrupted_pct,
        "MAX": [len(corr_channels), max_chan_offset,  max_sane_pct, max_corrupted_pct],
    }
