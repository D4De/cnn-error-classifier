from typing import Dict, Iterable, Tuple, Optional

from coordinates import Coordinates
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass

def same_row_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    """
    Return True if a bullet Wake spatial distribution is recognized
    Same Row: multiple corrupted values lie in the same row (same feature map)
    """
    first_N = sparse_diff[0].N
    first_C = sparse_diff[0].C
    first_H = sparse_diff[0].H
    for coordinates in sparse_diff:
        if (
            coordinates.N != first_N
            or coordinates.C != first_C
            or coordinates.H != first_H
        ):
            return None
    all_ws = [coord.W for coord in sparse_diff]
    w_offset = max(all_ws) - min(all_ws)

    all_ws = [coord.W for coord in sparse_diff]
    w_skips = [curr - prev for prev, curr in zip(all_ws, next(all_ws))]    
    return SpatialClassParameters(SpatialClass.SAME_COLUMN, 
        keys = {
            "cardinality": len(sparse_diff),
            "value_offset": w_offset,
            "min_value_skip": min(w_skips),
            "max_value_skip": max(w_skips),
        },
        aggregate_values = {}
    )

