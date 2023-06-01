from typing import Dict, Iterable, Tuple, Optional

from coordinates import Coordinates
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass
from utils import quantize_percentage

def same_column_pattern(
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
    first_W = sparse_diff[0].W
    for coordinates in sparse_diff:
        if (
            coordinates.N != first_N
            or coordinates.C != first_C
            or coordinates.W != first_W
        ):
            return None
        

    all_hs = [coord.W for coord in sparse_diff]
    h_offset = max(all_hs) - min(all_hs)
    col_corruption_pct = quantize_percentage(len(sparse_diff) / h_offset)    
    h_skips = [curr - prev for prev, curr in zip(all_hs, next(all_hs))]    
    return SpatialClassParameters(SpatialClass.SAME_COLUMN, 
        keys = {
            "col_corruption_pct": col_corruption_pct,
        },
        stats = {
            "max_cardinality": (len(sparse_diff), MaxAggregator()),
            "min_value_skip": (min(h_skips), MinAggregator()),
            "max_value_skip": (max(h_skips), MaxAggregator())
        }
    )
