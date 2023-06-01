from collections import defaultdict
from operator import itemgetter
from typing import Dict, Iterable, Tuple, Optional

from coordinates import Coordinates, raveled_channel_index
from spatial_classifier.aggregators import MaxAggregator, MinAggregator
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass
from utils import quantize_percentage


def single_channel_random_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    
    if len(corr_channels) != 1:
        return None

    error_cardinality = len(sparse_diff)
    channel_corruption_pct = quantize_percentage(error_cardinality / (shape.H * shape.W))

    raveled_value_idxs = sorted([raveled_channel_index(shape, coord) for coord in sparse_diff])
    value_skips = [curr - prev for prev, curr in zip(raveled_value_idxs, raveled_value_idxs[1:])]  


    return SpatialClassParameters(SpatialClass.SINGLE_CHANNEL_RANDOM, 
                keys = {
                    "channel_corruption_pct": channel_corruption_pct
                },
                stats = {
                    "max_cardinality": (error_cardinality, MaxAggregator()),
                    "min_skip": (min(value_skips), MinAggregator()),
                    "max_skip": (max(value_skips), MaxAggregator()),

                }
            )
