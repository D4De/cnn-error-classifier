from typing import Dict, Iterable, Tuple, Optional

from coordinates import Coordinates
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass


def single_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Optional[SpatialClassParameters]:
    if len(sparse_diff) == 1:
        return SpatialClassParameters(SpatialClass.SINGLE, keys={}, aggregate_values={})
    else:
        return None

