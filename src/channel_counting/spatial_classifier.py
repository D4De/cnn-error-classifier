
import numpy as np

from collections import OrderedDict


from coordinates import Coordinates, map_to_coordinates, numpy_coords_to_python_coord, TensorLayout

from spatial_classifier.spatial_class import SpatialClass
from spatial_classifier.spatial_class_parameters import SpatialClassParameters

from spatial_classifier.classifiers.single import single_pattern
from spatial_classifier.classifiers.skip_4 import skip_4_pattern
from spatial_classifier.classifiers.same_row import same_row_pattern
from spatial_classifier.classifiers.rectangles import rectangles_pattern
from spatial_classifier.classifiers.bullet_wake import bullet_wake_pattern
from spatial_classifier.classifiers.single_block import single_block_pattern
from spatial_classifier.classifiers.full_channels import full_channels_pattern
from spatial_classifier.classifiers.shattered_channel import shattered_channel_pattern
from spatial_classifier.classifiers.single_channel_random import single_channel_random_pattern
from spatial_classifier.classifiers.quasi_shattered_channel import quasi_shattered_channel_pattern
from spatial_classifier.classifiers.multi_channel_multi_block import multi_channel_multi_block_pattern
from spatial_classifier.classifiers.multiple_channels_uncategorized import multiple_channels_uncategorized_pattern
from spatial_classifier.classifiers.single_channel_alternated_blocks import single_channel_alternated_blocks_pattern


SINGLE_CHANNEL_CLASSIFIERS_NEW = OrderedDict(
    [
        (SpatialClass.SINGLE, single_pattern),
        (SpatialClass.SKIP_4, skip_4_pattern),
#       (SpatialClass.SKIP_2, skip_2_pattern),
        (SpatialClass.SINGLE_BLOCK, single_block_pattern),
        (SpatialClass.SINGLE_CHANNEL_ALTERNATED_BLOCKS, single_channel_alternated_blocks_pattern),
        (SpatialClass.SAME_ROW, same_row_pattern),
#       (SpatialClass.FULL_CHANNELS, full_channels_pattern),
        (SpatialClass.RECTANGLES, rectangles_pattern),
        (SpatialClass.SINGLE_CHANNEL_RANDOM, single_channel_random_pattern),
    ]
)
"""
Defines how a faulty tensor with a single corrupted channel must be processed in order to determine his spatial class.
"""

MULTI_CHANNEL_CLASSIFIERS_NEW = OrderedDict(
    [
        (SpatialClass.SKIP_4, skip_4_pattern),
#        (SpatialClass.SKIP_2, skip_2_pattern),
        (SpatialClass.SINGLE_BLOCK, single_block_pattern),
        (SpatialClass.MULTI_CHANNEL_BLOCK, multi_channel_multi_block_pattern),
        (SpatialClass.BULLET_WAKE, bullet_wake_pattern),
#       (SpatialClass.FULL_CHANNELS, full_channels_pattern),
        (SpatialClass.RECTANGLES, rectangles_pattern),
        (SpatialClass.SHATTERED_CHANNEL, shattered_channel_pattern),
        (SpatialClass.QUASI_SHATTERED_CHANNEL, quasi_shattered_channel_pattern),
        (SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED, multiple_channels_uncategorized_pattern),
    ]
)
"""
Defines how a faulty tensor with multiple corrupted channels must be processed in order to determine his spatial class.
"""

SPATIAL_CLASS_NAMES = set(SINGLE_CHANNEL_CLASSIFIERS_NEW.keys()).union(set(MULTI_CHANNEL_CLASSIFIERS_NEW.keys()))


def spatial_classification(
    diff_mask: np.ndarray,
    shape: Coordinates,
    layout: TensorLayout,
) -> tuple[SpatialClass, SpatialClassParameters, list[int]]:
    # corrupted_channels = sorted(list({x.C for x in sparse_diff}))

    # find indices of the corrupted channels
    corrupted_channels: list[int] = np.unique(np.nonzero(diff_mask)[1]).tolist()  # channel dimension has index 1

    # test for full_channels first to optimize
    full_channels_result = full_channels_pattern(diff_mask, shape, corrupted_channels)
    if full_channels_result is not None:
        return SpatialClass.FULL_CHANNELS, full_channels_result, corrupted_channels

    # test all other patterns
    # convert diff_mask to a sparse representation of the corrupted elements
    sparse_diff_native_coords = list(zip(*np.nonzero(diff_mask)))
    sparse_diff = [
        map_to_coordinates(numpy_coords_to_python_coord(coords), layout)
        for coords in sparse_diff_native_coords
    ]

    if len(corrupted_channels) == 1:
        for sp_class, classifier in SINGLE_CHANNEL_CLASSIFIERS_NEW.items():
            result = classifier(sparse_diff, shape, corrupted_channels)
            if result is not None:
                return sp_class, result, corrupted_channels
    else:
        for sp_class, classifier in MULTI_CHANNEL_CLASSIFIERS_NEW.items():
            result = classifier(sparse_diff, shape, corrupted_channels)
            if result is not None:
                return sp_class, result, corrupted_channels
