from dataclasses import dataclass
from typing import Iterable, List, Tuple


from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_class import SpatialClass

from spatial_classifier.classifiers.bullet_wake import bullet_wake_pattern
from spatial_classifier.classifiers.full_channels import full_channels_pattern
from spatial_classifier.classifiers.multi_channel_multi_block import multi_channel_multi_block_pattern
from spatial_classifier.classifiers.quasi_shattered_channel import quasi_shattered_channel_pattern
from spatial_classifier.classifiers.rectangles import rectangles_pattern
from spatial_classifier.classifiers.multiple_channels_uncategorized import multiple_channels_uncategorized_pattern
from spatial_classifier.classifiers.single_channel_random import single_channel_random_pattern
from spatial_classifier.classifiers.same_column import same_column_pattern
from spatial_classifier.classifiers.same_row import same_row_pattern
from spatial_classifier.classifiers.shattered_channel import shattered_channel_pattern
from spatial_classifier.classifiers.single import single_pattern
from spatial_classifier.classifiers.single_block import single_block_pattern
from spatial_classifier.classifiers.single_channel_alternated_blocks import (
    single_channel_alternated_blocks_pattern,
)
from spatial_classifier.classifiers.skip_2 import skip_2_pattern
from spatial_classifier.classifiers.skip_4 import skip_4_pattern

from collections import OrderedDict
from coordinates import Coordinates
from enum import Enum
import os
import shutil



SINGLE_CHANNEL_CLASSIFIERS_NEW = OrderedDict(
    [
        (SpatialClass.SINGLE, single_pattern),
        (SpatialClass.SKIP_4, skip_4_pattern),
#        (SpatialClass.SKIP_2, skip_2_pattern),
        (SpatialClass.SINGLE_BLOCK, single_block_pattern),
        (
            SpatialClass.SINGLE_CHANNEL_ALTERNATED_BLOCKS,
            single_channel_alternated_blocks_pattern,
        ),
        (SpatialClass.SAME_ROW, same_row_pattern),
        (SpatialClass.FULL_CHANNELS, full_channels_pattern),
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
        (SpatialClass.FULL_CHANNELS, full_channels_pattern),
        (SpatialClass.RECTANGLES, rectangles_pattern),
        (SpatialClass.SHATTERED_CHANNEL, shattered_channel_pattern),
#        (SpatialClass.QUASI_SHATTERED_CHANNEL, quasi_shattered_channel_pattern),
        (SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED, multiple_channels_uncategorized_pattern),
    ]
)
"""
Defines how a faulty tensor with multiple corrupted channels must be processed in order to determine his spatial class.
"""

# Old spatial classes
"""
SINGLE_CHANNEL_CLASSIFIERS_OLD = OrderedDict(
    [
        (SpatialClass.SINGLE, single_pattern),
        (SpatialClass.SAME_ROW, same_row_pattern),
        (SpatialClass.SAME_COLUMN, same_column_pattern),
        (SpatialClass.SINGLE_MAP_RANDOM, random_pattern),
    ]
)
"""
"""
Defines how a faulty tensor with a single corrupted channel must be processed in order to determine his spatial class, using the old classification
made by Toschi
"""
"""
MULTI_CHANNEL_CLASSIFIERS_OLD = OrderedDict(
    [
        (SpatialClass.BULLET_WAKE, bullet_wake_pattern),
        (SpatialClass.SHATTERED_GLASS, shattered_glass_pattern),
        (SpatialClass.QUASI_SHATTERED_GLASS, quasi_shattered_glass_pattern),
        (SpatialClass.MULTIPLE_MAP_RANDOM, random_pattern),
    ]
)
"""
"""
Defines how a faulty tensor with multiple corrupted channels must be processed in order to determine his spatial class, using the old classification
made by Toschi
"""




def spatial_classification(
    sparse_diff: Iterable[Coordinates], shape: Coordinates
) -> Tuple[SpatialClass, SpatialClassParameters, List[int]]:
    corrupted_channels = sorted(list({x.C for x in sparse_diff}))
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


def create_visual_spatial_classification_folders(visualize_output_folder: str):
    if not os.path.isdir(visualize_output_folder):
        os.mkdir(visualize_output_folder)
    for sp_class in SpatialClass:
        class_path = sp_class.class_folder(visualize_output_folder)
        if not os.path.isdir(class_path):
            os.mkdir(class_path)


def clear_spatial_classification_folders(output_path: str):
    for sp_class in SpatialClass:
        class_path = sp_class.class_folder(output_path)
        shutil.rmtree(class_path, ignore_errors=True)


def accumulate_max(curr_max_list: List[int], new_list: List[int]) -> List[int]:
    """
    Returns a list containing, at each position, the maximum value between the two same positions at curr_max_list and new_list.
    The maximum is calculated between the absolute values of two number. But in the output the sign is kept as the original

    (Example max'(-4, 2) = -4. max'(-1, 7) = 7)

    new_list must be at least long as curr_max_list. If the values of new_list in the positions exceeding exceeding the length of curr_max_list
    will be automatically put in the output list with their original sings.

    Example:
    ```
    accumulate_max([2, -1, 7], [-1, 4, 6, -9, 3]) = [2, 4, 7, -9, 3]

    accumulate_max([-1, 4, 6, -9, 3], [2, -1, 7])  -> Error
    ```
    """
    res = [0 for _ in range(len(new_list))]
    for i in range(len(new_list)):
        if i >= len(curr_max_list):
            res[i] = new_list[i]
        else:
            a, b = abs(curr_max_list[i]), abs(new_list[i])
            if a > b:
                res[i] = curr_max_list[i]
            else:
                res[i] = new_list[i]
    return res
