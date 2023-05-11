from typing import Callable, Dict, Iterable, List, Tuple
from spatial_classes.bullet_wake import bullet_wake_pattern
from spatial_classes.channel_aligned_same_block import (
    channel_aligned_same_block_pattern,
)
from spatial_classes.full_single_channel import full_single_channel_pattern
from spatial_classes.multi_channel_multi_block import multi_channel_multi_block_pattern
from spatial_classes.quasi_shattered_channel import quasi_shattered_channel_pattern
from spatial_classes.quasi_shattered_glass import quasi_shattered_glass_pattern
from spatial_classes.random import random_pattern
from spatial_classes.same_column import same_column_pattern
from spatial_classes.same_row import same_row_pattern
from spatial_classes.shattered_channel import shattered_channel_pattern
from spatial_classes.shattered_glass import shattered_glass_pattern
from spatial_classes.single import single_pattern
from spatial_classes.single_block import single_block_pattern
from spatial_classes.single_channel_alternated_blocks import (
    single_channel_alternated_blocks_pattern,
)
from spatial_classes.skip_2 import skip_2_pattern
from spatial_classes.skip_4 import skip_4_pattern
from spatial_classes.tensor_aligned_single_block import (
    tensor_aligned_single_block_pattern,
)
from collections import OrderedDict
from coordinates import Coordinates
from enum import Enum
import os
import shutil


def to_classes_id(name) -> str:
    """
    Maps a SpatialClass enum instance to his id used in the CLASSES framework.
    """
    if name == SpatialClass.SAME_ROW.display_name():
        return "0"
    elif name == SpatialClass.SAME_COLUMN.display_name():
        return "1"
    elif name == SpatialClass.SINGLE_MAP_RANDOM.display_name():
        return "3"
    elif name == SpatialClass.BULLET_WAKE.display_name():
        return "4"
    elif name == SpatialClass.SHATTERED_GLASS.display_name():
        return "6"
    elif name == SpatialClass.QUASI_SHATTERED_GLASS.display_name():
        return "7"
    elif name == SpatialClass.SKIP_4.display_name():
        return "1001"
    elif (
        name == SpatialClass.CHANNEL_ALIGNED_SAME_BLOCK.display_name()
        or name == SpatialClass.CHANNEL_ALIGNED_SINGLE_BLOCK.display_name()
    ):
        return "1002"
    elif name == SpatialClass.MULTIPLE_MAP_RANDOM.display_name():
        return "8"
    elif name == SpatialClass.TENSOR_ALIGNED_SINGLE_BLOCK.display_name():
        return "1003"
    elif name == SpatialClass.SINGLE_BLOCK.display_name():
        return "1004"
    elif name == SpatialClass.MULTI_CHANNEL_BLOCK.display_name():
        return "1005"
    elif name == SpatialClass.SHATTERED_CHANNEL.display_name():
        return "1006"
    elif name == SpatialClass.QUASI_SHATTERED_CHANNEL.display_name():
        return "1007"
    elif name == SpatialClass.SINGLE_CHANNEL_ALTERNATED_BLOCKS.display_name():
        return "1008"
    elif name == SpatialClass.SKIP_2.display_name():
        return "1009"
    elif name == SpatialClass.FULL_SINGLE_CHANNEL.display_name():
        return "1010"
class SpatialClass(Enum):
    SAME = 0
    SINGLE = 1
    SAME_COLUMN = 2
    SAME_ROW = 3
    SINGLE_MAP_RANDOM = 4
    BULLET_WAKE = 5
    SHATTERED_GLASS = 6
    QUASI_SHATTERED_GLASS = 7
    SKIP_4 = 8
    CHANNEL_ALIGNED_SAME_BLOCK = 9
    MULTIPLE_MAP_RANDOM = 10
    CHANNEL_ALIGNED_SINGLE_BLOCK = 11
    TENSOR_ALIGNED_SINGLE_BLOCK = 12
    SINGLE_BLOCK = 13
    MULTI_CHANNEL_BLOCK = 14
    SHATTERED_CHANNEL = 15
    QUASI_SHATTERED_CHANNEL = 16
    SINGLE_CHANNEL_ALTERNATED_BLOCKS = 17
    FULL_SINGLE_CHANNEL = 19

    def display_name(self) -> str:
        """
        Returns the display name of the Spatial Class. The Display is the name of the class in snake case
        """
        return self.name.lower()

    def class_folder(self, output_path) -> str:
        """
        Returns the path of the subfolder (inside output_path) where the visualizations of tensors classified under this class are stored
        """
        return os.path.join(output_path, self.display_name())

    def output_path(self, output_path, basename) -> str:
        return os.path.join(output_path, self.display_name(), basename)


SINGLE_CHANNEL_CLASSIFIERS_NEW = OrderedDict(
    [
        (SpatialClass.SINGLE, single_pattern),
        (SpatialClass.SKIP_4, skip_4_pattern),
        (SpatialClass.SINGLE_BLOCK, single_block_pattern),
        (
            SpatialClass.SINGLE_CHANNEL_ALTERNATED_BLOCKS,
            single_channel_alternated_blocks_pattern,
        ),
        (SpatialClass.SAME_ROW, same_row_pattern),
        (SpatialClass.FULL_SINGLE_CHANNEL, full_single_channel_pattern),
        (SpatialClass.SINGLE_MAP_RANDOM, random_pattern),
    ]
)
"""
Defines how a faulty tensor with a single corrupted channel must be processed in order to determine his spatial class.
"""

MULTI_CHANNEL_CLASSIFIERS_NEW = OrderedDict(
    [
        (SpatialClass.SKIP_4, skip_4_pattern),
        (SpatialClass.SINGLE_BLOCK, single_block_pattern),
        (SpatialClass.MULTI_CHANNEL_BLOCK, multi_channel_multi_block_pattern),
        (SpatialClass.BULLET_WAKE, bullet_wake_pattern),
        (SpatialClass.SHATTERED_CHANNEL, shattered_channel_pattern),
        (SpatialClass.QUASI_SHATTERED_CHANNEL, quasi_shattered_channel_pattern),
        (SpatialClass.MULTIPLE_MAP_RANDOM, random_pattern),
    ]
)
"""
Defines how a faulty tensor with multiple corrupted channels must be processed in order to determine his spatial class.
"""

# Old spatial classes

SINGLE_CHANNEL_CLASSIFIERS_OLD = OrderedDict(
    [
        (SpatialClass.SINGLE, single_pattern),
        (SpatialClass.SAME_ROW, same_row_pattern),
        (SpatialClass.SAME_COLUMN, same_column_pattern),
        (SpatialClass.SINGLE_MAP_RANDOM, random_pattern),
    ]
)
"""
Defines how a faulty tensor with a single corrupted channel must be processed in order to determine his spatial class, using the old classification
made by Toschi
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
Defines how a faulty tensor with multiple corrupted channels must be processed in order to determine his spatial class, using the old classification
made by Toschi
"""


SPATIAL_CLASSES = list(SpatialClass)


def spatial_classification(
    sparse_diff: Iterable[Coordinates], shape: Coordinates
) -> Tuple[SpatialClass, dict[str, any]]:
    corrupted_channels = list({x.C for x in sparse_diff})

    if len(corrupted_channels) == 1:
        for sp_class, classifier in SINGLE_CHANNEL_CLASSIFIERS_NEW.items():
            is_class, pattern_data = classifier(sparse_diff, shape, corrupted_channels)
            if is_class:
                return sp_class, pattern_data
    else:
        for sp_class, classifier in MULTI_CHANNEL_CLASSIFIERS_NEW.items():
            is_class, pattern_data = classifier(sparse_diff, shape, corrupted_channels)
            if is_class:
                return sp_class, pattern_data


def create_visual_spatial_classification_folders(visualize_output_folder: str):
    if not os.path.isdir(visualize_output_folder):
        os.mkdir(visualize_output_folder)
    for sp_class in SPATIAL_CLASSES:
        class_path = sp_class.class_folder(visualize_output_folder)
        if not os.path.isdir(class_path):
            os.mkdir(class_path)


def clear_spatial_classification_folders(output_path: str):
    for sp_class in SPATIAL_CLASSES:
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
