from itertools import groupby
from operator import itemgetter
from typing import Callable, Dict, Iterable, List, Literal, Set, Tuple, Union
from collections import OrderedDict, defaultdict
import logging as log
import numpy as np
from coordinates import Coordinates, TensorLayout, map_to_coordinates
from enum import Enum
import os
import shutil


def to_classes_id(name) -> str:
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
        return "9"
    elif name == SpatialClass.CHANNEL_ALIGNED_SAME_BLOCK.display_name() or name == SpatialClass.CHANNEL_ALIGNED_SINGLE_BLOCK.display_name():
        return "10"
    elif name == SpatialClass.MULTIPLE_MAP_RANDOM.display_name():
        return "8"
    elif name == SpatialClass.TENSOR_ALIGNED_SINGLE_BLOCK.display_name():
        return "11"


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

    def display_name(self) -> str:
        return self.name.lower()

    def class_folder(self, output_path) -> str:
        return os.path.join(output_path, self.display_name())

    def output_path(self, output_path, basename) -> str:
        return os.path.join(output_path, self.display_name(), basename)


def single_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(sparse_diff) == 1:
        return True, {"error_pattern": 0}
    else:
        return False, {}


def same_row_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
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
            return False, {}
    all_ws = [coord.W for coord in sparse_diff]
    min_w = min(all_ws)
    max_w_offset = max(all_ws) - min_w
    return True, {
        "error_pattern": tuple(sorted(coord.W - min_w for coord in sparse_diff)),
        "max_w_offset": max_w_offset,
        "MAX": [max_w_offset],
    }

def same_column_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
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
            return False, {}
    all_hs = [coord.W for coord in sparse_diff]
    min_h = min(all_hs)
    max_h_offset = max(all_hs) - min_h
    return True, {
        "error_pattern": tuple(sorted(coord.W - min_h for coord in sparse_diff)),
        "max_h_offset": max_h_offset,
        "MAX": [max_h_offset],
    }


def bullet_wake_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Return True if a bullet Wake spatial distribution is recognized
    Bullet Wake: the same location is corrupted in all (or in multiple) feature maps
    """
    first_N = sparse_diff[0].N
    first_W = sparse_diff[0].W
    first_H = sparse_diff[0].H
    for coordinates in sparse_diff:
        if (
            coordinates.N != first_N
            or coordinates.H != first_H
            or coordinates.W != first_W
        ):
            return False, {}
    if len(sparse_diff) > 1:
        min_c = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - min_c
        return True, {
            "error_pattern": tuple(coord.C - min_c for coord in sparse_diff),
            "max_c_offset": max_c_offset,
            "MAX": [max_c_offset],
        }
    else:
        return False, {}


def shattered_glass_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Return True if a Shattered Glass spatial distribution is recognized.
    Shattered glass: like one or more Bullet wake errors, but in one or multiple feature maps the corruption spreads over a row (or part of the row)
    """
    # Common Row Index
    first_H = sparse_diff[0].H
    cols_by_channels = defaultdict(lambda: set())
    for coord in sparse_diff:
        # To be Shattered Glass all corruption must stay on the same row of different feature map
        if coord.H != first_H:
            return False, {}
        cols_by_channels[coord.C].add(coord.W)
    if len(cols_by_channels) == 0:
        return False, {}
    common_cols = cols_by_channels[list(cols_by_channels.keys())[0]]
    # Check if there is a common corrupted position in all corrupted feature maps
    for cols_set in cols_by_channels.values():
        common_cols &= cols_set
    if len(common_cols) > 0:
        common_element_col = next(iter(common_cols))
        smallest_chan = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - smallest_chan
        error_pattern = tuple(
            (
                chan - smallest_chan,
                tuple(col - common_element_col for col in sorted(cols)),
            )
            for chan, cols in sorted(cols_by_channels.items(), key=itemgetter(0))
        )
        min_w_offset = min(coord.W - common_element_col for coord in sparse_diff)
        max_w_offset = max(coord.W - common_element_col for coord in sparse_diff)
        feature_maps_count = len(cols_by_channels)
        return True, {
            "error_pattern": error_pattern,
            "min_w_offset": min_w_offset,
            "max_w_offset": max_w_offset,
            "max_c_offset": max_c_offset,
            "feature_maps_count": feature_maps_count,
            "MAX": [feature_maps_count, max_c_offset, min_w_offset, max_w_offset],
        }
    else:
        return False, {}


def quasi_shattered_glass_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Return True if a Shattered Glass spatial distribution is recognized.
    Shattered glass: like one or more Bullet wake errors, but in one or multiple feature maps the corruption spreads over a row (or part of the row)
    """
    # Common Row Index
    first_H = sparse_diff[0].H
    cols_by_channels : Dict[int, Set[int]] = defaultdict(lambda: set())
    channels_by_cols : Dict[int, Set[int]] = defaultdict(lambda: set())
    for coord in sparse_diff:
        # If the corruption expands to different rows, then it is not shattered glass
        if coord.H != first_H:
            return False, {}
        cols_by_channels[coord.C].add(coord.W)
        channels_by_cols[coord.W].add(coord.C)
    if len(cols_by_channels) == 0:
        return False, {}
    common_cols = {col : len(channel_set) for col, channel_set in channels_by_cols.items() if len(channel_set) >= 1}
    if len(common_cols) > 0:
        common_element_col, rows_in_common = max(common_cols.items(), key=itemgetter(1))
        smallest_chan = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - smallest_chan
        error_pattern = tuple(
            (
                chan - smallest_chan,
                tuple(col - common_element_col for col in sorted(cols)),
            )
            for chan, cols in sorted(cols_by_channels.items(), key=itemgetter(0))
        )
        min_w_offset = min(coord.W - common_element_col for coord in sparse_diff)
        max_w_offset = max(coord.W - common_element_col for coord in sparse_diff)
        feature_maps_count = len(cols_by_channels)
        return True, {
            "error_pattern": error_pattern,
            "min_w_offset": min_w_offset,
            "max_w_offset": max_w_offset,
            "max_c_offset": max_c_offset,
            "feature_maps_count": feature_maps_count,
            "MAX": [feature_maps_count, max_c_offset, min_w_offset, max_w_offset],
        }
    else:
        return False, {}


def raveled_channel_index(shape: Coordinates, coordinate: Coordinates) -> int:
    """
    Takes in input the shape of a tensor and a coordinate. coordinate must be within the shape
    Returns the raveled index inside the  channel. For example if the channel HxW dimensions are 8x16
    and the coordinates are H: 4, W: 8 (the other dimensions are ignored) the raveled index will be 8 * 16 + 4.

    This index reflects the fact that the channel is raveled inside the memory using a row major order, so the last item of a row is
    followed by the first item of the next row.
    """
    return coordinate.W + shape.W * coordinate.H

def raveled_tensor_index(shape: Coordinates, coordinate: Coordinates) -> int:
    """
    Takes in input the shape of a tensor and a coordinate. coordinate must be within the shape
    Returns the raveled index inside the tensor. For example if the channel HxW dimensions are 8x16 and the
    and the coordinates are C: 2 H: 4, W: 8 (the other dimensions are ignored) the raveled index will be 8 * 16 + 4 + 2 * 4 * 8.

    This index reflects the fact that the channel is raveled inside the memory using a row major order, so the last item of a row is
    followed by the first item of the next row, and that the channel are stored in memory contigously
    """
    return coordinate.C * (shape.H * shape.W) + shape.W * coordinate.H + coordinate.W 


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


def channel_aligned_multiple_blocks(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    """
    if len(corr_channels) != 1:
        return False, {}
    align = 32
    raveled_fault_indexes = sorted(raveled_channel_index(shape, coord) for coord in sparse_diff)
    faults_by_block = defaultdict(int)
    n_blocks = (shape.H * shape.W + align - 1) // align
    for index in raveled_fault_indexes:
        faults_by_block[index // align] += 1
    matches_pattern = all(corr_values >= align // 2 + 1 for block_id, corr_values in faults_by_block if block_id != n_blocks - 1)
    if matches_pattern:
        first_block = min(faults_by_block.keys())
        last_block = max(faults_by_block.keys())
        block_begin = first_block * align
        error_pattern = (align,  tuple(idx - block_begin for idx in raveled_fault_indexes))
        return True, {
            "error_pattern": error_pattern,
            "align": align,
            "MAX": [last_block - first_block],
        }
    else:
        return False, {}


def tensor_aligned_single_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Returns True if a Channel Aligned Same Block spatial distribution is recognized.

    """
    raveled_fault_indexes = sorted(raveled_tensor_index(shape, coord) for coord in sparse_diff)
    corrupted_items_count = len(sparse_diff)
    aligments = [16,32,64]
    if corrupted_items_count < min(aligments) // 2:
        return False, {}

    min_align = None
    block_id = None
    for align in aligments:
        align_set = {x // align for x in raveled_fault_indexes}
        if len(align_set) != 1:
            continue
        else:
            min_align = align
            block_id = next(iter(align_set))
            break
    if min_align is None or corrupted_items_count < min_align // 2:
        return False, {}
    block_begin = block_id * min_align
    error_pattern = (min_align,  tuple(idx - block_begin for idx in raveled_fault_indexes))
    return True, {
        "error_pattern": error_pattern,
        "align": min_align,
        "MAX": [],
    }
    

def multi_channel_skip_4_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Return True if a Skip4 pattern is identified

    A skip 4 pattern happens when there are errors every 4 values (using raveled channel index) in each corrupeted channel.
    This algorithm finds the lowest raveled channel index of a corrupted values, called smallest_coordinate (the same for all channels).

    Errors are expected only in the values that have distance divisible by 4 from smallest_coordinate.

    To match there must be globally at least 3 raveled channel index that follow the criteria (good_positions), and there must not be more than 3
    positions that don't respect the criteria (bad_positions).

    The position 0 does not count either for the good_positions and for the wrong_positions
    """
    coordinates = set([raveled_channel_index(shape, coord) for coord in sparse_diff])
    smallest_coordinate = min(coordinates)
    # to be a skip 4 error it should be true that (raveled_channel_index(error) === smallest_coordinate mod 4)
    # 0 is always allowed inside the skip4 pattern
    candidate_positions = set(range(smallest_coordinate, shape.H * shape.W, 4))
    # All the errors in the tensor that have a distance multiple of 4
    good_positions = coordinates & candidate_positions
    # all the other positions
    wrong_positions = coordinates - candidate_positions

    if len(good_positions) >= 2 and len(wrong_positions) <= 1:
        # Generate the error_pattern
        indexes_by_channel = defaultdict(list)
        min_c = min(coord.C for coord in sparse_diff)
        max_c_offset = max(coord.C for coord in sparse_diff) - min_c
        min_index = smallest_coordinate
        for coord in sparse_diff:
            indexes_by_channel[coord.C - min_c].append(
                (raveled_channel_index(shape, coord) - min_index)
            )
            error_pattern = tuple(
                (chan, tuple(idx for idx in sorted(indexes)))
                for chan, indexes in sorted(
                    indexes_by_channel.items(), key=itemgetter(0)
                )
            )
        max_idx_offset = max(coordinates) - smallest_coordinate
        return True, {
            "error_pattern": error_pattern,
            "max_c_offset": max_c_offset,
            "max_idx_offset": max_idx_offset,
            "MAX": [max_c_offset, max_idx_offset],
        }
    else:
        return False, {}


def random_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    indexes_by_channel = defaultdict(list)
    min_c = min(coord.C for coord in sparse_diff)
    min_index = min(raveled_channel_index(shape, coord) for coord in sparse_diff)
    for coord in sparse_diff:
        indexes_by_channel[coord.C - min_c].append(
            raveled_channel_index(shape, coord) - min_index
        )
        error_pattern = tuple(
            (chan, tuple(idx for idx in sorted(indexes)))
            for chan, indexes in sorted(indexes_by_channel.items(), key=itemgetter(0))
        )
    return True, {"error_pattern": error_pattern}

SINGLE_CHANNEL_CLASSIFIERS: Dict[SpatialClass,     Callable[
        [Iterable[Coordinates], Coordinates, Iterable[int]],
        Tuple[bool, Dict[str, any]],
    ],] = OrderedDict([
            (SpatialClass.SINGLE, single_classifier),
            (SpatialClass.SKIP_4, multi_channel_skip_4_pattern),
            (SpatialClass.TENSOR_ALIGNED_SINGLE_BLOCK, tensor_aligned_single_block_pattern),
            (SpatialClass.CHANNEL_ALIGNED_SINGLE_BLOCK, channel_aligned_same_block_pattern),
            (SpatialClass.SAME_ROW, same_row_classifier),
            (SpatialClass.SINGLE_MAP_RANDOM, random_classifier),
    ])

SINGLE_CHANNEL_CLASSIFIERS_OLD: Dict[
    SpatialClass,
    Callable[
        [Iterable[Coordinates], Coordinates, Iterable[int]],
        Tuple[bool, Dict[str, any]],
    ],
] = OrderedDict(
    [
        (SpatialClass.SINGLE, single_classifier),
        (SpatialClass.SAME_ROW, same_row_classifier),
        (SpatialClass.SAME_COLUMN, same_column_classifier),
        (SpatialClass.SINGLE_MAP_RANDOM, random_classifier),
    ]
) 


MULTI_CHANNEL_CLASSIFIERS_OLD: Dict[
    SpatialClass,
    Callable[
        [Iterable[Coordinates], Coordinates, Iterable[int]],
        Tuple[bool, Dict[str, any]],
    ],
] = OrderedDict(
    [
        (SpatialClass.BULLET_WAKE, bullet_wake_classifier),
        (SpatialClass.SHATTERED_GLASS, shattered_glass_classifier),
        (SpatialClass.QUASI_SHATTERED_GLASS, quasi_shattered_glass_classifier),
        (SpatialClass.MULTIPLE_MAP_RANDOM, random_classifier),
    ]
)

MULTI_CHANNEL_CLASSIFIERS: Dict[
    SpatialClass,
    Callable[
        [Iterable[Coordinates], Coordinates, Iterable[int]],
        Tuple[bool, Dict[str, any]],
    ],
] = OrderedDict(
    [
        (SpatialClass.BULLET_WAKE, bullet_wake_classifier),
        (SpatialClass.SKIP_4, multi_channel_skip_4_pattern),
        (SpatialClass.CHANNEL_ALIGNED_SAME_BLOCK, channel_aligned_same_block_pattern),
        (SpatialClass.TENSOR_ALIGNED_SINGLE_BLOCK, tensor_aligned_single_block_pattern),
        (SpatialClass.SHATTERED_GLASS, shattered_glass_classifier),
        (SpatialClass.QUASI_SHATTERED_GLASS, quasi_shattered_glass_classifier),
        (SpatialClass.MULTIPLE_MAP_RANDOM, random_classifier),
    ]
)


SPATIAL_CLASSES = list(SpatialClass)


def spatial_classification(
    sparse_diff: Iterable[Coordinates], shape: Coordinates
) -> Tuple[SpatialClass, dict[str, any]]:
    corrupted_channels = list({x.C for x in sparse_diff})

    if len(corrupted_channels) == 1:
        for sp_class, classifier in SINGLE_CHANNEL_CLASSIFIERS.items():
            is_class, pattern_data = classifier(sparse_diff, shape, corrupted_channels)
            if is_class:
                return sp_class, pattern_data
    else:
        for sp_class, classifier in MULTI_CHANNEL_CLASSIFIERS.items():
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


def accumulate_max(curr_max_list : List[int], new_list : List[int]) -> List[int]:
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
