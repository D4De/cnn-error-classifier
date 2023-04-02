from itertools import groupby
from operator import itemgetter
from typing import Callable, Dict, Iterable, Literal, Tuple, Union
from collections import OrderedDict, defaultdict

import numpy as np
from coordinates import Coordinates, TensorLayout, map_to_coordinates
from enum import Enum
import os
import shutil


class SpatialClass(Enum):
    SAME = 0
    SINGLE = 1
    SAME_ROW = 3
    BULLET_WAKE = 4
    SHATTERED_GLASS = 5
    #    SINGLE_BLOCK_SINGLE_CHANNEL = 6
    #    SINGLE_BLOCK_MULTI_CHANNEL = 7
    #    SKIP_4_MULTI_CHANNEL = 9
    SKIP_4 = 6
    SINGLE_BLOCK = 7
    RANDOM = 10

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
) -> Union[Literal[False], Tuple[Literal[True], Dict[str, any]]]:
    if len(sparse_diff) == 1:
        return True, {"error_pattern": 0}
    else:
        return False, {}


def same_row_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Union[Literal[False], Tuple[Literal[True], Dict[str, any]]]:
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
    min_w = min(coord.W for coord in sparse_diff)
    return True, {
        "error_pattern": tuple(sorted(coord.W - min_w for coord in sparse_diff))
    }


def bullet_wake_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Union[Literal[False], Tuple[Literal[True], Dict[str, any]]]:
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
        return True, {"error_pattern": tuple(coord.C - min_c for coord in sparse_diff)}
    else:
        return False, {}


def shattered_glass_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Union[Literal[False], Tuple[Literal[True], Dict[str, any]]]:
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
        error_pattern = tuple(
            (
                chan - smallest_chan,
                tuple(col - common_element_col for col in sorted(cols)),
            )
            for chan, cols in sorted(cols_by_channels.items(), key=itemgetter(0))
        )
        return True, {"error_pattern": error_pattern}
    else:
        return False, {}


def unraveled_channel_index(shape: Coordinates, coordinate: Coordinates) -> int:
    return coordinate.W + shape.W * coordinate.H


def multi_channel_single_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Union[Literal[False], Tuple[Literal[True], Dict[str, any]]]:
    channel_with_max_align = None
    max_align = 0
    for channel in corr_channels:
        corrupted_items = sum(1 for coord in sparse_diff if coord.C == channel)
        if 7 <= corrupted_items <= 8:
            align = 8
        elif 14 <= corrupted_items <= 16:
            align = 16
        elif 28 <= corrupted_items <= 32:
            align = 32
        else:
            continue
        if align > max_align:
            channel_with_max_align = channel
            max_align = align
    if channel_with_max_align is None:
        return False, {}
    coordinates = [unraveled_channel_index(shape, coord) for coord in sparse_diff]
    block_id = {coord // align for coord in coordinates}
    if len(block_id) == 1:
        the_block_id = next(iter(block_id))
        indexes_by_channel = defaultdict(list)
        min_c = min(coord.C for coord in sparse_diff)
        min_index = the_block_id * max_align
        for coord in sparse_diff:
            indexes_by_channel[coord.C - min_c].append(
                unraveled_channel_index(shape, coord) - min_index
            )
            error_pattern = tuple(
                (chan, tuple(idx for idx in sorted(indexes)))
                for chan, indexes in sorted(
                    indexes_by_channel.items(), key=itemgetter(0)
                )
            )
        return True, {"error_pattern": error_pattern, "align": max_align}
    else:
        return False, {}


def multi_channel_skip_4_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Union[Literal[False], Tuple[Literal[True], Dict[str, any]]]:
    coordinates = set([unraveled_channel_index(shape, coord) for coord in sparse_diff])
    smallest_coordinate = min(coordinates)
    candidate_positions = set(range(smallest_coordinate, shape.H * shape.W, 4))
    good_positions = coordinates & candidate_positions
    wrong_positions = coordinates - candidate_positions
    # print(f'{smallest_coordinate=} {candidate_positions=} {good_positions=} {wrong_positions=}')
    if len(good_positions) >= 3 and len(wrong_positions) <= 3:
        indexes_by_channel = defaultdict(list)
        min_c = min(coord.C for coord in sparse_diff)
        min_index = smallest_coordinate
        for coord in sparse_diff:
            indexes_by_channel[coord.C - min_c].append(
                (unraveled_channel_index(shape, coord) - min_index)
            )
            error_pattern = tuple(
                (chan, tuple(idx for idx in sorted(indexes)))
                for chan, indexes in sorted(
                    indexes_by_channel.items(), key=itemgetter(0)
                )
            )
        return True, {"error_pattern": error_pattern}
    else:
        return False, {}


'''
# A lot frequent
def is_new_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> bool:
    """
    Validate if the given list of indexes represents the new pattern we are observing.

    Patameters:
        indexes: a list of indexes representing differences between a faulty tensor and the corresponding golden one

    Return values:
        A boolean determining if is indeed a new pattern or not.
        An integer representing the number of errors in a single channel if we are dealing with the new pattern, 0 otherwise
    """
    errors_channel = sorted(list(set([x.C for x in sparse_diff])))

    if len(errors_channel) == 0:
        return False, 0

    base_channel = errors_channel[0]
    base_channel_errors = {(x.H, x.W) for x in sparse_diff if x.C == base_channel}

    errors_count = len(base_channel_errors)
    has_common_errors = False

    for c in errors_channel:
        curr_channel_idxs = {(x.H, x.W) for x in sparse_diff if x.C == c}
        errors_count = max(errors_count, len(curr_channel_idxs))
        if not has_common_errors:
            inters = curr_channel_idxs & base_channel_errors
            has_common_errors = len(inters) >= 2

    return has_common_errors
'''


def random_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Union[Literal[False], Tuple[Literal[True], Dict[str, any]]]:
    indexes_by_channel = defaultdict(list)
    min_c = min(coord.C for coord in sparse_diff)
    min_index = min(unraveled_channel_index(shape, coord) for coord in sparse_diff)
    for coord in sparse_diff:
        indexes_by_channel[coord.C - min_c].append(
            unraveled_channel_index(shape, coord) - min_index
        )
        error_pattern = tuple(
            (chan, tuple(idx for idx in sorted(indexes)))
            for chan, indexes in sorted(indexes_by_channel.items(), key=itemgetter(0))
        )
    return True, {"error_pattern": error_pattern}


SINGLE_CHANNEL_CLASSIFIERS: Dict[
    SpatialClass,
    Callable[
        [Iterable[Coordinates], Coordinates, Iterable[int]],
        Union[Literal[False], Tuple[Literal[True], Dict[str, any]]],
    ],
] = OrderedDict(
    [
        (SpatialClass.SINGLE, single_classifier),
        (SpatialClass.SKIP_4, multi_channel_skip_4_pattern),
        (SpatialClass.SINGLE_BLOCK, multi_channel_single_block_pattern),
        (SpatialClass.SAME_ROW, same_row_classifier),
        (SpatialClass.RANDOM, random_classifier),
    ]
)

MULTI_CHANNEL_CLASSIFIERS: Dict[
    SpatialClass,
    Callable[
        [Iterable[Coordinates], Coordinates, Iterable[int]],
        Union[Literal[False], Tuple[Literal[True], Dict[str, any]]],
    ],
] = OrderedDict(
    [
        (SpatialClass.BULLET_WAKE, bullet_wake_classifier),
        (SpatialClass.SKIP_4, multi_channel_skip_4_pattern),
        (SpatialClass.SINGLE_BLOCK, multi_channel_single_block_pattern),
        (SpatialClass.SHATTERED_GLASS, shattered_glass_classifier),
        (SpatialClass.RANDOM, random_classifier),
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
