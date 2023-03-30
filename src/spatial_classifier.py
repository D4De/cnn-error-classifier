from itertools import groupby
from typing import Callable, Dict, Iterable, Tuple
from collections import OrderedDict, defaultdict

import numpy as np
from coordinates import Coordinates, TensorLayout
from enum import Enum
import os
import shutil


class SpatialClass(Enum):
    SAME = 0
    SINGLE = 1
    PHOTOCOPY = 2
    SAME_ROW = 3
    BULLET_WAKE = 4
    SHATTERED_GLASS = 5
    RANDOM = 6

    def display_name(self) -> str:
        return self.name.lower()

    def class_folder(self, output_path) -> str:
        return os.path.join(output_path, self.display_name())

    def output_path(self, output_path, basename) -> str:
        return os.path.join(output_path, self.display_name(), basename)


def single_classifier(diff: Iterable[Coordinates], dense_diff: Tuple[np.ndarray, TensorLayout]) -> bool:
    return len(diff) == 1

def photocopy(diff: Iterable[Coordinates], dense_diff: Tuple[np.ndarray, TensorLayout]) -> bool:
    tensor, layout = dense_diff
    is_correct_tensor = (tensor == 0)[layout.N_index()]
    layers_with_error = is_correct_tensor[~np.all(is_correct_tensor, axis=(layout.H_index() - 1, layout.W_index() - 1))]
    if layers_with_error.shape[0] <= 1:
        return False
    if layout == TensorLayout.NCHW:
        return np.all(layers_with_error == layers_with_error[0,:,:])
    else:
        return np.all(layers_with_error == layers_with_error[:,:,0])            


def same_row_classifier(diff: Iterable[Coordinates], dense_diff: Tuple[np.ndarray, TensorLayout]) -> bool:
    """
    Return True if a bullet Wake spatial distribution is recognized
    Same Row: multiple corrupted values lie in the same row (same feature map)
    """
    first_N = diff[0].N
    first_C = diff[0].C
    first_H = diff[0].H
    for coordinates in diff:
        if (
            coordinates.N != first_N
            or coordinates.C != first_C
            or coordinates.H != first_H
        ):
            return False
    return True


def bullet_wake_classifier(diff: Iterable[Coordinates], dense_diff: Tuple[np.ndarray, TensorLayout]) -> bool:
    """
    Return True if a bullet Wake spatial distribution is recognized
    Bullet Wake: the same location is corrupted in all (or in multiple) feature maps
    """
    first_N = diff[0].N
    first_W = diff[0].W
    first_H = diff[0].H
    for coordinates in diff:
        if (
            coordinates.N != first_N
            or coordinates.H != first_H
            or coordinates.W != first_W
        ):
            return False
    if len(diff) > 1:
        return True
    else:
        return False


def shattered_glass_classifier(diff: Iterable[Coordinates], dense_diff: Tuple[np.ndarray, TensorLayout]) -> bool:
    """
    Return True if a Shattered Glass spatial distribution is recognized.
    Shattered glass: like one or more Bullet wake errors, but in one or multiple feature maps the corruption spreads over a row (or part of the row)
    """
    # Common Row Index
    first_H = diff[0].H
    cols_by_channels = defaultdict(lambda: set())
    for coord in diff:
        # To be Shattered Glass all corruption must stay on the same row of different feature map
        if coord.H != first_H:
            return False
        cols_by_channels[coord.C].add(coord.W)
    if len(cols_by_channels) == 0:
        return False
    common_cols = cols_by_channels[list(cols_by_channels.keys())[0]]
    # Check if there is a common corrupted position in all corrupted feature maps
    for cols_set in cols_by_channels.values():
        common_cols &= cols_set
    if len(common_cols) > 0:
        return True
    else:
        return False


def random_classifier(_: Iterable[Coordinates], x : Tuple[np.ndarray, TensorLayout]) -> bool:
    return True


CLASSIFIERS: Dict[SpatialClass, Callable[[Iterable[Coordinates], Tuple[np.ndarray, TensorLayout]], bool]] = OrderedDict(
    [
        (SpatialClass.SINGLE, single_classifier),
        (SpatialClass.SAME_ROW, same_row_classifier),
        (SpatialClass.BULLET_WAKE, bullet_wake_classifier),
        (SpatialClass.SHATTERED_GLASS, shattered_glass_classifier),
        (SpatialClass.PHOTOCOPY, photocopy),
        (SpatialClass.RANDOM, random_classifier),
    ]
)

SPATIAL_CLASSES = list(CLASSIFIERS.keys())


def spatial_classification(diff : Iterable[Coordinates], sparse_diff : Tuple[np.ndarray, TensorLayout]) -> SpatialClass:
    for sp_class, classifier in CLASSIFIERS.items():
        if classifier(diff, sparse_diff):
            return sp_class
    return SpatialClass.RANDOM


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
