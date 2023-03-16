from typing import Callable, Dict, List, Tuple
from collections import OrderedDict, defaultdict
from coordinates import Coordinates
from enum import Enum
import os


class SpatialClass(Enum):
    SAME = 0
    SINGLE = 1
    SAME_ROW = 2
    BULLET_WAKE = 3
    SHATTERED_GLASS = 4
    RANDOM = 5

    def display_name(self):
        return self.name.lower()

    def class_folder(self, output_path) -> str:
        return os.path.join(output_path, self.display_name())

    def output_path(self, output_path, basename) -> str:
        return os.path.join(output_path, self.display_name(), basename)


def single_classifier(diff: List[Coordinates]) -> bool:
    return len(diff) == 1


def same_row_classifier(diff: List[Coordinates]) -> bool:
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


def bullet_wake_classifier(diff: List[Coordinates]) -> bool:
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


def shattered_glass_classifier(diff: List[Coordinates]) -> bool:
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
    common_cols = set(cols_by_channels.keys())
    # Check if there is a common corrupted position in all corrupted feature maps
    for cols_set in cols_by_channels.values():
        common_cols &= cols_set
    if len(common_cols) > 0:
        return True
    else:
        return False


def random_classifier(_: List[Coordinates]) -> bool:
    return True


CLASSIFIERS: Dict[SpatialClass, Callable[[List[Coordinates]], bool]] = OrderedDict(
    [
        (SpatialClass.SINGLE, single_classifier),
        (SpatialClass.SAME_ROW, same_row_classifier),
        (SpatialClass.BULLET_WAKE, bullet_wake_classifier),
        (SpatialClass.SHATTERED_GLASS, shattered_glass_classifier),
        (SpatialClass.RANDOM, random_classifier),
    ]
)

SPATIAL_CLASSES = list(CLASSIFIERS.keys())


def spatial_classification(diff) -> SpatialClass:
    for sp_class, classifier in CLASSIFIERS.items():
        if classifier(diff):
            return sp_class
    return SpatialClass.RANDOM


def create_spatial_classification_folders(output_path: str):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    for sp_class in SPATIAL_CLASSES:
        class_path = sp_class.class_folder(output_path)
        if not os.path.isdir(class_path):
            os.mkdir(class_path)
