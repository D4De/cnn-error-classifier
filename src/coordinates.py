from collections import namedtuple


Coordinates = namedtuple("Coordinates", ["N", "H", "W", "C"])

from enum import Enum


class TensorLayout(Enum):
    NHWC = 0
    NCHW = 1


LAYOUTS = {
    TensorLayout.NHWC: Coordinates(0, 1, 2, 3),
    TensorLayout.NCHW: Coordinates(0, 2, 3, 1),
}


def map_to_coordinates(native_coord: tuple, layout: TensorLayout) -> Coordinates:
    selected_layout = LAYOUTS[layout]
    return Coordinates(*[native_coord[new_idx] for new_idx in selected_layout])
