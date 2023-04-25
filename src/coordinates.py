from collections import namedtuple
from typing import Iterable, Tuple, Union


Coordinates = namedtuple("Coordinates", ["N", "H", "W", "C"])

from enum import Enum


class TensorLayout(Enum):
    NHWC = 0
    NCHW = 1

    def N_index(self):
        return self.name.index("N")

    def H_index(self):
        return self.name.index("H")

    def C_index(self):
        return self.name.index("C")

    def W_index(self):
        return self.name.index("W")


LAYOUTS = {
    TensorLayout.NHWC: Coordinates(0, 1, 2, 3),
    TensorLayout.NCHW: Coordinates(0, 2, 3, 1),
}


def map_to_coordinates(native_coord: tuple, layout: TensorLayout) -> Coordinates:
    selected_layout = LAYOUTS[layout]
    return Coordinates(*[native_coord[new_idx] for new_idx in selected_layout])


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

def identify_block_length(
    indexes: Iterable[int],
) -> Union[Tuple[int, int, int], None]:
    cardinality = len(indexes)
    if cardinality <= 10:
        return None
    # span is the distance between the error with the lowest index and the error with the highest index
    block_begin = indexes[0]
    block_end = indexes[-1]
    span = block_end - block_begin

    # check that errors have the right cardinality for a block length of 16,32,64 and that they are all within the block
    if 10 < cardinality <= 16 and span <= 16:
        return 16, block_begin % 16, block_begin // 16
    elif 16 < cardinality <= 32 and span <= 32:
        return 32, block_begin % 32, block_begin // 32
    elif 32 < cardinality <= 64 and span <= 64:
        return 64, block_begin % 64, block_begin // 64
    else:
        return None

