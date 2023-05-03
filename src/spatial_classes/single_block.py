from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, identify_block_length, raveled_tensor_index


def single_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    indexes = sorted(raveled_tensor_index(shape, coord) for coord in sparse_diff)
    block_begin = indexes[0]
    result = identify_block_length(indexes)
    if result is None:
        return False, {}
    block_length, aligment_offset, block_id = result
    # Pattern: (<Block Length>, ([... <tensor indices based from begin of the block>]))
    error_pattern = (block_length, tuple(idx - block_begin for idx in indexes))
    return True, {
        "error_pattern": error_pattern,
        "block_length": block_length,
        "MAX": [],
    }
