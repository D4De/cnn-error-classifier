from typing import Dict, Iterable, Tuple

from coordinates import Coordinates, raveled_tensor_index


def tensor_aligned_single_block_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    Returns True if a Channel Aligned Same Block spatial distribution is recognized.

    """
    raveled_fault_indexes = sorted(
        raveled_tensor_index(shape, coord) for coord in sparse_diff
    )
    corrupted_items_count = len(sparse_diff)
    aligments = [16, 32, 64]
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
    error_pattern = (
        min_align,
        tuple(idx - block_begin for idx in raveled_fault_indexes),
    )
    return True, {
        "error_pattern": error_pattern,
        "align": min_align,
        "MAX": [],
    }
