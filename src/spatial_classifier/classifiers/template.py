from typing import Dict, Iterable, Tuple

from coordinates import Coordinates


def template_pattern(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    """
    This is an example of a classifier for a spatial pattern
    Copy file and write your classifier under this docstring

    sparse_diff
    ---
    An iterable of Coordinates containing the coordinates of the values of the faulty that are different from the golden one.
    This ite

    shape
    ---
    The shape of the faulty and the golden tensor here analyzed

    corr_channels
    ---
    An iterable that contains the number of channels that are corrupted

    Returns
    ---
    * If the pattern matches:
        A tuple containing (True, dict) where the dict has at least the following keys:
            * "error_pattern": A json serializable pattern that represent the spatial distribution
            * "MAX": A list (or a single value) that will be accumulated togheter with other values matching this pattern. The accumulator function is
            accumulate_max inside tensor_classifier.py. If in doubt put an empty list here
    * If the pattern not matches:
        A tuple containing (False, {})
    False
    """
    ...
    if ...:
        return True, {"error_pattern": ..., "MAX": ...}
    else:
        return None
