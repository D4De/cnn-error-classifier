from typing import Dict, Iterable, Tuple

from coordinates import Coordinates


def single_classifier(
    sparse_diff: Iterable[Coordinates],
    shape: Coordinates,
    corr_channels: Iterable[int],
) -> Tuple[bool, Dict[str, any]]:
    if len(sparse_diff) == 1:
        return True, {"error_pattern": 0}
    else:
        return False, {}