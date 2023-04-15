from collections import defaultdict
from typing import Any


def int_defaultdict() -> defaultdict[Any, int]:
    return defaultdict(int)

def double_int_defaultdict() -> defaultdict[Any, int]:
    return defaultdict(int_defaultdict)

def list_defaultdict() -> defaultdict[Any, list]:
    return defaultdict(list)