from collections import OrderedDict, defaultdict
from typing import Any, Callable, Tuple


def int_defaultdict() -> defaultdict[Any, int]:
    return defaultdict(int)


def double_int_defaultdict() -> defaultdict[Any, int]:
    return defaultdict(int_defaultdict)


def list_defaultdict() -> defaultdict[Any, list]:
    return defaultdict(list)


def sort_dict(
    data: dict,
    sort_key: Callable[[Tuple[Any, Any]], Any] = lambda x: x[1],
    reverse=True,
):
    """
    Returns an OrderedDict. Items are sorted and inserted inside the OrderedDict following a sorting order specified in the parameters of this function

    data : dict
    ---
    Any dictionary to sort

    sort_key : Callable[[Tuple[Any, Any]], Any]
    ---
    A function that takes in input a tuple (key and value of a dictionary entry) and returns a value that will be used as sorting key
    If not specified the dictionary will be sorted against the values

    reverse : boolean
    ---
    If true, the insertion order will be ascendent on the keys. Defaults to true
    """
    return OrderedDict(
        sorted(
            [(key, value) for key, value in data.items()], key=sort_key, reverse=reverse
        )
    )
