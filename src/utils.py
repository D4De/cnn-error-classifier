from collections import OrderedDict, defaultdict
import math
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar
import numpy as np
import zipfile

def int_defaultdict() -> defaultdict[Any, int]:
    return defaultdict(int)


def double_int_defaultdict() -> defaultdict[Any, int]:
    return defaultdict(int_defaultdict)


def list_defaultdict() -> defaultdict[Any, list]:
    return defaultdict(list)

K = TypeVar("K")
V = TypeVar("V")

def sort_dict(
    data: Dict[K, V],
    sort_key: Callable[[Tuple[K, V]], Any] = lambda x: x[1],
    reverse=False,
) -> OrderedDict[K, V]:
    """
    Returns an OrderedDict. Items are sorted and inserted inside the OrderedDict following a sorting order specified in the parameters of this function

    data : Dict[K, V]
    ---
    Any dictionary to sort

    sort_key : Callable[[Tuple[K, V]], Any]
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

T = TypeVar("T")
S = TypeVar("S")


def group_by(results : Iterable[S], key : Callable[[S],T]) -> Dict[T, List[S]]:
    """
    Takes an Iterable of a type S and a grouping function that maps an object of type S to its group, represented by a variable of type T.
    Returns  
    """
    groups : defaultdict[T, List[S]] = defaultdict(list)

    for result in results:
        groups[key(result)].append(result)

    return groups


def count_by(results : Iterable[S], key : Callable[[S],T], count_funct : Callable[[S], int] = lambda x: 1) -> Dict[T, int]:
    counts : defaultdict[T, int] = defaultdict(int)

    for result in results:
        counts[key(result)] += count_funct(result)

    return counts

def quantize_percentage(proportion : float, quantization_levels : int = 10) -> Tuple[float, float]:
    step = 100 / quantization_levels
    top_value = float(min(math.ceil(proportion * quantization_levels) * step, 100))
    bot_value = float(max(0, top_value - step))
    return (bot_value, top_value)


#--NUMPY UTILS--
def read_npy_size(npy_path):
    """Reads just the header of a .npy file to determine the array size."""
    with open(npy_path, 'rb') as fobj:
        version = np.lib.format.read_magic(fobj)
        func_name = 'read_array_header_' + '_'.join(str(v) for v in version)
        func = getattr(np.lib.format, func_name)
        shape, _, _ = func(fobj)
        return shape
    
def read_npz_sizes(npz_path):
    """Reads the headers of the .npy files within a .npz archive and yields the sizes."""
    with zipfile.ZipFile(npz_path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
                
            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            func_name = 'read_array_header_' + '_'.join(str(v) for v in version)
            func = getattr(np.lib.format, func_name)
            shape, _, _ = func(npy)
            yield shape