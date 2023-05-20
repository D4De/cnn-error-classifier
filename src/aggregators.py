from collections import defaultdict
from functools import reduce
from operator import itemgetter
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypeVar

from analyzed_tensor import AnalyzedTensor
from spatial_classifier import SpatialClass, accumulate_max, to_classes_id
from utils import sort_dict

T = TypeVar("T")
S = TypeVar("S")

def group_by(results : Iterable[S], key : Callable[[S],T]) -> Dict[T, List[S]]:
    groups : defaultdict[T, List[S]] = defaultdict(list)

    for result in results:
        groups[key(result)].append(result)
    
    return groups

def count_by(results : Iterable[S], key : Callable[[S],T]) -> Dict[T, int]:
    counts : defaultdict[T, int] = defaultdict(int)
    
    for result in results:
        counts[key(result)] += 1

    return counts

def spatial_classes_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    counts = count_by(results, key=lambda x: x.spatial_class)
    counts_names = {sp_class.display_name() : count for sp_class, count in counts.items()}
    return sort_dict(counts_names, sort_key=itemgetter(1), reverse=True)

def faulty_tensors_count(results : List[AnalyzedTensor]) -> int:
    return len(results)


def cardinalities_counts(results : List[AnalyzedTensor]) -> Dict[int, int]:
    counts = count_by(results, key=lambda x: x.corrupted_values_count)
    
    return sort_dict(counts, sort_key=itemgetter(1), reverse=True)

