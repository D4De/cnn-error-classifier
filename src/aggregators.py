from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Tuple, TypeVar, Union

from analyzed_tensor import AnalyzedTensor
from coordinates import coordinates_to_tuple
from domain_classifier import ValueClass
from utils import count_by, group_by, sort_dict


def spatial_classes_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    counts = count_by(results, key=lambda x: x.spatial_class)
    counts_names = {sp_class.display_name() : count for sp_class, count in counts.items()}
    return sort_dict(counts_names, sort_key=itemgetter(1), reverse=True)

def faulty_tensors_count(results : List[AnalyzedTensor]) -> int:
    return len(results)


def cardinalities_counts(results : List[AnalyzedTensor]) -> Dict[int, int]:
    counts = count_by(results, key=lambda x: x.corrupted_values_count)
    
    return sort_dict(counts, sort_key=itemgetter(1), reverse=True)

def cardinalities_counts_by_sp_class(results : List[AnalyzedTensor]) -> Dict[str, Dict[int, int]]:
    result = {}
    groups = group_by(results, key=lambda x: x.spatial_class.name)

    for group, tensors in groups.items():
        result[group] = sort_dict(count_by(tensors, key=lambda x: x.corrupted_values_count), sort_key=itemgetter(1), reverse=True)
    
    
    return result


def domain_classes_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    counts = defaultdict(int)
    for result in results:
        for dom_class, count in result.value_classes_counts.items():
            counts[dom_class.display_name()] += count

    return counts

def experiment_counts(metadata_list : List[dict]) -> Union[Tuple[int, Dict[str, int]], Tuple[None, None]]:
    counts = defaultdict(int)
    total_count = 0
    for metadata in metadata_list:
        if "experiment_counts" not in metadata:
            return None, None
        batch_counts = metadata["experiment_counts"]
        for type, exps in batch_counts.items():
            counts[type] += exps
            total_count += exps
    return total_count, counts

def tensor_count_by_sub_batch(results : List[AnalyzedTensor]) -> Dict[str, int]:
    return count_by(results, key=lambda x: x.metadata.get("sub_batch_name"))


def tensor_count_by_shape(results : List[AnalyzedTensor]) -> Dict[str, int]:
    return count_by(results, key=lambda x: str(coordinates_to_tuple(x.shape)))

