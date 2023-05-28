from collections import defaultdict
from operator import itemgetter
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar, Union

from analyzed_tensor import AnalyzedTensor
from coordinates import coordinates_to_tuple
from domain_classifier import ValueClass
from utils import sort_dict

T = TypeVar("T")
S = TypeVar("S")

def group_by(results : Iterable[S], key : Callable[[S],T]) -> Dict[T, List[S]]:
    groups : defaultdict[T, List[S]] = defaultdict(list)

    for result in results:
        groups[key(result)].append(result)
    
    return groups

def count_by(results : Iterable[S], key : Callable[[S],T], count_funct : Callable[[S], int] = lambda x: 1) -> Dict[T, int]:
    counts : defaultdict[T, int] = defaultdict(int)
    
    for result in results:
        counts[key(result)] += count_funct(result)

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


def classes_triple_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    def triple_maker(x : AnalyzedTensor):
        if x.corrupted_values_count == 1:
            return "(1,-1,0)"
        return str((x.corrupted_values_count, x.spatial_class.display_name(), x.spatial_pattern))
    return sort_dict(count_by(results, key=triple_maker), sort_key=itemgetter(1), reverse=True)

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


def domain_classes_types_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    type_counts = {}

    for type in ["(RANDOM)", "(OFF_BY_ONE)", "(NAN)", "UNCATEGORIZED", "(RANDOM, OFF_BY_ONE)", "(RANDOM, SINGLE_NAN)", "(RANDOM, MULTIPLE_NAN)"]:
        type_counts[type] = 0
    for result in results:
        dom_classes_counts = result.value_classes_counts
        non_zero_dom_classes = [dom_class for dom_class, count in dom_classes_counts.items() if count > 0 and dom_class != ValueClass.SAME and dom_class != ValueClass.ALMOST_SAME]
        if len(non_zero_dom_classes) == 1:
            the_dom_class = non_zero_dom_classes[0]
            if the_dom_class == ValueClass.RANDOM:
                type_counts["(RANDOM)"] += 1
            elif the_dom_class == ValueClass.OFF_BY_ONE:
                type_counts["(OFF_BY_ONE)"] += 1                
            elif the_dom_class == ValueClass.NAN:
                type_counts["(NAN)"] += 1
            else:
                type_counts["UNCATEGORIZED"] += 1
        elif len(non_zero_dom_classes) == 2:
            non_zero_dom_classes_set = set(non_zero_dom_classes)
            if ValueClass.OFF_BY_ONE in non_zero_dom_classes_set and ValueClass.RANDOM in non_zero_dom_classes_set:
                type_counts["(RANDOM, OFF_BY_ONE)"] += 1
            elif ValueClass.NAN in non_zero_dom_classes_set and ValueClass.RANDOM in non_zero_dom_classes_set:
                nan_count = dom_classes_counts[ValueClass.NAN]
                if nan_count == 1:
                    type_counts["(RANDOM, SINGLE_NAN)"] += 1
                else:
                    type_counts["(RANDOM, MULTIPLE_NAN)"] += 1
        else:
            type_counts["UNCATEGORIZED"] += 1

        
    return type_counts

def domain_class_type_per_spatial_class(result : List[AnalyzedTensor]) -> Dict[str, Dict[str, int]]:
    sp_classes_group = group_by(result, lambda x: x.spatial_class.display_name())
    result_dict = {}
    for sp_class, tensors in sp_classes_group.items():
        result_dict[sp_class] = domain_classes_types_counts(tensors)
    return result_dict