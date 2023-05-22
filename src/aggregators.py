from collections import defaultdict
from operator import itemgetter
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

from analyzed_tensor import AnalyzedTensor
from domain_classifier import DomainClass
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


def classes_triple_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    def triple_maker(x : AnalyzedTensor):
        if x.corrupted_channels_count == 1:
            return "(1,-1,0)"
        return str((x.corrupted_values_count, x.spatial_class.display_name(), x.spatial_pattern))
    return sort_dict(count_by(results, key=triple_maker), sort_key=itemgetter(1), reverse=True)

def domain_classes_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    counts = defaultdict(int)
    for result in results:
        for dom_class, count in result.domain_classes_counts.items():
            counts[dom_class.display_name()] += count

    return counts

def domain_classes_types_counts(results : List[AnalyzedTensor]) -> Dict[str, int]:
    type_counts = defaultdict(int)
    for result in results:
        dom_classes_counts = result.domain_classes_counts
        non_zero_dom_classes = [dom_class for dom_class, count in dom_classes_counts.items() if count > 0 and dom_class != DomainClass.SAME and dom_class != DomainClass.ALMOST_SAME]
        if len(non_zero_dom_classes) == 1:
            the_dom_class = non_zero_dom_classes[0]
            if the_dom_class == DomainClass.RANDOM:
                type_counts["(RANDOM)"] += 1
            elif the_dom_class == DomainClass.OFF_BY_ONE:
                type_counts["(OFF_BY_ONE)"] += 1                
            elif the_dom_class == DomainClass.NAN:
                type_counts["(NAN)"] += 1
            elif the_dom_class == DomainClass.FLIP:
                type_counts["UNCATEGORIZED"] += 1
        elif len(non_zero_dom_classes) == 2:
            non_zero_dom_classes_set = set(non_zero_dom_classes)
            if DomainClass.OFF_BY_ONE in non_zero_dom_classes_set and DomainClass.RANDOM in non_zero_dom_classes_set:
                type_counts["(RANDOM, OFF_BY_ONE)"] += 1
            elif DomainClass.NAN in non_zero_dom_classes_set and DomainClass.RANDOM in non_zero_dom_classes_set:
                nan_count = dom_classes_counts[DomainClass.NAN]
                if nan_count == 1:
                    type_counts["(RANDOM, SINGLE_NAN)"] += 1
                else:
                    type_counts["(RANDOM, MULTIPLE_NAN)"] += 1
        
    return type_counts

            