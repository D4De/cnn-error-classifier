from collections import OrderedDict, defaultdict
from functools import reduce
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from aggregators import count_by, group_by
from analyzed_tensor import AnalyzedTensor
from args import Args
from domain_classifier import DomainClass
from spatial_classifier import accumulate_max, to_classes_id
from utils import sort_dict


def generate_classes_models_old(results : Iterable[AnalyzedTensor], args : Args):
    """
    Calculates and create the three files containing the CLASSES framework model

    results
    ---
    List of all the analyzed experimental results (AnalyzedTensor)

    args
    ---
    The Args object containing all the program options, used for getting the output folder
    """
    cardinalities, spatial_models = calculate_classes_cardinalities_and_spatial(results)
    value_analysis = calculate_value_analysis(results)
    save_classes_models(cardinalities, spatial_models, value_analysis, args)

def save_classes_models(cardinality_model_json : dict, spatial_model_json : dict, value_analisys_txt : str, args : Args):
    classes_output_dir = os.path.join(
        args.output_dir, f"{args.classes[0]}_{args.classes[1]}"
    )
    if not os.path.exists(classes_output_dir):
        os.mkdir(classes_output_dir)
    
    with open(os.path.join(classes_output_dir, "value_analysis.txt"), "w") as f:
        f.write(value_analisys_txt)
    
    with open(
        os.path.join(
            classes_output_dir,
            f"{args.classes[0]}_{args.classes[1]}_spatial_model.json",
        ),
        "w",
    ) as f:
        f.write(json.dumps(spatial_model_json, indent=2))
    
    with open(
        os.path.join(
            classes_output_dir,
            f"{args.classes[1]}_{args.classes[0]}_anomalies_count.json",
        ),
        "w",
    ) as f:
        f.write(json.dumps(cardinality_model_json, indent=2))

def calculate_classes_cardinalities_and_spatial(results : Iterable[AnalyzedTensor]) -> Tuple[Dict[int, list], Dict[int, Dict[str, Dict[str, float]]]]:
    total_count = len(results)
    # Dictionary containing all the conts 
    cardinalities = count_by(results, key=lambda x: x.corrupted_values_count)    
    classes_cardinalities_json = {cardinality : [count, count / total_count] for cardinality, count in cardinalities.items()} 

    results_by_cardinality = group_by(results, key=lambda x: x.corrupted_values_count)
    # Maximum threshold for relative frequencies inside the patterns
    # All the patterns that have a relative frequency lower than that will be considered random
    RANDOM_THRESHOLD = 0.05

    classes_spatial_json : Dict[int, Dict[str, Dict[str, float]]] = OrderedDict()

    for cardinality, cardinality_results in results_by_cardinality.items():
        if cardinality == 1:
            classes_spatial_json[1] = {"RANDOM": 1.0}
            continue

        cardinality_total_count = cardinalities[cardinality]
        results_by_sp_class = group_by(cardinality_results, key=lambda x: x.spatial_class)
        # Key: Spatial Class ID, Value: Relative Frequqncy of the spatial class given the current cardinality
        ff_spatial_json = OrderedDict()
        # Key: Spatial Class ID, Value: A dictionary
        # The sub-dictionary contains
        # Keys: The spatial pattern (parameters of the pattern) or "RANDOM" or "MAX"
        # Values: Relative frequency of the spatial pattern given the current cardinality and the spatial pattern
        pf_spatial_json = OrderedDict()
        for sp_class, sp_class_results in results_by_sp_class.items():
            # Get the key of the FF/PF dicts: The id of the spatial pattern used in CLASSES
            sp_class_classes_id = to_classes_id(sp_class.display_name())
            # Set the relative frequency of the pattern
            ff_spatial_json[sp_class_classes_id] = len(sp_class_results) / cardinality_total_count
            # Number of tensors that have the current cardinality and current spatial class
            sp_pattern_total_count = len(sp_class_results)
            # Count the number of occurence of each pattern
            sp_pattern_counts = count_by(sp_class_results, key=lambda x: x.spatial_pattern)
            sp_pattern_counts_classes = {str(pattern) : count / sp_pattern_total_count for pattern, count in sp_pattern_counts.items() if count / sp_pattern_total_count >= RANDOM_THRESHOLD}
            # Count the number of patterns that fall into the wildcard category "RANDOM" (relative frequency < RANDOM_THRESHOLD)
            random_counts = sum(count for count in sp_pattern_counts.values() if count / sp_pattern_total_count < RANDOM_THRESHOLD)

            sp_class_results_max = [x.spatial_class_params.get("MAX", []) for x in sp_class_results]
            sp_class_results_max = [x if isinstance(x, list) else [x] for x in sp_class_results_max]
            # Calculate the "MAX" vector, useful for generating "RANDOM" patterns inside CLASSES
            max_vector = reduce(accumulate_max, sp_class_results_max)
            sp_pattern_counts_classes = sp_pattern_counts_classes | {"MAX": max_vector, "RANDOM": random_counts / sp_pattern_total_count}

            pf_spatial_json[sp_class_classes_id] = sp_pattern_counts_classes
        
        classes_spatial_json[str(cardinality)] = {"FF": ff_spatial_json, "PF": pf_spatial_json}
    return sort_dict(classes_cardinalities_json, sort_key=lambda k: int(k[0])), sort_dict(classes_spatial_json, sort_key=lambda k: int(k[0]))



def calculate_value_analysis(results : Iterable[AnalyzedTensor]) -> str:
    counts : defaultdict[DomainClass, int] = defaultdict(int)
    corrupted_values = 0
    for result in results:
        for dom_class, count in result.domain_classes_counts.items():
            if dom_class == DomainClass.SAME or dom_class == DomainClass.ALMOST_SAME:
                continue
            counts[dom_class] += count
            corrupted_values += count
    counts = {clz : cnt / corrupted_values for clz, cnt in counts.items()}
    values_txt = f'There have been {corrupted_values} faults\n[-1, 1]: {counts[DomainClass.OFF_BY_ONE]}\nOthers: {counts[DomainClass.RANDOM]}\nNan: {counts[DomainClass.NAN]}\nZeros: {counts[DomainClass.ZERO]}\nValid: 1.00000' 
    return values_txt
