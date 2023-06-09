from collections import defaultdict
import json
from operator import itemgetter
import os
from typing import Any, Dict, Iterable, List, Literal
from domain_classifier import ValueClass

from utils import sort_dict
from analyzed_tensor import AnalyzedTensor
from args import Args
from utils import group_by
import collections.abc

def generate_classes_models(results : Iterable[AnalyzedTensor], args: Args):
    """
    Generate a json file containing that can be used by the CLASSES framework
    to simulate the errors.

    Parameters
    ---
    results : Iterable[AnalyzedTensor]
        Result of the analysis for each corrupted tensor.
    args : Args
        Object containing all the user preferences supplied by the command line arguments

    Returns
    ---
    None. A json file named "{args.classes[0]}_{args.classes[1]}.json" will be created in the result folder

    Raises
    ---
    FileNotFoundError: If the folder where the file is written does not exist

    """
    tensor_by_sp_class = group_by(results, key=lambda x: x.spatial_class)

    classes_model = {
        "_tensor_count_pre_pruning": len(results)
    }

    for sp_class, tensors in tensor_by_sp_class.items():
        parameter_list, categories_count = generate_parameter_list(tensors, len(results))
        classes_model[sp_class.display_name()] = {
            "count": len(tensors),
            "frequency": len(tensors) / len(results),
            "categories_count": categories_count,
            "domain_classes": generate_domain_class_freq(tensors, len(tensors) / len(results)),
            "parameters": parameter_list
        }


    classes_model = prune_classes_model(classes_model, args)

    classes_output_dir = os.path.join(
        args.output_dir, f"{args.classes[0]}_{args.classes[1]}"
    )

    if not os.path.exists(classes_output_dir):
        os.mkdir(classes_output_dir)

    with open(os.path.join(classes_output_dir, f"{args.classes[0]}_{args.classes[1]}.json"), 'w') as f:
        json.dump(classes_model, f, indent=3)
    

def prune_classes_model(classes_model : Dict[str, dict], args: Args):
    pruned_classes_model = {}
    updated_tensor_count = 0
    updated_category_count = 0
    for sp_class, sp_class_dict in classes_model.items():

        if sp_class.startswith("_"):
            continue

        sp_class_count = 0
        old_tensor_count = sp_class_dict["count"]

        if old_tensor_count < args.classes_category_absolute_cutoff:
            continue

        big_categories = [param for param in sp_class_dict["parameters"] if param["overall_frequency"] >= args.classes_category_relative_cutoff]
        big_categories_count = sum(cat["count"] for cat in big_categories)
        updated_tensor_count += big_categories_count
        sp_class_count += big_categories_count

        cutoff_categories = [param for param in sp_class_dict["parameters"] if param["overall_frequency"] < args.classes_category_relative_cutoff]
        
        new_categories = big_categories

        if len(cutoff_categories) >= 1:
            merged_cat = merge_categories(cutoff_categories)
            if merged_cat["count"] >= args.classes_category_absolute_cutoff:
                updated_tensor_count += merged_cat["count"]
                sp_class_count += merged_cat["count"]
                updated_category_count += 1
                new_categories.append(merged_cat)

        pruned_classes_model[sp_class] = {
            "count": sp_class_count,
            "frequency": None,
            "categories_count": len(new_categories),
            "domain_classes": sp_class_dict["domain_classes"],
            "parameters": new_categories
        }
    
    redone_classes_model = {}
    redone_classes_model["_rejected_tensors_proportion"] = 1 - (updated_tensor_count / classes_model["_tensor_count_pre_pruning"])
    redone_classes_model["_tensor_count"] = updated_tensor_count
    redone_classes_model["_categories_count"] = updated_category_count

    for sp_class, sp_class_dict in pruned_classes_model.items():
        if sp_class.startswith("_"):
            continue
        redone_classes_model[sp_class] = sp_class_dict
        redone_classes_model[sp_class]["frequency"] = redone_classes_model[sp_class]["count"] / updated_tensor_count
        parameters = redone_classes_model[sp_class]["parameters"]
        for category in parameters:
            category["conditional_frequency"] = category["count"] / sp_class_dict["count"]
            category["overall_frequency"] = category["count"] / updated_tensor_count
    
    return redone_classes_model


def merge_categories(categories : List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(categories) == 1:
        return categories[0]
    
    if len(categories) < 1:
        raise ValueError("Categories argument list should have at least one argument")

    merged_keys = merge_parameters_dict(categories, "keys")

    merged_stats = merge_parameters_dict(categories, "stats")


    unified_cat = {
        "keys": merged_keys,
        "stats": merged_stats,
        "conditional_frequency": None,
        "overall_frequency": None,
        "count": sum(cat["count"] for cat in categories)
    }
    return unified_cat


def merge_parameters_dict(categories : List[Dict[str, Any]], param_key : Literal["keys", "stats"]):
    
    merged_dict = {}
    
    for key in categories[0][param_key].keys():
        possible_values = []
        for category in categories:
            value = category[param_key][key]
            if isinstance(value, dict):
                if "RANDOM" in value:
                    possible_values += value["RANDOM"]
                else:
                    raise ValueError("A dict as parameter value is accepted only when contains 'RANDOM' key" ) 
            else:
                possible_values.append(value)
            
        unique_values = list(set(possible_values))
        if len(unique_values) == 1:
            merged_dict[key] = unique_values[0]
        elif len(unique_values) > 1:
            merged_dict[key] = {"RANDOM": unique_values}
        else:
            raise ValueError("Inconsistent State: There must be some value in the unique_value list")

    return merged_dict

def generate_parameter_list(sp_class_results : Iterable[AnalyzedTensor], total_analyzed_tensor_count : int):
    def make_hashable_key_value_pairs(t : AnalyzedTensor):
        hashable_key = tuple(sorted(t.spatial_class_params.keys.items(), key=itemgetter(0)))
        return hashable_key

    tensor_by_parameters = group_by(sp_class_results, key=make_hashable_key_value_pairs)
    parameters_category_count = len(tensor_by_parameters)
    total_sp_class_items = len(sp_class_results)
    parameters_list : List[Dict[str, Any]] = []
    for analyzed_tensors in tensor_by_parameters.values():
        stats_dict = defaultdict(list)
        for aggregated_key, value in analyzed_tensors[0].spatial_class_params.stats.items():
            if not isinstance(value, collections.abc.Sized):
                raise ValueError(f"Spatial class classifier for {analyzed_tensors[0].spatial_class.display_name()} is malformed: values of SpatialClassParameter.stats dict must be a (<value>, <aggregator>) tuple")
            _, aggregator = value
            aggr_keys = []
            for tensor in analyzed_tensors:
                aggr_keys.append(tensor.spatial_class_params.stats[aggregated_key][0])
            stats_dict[aggregated_key] = aggregator(*aggr_keys)

        param_dict = {
            "keys": analyzed_tensors[0].spatial_class_params.keys,
            "stats": stats_dict,
            "conditional_frequency": len(analyzed_tensors) / total_sp_class_items,
            "overall_frequency": len(analyzed_tensors) / total_analyzed_tensor_count,
            "count": len(analyzed_tensors)
        }
        parameters_list.append(param_dict)

    return sorted(parameters_list, key= lambda x: x["count"], reverse=True), parameters_category_count

def generate_domain_class_freq(sp_class_results : Iterable[AnalyzedTensor], sp_class_freq : float):
    def dom_classes_hashable_keys(t : AnalyzedTensor):
        hashable_key = tuple(sorted(t.domain_class.items(), key=itemgetter(0)))
        return hashable_key
    tensors_count_by_dom_class = group_by(sp_class_results, key=dom_classes_hashable_keys)
    random_count = 0
    dom_classes = []
    for tensors_in_dom_class in tensors_count_by_dom_class.values():
        if len(tensors_in_dom_class) == 0:
            continue
        dom_class = tensors_in_dom_class[0].domain_class
        dom_class_count = len(tensors_in_dom_class)
        dom_class_rel_freq = dom_class_count / len(sp_class_results)
        if dom_class_rel_freq * sp_class_freq < 0.01 or dom_class_count < 5 or "random" in dom_class:
            random_count += dom_class_count
        else:
            sorted_dom_class = sort_dict(dom_class, sort_key=lambda x: x[1][0], reverse=True)
            dom_classes.append({
                **sorted_dom_class,
                "count": dom_class_count,
                "frequency": dom_class_rel_freq
            })
    dom_classes = sorted(dom_classes, key=lambda x: x["frequency"], reverse=True)
    if random_count > 0:

        values_distribution = value_class_distribution(sp_class_results)

        dom_classes.append({
            "random": (100.0, 100.0),
            "count": random_count,
            "values": values_distribution,
            "frequency": random_count / len(sp_class_results)       
        })

    return dom_classes


def value_class_distribution(sp_class_results : Iterable[AnalyzedTensor]) -> Dict[str, float]:
    value_classes_counts : defaultdict[str, int] = defaultdict(int)
    total_count = 0
    for cl in sp_class_results:
        for val_class, cnt in cl.value_classes_counts.items():
            if val_class == ValueClass.SAME:
                continue
            value_classes_counts[val_class.display_name()] += cnt
            total_count += cnt
    return {val_class : cnt / total_count for val_class, cnt in value_classes_counts.items()}