from collections import OrderedDict, defaultdict
from functools import reduce
import json
from operator import itemgetter
import os
from typing import Any, Dict, Iterable, List, Tuple

from utils import count_by
from analyzed_tensor import AnalyzedTensor
from args import Args
from utils import group_by
import pprint as pp
import collections.abc

def generate_classes_models(results : Iterable[AnalyzedTensor], args: Args):
    tensor_by_sp_class = group_by(results, key=lambda x: x.spatial_class)

    classes_model = {}

    for sp_class, tensors in tensor_by_sp_class.items():
        parameter_list, categories_count = generate_parameter_list(tensors, len(results))
        classes_model[sp_class.display_name()] = {
            "count": len(tensors),
            "frequency": len(tensors) / len(results),
            "categories_count": categories_count,
            "domain_classes": generate_domain_class_freq(tensors),

            "parameters": parameter_list
        }

    classes_output_dir = os.path.join(
        args.output_dir, f"{args.classes[0]}_{args.classes[1]}"
    )

    if not os.path.exists(classes_output_dir):
        os.mkdir(classes_output_dir)

    with open(os.path.join(classes_output_dir, f"{args.classes[0]}_{args.classes[1]}.json"), 'w') as f:
        json.dump(classes_model, f, indent=3)
    



def generate_parameter_list(sp_class_results : Iterable[AnalyzedTensor], total_analyzed_tensor_count : int):
    def grouper_key_func(t : AnalyzedTensor):
        hashable_key = tuple(sorted(t.spatial_class_params.keys.items(), key=itemgetter(0)))
        return hashable_key

    tensor_by_parameters = group_by(sp_class_results, key=grouper_key_func)
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

def generate_domain_class_freq(sp_class_results : Iterable[AnalyzedTensor]):
    domain_classes_count = count_by(sp_class_results, key=lambda x: x.domain_class.display_name())
    domain_classes_relative = {dom_class : freq / len(sp_class_results) for dom_class, freq in domain_classes_count.items()}
    return domain_classes_relative
