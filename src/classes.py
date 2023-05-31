from collections import OrderedDict, defaultdict
from functools import reduce
import json
from operator import itemgetter
import os
from typing import Any, Dict, Iterable, List, Tuple

from utils import count_by
from analyzed_tensor import AnalyzedTensor
from args import Args
from domain_classifier import ValueClass
from utils import group_by, sort_dict


def generate_classes_models(results : Iterable[AnalyzedTensor], args: Args):
    tensor_by_sp_class = group_by(results, key=lambda x: x.spatial_class)

    classes_model = {}

    for sp_class, tensors in tensor_by_sp_class.items():
        classes_model[sp_class.display_name()] = {
            "count": len(tensors),
            "frequency": len(tensors) / len(results),
            "domain_classes": generate_domain_class_freq(tensors),
            "parameters": generate_parameter_list(tensors)
        }

    classes_output_dir = os.path.join(
        args.output_dir, f"{args.classes[0]}_{args.classes[1]}"
    )

    if not os.path.exists(classes_output_dir):
        os.mkdir(classes_output_dir)

    with open(os.path.join(classes_output_dir, f"{args.classes[0]}_{args.classes[1]}.json"), 'w') as f:
        json.dump(classes_model, f, indent=2)
    



def generate_parameter_list(sp_class_results : Iterable[AnalyzedTensor]):
    def grouper_key_func(t : AnalyzedTensor):
        hashable_key = tuple(sorted(t.spatial_class_params.keys.items(), key=itemgetter(0)))
        return hashable_key


    tensor_by_parameters = group_by(sp_class_results, key=grouper_key_func)
    total_sp_class_items = len(sp_class_results)
    parameters_list : List[Dict[str, Any]] = []
    for key, analyzed_tensor in tensor_by_parameters.items():
        param_dict = {
            **(analyzed_tensor[0].spatial_class_params.keys),
            "frequency": len(analyzed_tensor) / total_sp_class_items,
            "count": len(analyzed_tensor)
        }
        parameters_list.append(param_dict)

    return sorted(parameters_list, key= lambda x: x["count"], reverse=True)

def generate_domain_class_freq(sp_class_results : Iterable[AnalyzedTensor]):
    domain_classes_count = count_by(sp_class_results, key=lambda x: x.domain_class.display_name())
    domain_classes_relative = {dom_class : freq / len(sp_class_results) for dom_class, freq in domain_classes_count.items()}
    return domain_classes_relative
