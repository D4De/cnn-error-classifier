import json

from typing import Any, Dict, Tuple
from dataclasses import dataclass
from domain_classifier import ValueClass
from spatial_classifier.spatial_classifier import SpatialClass
from spatial_classifier.spatial_class_parameters import SpatialClassParameters


@dataclass
class AnalyzedTensorConv:
    group : str
    hw_unit : str
    error_number: int
    injection_number: int
    shape : tuple
    spatial_class : SpatialClass    
    spatial_class_params : SpatialClassParameters
    value_classes_counts : Dict[ValueClass, int]
    domain_class : Dict[str, Tuple[float, float]]
    corrupted_values_count : int
    corrupted_channels_count : int

    def as_insert_param_list(self) -> Dict[str, Any]:

        domain_class_count_str = {
            dom.display_name() : count for dom, count in self.value_classes_counts.items()
        }

        return {
            "group": self.group,
            "hw_unit": self.hw_unit,
            "error_number": self.error_number,
            "injection_number": self.injection_number,
            "shape": self.shape,
            "spatial_class": self.spatial_class.display_name(),
            "spatial_class_params": self.spatial_class_params.to_json(),
            "value_classes_counts": json.dumps(domain_class_count_str),
            "domain_class": json.dumps(self.domain_class),
            "corrupted_values_count": self.corrupted_values_count,
            "corrupted_channels_count": self.corrupted_channels_count,
        }


@dataclass
class AnalyzedTensorFC:
    group : str
    hw_unit : str
    error_number: int
    injection_number: int
    shape : tuple
    corrupted_values_count : int
    L1_distance: float
    L2_distance: float


    def as_insert_param_list(self) -> Dict[str, Any]:
        return {
            "group": self.group,
            "hw_unit": self.hw_unit,
            "error_number": self.error_number,
            "injection_number": self.injection_number,
            "shape": self.shape,
            "corrupted_values_count": self.corrupted_values_count,
            "L1_distance": self.L1_distance,
            "L2_distance": self.L2_distance
        }