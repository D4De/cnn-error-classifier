


from dataclasses import dataclass
from typing import Any, Dict

from coordinates import Coordinates, TensorLayout
from domain_classifier import DomainClass
from spatial_classifier import SpatialClass


@dataclass
class AnalyzedTensor:
    batch : str
    sub_batch : str
    file_name : str
    file_path : str
    shape : Coordinates
    spatial_class : SpatialClass    
    spatial_class_params : Dict[str, Any]
    spatial_pattern : Any
    domain_classes_counts : Dict[DomainClass, int]
    corrupted_values_count : int
    corrupted_channels_count : int
    layout : TensorLayout
    metadata : Dict[str, Any]