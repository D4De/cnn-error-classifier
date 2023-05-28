


from dataclasses import dataclass
import json
from typing import Any, Dict, List

from coordinates import Coordinates, TensorLayout
from domain_classifier import DomainClass, ValueClass
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
    value_classes_counts : Dict[ValueClass, int]
    domain_class : DomainClass
    corrupted_values_count : int
    corrupted_channels_count : int
    layout : TensorLayout
    metadata : Dict[str, Any]


    def as_insert_param_list(self) -> List[Any]:

        domain_class_count_str = {
            dom.display_name() : count for dom, count in self.value_classes_counts.items()

        }

        return [
            self.batch,
            self.sub_batch,
            self.file_name,
            self.file_path,
            self.metadata.get("igid"),
            self.metadata.get("bfm"),
            self.shape.N,
            self.shape.H,
            self.shape.C,
            self.shape.W,
            self.spatial_class.display_name(),
            json.dumps(self.spatial_class_params),
            str(self.spatial_pattern),
            json.dumps(domain_class_count_str),
            self.domain_class.display_name(),
            self.corrupted_values_count,
            self.corrupted_channels_count,
            self.layout.name,
            json.dumps(self.metadata)
        ]