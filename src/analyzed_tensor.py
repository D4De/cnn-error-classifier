


from dataclasses import dataclass
import json
from typing import Any, Dict, List

from coordinates import Coordinates, TensorLayout
from domain_classifier import DomainClass, ValueClass
from spatial_classifier.spatial_class_parameters import SpatialClassParameters
from spatial_classifier.spatial_classifier import SpatialClass


@dataclass
class AnalyzedTensor:
    batch : str
    sub_batch : str
    file_name : str
    file_path : str
    shape : Coordinates
    spatial_class : SpatialClass    
    spatial_class_params : SpatialClassParameters
    value_classes_counts : Dict[ValueClass, int]
    domain_class : DomainClass
    corrupted_values_count : int
    corrupted_channels_count : int
    layout : TensorLayout
    metadata : Dict[str, Any]


    def as_insert_param_list(self) -> Dict[str, Any]:

        domain_class_count_str = {
            dom.display_name() : count for dom, count in self.value_classes_counts.items()
        }


        return {
            "batch_name": self.batch,
            "sub_batch_name": self.sub_batch,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "igid": self.metadata.get("igid"),
            "bfm": self.metadata.get("bfm"),
            "N": self.shape.N,
            "H": self.shape.H,
            "C": self.shape.C,
            "W": self.shape.W,
            "spatial_class": self.spatial_class.display_name(),
            "spatial_class_params": self.spatial_class_params.to_json(),
            "value_classes_counts": json.dumps(domain_class_count_str),
            "domain_class": self.domain_class.display_name(),
            "corrupted_values_count": self.corrupted_values_count,
            "corrupted_channels_count": self.corrupted_channels_count,
            "layout": self.layout.name,
            "metadata": json.dumps(self.metadata)
        }