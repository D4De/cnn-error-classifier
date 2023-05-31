from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, Tuple
from spatial_classifier.spatial_class import SpatialClass

SPATIAL_CLASSES = list(SpatialClass)

@dataclass
class SpatialClassParameters:
    spatial_class : SpatialClass
    keys : Dict[str, Any]
    aggregate_values : Dict[str, Tuple[Any, Callable]]


    def to_json(self) -> str:
        json_dict = {
            "spatial_class": self.spatial_class.display_name(),
            "keys": self.keys,
            "aggregate_values": {
                k : {
                    "value": v[0],
                    "aggregator": v[1].name
                } for k, v in self.aggregate_values.items()
            }
        }
        return json.dumps(json_dict)

