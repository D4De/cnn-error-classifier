import math
import struct
from enum import Enum
from typing import Dict, Tuple

import numpy as np

from utils import quantize_percentage


class ValueClass(Enum):
    SAME = 0
    ALMOST_SAME = 1
    IN_RANGE = 2
    ZERO = 3
    FLIP = 4
    OUT_OF_RANGE = 5
    NAN = 6

    def display_name(self) -> str:
        return self.name.lower()


class DomainClass(Enum):
    ONLY_RANDOM = 0
    ONLY_OFF_BY_ONE = 1
    ONLY_NAN = 2
    ONLY_ZERO = 3
    RANDOM_OFF_BY_ONE = 4
    RANDOM_SINGLE_NAN = 5
    RANDOM_MULTIPLE_NAN = 6
    UNCATEGORIZED = 7


    def display_name(self) -> str:
        return self.name.lower()

def binary(num):
    return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))


def value_classification(
    golden_value: float,
    faulty_value: float,
    golden_range_min : float,
    golden_range_max : float,
    eps: float = 10e-3,
    almost_same: bool = False,
) -> ValueClass:
    if math.isnan(faulty_value) or math.isinf(faulty_value):
        return ValueClass.NAN
    elif golden_value == faulty_value:
        return ValueClass.SAME
    elif abs(golden_value - faulty_value) < eps:
        # TODO
        return ValueClass.ALMOST_SAME if almost_same else ValueClass.SAME
    elif faulty_value == 0:
        return ValueClass.ZERO
    bit_diff = [
        i for i in range(32) if binary(golden_value)[i] != binary(faulty_value)[i]
    ]
    if len(bit_diff) == 1:
        return ValueClass.FLIP
    elif golden_range_min <= faulty_value <= golden_range_max:
        return ValueClass.IN_RANGE
    else:
        return ValueClass.OUT_OF_RANGE


def domain_classification(value_classes_counts : Dict[ValueClass, int]) -> Dict[str, Tuple[float, float]]:
    present_value_classes = [dom_class for dom_class, count in value_classes_counts.items() if count > 0 and dom_class != ValueClass.SAME and dom_class != ValueClass.ALMOST_SAME]
    val_classes_freq_sum = sum(value_classes_counts[val_class] for val_class in present_value_classes)
    if len(present_value_classes) == 1:
        only_value_class = present_value_classes[0]
        return {only_value_class.display_name() : (100.0, 100.0)}
    elif len(present_value_classes) == 2:
        class_1, class_2 = present_value_classes
        quant_levels = 8
        class_1_range = quantize_percentage(value_classes_counts[class_1] / val_classes_freq_sum, quantization_levels=quant_levels)
        class_2_range_bot = 100 - class_1_range[1]
        class_2_range_top = max(100, class_2_range_bot + 100 / quant_levels)
        return {
            class_1.display_name(): class_1_range,
            class_2.display_name(): (class_2_range_bot, class_2_range_top)
        }
    else:
        return {
            "random": (100.0, 100.0)
        }
