import math
import struct
from enum import Enum
from typing import Dict

import numpy as np


class ValueClass(Enum):
    SAME = 0
    ALMOST_SAME = 1
    OFF_BY_ONE = 2
    ZERO = 3
    FLIP = 4
    RANDOM = 5
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
    elif abs(golden_value - faulty_value) <= 1:
        return ValueClass.OFF_BY_ONE
    else:
        return ValueClass.RANDOM


def value_classification_int(
    golden_value: float,
    faulty_value: float,
    eps: float = 10e-3,
    almost_same: bool = False,
) -> int:
    return value_classification(golden_value, faulty_value, eps, almost_same).value


value_classification_vect = np.vectorize(
    value_classification_int, excluded=["eps", "almost_same"]
)

def domain_classification(value_classes_counts : Dict[ValueClass, int]) -> DomainClass:
    present_value_classes = [dom_class for dom_class, count in value_classes_counts.items() if count > 0 and dom_class != ValueClass.SAME and dom_class != ValueClass.ALMOST_SAME]
    if len(present_value_classes) == 1:
        only_value_class = present_value_classes[0]
        if only_value_class == ValueClass.RANDOM:
            return DomainClass.ONLY_RANDOM
        elif only_value_class == ValueClass.OFF_BY_ONE:
            return DomainClass.ONLY_OFF_BY_ONE              
        elif only_value_class == ValueClass.NAN:
            return DomainClass.ONLY_NAN
        elif only_value_class == ValueClass.ZERO:
            return DomainClass.ONLY_ZERO
        else:
            return DomainClass.UNCATEGORIZED
    elif len(present_value_classes) == 2:
        non_zero_dom_classes_set = set(present_value_classes)
        if ValueClass.OFF_BY_ONE in non_zero_dom_classes_set and ValueClass.RANDOM in non_zero_dom_classes_set:
            return DomainClass.RANDOM_OFF_BY_ONE
        elif ValueClass.NAN in non_zero_dom_classes_set and ValueClass.RANDOM in non_zero_dom_classes_set:
            nan_count = value_classes_counts[ValueClass.NAN]
            if nan_count == 1:
                return DomainClass.RANDOM_SINGLE_NAN
            else:
                return DomainClass.RANDOM_MULTIPLE_NAN
    else:
        return DomainClass.UNCATEGORIZED
