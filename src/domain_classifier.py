import math
import struct
import numpy as np

from enum import Enum
from typing import Dict, Tuple
from collections import defaultdict

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


def count_differing_bits(a: float, b: float):
    """Counts the number of corresponding bits that differ between the binary representations of two floats."""
    return int.bit_count(                   # count number of 1 bits in
        np.bitwise_xor(                     # (a XOR b),
            np.float32(a).view(np.int32),   # viewing a as a 32-bit integer
            np.float32(b).view(np.int32)    # and doing the same for b
        ).item()
    )


def value_classification(
    faulty_values: np.ndarray,
    golden_values: np.ndarray,
    golden_range_min : float,
    golden_range_max : float,
) -> tuple[defaultdict[ValueClass, int], np.ndarray]:
    """
    Takes all the faulty and golden values corresponding to spots where the tensors differ and classifies the values.
    Returns a dictionary with the number of elements in each value class and a numpy array, shaped like the inputs, indicating
    what value class each position belongs to.
    """

    value_class_counts = defaultdict(int)
    faulty_value_classes = np.zeros_like(faulty_values, dtype=np.int8)

    # check what values are NaN of Inf
    nan_inf_vec = np.vectorize(lambda x: math.isnan(x) or math.isinf(x))
    nan_mask = nan_inf_vec(faulty_values)
    faulty_value_classes[nan_mask] = ValueClass.NAN.value
    value_class_counts[ValueClass.NAN] = np.count_nonzero(nan_mask)

    # check what values are 0
    zero_mask = (faulty_values == 0)
    faulty_value_classes[zero_mask] = ValueClass.ZERO.value
    value_class_counts[ValueClass.ZERO] = np.count_nonzero(zero_mask)
    
    # OR the two masks and invert to get the remaining elements. Return immediately if there are no more elements
    rest_mask = ~(nan_mask | zero_mask)
    if not np.any(rest_mask):
        return value_class_counts, faulty_value_classes

    rest_faulty = faulty_values[rest_mask]
    rest_golden = golden_values[rest_mask]

    rest_temp = np.zeros_like(rest_faulty, dtype=faulty_value_classes.dtype)

    # check bitflips among the remaining elements
    bitflip_vec = np.vectorize(lambda a,b: count_differing_bits(a,b) == 1)
    bitflip_mask = bitflip_vec(rest_faulty, rest_golden)
    rest_temp[bitflip_mask] = ValueClass.FLIP.value
    value_class_counts[ValueClass.FLIP] = np.count_nonzero(bitflip_mask)

    # take the last remaining elements
    rest2_faulty = rest_faulty[~bitflip_mask]
    # the only remaining possibilities are in_range and out_of_range. Initialize with out_of_range
    rest2_temp = np.full_like(rest2_faulty, ValueClass.OUT_OF_RANGE.value)

    # check what values are in range
    in_range_mask = np.logical_or(golden_range_min <= rest2_faulty, rest2_faulty <= golden_range_max)
    rest2_temp[in_range_mask] = ValueClass.IN_RANGE.value
    value_class_counts[ValueClass.IN_RANGE] = np.count_nonzero(in_range_mask)
    value_class_counts[ValueClass.OUT_OF_RANGE] = np.count_nonzero(in_range_mask == 0)

    # insert rest2_temp into rest_temp
    rest_temp[~bitflip_mask] = rest2_temp
    # insert rest_temp into faulty_value_classes
    faulty_value_classes[rest_mask] = rest_temp

    return value_class_counts, faulty_value_classes


def domain_classification(
    value_class_counts : Dict[ValueClass, int]
) -> Dict[str, Tuple[float, float]]:
    present_value_classes = [dom_class for dom_class, count in value_class_counts.items() if count > 0 and dom_class != ValueClass.SAME and dom_class != ValueClass.ALMOST_SAME]
    val_classes_freq_sum = sum(value_class_counts[val_class] for val_class in present_value_classes)
    if len(present_value_classes) == 1:
        only_value_class = present_value_classes[0]
        return {only_value_class.display_name() : (100.0, 100.0)}
    elif len(present_value_classes) == 2:
        class_1, class_2 = present_value_classes
        quant_levels = 8
        class_1_range = quantize_percentage(value_class_counts[class_1] / val_classes_freq_sum, quantization_levels=quant_levels)
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
