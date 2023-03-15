import math
import struct
from enum import Enum

import numpy as np

class DomainClass(Enum):
    SAME = 0
    ALMOST_SAME = 1
    OFF_BY_ONE = 2
    ZERO = 3
    FLIP = 4
    RANDOM = 5
    NAN = 6

    def display_name(self):
        return self.name.lower()


def binary(num):
    return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))


def domain_classification(golden_value: float, faulty_value: float) -> DomainClass:
    if math.isnan(faulty_value) or math.isinf(faulty_value):
        return DomainClass.NAN
    if golden_value == faulty_value:
        return DomainClass.SAME
    if abs(golden_value - faulty_value) < 10e-3:
        return DomainClass.ALMOST_SAME
    if faulty_value == 0:
        return DomainClass.ZERO
    bit_diff = [
        i for i in range(32) if binary(golden_value)[i] != binary(faulty_value)[i]
    ]
    if len(bit_diff) == 1:
        return DomainClass.FLIP
    if abs(golden_value - faulty_value) <= 1:
        return DomainClass.OFF_BY_ONE
    else:
        return DomainClass.RANDOM

def domain_classification_int(golden_value: float, faulty_value: float) -> int:
    return domain_classification(golden_value, faulty_value).value

domain_classification_vect = np.vectorize(domain_classification_int)