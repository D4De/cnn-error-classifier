
from typing import Any

def identity(arg):
    return arg

class MaxAggregator:
    def __init__(self, key=identity):
        self.key = key
        self.name = "MAX"
    def __call__(self, *args: Any) -> Any:
        return max(args, key=self.key)
    
class MinAggregator:
    def __init__(self, key=identity):
        self.key = key
        self.name = "MIN"
    def __call__(self, *args: Any) -> Any:
        return min(args, key=self.key)
    
class AvgAggregator:
    def __init__(self, mapper=identity):
        self.mapper = mapper
        self.name = "AVG"
    def __call__(self, *args: Any) -> Any:
        return sum(self.mapper(arg) for arg in args) / len(args)