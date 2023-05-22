import os
from typing import Callable, List, Union
from analyzed_tensor import AnalyzedTensor
from args import Args
import numpy as np
import logging as log
from tensor_analyzer import analyze_tensor


def analyze_tensor_directory(
    faulty_path: str,
    golden: np.ndarray,
    args: Args,
    metadata: dict = {},
    on_tensor_completed: Union[Callable[[], None], None] = None,
):
    # Get list of all file ending in .npy
    faulty_files_path = [
        os.path.join(faulty_path, entry)
        for entry in os.listdir(faulty_path)
        if entry.split(".")[1] == "npy"
    ]

    log.info(f"Found {len(faulty_files_path)} faulty tensors to analize")

    results : List[AnalyzedTensor]= []
    classified_tensors = 0

    # Iterate over all tensor files
    for file_path in sorted(faulty_files_path):
        sp_class, result = analyze_tensor(
            file_path=file_path,
            golden=golden,
            args=args,
            metadata=metadata,
        )
        if on_tensor_completed is not None:
            on_tensor_completed()

        if result is not None and sp_class != "masked" and sp_class != "skipped":
            classified_tensors += 1
            results.append(result)


    if classified_tensors == 0:
        log.warn(f"No tensors were classified in {faulty_path}")
    
    return results