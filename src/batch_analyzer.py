
import json
from multiprocessing import Queue
import os
from typing import Any, Dict, List, Tuple, Union
from analyzed_tensor import AnalyzedTensor

from args import Args
import logging as log
import numpy as np
from coordinates import map_to_coordinates
from sub_batch_analyzer import analyze_tensor_directory
import re

def get_igprofile_kernels(text : str) -> List[str]:
    kernel_names = []

    pattern = r'kernel_name: ([^<\(]+)'
    matches = re.findall(pattern, text)
    
    for match in matches:
        kernel_names.append(match.strip())
    
    return kernel_names

def analyze_batch(
    batch_path: str, args: Args, queue: Union[Queue, None]
) -> Union[Tuple[List[AnalyzedTensor], Dict[str, Any]], None]:
    """
    Analyze a single batch of tensors

    This function processes a batch independently from the others, returning a list of results (AnalyzedTensors)
    Multiple instances of this function can be run in parallel using multiprocessing, for speeding up the analysis
    ---
    Parameters

    batch_path: The absolute path of the batch path
    args: The args passed from the command line
    queue: Optional, if specified a "processed" message is put into that queue every time a tensor is analyzed
    """
    golden_path = os.path.join(batch_path, args.golden_path)
    batch_name = os.path.basename(batch_path)

    # No gold found
    if not os.path.exists(golden_path):
        log.error(
            f"Skipping {batch_name} batch since it does not contain golden tensor"
        )
        return None

    # Load golden file for the batch
    try:
        golden: np.ndarray = np.load(golden_path)
    except:
        log.error(f"Skipping {batch_name} batch. Could not read golden")
        return None

    if golden is not None and len(golden.shape) != 4:
        log.error(
            f"Skipping {batch_name} batch. Dimension of golden not supported {golden.shape}"
        )
        return None

    # Remap coordinates using Coordinate namedtuple (to be compatible with NHWC and NCHW)
    golden_shape = map_to_coordinates(golden.shape, args.layout)
    log.info(f"Golden tensor ({args.golden_path}) loaded. Shape {golden_shape}")

    # Absolute path to faulty tensors folders
    faulty_path = os.path.join(batch_path, args.faulty_path)

    if not os.path.exists(faulty_path):
        log.warning(f"Skipping {batch_name} batch. Could not open path to faulty path")
        return None

    # Retrieve metadata from info.json file (if they exist)
    metadata_path = os.path.join(faulty_path, "info.json")
    stats_path = os.path.join(faulty_path, "injection-counts.json")
    nvbitfi_igprofile_path = os.path.join(faulty_path, "nvbitfi-igprofile.txt")

    metadata = {"batch_name": batch_name}

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata |= json.load(f)
    
    if os.path.exists(stats_path):
         with open(stats_path, "r") as f:
            metadata["experiment_counts"] = json.load(f)
    
    if os.path.exists(nvbitfi_igprofile_path):
        with open(nvbitfi_igprofile_path, "r") as f:
            text = f.read()
            metadata["igprofile_kernels"] = get_igprofile_kernels(text)
            
    

    sub_batch_dir = [
        os.path.join(faulty_path, dir)
        for dir in os.listdir(faulty_path)
        if os.path.isdir(os.path.join(faulty_path, dir))
    ]

    # If there is a queue specified prepare the lambda for signalling to the progress bar process that a tensor was processed
    if queue is not None:
        on_tensor_completed = lambda: queue.put(("processed", 1), block=False)
    else:
        on_tensor_completed = None

    batch_analyzed_tensors = []

    # Read batch subdirectory
    for faulty_path in sorted(sub_batch_dir):
        sub_batch_name = os.path.basename(faulty_path)
        # (I)struction (G)roup (ID) and (B)it (F)lip (M)odel are inferend from the folder name (separated by _)
        sub_batch_tokens = sub_batch_name.split("_")
        if len(sub_batch_tokens) == 2:
            igid, bfm = sub_batch_tokens
        else:
            igid, bfm = None, None
        # Append sub batch metadata to the batch metadata
        sub_batch_metadata = metadata | {
            "igid": igid,
            "bfm": bfm,
            "shape": golden.shape,
            "batch_name": batch_name,
            "sub_batch_name": sub_batch_name
        }
        # Run analysis of a sub batch (i.e. fp32_wzv, ld_wrv, ... subdirectories)
        sub_batch_analyzed_tensors = analyze_tensor_directory(
            faulty_path=faulty_path,
            golden=golden,
            args=args,
            on_tensor_completed=on_tensor_completed,
            metadata=sub_batch_metadata,
        )

        batch_analyzed_tensors += sub_batch_analyzed_tensors

    return batch_analyzed_tensors, metadata

    

