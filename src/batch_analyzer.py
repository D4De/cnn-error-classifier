from collections import OrderedDict, defaultdict, namedtuple
import json
from multiprocessing import Queue
import os
from typing import Any, Union

import tqdm
from args import Args
import logging as log
import numpy as np
from coordinates import map_to_coordinates
from spatial_classifier import accumulate_max

from tensor_analyzer import analyze_tensor_directory
from utils import double_int_defaultdict, int_defaultdict, list_defaultdict

BatchAnalyzeReturnType = namedtuple('BatchAnalyzeReturnType', ['batch_name', 'batch_report', 'batch_dom_classes', 'batch_sp_classes', 'batch_error_patterns', 'batch_cardinalities', 'batch_class_params'])


def analyze_batch(batch_path : str, args : Args, queue: Union[Queue, None]) -> Union[BatchAnalyzeReturnType, None]:
    """
    Analyze a single batch of tensors

    This function is compatible multiprocessing, for speeding up the analysis
    ---
    Parameters

    batch_path: The absolute path of the batch path
    args: The args passed from the command line
    queue: Optional, if a "processed" message is put into that queue at every completed tensor analyis 
    """
    golden_path = os.path.join(batch_path, args.golden_path)
    batch_name = os.path.basename(batch_path)

    # No gold found
    if not os.path.exists(golden_path):
        print(
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
        log.warning(
            f"Skipping {batch_name} batch. Could not open path to faulty path"
        )
        return None
    
    # Retrieve metadata from info.json file (if they exist)
    metadata_path = os.path.join(faulty_path, "info.json")


    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.loads(f.read())
    else:
        metadata = {}

    sub_batch_dir = [
        os.path.join(faulty_path, dir)
        for dir in os.listdir(faulty_path)
        if os.path.isdir(os.path.join(faulty_path, dir))
    ]
    
    batch_report = OrderedDict()
    # Stores the absolute frequency of the domain patterns for this batch
    batch_dom_classes = defaultdict(int)
    # Stores the absolute frequency of the spatial patterns for this batch
    batch_sp_classes = defaultdict(int)
    # Stores the absolute frequencies of the different spatial configuration of each pattern, grouped by error cardinalities
    # global_error_patterns[cardinality][spatial class name][pattern (stringfied tuple)] -> frequency of the pattern
    batch_error_patterns = defaultdict(
        double_int_defaultdict
    )
    # Store additional parameters for the spatial classes, grouped by cardinality
    batch_class_params = defaultdict(
        list_defaultdict
    )
    # global_error_patterns[cardinality][spatial class name] -> frequency of the spatial class with that cardinailty
    batch_cardinalities = defaultdict(int_defaultdict)

    # If there is a queue specified prepare the lambda for signalling to the progress bar process that a tensor was processed
    if queue is not None:
        on_tensor_completed = lambda: queue.put("processed", block=False)
    else:
        on_tensor_completed = None

    # Read batch subdirectory
    for faulty_path in sorted(sub_batch_dir):
        sub_batch_name = os.path.basename(faulty_path)
        # (I)struction (G)roup (ID) and (B)it (F)lip (M)odel are inferend from the folder name (separated by _)
        igid = sub_batch_name.split("_")[0]
        bfm = sub_batch_name.split("_")[1]
        # Append sub batch metadata to the batch metadata
        sub_batch_metadata = metadata | {"igid": igid, "bfm": bfm, "batch_name": batch_name}
        # Run analysis of a sub batch (i.e. fp32_wzv, ld_wrv, ... subdirectories)
        report = analyze_tensor_directory(
            faulty_path=faulty_path,
            golden=golden,
            report_output_path=os.path.join(
                args.reports_path, f"report_{batch_name}_{sub_batch_name}.json"
            ),
            layout=args.layout,
            epsilon=args.epsilon,
            almost_same=args.almost_same,
            image_output_dir=args.visualize_path,
            visualize_errors=args.visualize,
            save_report=args.partial_reports,
            on_tensor_completed=on_tensor_completed,
            metadata=sub_batch_metadata,
        )

        if len(report) == 0:
        # Empty report = Tensor not processed
            continue
        # accumulate values to add in batch global dictionaries
        # Accumulate domain classes absolute frequencies
        for dom_class, count in report["global_data"]["domain_classes"].items():
            batch_dom_classes[dom_class] += count
        # Accumulate spatial classes absolute frequencies
        for sp_class, count in report["global_data"]["spatial_classes"].items():
            batch_sp_classes[sp_class] += count

        for cardinality, sp_classes in report["global_data"][
            "error_patterns"
        ].items():
            total = 0
            for sp_class, patterns in sp_classes.items():
                # Sum total cardinalities frequencies
                total += len(patterns)
                batch_cardinalities[cardinality][sp_class] += len(patterns)
                for pattern, freq in patterns.items():
                    # Accumulate frequencies of the single spatial configurations of each spatial pattern 
                    batch_error_patterns[cardinality][sp_class][
                        pattern
                    ] += freq
            batch_cardinalities[cardinality]["sum"] = total
        for cardinality, sp_classes in report["global_data"]["class_params"].items():
            for sp_class, max_val in sp_classes.items():
                batch_class_params[cardinality][sp_class] = accumulate_max(batch_class_params[cardinality][sp_class], max_val)
        batch_report[sub_batch_name] = report["global_data"]
        # remove  tooverbose data from global report
        del batch_report[sub_batch_name]["raveled_patterns"]
        del batch_report[sub_batch_name]["error_patterns"]



    return BatchAnalyzeReturnType(batch_name, batch_report, batch_dom_classes, batch_sp_classes, batch_error_patterns, batch_cardinalities, batch_class_params)

