import os
import logging as log
import numpy as np

from multiprocessing import Queue
from typing import Any, Dict, List, Tuple, Union
from args import Args
from typing import Callable, Optional
from collections import defaultdict

from analyzed_tensor import AnalyzedTensor
from coordinates import map_to_coordinates, numpy_coords_to_python_coord, coordinates_to_tuple
from domain_classifier import ValueClass, domain_classification, value_classification
from spatial_classifier.spatial_classifier import spatial_classification
from visualizer import visualize
from classes import generate_classes_models


def analyze_batch(
    batch_path: str, args: Args, queue: Union[Queue, None]
) -> Union[Tuple[List[AnalyzedTensor], Dict[str, Any]], None]:
    """
    Analyze a single batch of tensors.
    This function processes a batch independently from the others, returning a list of results (AnalyzedTensors)
    Multiple instances of this function can be run in parallel using multiprocessing, for speeding up the analysis.
    """
    golden_path = os.path.join(batch_path, args.golden_path)
    
    topdir_name = os.path.basename(os.path.dirname(batch_path))
    batch_name = topdir_name + '_' + os.path.basename(batch_path)

    # No golden found
    if not os.path.exists(golden_path):
        log.error(
            f"Skipping {batch_name} since it does not contain a golden file."
        )
        return None

    # Load golden file for the batch
    try:
        golden: np.ndarray = np.load(golden_path)
    except:
        log.error(f"Skipping {batch_name} batch. Could not read golden file.")
        return None

    if golden is not None and len(golden.shape) != 4:
        log.error(
            f"Skipping {batch_name} batch. Dimension of golden not supported: {golden.shape}."
        )
        return None

    log.info(f"Golden tensor ({args.golden_path}) loaded. Shape {golden.shape}.")

    # Absolute path to faulty tensors archive
    errors_path = os.path.join(batch_path, args.faulty_path)

    if not os.path.exists(errors_path):
        log.warning(f"Skipping {batch_name} batch. Could not open errors archive.")
        return None
            

    # If there is a queue specified prepare the lambda for signalling to the progress bar process that a tensor was processed
    if queue is not None:
        on_tensor_completed = lambda: queue.put(("processed", 1), block=False)
    else:
        on_tensor_completed = None

    # Prepare metadata and start analysis
    batch_metadata = {
        "shape": golden.shape,
        "batch_name": batch_name
    }

    batch_analyzed_tensors = analyze_errors_archive(
        errors_path,
        golden,
        args,
        on_tensor_completed=on_tensor_completed,
        metadata=batch_metadata
    )

    # classification is done for this hardware unit; if requested, generate its model
    if args.classes_unit_models:
        generate_classes_models(batch_analyzed_tensors, args, batch_path)

    return batch_analyzed_tensors, batch_metadata


def analyze_errors_archive(
        errors_path: str,
        golden: np.ndarray,
        args: Args,
        on_tensor_completed: Union[Callable[[], None], None] = None,
        metadata: dict = {}
):
    results: List[AnalyzedTensor] = []
    classified_tensors = 0

    golden_range_min = float(np.min(golden))
    golden_range_max = float(np.max(golden))

    # load the npz archive
    errors_archive = np.load(errors_path)

    # iterate over the files in the archive
    for error_number, error in errors_archive.items():
        golden_tensor = golden[int(error_number)]
        # each file is a 5D tensor: iterate twice to get a single tensor
        for injection in error:
            for error_tensor in injection:
                sp_class, result = analyze_error_tensor(
                    errors_path=errors_path,
                    error_number=error_number,
                    tensor=error_tensor[np.newaxis, :], #reshape to 4D
                    golden=golden_tensor[np.newaxis, :], #reshape to 4D
                    args=args,
                    golden_range_min= golden_range_min,
                    golden_range_max= golden_range_max,
                    metadata=metadata,
                )

                if on_tensor_completed is not None:
                    on_tensor_completed()
                
                if result is not None and sp_class != "masked" and sp_class != "skipped":
                    classified_tensors += 1
                    results.append(result)

    if classified_tensors == 0:
        log.warning(f"No tensors were classified in {errors_path}")
    
    return results


def analyze_error_tensor(
    errors_path: str,
    error_number: str,
    tensor: np.ndarray,
    golden: np.ndarray,
    golden_range_min: float,
    golden_range_max: float,
    args: Args,
    metadata: dict = {},
) -> Tuple[str, Optional[AnalyzedTensor]]:

    error_shape = map_to_coordinates(tensor.shape, args.layout)
    golden_shape = map_to_coordinates(golden.shape, args.layout)

    # Check shape correctness
    if error_shape != golden_shape:
        log.warning(
            f"Skipping {errors_path} number {error_number}. Invalid shape (Faulty has shape: {error_shape}, Golden has shape: {golden_shape})"
        )
        return "skipped", None

    value_class_count = defaultdict(int)

    # Generate a list of all coordinates where a difference is observed (Sparse matrix)
    if args.almost_same:
        sparse_diff_native_coords = list(zip(*np.nonzero(tensor - golden)))
    else:
        sparse_diff_native_coords = list(zip(*np.where(np.abs(tensor - golden) >= args.epsilon)))

    tensor_diff = np.zeros(coordinates_to_tuple(error_shape), dtype=np.int8)    

    for coord in sparse_diff_native_coords:
        val_class = value_classification(golden[coord[0], coord[1], coord[2], coord[3]], tensor[coord[0], coord[1], coord[2], coord[3]], golden_range_min, golden_range_max,  args.epsilon, args.almost_same)
        tensor_diff[coord[0], coord[1], coord[2], coord[3]] = val_class.value
        value_class_count[val_class] += 1
    
    value_class_count[ValueClass.SAME] = golden.size - sum(value_class_count.values())

    # No diff = masked
    if len(sparse_diff_native_coords) == 0:
        log.info(f"{errors_path} number {error_number} has no diffs with golden")
        return "masked", None
    sparse_diff = [
        map_to_coordinates(numpy_coords_to_python_coord(coords), args.layout)
        for coords in sparse_diff_native_coords
    ]

    # Pefmorm spatial classifcation
    spatial_class, pattern_params, faulty_channels = spatial_classification(sparse_diff, golden_shape)
    domain_class = domain_classification(value_class_count)

    if args.visualize:
        folder_path = spatial_class.class_folder(args.visualize_path)
        file_count = len([x for x in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, x))])
        if args.visualize_limit == 0 or file_count < args.visualize_limit:
            visualize(
                tensor_diff,
                faulty_channels,
                args.layout,
                spatial_class.output_path(
                    args.visualize_path, f'{metadata["batch_name"]}_{error_number}'
                ),
                save=True,
                show=False,
                suptitile=f'{metadata.get("batch_name") or ""} {metadata.get("sub_batch_name") or "" or error_number} {golden_shape.C}x{golden_shape.H}x{golden_shape.W}',
                invalidate=True,
            )
        

    # Per tensor report generator
    return spatial_class.display_name(), AnalyzedTensor(
        batch=metadata["batch_name"],
        sub_batch=error_number,
        file_name=os.path.basename(errors_path),
        file_path=errors_path,
        shape=error_shape,
        spatial_class=spatial_class,
        spatial_class_params=pattern_params,
        value_classes_counts= value_class_count,
        corrupted_channels_count=len(faulty_channels),
        corrupted_values_count=len(sparse_diff),
        domain_class=domain_class,
        golden_range_min=golden_range_min,
        golden_range_max=golden_range_max,
        layout=args.layout,
        metadata=metadata
    )
