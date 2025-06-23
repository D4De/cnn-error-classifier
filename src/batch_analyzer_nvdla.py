import os
import logging as log
import numpy as np
import json
import csv

from multiprocessing import Queue
from typing import Any, Dict, List, Tuple, Union
from args import Args
from typing import Callable, Optional
from collections import defaultdict, OrderedDict
from statistics import mean, stdev

from analyzed_tensor import AnalyzedTensor, AnalyzedTensorFC
from coordinates import map_to_coordinates, numpy_coords_to_python_coord, coordinates_to_tuple
from aggregators import cardinalities_counts_by_sp_class, spatial_classes_counts, tensor_count_by_shape
from domain_classifier import ValueClass, domain_classification, value_classification
from spatial_classifier.spatial_classifier import spatial_classification
from spatial_classifier.spatial_class import SpatialClass
from visualizer import visualize
from classes import generate_classes_models


def analyze_batch(
    batch_path: str, args: Args, queue: Union[Queue, None]
) -> Tuple[List[AnalyzedTensor], Dict[str, Any]] | Tuple[List[AnalyzedTensorFC], Dict[str, Any]] | None:
    """
    Analyze a single batch of tensors.
    This function processes a batch independently from the others, returning a list of results (AnalyzedTensors)
    Multiple instances of this function can be run in parallel using multiprocessing, for speeding up the analysis.
    """
    golden_path = os.path.join(batch_path, args.golden_path)
    
    topdir_name = os.path.basename(os.path.dirname(batch_path))
    batch_name = topdir_name + '_' + os.path.basename(batch_path)

    # check if the errors archive exists
    errors_path = os.path.join(batch_path, args.faulty_path)

    if not os.path.exists(errors_path):
        log.warning(f"Skipping {batch_name} batch. Errors archive not found.")
        return None
            

    # if there is a queue specified prepare the lambda for signalling to the progress bar process that a tensor was processed
    if queue is not None:
        on_tensor_completed = lambda: queue.put(("processed", 1), block=False)
    else:
        on_tensor_completed = None


    # check if the golden file exists
    if not os.path.exists(golden_path):
        log.error(
            f"Skipping {batch_name} since it does not contain a golden file."
        )
        return None

    # load golden file for the batch
    try:
        golden: np.ndarray = np.load(golden_path)
    except:
        log.error(f"Skipping {batch_name} batch. Could not read golden file.")
        return None

    if golden is None:
        log.error(f'Skipping {batch_name} batch. Golden file is malformed or empty.')
        return None
    

    # prepare metadata for the analysis
    batch_metadata = {
        "shape": golden.shape,
        "batch_name": batch_name
    }


    # determine type of layer according to golden shape
    if len(golden.shape) == 4:
        log.info(f'{batch_name} batch: golden is 4D, assuming layer is convolutional.')

        batch_analyzed_tensors = analyze_errors_archive(
            errors_path,
            golden,
            args,
            on_tensor_completed=on_tensor_completed,
            metadata=batch_metadata
        )

        # classification is done for this hardware unit; if requested, generate its model and report
        if args.classes_unit_models:
            unit_dir = os.path.join(args.classes_output_dir, batch_name)
            if not os.path.isdir(unit_dir):
                os.makedirs(unit_dir, exist_ok=True)

            generate_classes_models(batch_analyzed_tensors, args, unit_dir)
            generate_batch_report(unit_dir, batch_analyzed_tensors)
            report_uncategorized_tensors(unit_dir, batch_analyzed_tensors)
    

    elif len(golden.shape) == 2:
        log.info(f'{batch_name} batch: golden is 2D, assuming layer is fully connected.')

        batch_analyzed_tensors = analyze_errors_archive_fc(
            errors_path,
            golden,
            args,
            on_tensor_completed=on_tensor_completed,
            metadata=batch_metadata
        )

        # classification is done for this hardware unit; if requested, generate its report
        if args.classes_unit_models:
            unit_dir = os.path.join(args.classes_output_dir, batch_name)
            if not os.path.isdir(unit_dir):
                os.makedirs(unit_dir, exist_ok=True)

            generate_batch_report_fc(unit_dir, batch_analyzed_tensors, args.last_fc)


    else:
        log.error(f"Skipping {batch_name} batch. Dimension of golden not supported: {golden.shape}.")
        return None 

    return batch_analyzed_tensors, batch_metadata

# ARCHIVE ANALYSIS ---------------------------------------------------------------------------------------------------

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
        for injection_number, injection in enumerate(error):
            for error_tensor in injection:
                sp_class, result = analyze_error_tensor(
                    errors_path=errors_path,
                    error_number=error_number,
                    injection_number=injection_number,
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

def analyze_errors_archive_fc(
    errors_path: str,
    golden: np.ndarray,
    args: Args,
    on_tensor_completed: Union[Callable[[], None], None] = None,
    metadata: dict = {}
):
    results: List[AnalyzedTensorFC] = []
    classified_tensors = 0

    golden_range_min = float(np.min(golden))
    golden_range_max = float(np.max(golden))

    # load the npz archive
    errors_archive = np.load(errors_path)

    # iterate over the files in the archive
    for error_number, error in errors_archive.items():
        golden_tensor = golden[int(error_number)]
        # each file is a 3D tensor: iterate twice to get a 1D tensor for comparison
        for injection_number, injection in enumerate(error):
            for error_tensor in injection:
                result = analyze_error_tensor_fc(
                    errors_path=errors_path,
                    error_number=error_number,
                    injection_number=injection_number,
                    tensor=error_tensor,
                    golden=golden_tensor,
                    golden_range_min=golden_range_min,
                    golden_range_max=golden_range_max,
                    args=args,
                    metadata=metadata
                )

                if on_tensor_completed is not None:
                    on_tensor_completed()
                
                if result is not None:
                    classified_tensors += 1
                    results.append(result)

    if classified_tensors == 0:
        log.warning(f"No tensors were classified in {errors_path}")
    
    return results

# SINGLE TENSOR ANALYSIS ---------------------------------------------------------------------------------------------

def analyze_error_tensor(
    errors_path: str,
    error_number: str,
    injection_number: int,
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
            f"Skipping {errors_path} number {error_number} injection {injection_number}. Invalid shape (Faulty has shape: {error_shape}, Golden has shape: {golden_shape})"
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
        log.info(f"{errors_path} number {error_number} injection {injection_number} has no diffs with golden")
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
                    args.visualize_path, f'{metadata["batch_name"]}_{error_number}_{injection_number}'
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
        injection_number=injection_number,
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

def analyze_error_tensor_fc(
    errors_path: str,
    error_number: str,
    injection_number: int,
    tensor: np.ndarray,
    golden: np.ndarray,
    golden_range_min: float,
    golden_range_max: float,
    args: Args,
    metadata: dict = {},
):
    # Check shape correctness
    if tensor.shape != golden.shape:
        log.warning(
            f"Skipping {errors_path} number {error_number} injection {injection_number}. Invalid shape (Faulty has shape: {tensor.shape}, Golden has shape: {golden.shape})"
        )
        return None
 
    num_equal_elements = np.sum(tensor == golden)
    num_different_elements = int(tensor.size - num_equal_elements)

    L1_dist = float(np.linalg.norm(tensor-golden, 1))
    L2_dist = float(np.linalg.norm(tensor-golden))

    result_tensor = AnalyzedTensorFC(
        batch=metadata["batch_name"],
        sub_batch=error_number,
        injection_number=injection_number,
        file_name=os.path.basename(errors_path),
        file_path=errors_path,
        shape=tensor.shape,
        corrupted_values_count=num_different_elements,
        golden_range_min=golden_range_min,
        golden_range_max=golden_range_max,
        layout=args.layout,
        metadata=metadata,
        L1_distance=L1_dist,
        L2_distance=L2_dist
    )

    if args.last_fc:
        # this is the last fc layer: check the ranking
        output_class = np.argmax(tensor)
        golden_class = np.argmax(golden)
        result_tensor.misclassified = (output_class != golden_class)

    return result_tensor

# REPORT GENERATION ---------------------------------------------------------------------------------------------------

def generate_batch_report(batch_dir: str, analyzed_tensors: list[AnalyzedTensor]):
    report = OrderedDict()

    report["classified_tensors"] = len(analyzed_tensors)
    report["tensors_by_shape"] = tensor_count_by_shape(analyzed_tensors)
    report["spatial_classes"] = spatial_classes_counts(analyzed_tensors)
    report["class_cardinalites"] = cardinalities_counts_by_sp_class(analyzed_tensors)

    report_path = os.path.join(batch_dir, 'unit_report.json')
    with open(report_path, 'w') as rf:
        json.dump(report, rf, indent=2)


def report_uncategorized_tensors(batch_dir: str, analyzed_tensors: list[AnalyzedTensor]):
    report_path = os.path.join(batch_dir, 'uncategorized_log.csv')

    with open(report_path, 'w', newline='') as csvlog:
        logwriter = csv.writer(csvlog)
        logwriter.writerow(['Type', 'Error Number', 'Injection Number'])

        for tensor in analyzed_tensors:
            if tensor.spatial_class == SpatialClass.SINGLE_CHANNEL_RANDOM:
                logwriter.writerow(['Single', tensor.sub_batch, str(tensor.injection_number)])
            elif tensor.spatial_class == SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED:
                logwriter.writerow(['Multiple', tensor.sub_batch, str(tensor.injection_number)])


def generate_batch_report_fc(batch_dir: str, analyzed_tensors: List[AnalyzedTensorFC], last_fc: bool = False):
    report = OrderedDict()

    num_tensors = len(analyzed_tensors)
    report["classified_tensors"] = num_tensors
    report["tensor_shape"] = analyzed_tensors[0].shape

    # determine min, max, avg and std. dev. of number of corrupted values, L1 distance and L2 distance
    corrupted_values_counts = []
    L1_distances = []
    L2_distances = []
    # count misclassification (unused if this is not the final layer)
    num_misclassifications = 0

    for tensor in analyzed_tensors:
        corrupted_values_counts.append(tensor.corrupted_values_count)
        L1_distances.append(tensor.L1_distance)
        L2_distances.append(tensor.L2_distance)
        num_misclassifications += tensor.misclassified

    report["num_corrupted_values"] = {
        "min": min(corrupted_values_counts),
        "max": max(corrupted_values_counts),
        "mean": mean(corrupted_values_counts),
        "stdev": stdev(corrupted_values_counts)
    }
    report["L1_distance"] = {
        "min": min(L1_distances),
        "max": max(L1_distances),
        "mean": mean(L1_distances),
        "stdev": stdev(L1_distances)
    }
    report["L2_distance"] = {
        "min": min(L2_distances),
        "max": max(L2_distances),
        "mean": mean(L2_distances),
        "stdev": stdev(L2_distances)
    }

    if last_fc:
        report["misclassification_rate"] = num_misclassifications / num_tensors

    report_path = os.path.join(batch_dir, 'unit_report.json')
    with open(report_path, 'w') as rf:
        json.dump(report, rf, indent=2)
