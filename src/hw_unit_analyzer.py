import os
import csv
import json
import numpy as np
import logging as log
import threading

from tqdm import tqdm
from typing import List
from statistics import mean, stdev
from collections import defaultdict, OrderedDict

from args import Args
from classes import generate_classes_models
from visualizer import visualize
from coordinates import map_to_coordinates, numpy_coords_to_python_coord
from aggregators import cardinalities_counts_by_sp_class, spatial_classes_counts, tensor_count_by_shape
from analyzed_tensor import AnalyzedTensorConv, AnalyzedTensorFC
from domain_classifier import ValueClass, domain_classification, value_classification
from spatial_classifier.spatial_class import SpatialClass
from spatial_classifier.spatial_classifier import spatial_classification

# SOME UTILITIES ---------------------------------------------------------------------------------------------------

def set_console_logging_level(level):
    root = log.getLogger()
    for handler in root.handlers:
        if isinstance(handler, type(log.StreamHandler)):
            handler.setLevel(level)

def precalculate_unit_workload(errors_archive) -> int:
    num_tensors = 0

    for key in errors_archive:
        errors_file = errors_archive[key]
        file_shape = errors_file.shape
        num_tensors += file_shape[0] * file_shape[1]
    
    return num_tensors

# ANALYSIS FUNCTIONS ---------------------------------------------------------------------------------------------------

def analyze_hw_unit_directory(unit_dir: str, group_name: str, unit_name: str, args: Args, golden: np.ndarray):
    log.info(f'Starting analysis of {group_name}-{unit_name}')

    errors_path = os.path.join(unit_dir, args.errors_archive_filename)
    if not os.path.exists(errors_path):
        log.error(f"Skipping {unit_dir} hardware unit directory. Errors archive not found.")
        return []
    
    # load errors archive
    try:
        errors = np.load(errors_path)
    except:
        log.error(f'Errors archive {errors_path} could not be loaded.')
        raise ValueError()

    # compute the total number of tensors in the archive
    total_tensors = precalculate_unit_workload(errors)

    golden_range_min = float(np.min(golden))
    golden_range_max = float(np.max(golden))

    classified_tensors = []

    # disable console logging
    set_console_logging_level(log.WARN)

    with tqdm(total=total_tensors) as progress_bar:
        # determine layer type based on the golden size
        if len(golden.shape) == 4:
            # assume convolutional
            classified_tensors = analyze_errors_conv(
                golden=golden,
                golden_range_min=golden_range_min,
                golden_range_max=golden_range_max,
                errors=errors,
                args=args,
                group_name=group_name,
                unit_name=unit_name,
                progress_bar=progress_bar,
                errors_path=errors_path
            )
            
            # generate classes models and reports
            if args.classes:
                unit_dir = os.path.join(args.classes_output_dir, f'{group_name}_{unit_name}')
                if not os.path.isdir(unit_dir):
                    os.makedirs(unit_dir, exist_ok=True)

                generate_classes_models(classified_tensors, args, unit_dir)
                generate_unit_report_conv(unit_dir, classified_tensors)
                report_uncategorized_tensors(unit_dir, classified_tensors)

        elif len(golden.shape) == 2:
            # assume fully connected
            classified_tensors = analyze_errors_fc(
                golden=golden,
                errors=errors,
                args=args,
                group_name=group_name,
                unit_name=unit_name,
                progress_bar=progress_bar,
                errors_path=errors_path
            )

            unit_dir = os.path.join(args.classes_output_dir, unit_name)
            if not os.path.isdir(unit_dir):
                os.makedirs(unit_dir, exist_ok=True)

            generate_unit_report_fc(unit_dir, classified_tensors)

        else:
            log.error(f'Skipping {unit_dir} hardware unit directory. Unsupported golden shape {golden.shape}.')
    
    # enable console logging
    set_console_logging_level(log.INFO)
    
    return classified_tensors

# ARCHIVE ANALYSIS ---------------------------------------------------------------------------------------------------

def analyze_errors_conv(
    golden: np.ndarray,
    golden_range_min: float,
    golden_range_max: float,
    errors,
    args: Args,
    group_name: str,
    unit_name: str,
    progress_bar: tqdm,
    errors_path: str
):
    results: list[AnalyzedTensorConv] = []

    progress_bar_lock = threading.Lock()
    results_lock = threading.Lock()

    # create the threads
    threads: list[threading.Thread] = []
    for thread_idx in range(args.parallel):
        t = threading.Thread(
            target=analyze_error_tensors_conv,
            args=(
                thread_idx,
                args.parallel,
                progress_bar,
                progress_bar_lock,
                errors,
                golden,
                golden_range_min,
                golden_range_max,
                group_name,
                unit_name,
                errors_path,
                args,
                results,
                results_lock
            )
        )
        threads.append(t)
        t.start()
    
    # join the threads
    for t in threads:
        t.join()

    return results

def analyze_errors_fc(
    golden: np.ndarray,
    errors,
    args: Args,
    group_name: str,
    unit_name: str,
    progress_bar: tqdm,
    errors_path: str
):
    results: list[AnalyzedTensorFC] = []

    progress_bar_lock = threading.Lock()
    results_lock = threading.Lock()

    # create the threads
    threads: list[threading.Thread] = []
    for thread_idx in range(args.parallel):
        t = threading.Thread(
            target=analyze_error_tensors_fc,
            args=(
                thread_idx,
                args.parallel,
                progress_bar,
                progress_bar_lock,
                errors,
                golden,
                group_name,
                unit_name,
                errors_path,
                results,
                results_lock
            )
        )
        threads.append(t)
        t.start()
    
    # join the threads
    for t in threads:
        t.join()

    return results

# INDIVIDUAL TENSORS ANALYSIS ---------------------------------------------------------------------------------------------

def analyze_error_tensors_conv(
    thread_idx: int,
    thread_group_size: int,
    pbar: tqdm,
    pbar_lock: threading.Lock,
    errors,
    golden: np.ndarray,
    golden_range_min: float,
    golden_range_max: float,
    group_name: str,
    unit_name: str,
    errors_path: str,
    args: Args,
    shared_results: list[AnalyzedTensorConv],
    shared_results_lock: threading.Lock
):
    local_results = []

    keys = list(errors.keys())
    current_key_idx = 0

    # iterate over the files in the archive
    while current_key_idx < len(keys):
        current_key = keys[current_key_idx]
        current_file = errors[current_key]
        current_shape = current_file.shape

        # get the golden tensor associated to this key
        current_golden: np.ndarray = golden[int(current_key)]
        golden_shape = map_to_coordinates(current_golden.shape)

        # the current file is a 5D tensor with this shape (injection_number, batch_number, C, H, W)
        # we want to iterate through the injection numbers (and always get the first batch number, since it's 1)
        file_idx = thread_idx
        while file_idx < current_shape[0]:
            error: np.ndarray = current_file[file_idx][0]
            error_shape = map_to_coordinates(error.shape)

            # perform error analysis
            if golden_shape != error_shape:
                log.warning(
                    f"Skipping {errors_path} number {current_key} injection {file_idx}. Invalid shape (Faulty has shape: {error_shape}, Golden has shape: {golden_shape})"
                )
            else:
                value_class_count = defaultdict(int)

                # generate a list of all coordinates where a difference is observed (Sparse matrix)
                if args.almost_same:
                    sparse_diff_native_coords = list(zip(*np.nonzero(error - current_golden)))
                else:
                    sparse_diff_native_coords = list(zip(*np.where(np.abs(error - current_golden) >= args.epsilon)))

                tensor_diff = np.zeros(error.shape, dtype=np.int8)    

                # value classification
                for coord in sparse_diff_native_coords:
                    val_class = value_classification(current_golden[coord[0], coord[1], coord[2]], error[coord[0], coord[1], coord[2]], golden_range_min, golden_range_max, args.epsilon, args.almost_same)
                    tensor_diff[coord[0], coord[1], coord[2]] = val_class.value
                    value_class_count[val_class] += 1

                value_class_count[ValueClass.SAME] = current_golden.size - sum(value_class_count.values())

                # proceed only if there are real differences
                if len(sparse_diff_native_coords) != 0:
                    sparse_diff = [
                        map_to_coordinates(numpy_coords_to_python_coord(coords))
                        for coords in sparse_diff_native_coords
                    ]

                    # spatial classifcation
                    spatial_class, pattern_params, faulty_channels = spatial_classification(sparse_diff, golden_shape)
                    domain_class = domain_classification(value_class_count)

                    # create visualization
                    if args.visualize:
                        folder_path = spatial_class.class_folder(args.visualize_path)
                        file_count = len([x for x in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, x))])
                        if args.visualize_limit == 0 or file_count < args.visualize_limit:
                            visualize(
                                tensor_diff,
                                faulty_channels,
                                spatial_class.output_path(
                                    args.visualize_path, f'{group_name}_{unit_name}_{current_key}_{file_idx}'
                                ),
                                save=True,
                                show=False,
                                suptitile=f'{group_name} {unit_name} {current_key} {file_idx} {current_golden.shape[0]}x{current_golden.shape[1]}x{current_golden.shape[2]}',
                                invalidate=True,
                            )

                    # produce analyzed tensor
                    local_results.append(
                        AnalyzedTensorConv(
                            group=group_name,
                            hw_unit=unit_name,
                            error_number=int(current_key),
                            injection_number=file_idx,
                            shape=error.shape,
                            spatial_class=spatial_class,
                            spatial_class_params=pattern_params,
                            value_classes_counts=value_class_count,
                            domain_class=domain_class,
                            corrupted_values_count=len(sparse_diff_native_coords),
                            corrupted_channels_count=len(faulty_channels)
                        )
                    )

            # update progress bar
            pbar_lock.acquire()
            pbar.update()
            pbar_lock.release()

            file_idx += thread_group_size
        
        current_key_idx += 1

    # store final results
    shared_results_lock.acquire()
    shared_results += local_results
    shared_results_lock.release()

def analyze_error_tensors_fc(
    thread_idx: int,
    thread_group_size: int,
    pbar: tqdm,
    pbar_lock: threading.Lock,
    errors,
    golden: np.ndarray,
    group_name: str,
    unit_name: str,
    errors_path: str,
    shared_results: list[AnalyzedTensorConv],
    shared_results_lock: threading.Lock
):
    local_results = []

    keys = list(errors.keys())
    current_key_idx = 0

    while current_key_idx < len(keys):
        current_key = keys[current_key_idx]
        current_file = errors[current_key]
        current_shape = current_file.shape

        current_golden: np.ndarray = golden[int(current_key)]

        file_idx = thread_idx
        while file_idx < current_shape[0]:
            error: np.ndarray = current_file[file_idx][0]
            
            # perform error analysis
            if error.shape != current_golden.shape:
                log.warning(
                    f"Skipping {errors_path} number {current_key} injection {file_idx}. Invalid shape (Faulty has shape: {error.shape}, Golden has shape: {current_golden.shape})"
                )
            else:
                num_equal_elements = np.sum(error == current_golden)
                num_different_elements = error.size - num_equal_elements

                L1_dist = np.linalg.norm(error-current_golden, 1)
                L2_dist = np.linalg.norm(error-current_golden)

                # Per tensor report generator
                local_results.append(
                    AnalyzedTensorFC(
                        group=group_name,
                        hw_unit=unit_name,
                        error_number=int(current_key),
                        injection_number=file_idx,
                        shape=error.shape,
                        corrupted_values_count=num_different_elements,
                        L1_distance=L1_dist,
                        L2_distance=L2_dist
                    )
                )
                
            # update progress bar
            pbar_lock.acquire()
            pbar.update()
            pbar_lock.release()

            file_idx += thread_group_size
        
        current_key_idx += 1

    # store final results
    shared_results_lock.acquire()
    shared_results += local_results
    shared_results_lock.release()

# REPORT GENERATION ---------------------------------------------------------------------------------------------------

def generate_unit_report_conv(unit_dir: str, analyzed_tensors: list[AnalyzedTensorConv]):
    report = OrderedDict()

    report["classified_tensors"] = len(analyzed_tensors)
    report["tensors_by_shape"] = tensor_count_by_shape(analyzed_tensors)
    report["spatial_classes"] = spatial_classes_counts(analyzed_tensors)
    report["class_cardinalites"] = cardinalities_counts_by_sp_class(analyzed_tensors)

    report_path = os.path.join(unit_dir, 'unit_report.json')
    with open(report_path, 'w') as rf:
        json.dump(report, rf, indent=2)

def generate_unit_report_fc(unit_dir: str, analyzed_tensors: List[AnalyzedTensorFC]):
    report = OrderedDict()

    report["classified_tensors"] = len(analyzed_tensors)
    report["tensor_shape"] = analyzed_tensors[0].shape

    # determine min, max, avg and std. dev. of number of corrupted values, L1 distance and L2 distance
    corrupted_values_counts = []
    L1_distances = []
    L2_distances = []

    for tensor in analyzed_tensors:
        corrupted_values_counts.append(tensor.corrupted_values_count)
        L1_distances.append(tensor.L1_distance)
        L2_distances.append(tensor.L2_distance)

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

    report_path = os.path.join(unit_dir, 'unit_report.json')
    with open(report_path, 'w') as rf:
        json.dump(report, rf, indent=2)

def report_uncategorized_tensors(unit_dir: str, analyzed_tensors: list):
    report_path = os.path.join(unit_dir, 'uncategorized_log.csv')

    with open(report_path, 'w', newline='') as csvlog:
        logwriter = csv.writer(csvlog)
        logwriter.writerow(['Type', 'Error Number', 'Injection Number'])

        for tensor in analyzed_tensors:
            if tensor.spatial_class == SpatialClass.SINGLE_CHANNEL_RANDOM:
                logwriter.writerow(['Single', tensor.error_number, str(tensor.injection_number)])
            elif tensor.spatial_class == SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED:
                logwriter.writerow(['Multiple', tensor.error_number, str(tensor.injection_number)])
