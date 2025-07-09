import os
import numpy as np
import logging as log
import threading

import report_utils

from collections import defaultdict

from args import Args
from classes import generate_classes_models
from coordinates import map_to_coordinates, numpy_coords_to_python_coord

from analyzed_tensor import AnalyzedTensorConv, AnalyzedTensorFC
from domain_classifier import ValueClass, domain_classification, value_classification
from spatial_classifier.spatial_classifier import spatial_classification

# ANALYSIS FUNCTIONS ---------------------------------------------------------------------------------------------------

def analyze_hw_unit_directory(unit_dir: str, group_name: str, unit_name: str, args: Args, golden: np.ndarray, golden_range_min: float, golden_range_max: float):
    log.info(f'Starting analysis of {group_name}/{unit_name}')

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

    classified_tensors = []

    # determine layer type based on the golden size
    if len(golden.shape) == 4:
        # assume convolutional
        log.info('Golden is 4D: assuming convolutional layer.')
        classified_tensors = analyze_errors_conv(
            golden=golden,
            golden_range_min=golden_range_min,
            golden_range_max=golden_range_max,
            errors=errors,
            args=args,
            group_name=group_name,
            unit_name=unit_name,
            errors_path=errors_path
        )
        
        log.info(f'Analysis done for {group_name}/{unit_name}')
        # generate classes models and reports
        if args.classes:
            log.info('Generating CLASSES model and unit report.')
            unit_dir = os.path.join(args.classes_output_dir, f'{group_name}_{unit_name}')
            if not os.path.isdir(unit_dir):
                os.makedirs(unit_dir, exist_ok=True)

            generate_classes_models(classified_tensors, args, unit_dir)
            report_utils.generate_unit_report_conv(unit_dir, classified_tensors)

    elif len(golden.shape) == 2:
        # assume fully connected
        log.info('Golden is 2D: assuming fully connected layer.')
        classified_tensors = analyze_errors_fc(
            golden=golden,
            errors=errors,
            args=args,
            group_name=group_name,
            unit_name=unit_name,
            errors_path=errors_path
        )

        log.info(f'Analysis done for {group_name}/{unit_name}')
        unit_dir = os.path.join(args.classes_output_dir, unit_name)
        if not os.path.isdir(unit_dir):
            os.makedirs(unit_dir, exist_ok=True)

        log.info('Generating unit report.')
        report_utils.generate_unit_report_fc(unit_dir, classified_tensors)

    else:
        log.error(f'Skipping {unit_dir} hardware unit directory. Unsupported golden shape {golden.shape}.')
    
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
    errors_path: str
):
    results: list[AnalyzedTensorConv] = []
    results_lock = threading.Lock()

    # create the threads
    threads: list[threading.Thread] = []
    for thread_idx in range(args.parallel):
        t = threading.Thread(
            target=analyze_error_tensors_conv,
            args=(
                thread_idx,
                args.parallel,
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
    errors_path: str
):
    results: list[AnalyzedTensorFC] = []
    results_lock = threading.Lock()

    # create the threads
    threads: list[threading.Thread] = []
    for thread_idx in range(args.parallel):
        t = threading.Thread(
            target=analyze_error_tensors_fc,
            args=(
                thread_idx,
                args.parallel,
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
            log.info(f'File {current_key_idx} - Tensor {file_idx}')
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

                    # produce analyzed tensor
                    local_results.append(
                        AnalyzedTensorConv(
                            group=group_name,
                            hw_unit=unit_name,
                            error_number=int(current_key),
                            injection_number=file_idx,
			                corrupted_channels=faulty_channels,
                            shape=error.shape,
                            spatial_class=spatial_class,
                            spatial_class_params=pattern_params,
                            value_classes_counts=value_class_count,
                            domain_class=domain_class,
                            corrupted_values_count=len(sparse_diff_native_coords),
                            corrupted_channels_count=len(faulty_channels)
                        )
                    )

            # move to the next tensor in the file, skipping the rest of the thread group
            file_idx += thread_group_size
        
        # move to the next file in the archive
        current_key_idx += 1

    # store final results
    shared_results_lock.acquire()
    shared_results += local_results
    shared_results_lock.release()

def analyze_error_tensors_fc(
    thread_idx: int,
    thread_group_size: int,
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
            log.info(f'File {current_key_idx} - Tensor {file_idx}')
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

            file_idx += thread_group_size
        
        current_key_idx += 1

    # store final results
    shared_results_lock.acquire()
    shared_results += local_results
    shared_results_lock.release()